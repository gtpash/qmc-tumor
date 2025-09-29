################################################################################
# This script pushes forward parametric uncertainty through the forward model.
# The prior is sampled, the forward model is solved, and the quantity of interest (QoI) is computed.
# The script writes out the mean state and the QoI.
#
# NOTE: This script is for UNIFORM random fields for the parameters.
# NOTE: the reduced basis prior classes are not (mesh) MPI parallelized.
# NOTE: output will hang if the computational load is not balanced across MPI ranks.
################################################################################

import os
import sys
import time
import argparse
import math

import numpy as np
import dolfin as dl

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp
from mpi4py import MPI  # import after dolfin to avoid MPI initialization issues

from modeling.experiments import FisherKPP, TXFisherKPP
from modeling.models import MollifiedInitialCondition
from modeling.sampling import RBPrior, RBPriorQMC
from modeling.latticeseq_b2 import latticeseq_b2
from modeling.utils import root_print, allocate_samples_per_proc, MultipleSerialPDECollective, NullCollective, report_mesh_info
from modeling.qoi import computeNTV, computeTTV, computeTTC
from modeling.utils import mv_to_dense


def main(args) -> None:
    ############################################################
    # 0. General setup.
    ############################################################

    # Logging.
    dl.set_log_level(dl.LogLevel.WARNING)  # suppress dolfin output

    if args.sampler == "kle":
        np.random.seed(args.seed)  # set the random seed for KLE-based MC sampling
    elif args.sampler == "qmc":
        np.random.seed(args.qmc_shift)  # set the random seed for QMC random shift

    # Paths for data.
    OUT_DIR = args.outdir

    L = 100.0  # length of the domain [mm]
    NX = 100  # number of grid points in the x-direction
    NY = 100  # number of grid points in the y-direction
    SEED_WIDTH = 5.0  # width of the seed tumor [mm]
    SEED_CELLULARITY = 0.8  # cellular density of the seed tumor
    t0 = 0.0  # initial time
    tf = args.tf  # final time
    dt = args.dt  # time step size [day]
    THRESHOLD = 0.2  # for QoI computation

    # for the reduced basis
    NU = args.nu
    sdim = args.sdim

    # MPI setup.
    COMM_MESH = dl.MPI.comm_self  # FIXME: why does ibrun not like MPI.COMM_SELF
    COMM_SAMPLER = dl.MPI.comm_world  # FIXME: why does ibrun not like MPI.COMM_WORLD

    if COMM_SAMPLER.rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)  # make the output directory

    sample_size_allprocs = allocate_samples_per_proc(args.num_samples, COMM_SAMPLER)
    sample_size_proc = sample_size_allprocs[COMM_SAMPLER.rank]
    n_burn = int(np.sum(sample_size_allprocs[: COMM_SAMPLER.rank]))
    nfwd = int(np.sum(sample_size_allprocs))  # total number of forward solves across all processes

    # set up the model depending on the control option
    if args.tx:
        prob = TXFisherKPP(COMM_MESH)
    else:
        prob = FisherKPP(COMM_MESH)

    # -----------------------------------------------------------
    # 1. Set up the experiment.
    # -----------------------------------------------------------
    prob.mesh = dl.RectangleMesh(COMM_MESH, dl.Point(0.0, 0.0), dl.Point(L, L), NX, NY)
    if COMM_SAMPLER.rank == 0:
        report_mesh_info(prob.mesh)
    prob.setup_function_spaces_and_bcs()
    Vh, Vhmi = prob.Vh, prob.Vhmi  # get the function spaces

    # Set up a gaussian bump initial condition.
    # ctr = np.mean(prob.mesh.coordinates(), axis=0)  # center of the mesh.
    # u0_expr = MollifiedInitialCondition(center=ctr, r=SEED_WIDTH, v=SEED_CELLULARITY, degree=2)
    # u0 = dl.interpolate(u0_expr, prob.Vh[hp.STATE])
    u0 = dl.interpolate(dl.Expression("std::exp(-(100*(pow((x[0] - Lx/2)/Lx, 2) + pow((x[1] - Ly/2)/Ly, 2))))", Lx=L, Ly=L, element=Vh[0].ufl_element()), Vh[0])  # Rockne et al. initial condition

    # Set up the variational problem.
    pde = prob.setup_variational_problem(u0, t0, tf, dt)

    VIZ_FILE = os.path.join(OUT_DIR, f"mean_state.xdmf")
    if os.path.exists(VIZ_FILE):
        print(f"Output file {VIZ_FILE} already exists. Please remove it before running the script.")
        sys.exit(1)

    # -----------------------------------------------------------
    # 2. Set up the prior.
    # -----------------------------------------------------------

    assert args.sampler in ["mc", "qmc"], "Sampler must be one of 'mc', or 'qmc'."

    param_assigner = dl.FunctionAssigner(Vh[hp.PARAMETER], [Vhmi, Vhmi])  # assigner for the mixed space

    D0 = prob.D0  # mean diffusion coefficient [mm^2 / day]
    K0 = prob.K0  # mean proliferation rate [1/day]

    mud = dl.Function(Vhmi)
    mud.assign(dl.Constant(D0))  # mean for diffusion
    muk = dl.Function(Vhmi)
    muk.assign(dl.Constant(K0))  # mean for proliferation

    # build the reduced basis
    tmp = dl.Function(Vhmi)
    decoder = hp.MultiVector(tmp.vector(), sdim)
    for i in range(sdim):
        expr = dl.Expression("0.5*pow(j,-nu)*sin(pi*x[0]*j / Lx)*sin(pi*x[1]*j / Ly)", nu=NU, j=(i + 1), Lx=L, Ly=L, element=Vhmi.ufl_element())
        vv = dl.interpolate(expr, Vhmi)
        decoder[i].axpy(1.0, vv.vector())

    decoder_np = mv_to_dense(decoder)  # convert the decoder to a numpy array
    ones = np.ones(sdim)

    if args.sampler == "mc":
        # Monte Carlo sampling.
        md_prior = RBPrior(Vh=Vhmi, coeffs=ones * D0, mean=mud.vector(), decoder=decoder_np, mode="uniform")
        mk_prior = RBPrior(Vh=Vhmi, coeffs=ones * K0, mean=muk.vector(), decoder=decoder_np, mode="uniform")

        # burn-in.
        md_prior.burn_in(n_burn)
        mk_prior.burn_in(n_burn)
    elif args.sampler == "qmc":
        # Quasi-Monte Carlo sampling.
        lattice_gen_vec = np.loadtxt(args.lattice)[:, -1]  # load lattice point sequence for QMC sampling

        md_qmc_seq = latticeseq_b2(z=lattice_gen_vec[::2][:sdim], kstart=n_burn)
        md_shift = np.random.rand(sdim)  # random shift for QMC sampling
        md_prior = RBPriorQMC(Vh=Vhmi, coeffs=ones * D0, mean=mud.vector(), decoder=decoder_np, gen=iter(md_qmc_seq), shift=md_shift, mode="uniform")

        mk_qmc_seq = latticeseq_b2(z=lattice_gen_vec[1::2][:sdim], kstart=n_burn)
        mk_shift = np.random.rand(sdim)  # random shift for QMC sampling
        mk_prior = RBPriorQMC(Vh=Vhmi, coeffs=ones * K0, mean=muk.vector(), decoder=decoder_np, gen=iter(mk_qmc_seq), shift=mk_shift, mode="uniform")
    else:
        raise ValueError(f"Sampler '{args.sampler}' is not supported. Use 'mc' or 'qmc'.")

    # -----------------------------------------------------------
    # 3. Loop over samples, solve the forward model, and compute the QoI.
    # -----------------------------------------------------------

    u_mean = dl.Function(Vh[hp.STATE])  # to accumulate the mean state
    u_out = dl.Function(Vh[hp.STATE])  # helper function to write out the state
    u_solver = pde.generate_state()  # to store forward state

    md_sample = dl.Function(Vhmi)
    mk_sample = dl.Function(Vhmi)
    m0fun = dl.Function(Vh[hp.PARAMETER])

    # to store the QoI
    qoi_ttv = np.zeros((sample_size_proc, 1))
    qoi_ntv = np.zeros((sample_size_proc, 1))
    qoi_ttc = np.zeros((sample_size_proc, 1))

    # set up the collective for parallel I/O
    if COMM_SAMPLER.Get_size() == 1:
        collective = NullCollective()
    else:
        collective = MultipleSerialPDECollective(COMM_SAMPLER)

    all_start = time.perf_counter()
    for nn in range(sample_size_proc):

        np1 = nn + 1  # number of samples (1-indexed)

        # sample the prior(s).
        md_prior.sample(md_sample.vector())
        mk_prior.sample(mk_sample.vector())
        param_assigner.assign(m0fun, [md_sample, mk_sample])
        x0 = [u_solver, m0fun.vector(), None]

        # Solve the forward model
        try:
            pde.solveFwd(x0[hp.STATE], x0)
        except:
            raise RuntimeError(f"Forward model failed to solve at sample {np1} on Rank {COMM_SAMPLER.rank}.")
        u_mean.vector().axpy(1.0, x0[hp.STATE].view(pde.times[-1]))  # accumulate the end state

        # Compute the QoI
        xfun = hp.vector2Function(x0[hp.STATE].view(pde.times[-1]), Vh[hp.STATE])  # get the state at the final time
        qoi_ttv[nn] = computeTTV(xfun, THRESHOLD)
        qoi_ntv[nn] = computeNTV(xfun, THRESHOLD)
        qoi_ttc[nn] = computeTTC(xfun, carry_cap=1.0)

        # Compute the global sample count
        global_np1 = COMM_SAMPLER.allreduce(np1, op=MPI.SUM)  # FIXME: will this hang on the last sample?

        telapsed = (time.perf_counter() - all_start) / 60.0  # elapsed time in minutes
        root_print(COMM_SAMPLER, f"{global_np1}/{nfwd} samples completed in {telapsed:.2f} minutes (avg: {telapsed / global_np1:.2e} / sample).")
        # todo: checkpoint the qoi results periodically

        # Check if global_np1 is close to the next power of two
        next_power_of_two = 2 ** math.ceil(math.log2(global_np1))
        if abs(global_np1 - next_power_of_two) <= 2:  # Allow a small tolerance
            u_out.vector().zero()  # reset the output function
            u_out.vector().axpy(1.0, u_mean.vector())

            # Reduce the mean state across all processes
            collective.allReduce(u_out.vector(), op="sum")  # FIXME: this will hang if unbalanced allocation of work

            if COMM_SAMPLER.rank == 0:
                # Normalize the mean state
                u_out.vector().vec().scale(1.0 / global_np1)

                print(u_out.vector().get_local())
                print(global_np1)

                # todo: more robust I/O (for later readback with FEniCS)
                # Write the mean state to disk
                with dl.XDMFFile(COMM_MESH, os.path.join(OUT_DIR, f"mean_state.xdmf")) as fid:
                    fid.parameters["functions_share_mesh"] = True
                    fid.parameters["rewrite_function_mesh"] = False
                    fid.write_checkpoint(u_out, "mean", time_step=global_np1, append=True)

    # Gather QoIs on the root process
    qoi_ttv_global = COMM_SAMPLER.gather(qoi_ttv, root=0)
    qoi_ntv_global = COMM_SAMPLER.gather(qoi_ntv, root=0)
    qoi_ttc_global = COMM_SAMPLER.gather(qoi_ttc, root=0)

    if COMM_SAMPLER.rank == 0:
        # Combine results from all processes
        qoi_ttv = np.concatenate(qoi_ttv_global, axis=0)
        qoi_ntv = np.concatenate(qoi_ntv_global, axis=0)
        qoi_ttc = np.concatenate(qoi_ttc_global, axis=0)

        # Save the QoIs
        np.save(os.path.join(OUT_DIR, f"qoi_ttv.npy"), qoi_ttv)
        np.save(os.path.join(OUT_DIR, f"qoi_ntv.npy"), qoi_ntv)
        np.save(os.path.join(OUT_DIR, f"qoi_ttc.npy"), qoi_ttc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pushforward parametric uncertainty.")

    # Forward model options.
    parser.add_argument("--tx", action=argparse.BooleanOptionalAction, default=True, help="Apply chemoradiation therapy.")
    parser.add_argument("--tf", type=float, default=30.0, help="Final time [day].")
    parser.add_argument("--dt", type=float, default=0.125, help="Time step size [day].")

    # Sampling options.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_samples", type=int, default=16384, help="Number of samples to pushforward.")
    parser.add_argument("--lattice", type=str, default=None, help="Lattice file for QMC sampling.")
    parser.add_argument("--sdim", type=int, help="Dimension of the reduced basis.")
    parser.add_argument("--nu", type=float, default=2.0, help="Regularity of the random field.")
    parser.add_argument("--sampler", type=str, choices=["mc", "qmc"], default="mc", help="Sampler to use.")
    parser.add_argument("--qmc_shift", type=int, default=0, help="Seed value for the QMC sampling random shift.")

    # Output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")

    args = parser.parse_args()

    main(args)
