################################################################################
# This script pushes forward parametric uncertainty through the forward model.
# The prior is sampled, the forward model is solved, and the quantity of interest (QoI) is computed.
# The script writes out the mean state and the QoI.
#
# NOTE: the KLE-based prior classes are not (mesh) MPI parallelized.
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
from modeling.kle import KLEPrior, KLEPriorQMC
from modeling.latticeseq_b2 import latticeseq_b2
from modeling.utils import root_print, allocate_samples_per_proc, MultipleSerialPDECollective, NullCollective, report_mesh_info
from modeling.qoi import computeNTV, computeTTV, computeTTC


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
    MESH_FPATH = args.mesh
    OUT_DIR = args.outdir

    SEED_WIDTH = 5.0  # width of the seed tumor [mm]
    SEED_CELLULARITY = 0.8  # cellular density of the seed tumor
    t0 = 0.0  # initial time
    tf = args.tf  # final time
    dt = args.dt  # time step size [day]
    THRESHOLD = 0.2  # for QoI computation # todo: use Otsu method to determine this?

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
    prob.setup_mesh(MESH_FPATH)
    if COMM_SAMPLER.rank == 0:
        report_mesh_info(prob.mesh)
    prob.setup_function_spaces_and_bcs()
    Vh, Vhmi = prob.Vh, prob.Vhmi  # get the function spaces

    # Set up a gaussian bump initial condition.
    ctr = np.mean(prob.mesh.coordinates(), axis=0)  # center of the mesh.
    u0_expr = MollifiedInitialCondition(center=ctr, r=SEED_WIDTH, v=SEED_CELLULARITY, degree=2)
    u0 = dl.interpolate(u0_expr, prob.Vh[hp.STATE])

    # Set up the variational problem.
    pde = prob.setup_variational_problem(u0, t0, tf, dt)

    VIZ_FILE = os.path.join(OUT_DIR, f"mean_state.xdmf")
    if os.path.exists(VIZ_FILE):
        print(f"Output file {VIZ_FILE} already exists. Please remove it before running the script.")
        sys.exit(1)

    # -----------------------------------------------------------
    # 2. Set up the prior.
    # -----------------------------------------------------------

    if args.sampler == "mc":
        # Monte Carlo sampling from the full prior.

        # Set the prior mean.
        mu = dl.Function(Vh[hp.PARAMETER])
        mu.assign(dl.Constant([np.log(prob.D0), np.log(prob.K0)]))

        RHO_D = args.rho_d  # diffusion correlation length [mm]
        RHO_K = args.rho_k  # proliferation correlation length [mm]

        assert RHO_D is not None, "Diffusion correlation length must be specified for MC sampling."
        assert RHO_K is not None, "Proliferation correlation length must be specified for MC sampling."

        # Get the prior coefficients, and set up the prior.
        diff_coeffs = hp.BiLaplacianComputeCoefficients(prob.VAR_D, RHO_D, ndim=prob.PHYS_DIM)
        prolif_coeffs = hp.BiLaplacianComputeCoefficients(prob.VAR_K, RHO_K, ndim=prob.PHYS_DIM)
        mprior = hp.VectorBiLaplacianPrior(
            Vh[hp.PARAMETER], [diff_coeffs[0], prolif_coeffs[0]], [diff_coeffs[1], prolif_coeffs[1]], mean=mu.vector(), robin_bc=True, solver_type="lu"
        )  # Robin BC for boundary correction.
        noise = dl.Vector(COMM_MESH)  # use mesh communicator to create a vector for noise
        mprior.init_vector(noise, "noise")  # initialize the noise vector (this should also ensure appropriate MPI comm is used)
        rng = hp.Random(seed=args.seed + COMM_SAMPLER.rank)

    elif (args.sampler == "kle") or (args.sampler == "qmc"):
        assert args.kledir is not None, "KLE directory must be specified for KLE sampling."
        KLE_DIR = args.kledir

        # set up an assigner for the mixed space.
        param_assigner = dl.FunctionAssigner(Vh[hp.PARAMETER], [Vhmi, Vhmi])

        # Set the prior mean.
        mud = dl.Function(Vhmi)
        mud.assign(dl.Constant(np.log(prob.D0)))
        muk = dl.Function(Vhmi)
        muk.assign(dl.Constant(np.log(prob.K0)))

        # Get the prior coefficients, and set up the prior.
        md_kle_decoder = np.load(os.path.join(KLE_DIR, "diffusion", f"diffusion_decoder.npy"))
        md_evals = np.load(os.path.join(KLE_DIR, "diffusion", f"diffusion_evals.npy"))
        mk_kle_decoder = np.load(os.path.join(KLE_DIR, "reaction", f"reaction_decoder.npy"))
        mk_evals = np.load(os.path.join(KLE_DIR, "reaction", f"reaction_evals.npy"))

        # optionally truncate the KLE basis
        if args.rkle > 0:  # FIXME: allow different ranks for diffusion and reaction
            md_evals = md_evals[: args.rkle]
            md_kle_decoder = md_kle_decoder[:, : args.rkle]
            mk_evals = mk_evals[: args.rkle]
            mk_kle_decoder = mk_kle_decoder[:, : args.rkle]

        if args.sampler == "qmc":
            # Quasi-Monte Carlo sampling.
            lattice_gen_vec = np.loadtxt(args.lattice)[:, -1]  # load lattice point sequence for QMC sampling

            md_qmc_seq = latticeseq_b2(z=lattice_gen_vec[::2][: md_evals.shape[0]], kstart=n_burn)
            md_shift = np.random.rand(md_evals.shape[0])
            md_prior = KLEPriorQMC(Vhmi, md_evals, mud.vector(), md_kle_decoder, gen=iter(md_qmc_seq), shift=md_shift)

            mk_qmc_seq = latticeseq_b2(z=lattice_gen_vec[1::2][: mk_evals.shape[0]], kstart=n_burn)
            mk_shift = np.random.rand(mk_evals.shape[0])
            mk_prior = KLEPriorQMC(Vhmi, mk_evals, muk.vector(), mk_kle_decoder, gen=iter(mk_qmc_seq), shift=mk_shift)
        else:
            # Monte Carlo sampling from the KLE prior.
            md_prior = KLEPrior(Vhmi, md_evals, mud.vector(), md_kle_decoder)
            mk_prior = KLEPrior(Vhmi, mk_evals, muk.vector(), mk_kle_decoder)

            # burn in priors
            md_prior.burn_in(n_burn)
            mk_prior.burn_in(n_burn)

    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    # -----------------------------------------------------------
    # 3. Loop over samples, solve the forward model, and compute the QoI.
    # -----------------------------------------------------------

    u_mean = dl.Function(Vh[hp.STATE])  # to accumulate the mean state
    u_out = dl.Function(Vh[hp.STATE])  # helper function to write out the state
    u_solver = pde.generate_state()  # to store forward state

    if args.sampler in ["kle", "qmc"]:
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
        if args.sampler == "mc":
            rng.normal(1.0, noise)
            mprior.sample(noise, m0fun.vector())  # sample from the prior
            x0 = [u_solver, m0fun.vector(), None]
        else:
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

    # Directories for data.
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")

    # Forward model options.
    parser.add_argument("--tx", action=argparse.BooleanOptionalAction, default=True, help="Apply chemoradiation therapy.")
    parser.add_argument("--tf", type=float, default=30.0, help="Final time [day].")
    parser.add_argument("--dt", type=float, default=0.125, help="Time step size [day].")
    parser.add_argument("--rho_d", type=float, help="Diffusion correlation length [mm].")
    parser.add_argument("--rho_k", type=float, help="Proliferation correlation length [mm].")

    # Sampling options.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_samples", type=int, default=16384, help="Number of samples to pushforward.")
    parser.add_argument("--kledir", type=str, default=None, help="Directory to KLE basis.")
    parser.add_argument("--lattice", type=str, default=None, help="Lattice file for QMC sampling.")
    parser.add_argument("--rkle", type=int, default=-1, help="Number of KLE modes to use. If -1, use all modes.")
    parser.add_argument("--sampler", type=str, choices=["mc", "kle", "qmc"], default="mc", help="Sampler to use.")
    parser.add_argument("--qmc_shift", type=int, default=0, help="Seed value for the QMC sampling random shift.")

    # Output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")

    args = parser.parse_args()

    main(args)
