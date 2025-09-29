################################################################################
# This script is used to simulate the forward model for the reaction-diffusion
# tumor growth model. The script will load in a mesh, set up the variational
# problem, and solve the forward model. The script will write out the solution
# trajectory and the parameter field to an output directory.
#
# NOTE: This script is for UNIFORM random fields for the parameters.
# NOTE: the reduced basis prior classes are not MPI parllelized.
################################################################################

import os
import sys
import time
import argparse

import numpy as np
import dolfin as dl

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp

from modeling.experiments import FisherKPP, TXFisherKPP
from modeling.models import MollifiedInitialCondition, setupMisfit
from modeling.sampling import RBPrior, RBPriorQMC
from modeling.latticeseq_b2 import latticeseq_b2
from modeling.utils import root_print, report_mesh_info, mv_to_dense


def main(args) -> None:
    ############################################################
    # 0. General setup.
    ############################################################
    SEP = "\n" + "#" * 80 + "\n"
    dl.set_log_level(dl.LogLevel.WARNING)  # suppress dolfin output

    if args.sampler == "kle":
        np.random.seed(args.seed)  # set the random seed for KLE-based MC sampling
    elif args.sampler == "qmc":
        np.random.seed(args.qmc_shift)  # set the random seed for QMC random shift

    # Paths for data.
    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)  # make the output directory

    L = 100.0  # length of the domain [mm]
    NX = 100  # number of grid points in the x-direction
    NY = 100  # number of grid points in the y-direction
    SEED_WIDTH = 5.0  # width of the seed tumor [mm]
    SEED_CELLULARITY = 0.8  # cellular density of the seed tumor
    t0 = 0.0  # initial time
    tf = args.tf  # final time
    dt = args.dt  # time step size [day]

    NU = args.nu
    sdim = args.sdim

    # MPI setup.
    COMM = dl.MPI.comm_world
    nproc = COMM.size

    # set up the model depending on whether control is applied
    if args.tx:
        prob = TXFisherKPP(COMM)
    else:
        prob = FisherKPP(COMM)

    # Set the prior mean.
    D0 = prob.D0  # mean diffusion coefficient [mm^2 / day]
    K0 = prob.K0  # mean proliferation rate [1/day]

    ############################################################
    # 1. Set up the experiment.
    ############################################################
    root_print(COMM, SEP)
    root_print(COMM, f"There are {nproc} process(es).")
    root_print(COMM, f"Setting up the experiment...")
    root_print(COMM, SEP)

    prob.mesh = dl.RectangleMesh(COMM, dl.Point(0.0, 0.0), dl.Point(L, L), NX, NY)
    report_mesh_info(prob.mesh)

    prob.setup_function_spaces_and_bcs()
    Vh, Vhmi = prob.Vh, prob.Vhmi  # get the function spaces

    # Set up a gaussian bump initial condition.
    # ctr = np.mean(prob.mesh.coordinates(), axis=0)  # center of the mesh.
    # u0_expr = MollifiedInitialCondition(center=ctr, r=SEED_WIDTH, v=SEED_CELLULARITY, degree=2)
    # u0 = dl.interpolate(u0_expr, prob.Vh[hp.STATE])
    u0 = dl.interpolate(dl.Expression("std::exp(-(100*(pow((x[0] - Lx/2)/Lx, 2) + pow((x[1] - Ly/2)/Ly, 2))))", Lx=L, Ly=L, element=Vh[0].ufl_element()), Vh[0])  # Rockne et al. initial condition

    # Set up the variational problem.
    pde = prob.setup_variational_problem(u0, t0, tf, dt, logparam=False)

    ############################################################
    # 2. Set up the sampler.
    ############################################################
    root_print(COMM, f"Generating a parameter sample...")

    n_burn = args.n

    assert args.sampler in ["mc", "qmc"], "Sampler must be one of 'mc', or 'qmc'."

    param_assigner = dl.FunctionAssigner(Vh[hp.PARAMETER], [Vhmi, Vhmi])  # assigner for the mixed space

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

    ############################################################
    # 3. Set up solver objects.
    ############################################################
    u_solver = pde.generate_state()  # to store forward state
    md_sample = dl.Function(Vhmi)
    mk_sample = dl.Function(Vhmi)
    m0fun = dl.Function(Vh[hp.PARAMETER])

    md_prior.sample(md_sample.vector())
    mk_prior.sample(mk_sample.vector())
    param_assigner.assign(m0fun, [md_sample, mk_sample])
    x0 = [u_solver, m0fun.vector(), None]

    ############################################################
    # 4. Solve the forward model, write out the data.
    ############################################################
    root_print(COMM, f"Beginning the forward solve...")

    start = time.perf_counter()
    pde.solveFwd(x0[hp.STATE], x0)
    end = time.perf_counter() - start

    root_print(COMM, f"Forward solve took {end / 60:.2f} minutes.")
    root_print(COMM, f"Writing out the state...")

    pde.exportState(x0[hp.STATE], os.path.join(OUT_DIR, f"state_{n_burn}.xdmf"))

    root_print(COMM, "Writing out the parameter...")
    md_sample.rename("diffusion", "")
    mk_sample.rename("proliferation", "")
    with dl.XDMFFile(COMM, os.path.join(OUT_DIR, f"param_{n_burn}.xdmf")) as fid:
        fid.write(md_sample, 0)
        fid.write(mk_sample, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the forward model.")

    parser.add_argument("--sdim", type=int, help="Dimension of the reduced basis.")
    parser.add_argument("--nu", type=float, default=2.0, help="Regularity of the random field.")

    # Forward model options.
    parser.add_argument("--tx", action=argparse.BooleanOptionalAction, default=True, help="Apply chemoradiation therapy.")
    parser.add_argument("--tf", type=float, default=14.0, help="Final time [day].")
    parser.add_argument("--dt", type=float, default=0.125, help="Time step size [day].")

    # Sampling options.
    parser.add_argument("--sampler", type=str, choices=["mc", "qmc"], default="mc", help="Sampler to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n", type=int, default=0, help="Number of burn-in samples.")
    parser.add_argument("--lattice", type=str, default=None, help="Lattice file for QMC sampling.")
    parser.add_argument("--qmc_shift", type=int, default=0, help="Seed value for the QMC sampling random shift.")

    # Output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")

    args = parser.parse_args()

    main(args)
