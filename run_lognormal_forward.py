################################################################################
# This script is used to simulate the forward model for the reaction-diffusion
# tumor growth model. The script will load in a mesh, set up the variational
# problem, and solve the forward model. The script will write out the solution
# trajectory and the parameter field to an output directory.
# Optionally, it can also perform a finite difference check of the adjoint.
#
# NOTE: the KLE-based prior classes are not MPI parllelized.
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
from modeling.kle import KLEPrior, KLEPriorQMC
from modeling.latticeseq_b2 import latticeseq_b2
from modeling.utils import root_print, report_mesh_info


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
    MESH_FPATH = args.mesh
    OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)  # make the output directory

    SEED_WIDTH = 5.0  # width of the seed tumor [mm]
    SEED_CELLULARITY = 0.8  # cellular density of the seed tumor
    t0 = 0.0  # initial time
    tf = args.tf  # final time
    dt = args.dt  # time step size [day]

    # MPI setup.
    COMM = dl.MPI.comm_world
    nproc = COMM.size

    # set up the model depending on whether control is applied
    if args.tx:
        prob = TXFisherKPP(COMM)
    else:
        prob = FisherKPP(COMM)

    ############################################################
    # 1. Set up the experiment.
    ############################################################
    root_print(COMM, SEP)
    root_print(COMM, f"There are {nproc} process(es).")
    root_print(COMM, f"Setting up the experiment...")
    root_print(COMM, SEP)

    prob.setup_mesh(MESH_FPATH)
    report_mesh_info(prob.mesh)
    prob.setup_function_spaces_and_bcs()
    Vh, Vhmi = prob.Vh, prob.Vhmi  # get the function spaces

    # Set up a gaussian bump initial condition.
    ctr = np.mean(prob.mesh.coordinates(), axis=0)  # center of the mesh.
    u0_expr = MollifiedInitialCondition(center=ctr, r=SEED_WIDTH, v=SEED_CELLULARITY, degree=2)
    u0 = dl.interpolate(u0_expr, prob.Vh[hp.STATE])

    # Set up the variational problem.
    pde = prob.setup_variational_problem(u0, t0, tf, dt)

    ############################################################
    # 2. Set up the sampler.
    ############################################################
    root_print(COMM, f"Generating a parameter sample...")

    n_burn = args.n

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
        noise = dl.Vector(COMM)  # use mesh communicator to create a vector for noise
        mprior.init_vector(noise, "noise")  # initialize the noise vector (this should also ensure appropriate MPI comm is used)
        rng = hp.Random(seed=args.seed)

        # burn in the prior.
        for _ in range(n_burn):
            rng.normal(1.0, noise)

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

    ############################################################
    # 3. Set up solver objects.
    ############################################################
    u_solver = pde.generate_state()  # to store forward state
    if args.sampler in ["kle", "qmc"]:
        md_sample = dl.Function(Vhmi)
        mk_sample = dl.Function(Vhmi)
    m0fun = dl.Function(Vh[hp.PARAMETER])

    if args.sampler == "mc":
        rng.normal(1.0, noise)
        mprior.sample(noise, m0fun.vector())  # sample from the prior
        x0 = [u_solver, m0fun.vector(), None]
    else:
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

    if args.sampler == "mc":
        md_sample, mk_sample = m0fun.split(deepcopy=True)
    md_sample.rename("diffusion", "")
    mk_sample.rename("proliferation", "")
    with dl.XDMFFile(COMM, os.path.join(OUT_DIR, f"param_{n_burn}.xdmf")) as fid:
        fid.write(md_sample, 0)
        fid.write(mk_sample, 0)

    ############################################################
    # Optionally, perform a FD check of the adjoint.
    ############################################################
    if args.adjoint:
        root_print(COMM, f"Performing finite difference check of data-misfit adjoint...")

        misfit = setupMisfit(pde, x0)
        model = hp.Model(pde, mprior, misfit)  # create hIPPYlib model object.

        # Run the FD check.
        eps_list = np.logspace(-10, -1, num=10)
        start = time.perf_counter()
        eps, err_grad, err_H = hp.modelVerify(model, mu.vector(), is_quadratic=False, misfit_only=True, verbose=True, eps=eps_list)
        end = time.perf_counter()

        root_print(COMM, f"FD check took {(end - start) / 60:.2f} minutes.")
        root_print(COMM, SEP)

        import matplotlib.pyplot as plt

        plt.savefig(os.path.join(OUT_DIR, f"modelVerify.png"))

    root_print(COMM, SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the forward model.")

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
    parser.add_argument("--n", type=int, default=0, help="Number of burn-in samples.")
    parser.add_argument("--kledir", type=str, default=None, help="Directory to KLE basis.")
    parser.add_argument("--lattice", type=str, default=None, help="Lattice file for QMC sampling.")
    parser.add_argument("--rkle", type=int, default=-1, help="Number of KLE modes to use. If -1, use all modes.")
    parser.add_argument("--sampler", type=str, choices=["mc", "kle", "qmc"], default="mc", help="Sampler to use.")
    parser.add_argument("--qmc_shift", type=int, default=0, help="Seed value for the QMC sampling random shift.")

    # Output options.
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--adjoint", action=argparse.BooleanOptionalAction, default=False, help="Test the data-misfit adjoint.")

    args = parser.parse_args()

    main(args)
