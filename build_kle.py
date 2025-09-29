# This script is used to generate a Karhunen-Loeve expansion co/basis
# for the diffusion and proliferation rate parameters.
# The script will load in a mesh, set up the variational problem, and solve the forward model.
#
# NOTE: The parameters are modeled independently, so we generate separate bases.
# NOTE: This script is designed to be run in SERIAL.
#
# An example call to this script is:
# python3 -u build_kle.py --mesh meshes/box.xdmf --outdir ./kle/ --prefix box

import os
import sys
import argparse

import dolfin as dl
import numpy as np

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp

from modeling.utils import root_print, report_mesh_info
from modeling.kle import KLE
from modeling.experiments import FisherKPP


def main(args) -> None:
    ############################################################
    # 0. General setup.
    ############################################################
    SEP = "\n" + "#" * 80 + "\n"
    PREFIX = f"{args.prefix}_" if args.prefix is not None else ""

    # Logging.
    dl.set_log_level(dl.LogLevel.WARNING)  # suppress dolfin output
    COMM = dl.MPI.comm_world

    # Paths for data.
    MESH_FPATH = args.mesh
    KLE_DIR = args.outdir
    prob = FisherKPP(COMM)

    # Simulation parameters.
    PHYS_DIM = prob.PHYS_DIM  # physical dimension of the domain (only 2D for now)

    # override the diffusion and proliferation correlation lengths from CLI
    RHO_D = args.rho_d  # proliferation rate correlation length [mm]
    RHO_K = args.rho_k  # diffusion correlation length [mm]

    VAR_K = prob.VAR_K  # variance of the proliferation rate
    VAR_D = prob.VAR_D  # variance of the diffusion coefficient

    M_ORTH = False  # mass matrix orthogonalization for KLE
    RANK = args.rkle  # rank of the KLE expansion
    OVERSAMPLE = args.oversampling  # oversampling factor for the randomized eigensolver

    ############################################################
    # 1. Load mesh and define function spaces.
    ############################################################
    root_print(COMM, SEP)
    root_print(COMM, f"Loading in the mesh...")
    root_print(COMM, SEP)

    prob.setup_mesh(MESH_FPATH)

    root_print(COMM, f"Successfully loaded the mesh.")
    root_print(COMM, f"There are {COMM.size} process(es).")
    report_mesh_info(prob.mesh)

    #  Set up variational spaces for state and parameter.
    prob.setup_function_spaces_and_bcs()
    mu = dl.Function(prob.Vhmi)  # Set the prior mean.
    mu.assign(dl.Constant(0.0))  # Mean should be zero for KLE, add mean to samples later.

    ############################################################
    # 2. Set up the KLE for diffusion.
    ############################################################
    root_print(COMM, f"Setting up the prior for diffusion...")
    OUT_DIR = os.path.join(KLE_DIR, "diffusion")
    os.makedirs(OUT_DIR, exist_ok=True)
    root_print(COMM, f"Reduced bases will be saved to:\t{OUT_DIR}")

    # Get the prior coefficients, and set up the prior.
    diff_coeffs = hp.BiLaplacianComputeCoefficients(VAR_D, RHO_D, ndim=PHYS_DIM)
    mprior = hp.BiLaplacianPrior(prob.Vhmi, diff_coeffs[0], diff_coeffs[1], mean=mu.vector(), robin_bc=True)  # Robin BC for boundary correction.

    # Compute the KL expansion.
    diff_kle = KLE(prior=mprior, comm=COMM)
    diff_evals, diff_kle_decoder, diff_kle_encoder = diff_kle.construct_subspace(RANK, M_orthogonal=M_ORTH, oversampling=OVERSAMPLE)

    # Write output.
    np.save(os.path.join(OUT_DIR, f"{PREFIX}diffusion_evals.npy"), diff_evals)
    np.save(os.path.join(OUT_DIR, f"{PREFIX}diffusion_decoder.npy"), diff_kle_decoder)
    np.save(os.path.join(OUT_DIR, f"{PREFIX}diffusion_encoder.npy"), diff_kle_encoder)
    diff_kle.plot_spectrum(os.path.join(OUT_DIR, f"{PREFIX}diffusion_spectrum.png"))

    # Perform orthogonality check.
    root_print(COMM, "Orthogonality check for the diffusion KLE:")
    diff_kle.test_errors(ranks=[RANK])
    root_print(COMM, SEP)

    ############################################################
    # 2. Set up the KLE for the proliferation rate.
    ############################################################
    root_print(COMM, f"Setting up the prior for the reaction term...")
    OUT_DIR = os.path.join(KLE_DIR, "reaction")
    os.makedirs(OUT_DIR, exist_ok=True)
    root_print(COMM, f"Reduced bases will be saved to:\t{OUT_DIR}")

    # Proliferation second.
    react_coeffs = hp.BiLaplacianComputeCoefficients(VAR_K, RHO_K, ndim=PHYS_DIM)
    mprior = hp.BiLaplacianPrior(prob.Vhmi, react_coeffs[0], react_coeffs[1], mean=mu.vector(), robin_bc=True)  # Robin BC for boundary correction.

    # Compute the KL expansion.
    react_kle = KLE(prior=mprior, comm=COMM)
    react_evals, react_kle_decoder, react_kle_encoder = react_kle.construct_subspace(RANK, M_orthogonal=M_ORTH, oversampling=OVERSAMPLE)

    # Write output.
    np.save(os.path.join(OUT_DIR, f"{PREFIX}reaction_evals.npy"), react_evals)
    np.save(os.path.join(OUT_DIR, f"{PREFIX}reaction_decoder.npy"), react_kle_decoder)
    np.save(os.path.join(OUT_DIR, f"{PREFIX}reaction_encoder.npy"), react_kle_encoder)
    react_kle.plot_spectrum(os.path.join(OUT_DIR, f"{PREFIX}reaction_spectrum.png"))

    # Perform orthogonality check.
    root_print(COMM, "Orthogonality check for the diffusion KLE:")
    react_kle.test_errors(ranks=[RANK])
    root_print(COMM, SEP)

    # Write output for visualization.
    if args.viz:
        os.makedirs(os.path.join(KLE_DIR, "viz"), exist_ok=True)
        with dl.XDMFFile(COMM, os.path.join(KLE_DIR, "viz", f"{PREFIX}diffusion_kle.xdmf")) as fid:
            fid.parameters["functions_share_mesh"] = True
            fid.parameters["rewrite_function_mesh"] = False
            for i in range(RANK):
                fid.write(hp.vector2Function(diff_kle.kle_decoder[i], prob.Vhmi, name="kle"), i)

        with dl.XDMFFile(COMM, os.path.join(KLE_DIR, "viz", f"{PREFIX}reaction_kle.xdmf")) as fid:
            fid.parameters["functions_share_mesh"] = True
            fid.parameters["rewrite_function_mesh"] = False
            for i in range(RANK):
                fid.write(hp.vector2Function(react_kle.kle_decoder[i], prob.Vhmi, name="kle"), i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct KLE basis for model parameters.")

    # Directories for data.
    parser.add_argument("--mesh", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory.")
    parser.add_argument("--prefix", type=str, default=None, help="Name prefix for the output files.")
    parser.add_argument("--rkle", type=int, default=100, help="Number of requested KLE modes.")
    parser.add_argument("--oversampling", type=int, default=10, help="Oversampling factor for randomized eigensolver.")

    parser.add_argument("--rho_d", type=float, default=180.0, help="Diffusion correlation length [mm].")
    parser.add_argument("--rho_k", type=float, default=180.0, help="Proliferation correlation length [mm].")

    parser.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True, help="XDMF output for visualization.")

    args = parser.parse_args()

    main(args)
