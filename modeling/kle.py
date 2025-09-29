# Rewrite of the KLE class in hIPPYflow
#   see: https://github.com/hippylib/hippyflow/tree/main

import time
import dolfin as dl
import ufl
import numpy as np
import scipy
import hippylib as hp
import matplotlib.pyplot as plt

from .utils import mv_to_dense, root_print


class MassPreconditionedCovarianceOperator:
    def __init__(self, C, M):
        """
        Linear operator representing the mass matrix preconditioned
        covariance matrix :math:`M C M`
        """
        self.C = C
        self.M = M
        self.mpi_comm = self.M.mpi_comm()

        self.Mx = dl.Vector(self.mpi_comm)
        self.CMx = dl.Vector(self.mpi_comm)
        self.M.init_vector(self.Mx, 0)
        self.M.init_vector(self.CMx, 0)

    def init_vector(self, x, dim):
        self.M.init_vector(x, dim)

    def mult(self, x, y):
        self.M.mult(x, self.Mx)
        self.C.mult(self.Mx, self.CMx)
        self.M.mult(self.CMx, y)


class PriorPreconditionedProjector:
    """
    Prior preconditioned projector operator
    :math:`UU^TC^{-1}`
    """

    def __init__(self, U, Cinv, my_init_vector):
        """
        Constructor
            - :code:`U` - Array object must have :code:`dot_v` method implemented
            - :code:`Cinv` - Covariance operator must have :code:`mult` method implemented
            - :code:`my_init_vector` - lambda function for initialization of vectors compatible with :code:`Cinv`
        """
        self.U = U
        self.Cinv = Cinv
        self.my_init_vector = my_init_vector

        self.Cinvx = dl.Vector(self.Cinv.mpi_comm())
        self.my_init_vector(self.Cinvx, 0)

    def init_vector(self, x, dim):
        """
        Initialize :code:`x` to be compatible with the range (:code:`dim=0`) or domain (:code:`dim=1`) of :code:`PriorPreconditionedProjector`.
        """
        self.my_init_vector(x, dim)

    def mult(self, x, y):
        """
        Compute :math:`y = UU^TC^{-1} x`
        """
        self.Cinv.mult(x, self.Cinvx)
        UtCinvx = self.U.dot_v(self.Cinvx)
        y.zero()
        self.U.reduce(y, UtCinvx)


class KLE:
    """
    This class implements an input subspace projector based solely on the prior
    """

    def __init__(self, prior: hp.SqrtPrecisionPDE_Prior, comm):
        """Constructor

        Args:
            prior: The hIPPYlib prior object.
            comm: The MPI communicator.
            rank (int, optional): Requested rank. Defaults to 50.
            oversampling (int, optional): Oversampling for the randomized methods. Defaults to 20.
        """
        self.prior = prior
        self.comm = comm

        # set the covariance operator. R^{-1} is the A M^{-1} A action.
        self.C = hp.Solver2Operator(self.prior.Rsolver, mpi_comm=comm)

        self.evals = None
        self.kle_decoder = None
        self.kle_encoder = None
        self.M_orthogonal = None

    def _randomInputProjector(self, op, rank: int, oversampling: int) -> hp.MultiVector:
        """
        Compute and return a random projection (orthogonalized) basis.
        :code:`WARNING`: This method only works in serial currently.

        Inputs:
            :code:`op`: Operator to be compatible with. Must have an :code:`init_vector` method.

        Returns:
            :code:`Omega` (hp.MultiVector): Projection basis.
        """

        assert hasattr(op, "init_vector"), "Operator must have an init_vector method."

        # Initialize MultiVector of appropriate size.
        m_KLE = dl.Vector(self.comm)
        op.init_vector(m_KLE, 0)
        Omega = hp.MultiVector(m_KLE, rank + oversampling)

        # Generate random vectors and orthogonalize (on root process only).
        hp.parRandom.normal(1.0, Omega)
        Omega.orthogonalize()

        return Omega

    def construct_subspace(self, rank, M_orthogonal=False, oversampling: int = 20):
        """
        This method computes the KLE subspace.
            - :code:`M_orthogonal` - Boolean about whether the vectors are made to be mass matrix orthogonal

        Args:
            M_orthogonal (bool, optional): Whether or not the vectors should be made to be mass matrix orthogonal. Defaults to False.
        """
        start = time.time()

        # Ensure that the prior has the necessary attributes.
        assert hasattr(self.prior, "Rsolver"), "Prior must have an Rsolver attribute. This applies the inverse of the precision operator."
        assert hasattr(self.prior, "M"), "Prior must have an M attribute. This is the mass matrix in control space."
        assert hasattr(self.prior, "Msolver"), "Prior must have an Msolver attribute. This applies the inverse of the mass matrix in control space."

        if M_orthogonal:
            # Mass orthogonal, use the operator MCM = MR^{-1}M.
            KLEop = MassPreconditionedCovarianceOperator(self.C, self.prior.M)

            Omega = self._randomInputProjector(KLEop, rank, oversampling)

            # Solve generalized eigenvalue problem (GEVP) for mass matrix orthogonal basis.
            d_KLE, V_KLE = hp.doublePassG(KLEop, self.prior.M, self.prior.Msolver, Omega, rank, s=1)
            self.evals = d_KLE  # copy eigenvalues
            self.kle_decoder = V_KLE  # copy eigenvectors
            self.M_orthogonal = True

            # Build the projector with the Riesz map.
            self.kle_encoder = hp.MultiVector(self.kle_decoder)
            hp.MatMvMult(self.prior.M, self.kle_decoder, self.kle_encoder)
        else:
            # Identity orthogonal, use the operator C = R^{-1}.
            KLEop = self.C
            Omega = self._randomInputProjector(KLEop, rank, oversampling)

            # Solve eigenvalue problem.
            d_KLE, V_KLE = hp.doublePass(KLEop, Omega, rank, s=1)
            self.evals = d_KLE  # copy eigenvalues
            self.kle_decoder = V_KLE  # copy eigenvectors
            self.M_orthogonal = False

            self.kle_encoder = hp.MultiVector(self.kle_decoder)  # copy constructor

        self._subspace_construction_time = time.time() - start

        root_print(self.comm, f"Construction of input subspace took {self._subspace_construction_time:.2f} seconds")

        # return numpy objects
        return self.evals, mv_to_dense(self.kle_decoder), mv_to_dense(self.kle_encoder)

    def plot_spectrum(self, filepath: str) -> None:
        """Plot the spectrum and write to file.

        Args:
            filepath (str): Path to save spectrum to.
        """

        lambdas = self.evals
        lambdas = lambdas[lambdas > 1e-10]  # truncate below 1e-10

        fig, ax = plt.subplots()

        fig, ax = plt.subplots(figsize=(10, 5))
        indices = np.arange(lambdas.shape[0])
        ax.loglog(indices, lambdas, "kx", linewidth=2)
        ax.set_xlabel("i", fontsize=18)

        ax.set_ylabel("Eigenvalue", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.tick_params(axis="both", which="minor", labelsize=14)
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    def test_errors(self, ranks=[None], nsamples: int = 50, cut_off: float = 1e-12):
        """
        This method implements projection error tests for the KLE basis
            -:code:`ranks` - a python list of ints specifying the ranks for the projection error tests
            -:code:`cut_off` - where to truncate the ranks based on the spectral decay of KLE
        """

        # ranks assumed to be python list with sort in place member function
        ranks.sort()

        # Simple projection test
        if self.evals is None:
            raise ValueError("Input subspace not computed. Please run construct_subspace first.")
        elif len(self.evals) < ranks[-1]:
            root_print(self.comm, "Constructing input subspace because larger rank needed.")
            self.construct_subspace(rank=ranks[-1], M_orthogonal=self.M_orthogonal)
        else:
            root_print(self.comm, "Input subspace already computed proceeding with error tests")

        # Truncate eigenvalues for numerical stability
        numericalrank = np.where(self.evals > cut_off)[-1][-1] + 1  # due to 0 indexing
        ranks = ranks[: np.where(ranks <= numericalrank)[0][-1] + 1]  # due to inclusion

        # Set up numpy arrays to hold the global average relative errors and standard deviations.
        global_avg_rel_errors = np.ones_like(ranks, dtype=np.float64)
        global_std_rel_errors = np.zeros_like(ranks, dtype=np.float64)

        # Set up a multi-vector to store samples from the prior.
        projection_vector = dl.Vector(self.comm)
        self.prior.init_vector(projection_vector, 0)
        prior_samples = hp.MultiVector(projection_vector, nsamples)
        prior_samples.zero()

        # Generate smaples from the prior.
        for i in range(nsamples):
            noise = dl.Vector(self.comm)
            self.prior.init_vector(noise, "noise")
            hp.parRandom.normal(1.0, noise)
            self.prior.sample(noise, prior_samples[i])

        LocalErrors = hp.MultiVector(projection_vector, nsamples)

        for rank_index, rank in enumerate(ranks):
            LocalErrors.zero()  # zero out the erorrs for this rank

            if rank is None:
                # Use the full basis
                V_KLE = self.kle_decoder
                d_KLE = self.evals
            else:
                # Use only the first rank vectors
                V_KLE = hp.MultiVector(self.kle_decoder[0], rank)
                d_KLE = self.evals[0:rank]
                for i in range(rank):
                    V_KLE[i].axpy(1.0, self.kle_decoder[i])

            input_init_vector_lambda = lambda x, dim: self.prior.init_vector(x, dim=1)

            if self.M_orthogonal:
                InputProjectorOperator = PriorPreconditionedProjector(V_KLE, self.prior.M, input_init_vector_lambda)
            else:
                InputProjectorOperator = hp.LowRankOperator(np.ones_like(d_KLE), V_KLE, input_init_vector_lambda)

            rel_errors = np.zeros(LocalErrors.nvec())
            for i in range(LocalErrors.nvec()):
                LocalErrors[i].axpy(1.0, prior_samples[i])
                denominator = LocalErrors[i].norm("l2")
                projection_vector.zero()
                InputProjectorOperator.mult(LocalErrors[i], projection_vector)
                LocalErrors[i].axpy(-1.0, projection_vector)
                numerator = LocalErrors[i].norm("l2")
                rel_errors[i] = numerator / denominator

            avg_rel_error = np.mean(rel_errors)
            global_avg_rel_errors[rank_index] = avg_rel_error
            std_rel_error = np.std(rel_errors)
            global_std_rel_errors[rank_index] = std_rel_error

            root_print(self.comm, f"Naive relative error for rank {rank} is:\t{avg_rel_error:.4e} +/- {std_rel_error:.4e}")

        return global_avg_rel_errors, global_std_rel_errors


class KLEPrior:
    def __init__(self, Vh: dl.FunctionSpace, coeffs: np.ndarray, mean: dl.Vector, Psi: np.ndarray, M_orthogonal: bool = False):
        """Constructor

        Args:
            Vh (dl.FunctionSpace): The function space.
            coeffs (np.ndarray): The eigenvalues of the KLE.
            mean (dl.Vector): The mean vector.
            Psi (np.ndarray): The decoder.
        """
        self.Vh = Vh
        self.coeffs = coeffs
        self.mean = mean
        self.Psi = Psi

        trial = dl.TrialFunction(self.Vh)
        test = dl.TestFunction(self.Vh)

        varfM = ufl.inner(trial, test) * ufl.dx
        self.M = dl.assemble(varfM)

    def sample(self, x: dl.Vector) -> None:
        """Method to sample from the KLE prior.

        Args:
            x (dl.Vector): The vector to write the sample into.
        """
        x.zero()  # zero out the dl.Vector for the sample

        # generate samples from N(0,1)
        xi = np.random.randn(self.coeffs.shape[0])

        # compute sqrt(lambda) * xi
        scalings = np.multiply(np.sqrt(self.coeffs), xi)

        # apply the KLE modes to bring up to the full space
        x.set_local(self.Psi @ scalings)  # FIXME: not MPI safe

        # add mean
        x.axpy(1.0, self.mean)

    def burn_in(self, num_samples: int, manual: bool = True) -> None:
        """Burn-in process to stabilize the random sampling.

        Args:
            num_samples (int): Number of samples to discard.
        """

        if manual:
            for _ in range(num_samples):
                _ = np.random.randn(self.coeffs.shape[0])  # Generate and discard
        else:
            _ = np.random.randn(num_samples, self.coeffs.shape[0])  # Generate and discard all at once for efficiency


class KLEPriorQMC:
    def __init__(self, Vh: dl.FunctionSpace, coeffs: np.ndarray, mean: dl.Vector, Psi: np.ndarray, gen, M_orthogonal: bool = False, shift=None):
        """Constructor

        Args:
            Vh (dl.FunctionSpace): The function space.
            coeffs (np.ndarray): The eigenvalues of the KLE.
            mean (dl.Vector): The mean vector.
            Psi (np.ndarray): The decoder.
            gen: A QMC point generator on [0, 1]^d, d should match coeffs.shape[0].
            shift: a shift, in [0, 1]^d, for the QMC points (added modulo 1 to the points).

        Note we assume the eigenvalues are sorted, this is important for the efficiency of the QMC point generator.
        """
        self.Vh = Vh
        self.coeffs = coeffs
        self.mean = mean
        self.Psi = Psi
        self.gen = gen
        if shift is None:
            shift = np.random.rand(coeffs.shape[0])
        self.shift = shift

        is_sorted = np.all(np.diff(coeffs) <= 0)
        assert is_sorted, "We expect the eigenvalues to be sorted"

        trial = dl.TrialFunction(self.Vh)
        test = dl.TestFunction(self.Vh)

        varfM = ufl.inner(trial, test) * ufl.dx
        self.M = dl.assemble(varfM)

    def sample(self, x: dl.Vector) -> None:
        """Method to sample from the KLE prior.

        Args:
            x (dl.Vector): The vector to write the sample into.
        """
        x.zero()  # zero out the dl.Vector for the sample

        # draw next vector from the QMC generator (the QMC generator should give vectors of the correct dimension)
        xi = scipy.stats.norm.ppf(np.mod(next(self.gen) + self.shift, 1.0))

        # compute sqrt(lambda) * xi
        scalings = np.multiply(np.sqrt(self.coeffs), xi)

        # apply the KLE modes to bring up to the full space
        x.set_local(self.Psi @ scalings)  # FIXME: not MPI safe

        # add mean
        x.axpy(1.0, self.mean)
