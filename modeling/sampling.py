import dolfin as dl
import numpy as np
import scipy


class RBPrior:
    def __init__(self, Vh: dl.FunctionSpace, coeffs: np.ndarray, mean: dl.Vector, decoder: np.ndarray, mode="normal"):
        """Constructor

        Args:
            Vh (dl.FunctionSpace): The function space.
            coeffs (np.ndarray): The eigenvalues of the KLE.
            mean (dl.Vector): The mean vector.
            decoder (np.ndarray): The reduced basis decoder.
            base (str, optional): The base distribution for sampling. Defaults to "normal".
        """
        self.Vh = Vh
        self.coeffs = coeffs
        self.mean = mean
        self.decoder = decoder
        assert mode in ["normal", "uniform"], "Base distribution must be either 'normal' or 'uniform'."
        self.mode = mode

    def sample(self, x: dl.Vector) -> None:
        """Method to sample from the KLE prior.

        Args:
            x (dl.Vector): The vector to write the sample into.
        """
        x.zero()  # zero out the dl.Vector for the sample

        if self.mode == "normal":
            xi = np.random.randn(self.coeffs.shape[0])  # generate samples from N(0,1)
            scalings = np.multiply(np.sqrt(self.coeffs), xi)  # compute sqrt(lambda) * xi
        else:
            xi = np.random.uniform(-0.5, 0.5, self.coeffs.shape[0])  # generate samples from U(-0.5, 0.5)
            scalings = np.multiply(self.coeffs, xi)  # compute lambda * xi

        # apply the KLE modes to bring up to the full space
        x.set_local(self.decoder @ scalings)  # FIXME: not MPI safe

        # add mean
        x.axpy(1.0, self.mean)

    def burn_in(self, num_samples: int) -> None:
        """Burn-in process to stabilize the random sampling.

        Args:
            num_samples (int): Number of samples to discard.
        """

        # Generate and discard all at once for efficiency
        if self.mode == "normal":
            _ = np.random.randn(num_samples, self.coeffs.shape[0])
        else:
            _ = np.random.uniform(-0.5, 0.5, (num_samples, self.coeffs.shape[0]))


class RBPriorQMC:
    def __init__(self, Vh: dl.FunctionSpace, coeffs: np.ndarray, mean: dl.Vector, decoder: np.ndarray, gen, shift=None, mode="normal"):
        """Constructor

        Args:
            Vh (dl.FunctionSpace): The function space.
            coeffs (np.ndarray): The eigenvalues of the KLE.
            mean (dl.Vector): The mean vector.
            decoder (np.ndarray): The decoder.
            gen: A QMC point generator on [0, 1]^d, d should match coeffs.shape[0].
            shift: a shift, in [0, 1]^d, for the QMC points (added modulo 1 to the points).
            base (str, optional): The base distribution for sampling. Defaults to "normal".

        Note we assume the eigenvalues are sorted, this is important for the efficiency of the QMC point generator.
        """
        self.Vh = Vh
        self.coeffs = coeffs
        self.mean = mean
        self.decoder = decoder
        self.gen = gen
        if shift is None:
            shift = np.random.rand(coeffs.shape[0])
        self.shift = shift

        self.mode = mode

        is_sorted = np.all(np.diff(coeffs) <= 0)
        assert is_sorted, "We expect the eigenvalues to be sorted"

    def sample(self, x: dl.Vector) -> None:
        """Method to sample from the KLE prior.

        Args:
            x (dl.Vector): The vector to write the sample into.
        """
        x.zero()  # zero out the dl.Vector for the sample

        # draw next vector from the QMC generator (the QMC generator should give vectors of the correct dimension)
        if self.mode == "normal":
            xi = scipy.stats.norm.ppf(np.mod(next(self.gen) + self.shift, 1.0))  # Phi^{-1} ( frac( iz/N + \Delta ) )
            scalings = np.multiply(np.sqrt(self.coeffs), xi)  # compute sqrt(lambda) * xi
        else:
            xi = np.mod(next(self.gen) + self.shift, 1.0) - 0.5  # frac( iz/N + \Delta ) - 1/2
            scalings = np.multiply(self.coeffs, xi)  # compute lambda * xi

        # apply the KLE modes to bring up to the full space
        x.set_local(self.decoder @ scalings)  # FIXME: not MPI safe

        # add mean
        x.axpy(1.0, self.mean)
