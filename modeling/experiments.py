import dolfin as dl
import ufl
import numpy as np

from .utils import load_mesh
from .models import FKPPVarf, FKPPTXVarf, FKProblem, radioModel, chemoModel


class ApplicationProblem(object):
    """
    Base class for defining an application problem.
    """

    def setup_mesh(self, mesh_path):
        """
        Set up the mesh and geometry for the simulation.
        """
        raise NotImplementedError("Method should be implemented in child class.")

    def setup_function_spaces_and_bcs(self):
        """
        Set up function spaces and boundary conditions for the simulation.
        """
        raise NotImplementedError("Method should be implemented in child class.")

    def setup_variational_problem(self):
        """
        Set up the variational problem for the simulation.
        """
        raise NotImplementedError("Method should be implemented in child class.")


class FisherKPP(ApplicationProblem):
    """Fisher-KPP model for tumor growth."""

    def __init__(self, comm):
        self.comm = comm
        self.mesh = None
        self.Vh = None
        self.Vhmi = None
        self.bc = None
        self.bc0 = None

        # simulation parameters
        self.PHYS_DIM = 2  # physical dimension of the domain (only 2D for now)
        self.STATE_DEGREE = 1  # degree of the state finite element space
        self.PARAM_DEGREE = 1  # degree of the parameter finite element space
        self.D0 = 0.05  # diffusion coefficient [mm^2 / day]
        self.K0 = 0.3  # proliferation rate coefficient [1/day]
        # self.D0 = 0.392  # mean diffusion coefficient [mm^2 / day] from Rockne et al. 2009
        # self.K0 = 0.045  # mean proliferation rate [1/day] from Rockne et al. 2009
        self.RHO_K = 180.0  # proliferation rate correlation length [mm]
        self.RHO_D = 180.0  # diffusion correlation length [mm]
        self.VAR_K = 0.0682  # variance of the proliferation rate
        self.VAR_D = 0.2336  # variance of the diffusion coefficient

    def setup_mesh(self, mesh_fpath: str):
        """Load the mesh."""
        self.mesh = load_mesh(self.comm, mesh_fpath)

    def setup_function_spaces_and_bcs(self):
        """Setup function spaces for state, parameter, and adjoint."""
        Vhu = dl.FunctionSpace(self.mesh, "Lagrange", self.STATE_DEGREE)
        Vhmi = dl.FunctionSpace(self.mesh, "Lagrange", self.PARAM_DEGREE)

        # P1 x P1 for m_d, m_k
        mixed_element = ufl.MixedElement([Vhmi.ufl_element(), Vhmi.ufl_element()])
        Vhm = dl.FunctionSpace(self.mesh, mixed_element)
        self.Vh = [Vhu, Vhm, Vhu]
        self.Vhmi = Vhmi

        # define the boundary conditions
        self.bc = []  # homogeneous Neumann for state
        self.bc0 = []  # homogeneous Neumann for adjoint

    def setup_variational_problem(self, u0, t0, tf, dt, moll=True, clip=False, lumped=True, solver_type: str = "direct", logparam: bool = True):
        """Setup the variational problem for the PDE model."""

        # Set the variational form for the forward model.
        varf = FKPPVarf(dt, moll=moll, lumped=lumped, logparam=logparam)

        # Expecting solver parameters to be set from either CLI or .petscrc
        pde = FKProblem(self.Vh, varf, self.bc, self.bc0, u0, t0, tf, is_fwd_linear=False, clip=clip)

        # set solvers for the linear systems
        if solver_type == "direct":
            pass  # default
        elif solver_type == "iterative":
            pde.solverA = dl.PETScKrylovSolver("cg", "petsc_amg")
            pde.solverAadj = dl.PETScKrylovSolver("cg", "petsc_amg")
            pde.solver_fwd_inc = dl.PETScKrylovSolver("cg", "petsc_amg")
            pde.solver_adj_inc = dl.PETScKrylovSolver("cg", "petsc_amg")

            # Crank down the tolerance of the linear solvers.
            pde.solverA.parameters["relative_tolerance"] = 1e-12
            pde.solverA.parameters["absolute_tolerance"] = 1e-20
            pde.solverAadj.parameters = pde.solverA.parameters
            pde.solver_fwd_inc.parameters = pde.solverA.parameters
            pde.solver_adj_inc.parameters = pde.solverA.parameters
        else:
            raise ValueError(f"Unknown solver type '{solver_type}'. Choose 'direct' or 'iterative'.")

        return pde


class TXFisherKPP(FisherKPP):
    """Treated Fisher-KPP model for tumor growth"""

    def __init__(self, comm):
        super().__init__(comm)

        # treatment parameters, Stupp-like protocol
        self.RT_DOSE = 2.0  # radiotherapy dose [Gy]
        self.RT_ALPHA = 0.025  # LQ model alpha radiosensitivity parameter [1/day]
        self.RT_ALPHA_BETA_RATIO = 10.0  # radiosensitivity parameter ratio
        self.CT_EFFECT = 0.9  # chemotherapy surviving fraction
        self.CT_BETA = 24.0 / 1.8  # clearance rate of the chemotherapy [1/day]

    def _rt_timeline(self, start, end):
        """
        Generate radiotherapy timeline from start to end (inclusive), with 5 days of treatment per 7-day week.
        """
        # Compute all days in the interval
        all_days = np.arange(np.ceil(start), np.floor(end) + 1)
        # Only include days that are in the first 5 days of each week
        rt_days = [day for day in all_days if ((day - start) % 7) < 5]
        return np.array(rt_days)

    def _ct_timeline(self, start, end):
        """Generate chemotherapy timeline from start to end (inclusive), with treatment every day of the week."""
        return np.arange(np.ceil(start), np.floor(end) + 1)

    def setup_stupp_like(self, tx_start, tx_end):
        """Wrap the treatment timeline setup for Stupp-like protocol.
        Radiotherapy is given for 5 days per week, chemotherapy is given every day.
        """
        rt_days = self._rt_timeline(start=tx_start, end=tx_end)
        ct_days = self._ct_timeline(start=tx_start, end=tx_end)
        return rt_days, ct_days

    def setup_variational_problem(self, u0, t0, tf, dt, moll=True, clip=False, lumped=True, solver_type: str = "direct", logparam: bool = True):
        """Setup the variational problem for the PDE model."""

        # set up the treatment
        rt_days, ct_days = self.setup_stupp_like(t0, tf)
        rt_doses = self.RT_DOSE * np.ones_like(rt_days)  # todo: set this to a fixed, but spatially varying map?

        radio_model = radioModel(tx_days=rt_days, tx_doses=rt_doses, alpha=self.RT_ALPHA, alpha_beta_ratio=self.RT_ALPHA_BETA_RATIO)
        chemo_model = chemoModel(ct_days, ct_effect=self.CT_EFFECT, beta=self.CT_BETA)

        # Set the variational form for the forward model.
        varf = FKPPTXVarf(dt, rtmodel=radio_model, ctmodel=chemo_model, moll=moll, lumped=lumped, logparam=logparam)

        # Expecting solver parameters to be set from either CLI or .petscrc
        pde = FKProblem(self.Vh, varf, self.bc, self.bc0, u0, t0, tf, is_fwd_linear=False, clip=clip)

        # set solvers for the linear systems
        if solver_type == "direct":
            pass  # default
        elif solver_type == "iterative":
            pde.solverA = dl.PETScKrylovSolver("cg", "petsc_amg")
            pde.solverAadj = dl.PETScKrylovSolver("cg", "petsc_amg")
            pde.solver_fwd_inc = dl.PETScKrylovSolver("cg", "petsc_amg")
            pde.solver_adj_inc = dl.PETScKrylovSolver("cg", "petsc_amg")

            # Crank down the tolerance of the linear solvers.
            pde.solverA.parameters["relative_tolerance"] = 1e-12
            pde.solverA.parameters["absolute_tolerance"] = 1e-20
            pde.solverAadj.parameters = pde.solverA.parameters
            pde.solver_fwd_inc.parameters = pde.solverA.parameters
            pde.solver_adj_inc.parameters = pde.solverA.parameters
        else:
            raise ValueError(f"Unknown solver type '{solver_type}'. Choose 'direct' or 'iterative'.")

        return pde
