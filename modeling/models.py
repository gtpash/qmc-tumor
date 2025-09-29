import dolfin as dl
import ufl
import hippylib as hp
import numpy as np


class radioModel:
    """Class to compute the radiotherapy effect."""

    def __init__(self, tx_days: np.array, tx_doses: np.array, alpha: float = 3.0, alpha_beta_ratio: float = 10.0):
        self.tx_days = tx_days  # therapy days
        self.tx_doses = tx_doses  # therapy doses

        assert len(self.tx_days) == len(self.tx_doses), "Number of therapy days and doses must match."

        self.alpha = alpha  # radiosensitivity parameter alpha
        self.beta = alpha / alpha_beta_ratio  # radiosensitivity parameter beta

    def get_tx_factor(self, cur_t: float) -> dl.Constant:
        """Check if the current time is a radiotherapy time.
        If it is, returns the radiotherapy factor.
        This assumes an instantaneous effect.

        NOTE: this is not FEniCS differentiable wrt control varaible.
        """
        if cur_t in self.tx_days:
            # Therapy is applied at this time.
            idx = np.where(self.tx_days == cur_t)[0][0]  # get the index of the therapy
            dose = self.tx_doses[idx]

            # compute LQ model effect
            dose_factor = 1.0 - np.exp(-self.alpha * dose - self.beta * dose**2)
            return dl.Constant(dose_factor)
        else:
            # no therapy.
            return dl.Constant(0.0)


class chemoModel:
    """Class to compute the chemotherapy effect."""

    def __init__(self, tx_days: np.array, ct_effect: float, beta: float):
        self.tx_days = tx_days  # therapy days
        self.ct_effect = ct_effect  # chemotherapy effect (surviving fraction)
        self.beta = beta  # clearance rate

    def get_tx_factor(self, cur_t: float) -> dl.Constant:
        """Apply the decaying exponential chemotherapy effect.
        NOTE: this is not FEniCS differentiable wrt control varible.
        """
        time_since_applied = cur_t - self.tx_days

        if np.any(time_since_applied > 0):
            active_time = time_since_applied[time_since_applied >= 0]
            ct_decay = np.exp(-self.beta * active_time)  # clearance term
            ct_factor = (1.0 - self.ct_effect) * np.sum(ct_decay)  # 1 - surviving fraction
            return dl.Constant(ct_factor)
        else:
            return dl.Constant(0.0)


class FKPPVarf:
    """Variational form for the Fisher-KPP model of tumor growth.
    du/dt = div(exp(m1)*grad(u)) + exp(m2)*u*(1 - u)
    """

    def __init__(self, dt: float, lumped: bool = False, nudge: float = 1e-14, quad_degree: int = 5, moll: bool = True, logparam: bool = True):
        """Constructor.

        Args:
            dt (float): Time step [days].
            lumped (bool, optional): Whether or not to use mass lumping. Defaults to False.
            nudge (float, optional): Nudge parameter for mollification. Defaults to 1e-14.
            quad_degree (int, optional): Quadrature degree for integration. Defaults to 5.
            moll (bool, optional): Whether or not to mollify the reaction term. Defaults to True.
            logparam (bool, optional): Whether or not to use log-scale parameters for diffusion and reaction. Defaults to True.
        """
        self._dt = dt
        self.dt_inv = dl.Constant(1.0 / dt)
        self.nudge = nudge
        self.dX = ufl.dx(metadata={"quadrature_degree": quad_degree})
        self.moll = moll
        self.logparam = logparam

        if lumped:
            self.dX = ufl.dx(scheme="vertex", metadata={"quadrature_degree": 1, "representation": "quadrature"})

    @property
    def dt(self):
        return self._dt

    def diffusion(self, d, u, p):
        if self.logparam:
            diff = ufl.exp(d)
        else:
            diff = d
        return diff * ufl.inner(ufl.grad(u), ufl.grad(p)) * self.dX

    def reaction(self, k, u, p):
        if self.logparam:
            kappa = ufl.exp(k)
        else:
            kappa = k

        if self.moll:
            moll = (u + ufl.sqrt(u**2 + dl.Constant(self.nudge))) / 2  # mollify to enforce positivity
        else:
            moll = u

        return kappa * moll * (dl.Constant(1.0) - u) * p * self.dX

    # def mollify_clamp(u, eps):
    #     u_pos = 0.5 * (u + ufl.sqrt(u**2 + eps))
    #     u_clamped = 1.0 - 0.5 * ((1.0 - u_pos) + ufl.sqrt((1.0 - u_pos)**2 + eps))
    #     return u_clamped

    def __call__(self, u, u_old, m, p, t):
        d, k = ufl.split(m)

        return (u - u_old) * p * self.dt_inv * self.dX + self.diffusion(d, u, p) - self.reaction(k, u, p)


class FKPPTXVarf(FKPPVarf):
    """Variational form for the controlled Fisher-KPP model of tumor growth with radiotherapy and chemotherapy.
    du/dt = div(exp(m1)*grad(u)) + exp(m2)*u*(1 - u) - radio(u) - chemo(u)
    The reaction term is mollified to avoid spurious oscillations.
    """

    def __init__(self, dt: float, rtmodel: radioModel, ctmodel: chemoModel, lumped: bool = False, nudge: float = 1e-14, quad_degree: int = 5, moll: bool = True, logparam: bool = True):
        """Constructor

        Args:
            dt (float): Time step.
            rtmodel (radioModel): Radiotherapy model.
            ctmodel (chemoModel): Chemotherapy model.
            nudge (float, optional): Nudge parameter for mollification. Defaults to 1e-14.
            quad_degree (int, optional): Quadrature degree for integration. Defaults to 5.
        """
        super().__init__(dt, lumped=lumped, nudge=nudge, quad_degree=quad_degree, moll=moll, logparam=logparam)
        self.rtmodel = rtmodel  # radiotherapy model
        self.ctmodel = ctmodel  # chemotherapy model

    def radio(self, u, p, t):
        """Return the radiotherapy effect."""
        rteffect = dl.Constant(self.rtmodel.get_tx_factor(t))
        return self.dt_inv * rteffect * u * p * self.dX

    def chemo(self, u, p, t):
        """Return the radiotherapy effect."""
        cteffect = dl.Constant(self.ctmodel.get_tx_factor(t))
        return self.dt_inv * cteffect * u * p * self.dX

    def __call__(self, u, u_old, m, p, t):
        d, k = ufl.split(m)

        # be careful with the signs, we are in residual form LHS = 0
        return (u - u_old) * p * self.dt_inv * self.dX + self.diffusion(d, u, p) - self.reaction(k, u, p) + self.radio(u, p, t) + self.chemo(u, p, t)


def samplePrior(prior: hp.SqrtPrecisionPDE_Prior, n: int = 1, seed=None) -> dl.Vector:
    """Wrapper to sample from a :code:`hIPPYlib` prior.

    Args:
        prior: :code:`hIPPYlib` prior object.
        n: How long to burn in the RNG. Defaults to 1. Useful for drawing multiple samples from the same seed.
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        dl.Vector: sample from prior.
    """

    # Get a random normal sample.
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    if seed is not None:
        rng = hp.Random(seed=seed)
        for _ in range(n):
            # burn in the smapler.
            rng.normal(1.0, noise)
    else:
        for _ in range(n):
            # burn in the sampler.
            hp.parRandom.normal(1.0, noise)

    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)

    return mtrue


def MollifiedInitialCondition(center: list, r: float, v: float = 0.5, degree: int = 1) -> dl.Expression:
    """Gaussian blob initial condition.
        f(x) = a * exp( -(x-c)^2 / (2 b^2) )
            a: amplitude
            c: center
            b: width

    Args:
        center (list): Center location.
        r (float): _description_
        v (float, optional): Roughly the maximum value. Defaults to 0.5.
        degree (int, optional): Degree for dl.Expression interpolation. Defaults to 1.

    Raises:
        ValueError: If an unsupported geometric dimension is provided.

    Returns:
        dl.Expression: Dolfin Expression for the mollifier initial condition.
    """

    assert len(center) == 2, f"Geometric dimension does not equal coordinates in 'center'. Dimension should be 2."

    return dl.Expression("a * exp( -(pow(x[0]-cx,2)+pow(x[1]-cy,2) )/ (2*(pow(b,2))) )", cx=center[0], cy=center[1], a=v, b=r, degree=degree)


class FKProblem(hp.TimeDependentPDEVariationalProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, u0, t_init, t_final, is_fwd_linear=False, clip: bool = True):
        """
        Tailored class for the Fisher-KPP problem.
        Optional clipping to ensure positivity in the solution. NOTE: This method is not suitable when adjoint calculations are required.
        """
        super().__init__(Vh, varf_handler, bc, bc0, u0, t_init, t_final, is_fwd_linear)

        assert is_fwd_linear is False, "FKProblem is not suitable for linear forward problems."
        self.comm = self.mesh.mpi_comm()
        self.clip = clip

        self.parameters = dl.NonlinearVariationalSolver.default_parameters()
        self.parameters["nonlinear_solver"] = "snes"
        self.parameters["snes_solver"]["absolute_tolerance"] = 1e-16
        self.parameters["snes_solver"]["relative_tolerance"] = 1e-12
        self.parameters["snes_solver"]["maximum_iterations"] = 100
        self.parameters["snes_solver"]["report"] = False

    def solveFwd(self, out, x):
        """
        Solve the possibly nonlinear time dependent Fwd Problem:
        Given m, find u such that
        \delta_p F(u,m,p;\hat_p) = 0 \for all \hat_p
        """
        out.zero()

        if self.solverA is None:
            self.solverA = self._createLUSolver()

        u_old = dl.Function(self.Vh[hp.STATE])
        u_old.vector().axpy(1.0, self.init_cond.vector())
        out.store(u_old.vector(), self.t_init)

        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        u = dl.Function(self.Vh[hp.STATE])
        dp = dl.TestFunction(self.Vh[hp.ADJOINT])

        u.assign(u_old)

        u_old_old = dl.Function(self.Vh[hp.STATE])  # for 3-point extrapolation in time
        u_old_old.vector().axpy(1, u_old.vector())

        for t in self.times[1:]:

            # Richardson exptrapolation for initial guess, u = 2u_old - u_old_old
            u.vector().zero()
            u.vector().axpy(2.0, u_old.vector())
            u.vector().axpy(-1.0, u_old_old.vector())

            # set up residual form and solve
            res_form = self.varf(u, u_old, m, dp, t)
            self._set_time(self.fwd_bc, t)
            dl.solve(res_form == 0, u, self.fwd_bc, solver_parameters=self.parameters)

            # clip the solution
            if self.clip:
                tmp = dl.project(ufl.max_value(u, dl.Constant(0.0)), self.Vh[hp.STATE], solver_type="cg", preconditioner_type="jacobi")
                u.vector().zero()
                u.vector().axpy(1.0, tmp.vector())

            # store solution, update u_old and u_old_old
            out.store(u.vector(), t)
            u_old_old.assign(u_old)
            u_old.assign(u)


def setupMisfit(pde, x):
    """Set up the misfit object for the time-dependent problem.

    Args:
        pde: The PDE object.
        x: The list of solution vectors.

    Returns:
        misfit: The misfit object.
    """
    REL_NOISE = 0.01
    max_state = x[hp.STATE].norm("linf", "linf")
    noise_std_dev = REL_NOISE * max_state

    # Set up misfit object.
    misfits = []
    for t in pde.times:
        misfit_t = hp.ContinuousStateObservation(pde.Vh[hp.STATE], ufl.dx, pde.adj_bc)
        misfit_t.d.axpy(1.0, x[hp.STATE].view(t))
        hp.parRandom.normal_perturb(noise_std_dev, misfit_t.d)
        misfit_t.noise_variance = noise_std_dev * noise_std_dev
        misfits.append(misfit_t)

    misfit = hp.MisfitTD(misfits, pde.times)

    return misfit
