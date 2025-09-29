import dolfin as dl
import ufl


def computeNTV(u: dl.Function, threshold: float = 0.5) -> float:
    """Compute the normalized tumor volume of a function u.
    NTV := ( 1 / |Omega| ) * int_{Omega} u dx
    where |Omega| is the volume of the domain Omega.
    The NTV is a measure of the volume of the tumor in the domain Omega.

    Args:
        u (dl.Function): Function containing state.
        threshold (float, optional): Threshold value for state to compute the level set. Defaults to 0.5.

    Returns:
        float: The normalized tumor volume.
    """

    uth = ufl.conditional(ufl.ge(u, threshold), dl.Constant(1.0), dl.Constant(0.0))
    tumor = dl.assemble(uth * ufl.dx)

    dx = ufl.dx(u.function_space().mesh())  # measure of the domain

    vol = dl.assemble(dl.Constant(1.0) * dx)  # of the domain

    return tumor / vol


def computeTTV(u: dl.Function, threshold: float = 0.5) -> float:
    """Compute the total tumor volume of a function u.
    TTV := int_{Omega} u dx
    where |Omega| is the volume of the domain Omega.
    The TTV is a measure of the volume of the tumor in the domain Omega.

    Args:
        u (dl.Function): Function containing state.
        threshold (float, optional): Threshold value for state to compute the level set. Defaults to 0.5.

    Returns:
        float: The total tumor volume.
    """

    uth = ufl.conditional(ufl.ge(u, threshold), dl.Constant(1.0), dl.Constant(0.0))
    ttv = dl.assemble(uth * ufl.dx)

    return ttv


def computeTTC(u: dl.Function, carry_cap: float, threshold: float = None) -> float:
    """Compute the total tumor cellularity.

    Args:
        u (dl.Function): The tumor state.
        carry_cap (float): Carrying capacity [cells/mm^3]
        threshold (float, optional): Threshold value for state to compute level set. Defaults to None.

    Returns:
        float: total tumor cellularity.
    """

    if threshold is not None:
        uth = ufl.conditional(ufl.ge(u, threshold), dl.Constant(1.0), dl.Constant(0.0))
        ttc = dl.assemble(dl.Constant(carry_cap) * uth * u * ufl.dx)
    else:
        ttc = dl.assemble(dl.Constant(carry_cap) * u * ufl.dx)

    return ttc
