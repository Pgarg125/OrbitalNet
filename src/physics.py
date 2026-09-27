"""
physics.py
Two-body orbital mechanics: the governing ODE and ground-truth trajectory
simulation, used to build OrbitalNet's dataset (via SciPy's solve_ivp) and to
evaluate the trained models against numerically-integrated "true" orbits.

Canonical units
----------------
The central gravitational parameter GM is fixed to 1.0 for every trajectory
in this project. In the two-body problem, changing GM is mathematically
equivalent to rescaling length/time units, so fixing GM and instead varying
the initial position and velocity (semi-major axis, eccentricity,
orientation) produces the same diversity of circular and elliptical orbit
shapes -- without needing a 6th network input. This keeps the model
architecture a strict 5 -> ... -> 4 MLP (x0, y0, vx0, vy0, t -> x, y, vx, vy)
exactly as specified in the project synopsis, while still satisfying the
synopsis's intent of "randomised initial conditions ... and central mass M"
(mass is fixed to a canonical value; initial conditions carry the
randomisation). This design choice is documented here so it can be explained
clearly at viva.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

GM = 1.0  # canonical gravitational parameter (fixed across the whole dataset)


def two_body_ode(t: float, state: np.ndarray, gm: float = GM) -> list:
    """
    Right-hand side of the two-body ODE.

    state = [x, y, vx, vy]
    Returns d(state)/dt = [vx, vy, ax, ay], where acceleration follows
    Newton's law of gravitation: a = -GM * r_vec / |r|^3
    """
    x, y, vx, vy = state
    r = np.sqrt(x * x + y * y)
    r3 = r ** 3 + 1e-12  # guard against divide-by-zero at the origin
    ax = -gm * x / r3
    ay = -gm * y / r3
    return [vx, vy, ax, ay]


def sample_initial_condition(
    rng: np.random.Generator,
    a_range: tuple[float, float] = (0.8, 1.5),
    e_range: tuple[float, float] = (0.0, 0.7),
    gm: float = GM,
):
    """
    Sample a random bound orbit (circular when e=0, elliptical when e>0),
    returning its state at periapsis with a random in-plane orientation.

    Uses the vis-viva equation to get a physically-correct speed at
    periapsis for the sampled semi-major axis `a` and eccentricity `e`:
        v_periapsis = sqrt(GM * (2/r_periapsis - 1/a))

    Returns:
        state0: np.ndarray([x0, y0, vx0, vy0])
        period: float, orbital period T = 2*pi*sqrt(a**3 / GM)
    """
    a = rng.uniform(*a_range)
    e = rng.uniform(*e_range)
    phi = rng.uniform(0.0, 2.0 * np.pi)  # random orientation of the orbit in-plane

    r0 = a * (1.0 - e)  # periapsis distance
    v0 = np.sqrt(gm * (2.0 / r0 - 1.0 / a))  # vis-viva equation at periapsis

    # Position at periapsis, rotated by phi around the central body
    x0 = r0 * np.cos(phi)
    y0 = r0 * np.sin(phi)

    # Velocity at periapsis is purely tangential (perpendicular to position)
    vx0 = -v0 * np.sin(phi)
    vy0 = v0 * np.cos(phi)

    period = 2.0 * np.pi * np.sqrt(a ** 3 / gm)
    return np.array([x0, y0, vx0, vy0], dtype=np.float64), float(period)


def simulate_trajectory(
    state0: np.ndarray,
    t_eval: np.ndarray,
    gm: float = GM,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> np.ndarray:
    """
    Integrate the two-body ODE from state0 over the times in t_eval using
    SciPy's solve_ivp (adaptive-step RK45), matching the synopsis's stated
    use of SciPy's ODE solver as the ground-truth data generator.

    Returns:
        np.ndarray of shape (len(t_eval), 4): [x, y, vx, vy] at each time.
    """
    t_eval = np.asarray(t_eval, dtype=np.float64)
    sol = solve_ivp(
        two_body_ode,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=state0,
        t_eval=t_eval,
        args=(gm,),
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed to integrate trajectory: {sol.message}")
    return sol.y.T  # shape (len(t_eval), 4)
