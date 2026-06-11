"""
Double pendulum simulator using RK45 integration.

State space: [theta1, theta2, omega1, omega2]
  theta1/2 : angle from vertical (radians)
  omega1/2 : angular velocity (rad/s)

Model input/output uses Cartesian coordinates [x1, y1, x2, y2] so the
network never sees angle wrap-around discontinuities.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class PendulumParams:
    m1: float = 1.0
    m2: float = 1.0
    L1: float = 1.0
    L2: float = 1.0
    g: float = 9.81


def _derivatives(t, state, p):
    theta1, theta2, omega1, omega2 = state
    delta = theta1 - theta2
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    denom = 2 * p.m1 + p.m2 - p.m2 * np.cos(2 * delta)

    domega1 = (
        -(p.g * (2 * p.m1 + p.m2) * np.sin(theta1))
        - p.m2 * p.g * np.sin(theta1 - 2 * theta2)
        - 2 * sin_d * p.m2 * (omega2**2 * p.L2 + omega1**2 * p.L1 * cos_d)
    ) / (p.L1 * denom)

    domega2 = (
        2
        * sin_d
        * (
            omega1**2 * p.L1 * (p.m1 + p.m2)
            + p.g * (p.m1 + p.m2) * np.cos(theta1)
            + omega2**2 * p.L2 * p.m2 * cos_d
        )
    ) / (p.L2 * denom)

    return [omega1, omega2, domega1, domega2]


def simulate(
    theta1_0,
    theta2_0,
    omega1_0=0.0,
    omega2_0=0.0,
    t_end=20.0,
    dt=0.02,
    params=None,
):
    """
    Integrate a single trajectory.

    Returns
    -------
    np.ndarray of shape [T, 4]: columns are [theta1, theta2, omega1, omega2]
    """
    if params is None:
        params = PendulumParams()

    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(
        _derivatives,
        (0.0, t_end),
        [theta1_0, theta2_0, omega1_0, omega2_0],
        t_eval=t_eval,
        args=(params,),
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
    )
    return sol.y.T  # [T, 4]


def generate_trajectories(
    n=500,
    t_end=20.0,
    dt=0.02,
    params=None,
    seed=42,
):
    """
    Sample random initial conditions and simulate n trajectories.

    Returns
    -------
    list of np.ndarray, each [T, 4]
    """
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n):
        theta1 = rng.uniform(-np.pi, np.pi)
        theta2 = rng.uniform(-np.pi, np.pi)
        omega1 = rng.uniform(-2.0, 2.0)
        omega2 = rng.uniform(-2.0, 2.0)
        trajs.append(simulate(theta1, theta2, omega1, omega2, t_end, dt, params))
    return trajs


def to_cartesian(states, params=None):
    """
    Convert angle states [..., 4] to Cartesian joint positions [..., 4].

    Output columns: [x1, y1, x2, y2]
    Normalised so all values lie in [-1, 1] when L1 == L2 == 1.
    """
    if params is None:
        params = PendulumParams()

    theta1 = states[..., 0]
    theta2 = states[..., 1]

    x1 = params.L1 * np.sin(theta1)
    y1 = -params.L1 * np.cos(theta1)
    x2 = x1 + params.L2 * np.sin(theta2)
    y2 = y1 - params.L2 * np.cos(theta2)

    # Normalise to [-1, 1]
    scale = params.L1 + params.L2
    return np.stack([x1 / params.L1, y1 / params.L1, x2 / scale, y2 / scale], axis=-1)
