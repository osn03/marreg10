"""
Wind template

Students should compute generalized BODY-frame wind loads:
    tau_w6 = [Fx, Fy, Fz, Mx, My, Mz]

The simulator uses the 3-DOF subset [Fx, Fy, Mz] = tau_w6 indices [0, 1, 5]
and calls, once per step:

    wind.step(t, dt, eta, nu) -> (tau_w6, info)

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (gust spectra, time variation)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5])
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (RELATIVE wind: compute the loads from V_rw = V_wind - V_vessel,
            using the horizontal components nu[0], nu[1])

Outputs:
    tau_w6 : (6,) BODY loads
    info   : optional dict for logging, e.g.
             {"U": ambient speed, "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad)}
             Return {} (or None) if you do not need it.
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from" —
             convert before logging, do not log the raw constructor value.

Wind coefficient data
---------------------
The vessel wind coefficients C(alpha) = [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] are
provided in `data/wind_coeff.csv` (repository root), tabulated against the relative
wind angle alpha in degrees (0..360). Load them with:

    alpha_deg, C6 = load_wind_coefficients()

The wind loads are then computed as F_wind = U_rw^2 * C(alpha_rw), where U_rw
and alpha_rw are the relative wind speed and angle in the BODY frame.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

_WIND_COEFF_FILE = Path(__file__).resolve().parent.parent / "data" / "wind_coeff.csv"


def load_wind_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the vessel wind coefficient table.

    Returns
    -------
    alpha_deg : (M,) ndarray
        Relative wind angle grid [deg], from 0 to 360.
    C6 : (M, 6) ndarray
        Coefficients [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] at each angle.
    """
    table = np.loadtxt(_WIND_COEFF_FILE, delimiter=",", skiprows=1)
    return table[:, 0], table[:, 1:]


class Wind:
    """Template for student wind model.

    Constructor contract — the automated checks (``python check.py``,
    ``pytest``, ``notebooks/part_1_demo.ipynb``) construct your model with
    this signature, so keep it working:

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention —
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly-varying wind speed
        component [m/s] (required in Part 1; 0 disables it).
    tau_slow : time constant of the slow variation [s].
    seed : random seed for the slow component, so runs are reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, seed: int | None = None):
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.seed = seed
        if semantics not in ("from", "towards"):
            raise ValueError("semantics must be 'from' or 'towards'")
        if self.sigma_slow < 0.0 or self.tau_slow <= 0.0:
            raise ValueError("sigma_slow must be nonnegative and tau_slow positive")
        self._rng = np.random.default_rng(seed)
        self._slow_speed = 0.0
        self._alpha_deg, self._coefficients = load_wind_coefficients()

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.sigma_slow > 0.0 and dt > 0.0:
            decay = np.exp(-dt / self.tau_slow)
            scale = self.sigma_slow * np.sqrt(-np.expm1(-2.0 * dt / self.tau_slow))
            self._slow_speed = decay * self._slow_speed + scale * self._rng.standard_normal()

        speed = max(0.0, self.mean_speed + self._slow_speed)
        beta_ned = (self.beta + (np.pi if self.semantics == "from" else 0.0)) % (2.0 * np.pi)
        angle = beta_ned - eta[5]
        relative_u = speed * np.cos(angle) - nu[0]
        relative_v = speed * np.sin(angle) - nu[1]
        speed_squared = relative_u * relative_u + relative_v * relative_v
        alpha_body = np.arctan2(relative_v, relative_u) % (2.0 * np.pi)
        alpha_deg = np.degrees(alpha_body)
        index = np.searchsorted(self._alpha_deg, alpha_deg, side="right") - 1
        index = np.clip(index, 0, len(self._alpha_deg) - 2)
        weight = (alpha_deg - self._alpha_deg[index]) / (self._alpha_deg[index + 1] - self._alpha_deg[index])
        coefficients = (1.0 - weight) * self._coefficients[index] + weight * self._coefficients[index + 1]
        tau_w6 = speed_squared * coefficients
        info = {"U": speed, "beta_ned": beta_ned, "alpha_body": alpha_body}
        return tau_w6, info

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Loading the coefficient table.
    alpha_deg, coefficients = load_wind_coefficients()

    # A dense network of angles to show the linear interpolation.
    angles = np.linspace(0.0, 360.0, 1441)

    # Only cover surge, sway and yaw, corresponding to the 3-DOF model.
    indices = [0, 1, 5]
    labels = [r"$C_x$", r"$C_y$", r"$C_\psi$"]
    units = [
        r"N s$^2$/m$^2$",
        r"N s$^2$/m$^2$",
        r"N m s$^2$/m$^2$",
    ]

    # Størrelse og skrift tilpasset én kolonne i rapporten.
    # Adapted to the report
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    fig, axes = plt.subplots(
        3, 1,
        figsize=(3.5, 4.5),
        sharex=True,
    )

    for ax, index, label, unit in zip(axes, indices, labels, units):
        interpolated = np.interp(
            angles,
            alpha_deg,
            coefficients[:, index],
        )

        ax.plot(
            angles,
            interpolated,
            linewidth=1,
            label="Linear interpolation",
        )

        ax.plot(
            alpha_deg,
            coefficients[:, index],
            "o",
            markersize=2.5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Table values",
        )

        ax.set_ylabel(f"{label}\n[{unit}]")
        ax.set_xlim(0, 360)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper center", fontsize=6)
    axes[-1].set_xticks(np.arange(0, 361, 60))
    axes[-1].set_xlabel(
        r"Relative wind angle $\alpha_{rw}$ [deg]"
    )

    fig.tight_layout()

    # Saving the figure in the project folder.
    output_dir = (
        Path(__file__).resolve().parent.parent
        / "results_part1"
        / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / "wind_coefficients.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()