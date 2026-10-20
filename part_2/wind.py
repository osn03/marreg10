"""
Wind template (Project Part 2)

Part 2 extends the Part 1 wind model.  In addition to the mean and slowly
varying wind speed, the wind field must now contain

* a fluctuating **gust** component of the wind speed, generated from a wind
  spectrum of your choice (Harris, Davenport, NORSOK, ...), and
* a **slowly varying wind direction**, without a gust component, bounded to
  at most +/- 5 degrees around the mean direction.

You built such a wind generator in Assignment 2 -- reuse it.  The coefficient
lookup (relative wind -> loads) is unchanged from Part 1; only the ambient
wind field (speed and direction as functions of time) becomes richer.

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
             {"U": ambient speed (total), "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad),
              "U_mean": mean + slowly varying speed, "U_gust": gust speed}
             The Part 2 checks read "U_mean", "U_gust" and "beta_ned" to
             verify your wind field (gust present, direction within limit),
             so fill them in.
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
    """Template for the Part 2 wind model.

    Constructor contract -- the automated checks (``python check.py``,
    ``pytest``) and the Part 2 runner construct your model with this
    signature, so keep it working.  It is a superset of the Part 1 contract:
    your Part 1 model can be pasted in and extended.

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., tau_slow=...,
             gust=..., gust_params=..., sigma_dir=..., tau_dir=...,
             dir_limit=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : mean direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention --
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly varying wind speed
        component [m/s] (Part 1; 0 disables it).
    tau_slow : time constant of the slow speed variation [s].
    gust : switch the fluctuating (gust) speed component on.  The gust must
        be zero-mean and follow a wind spectrum; the spectrum and its
        parameters are your choice (see Assignment 2).  ``False`` gives the
        Part 1 wind field, which the checks use to isolate the gust.
    gust_params : dict with the parameters of *your* gust model (spectrum
        type, elevation, turbulence intensity, number of harmonics, ...).
        ``None`` means your own defaults.
    sigma_dir : standard deviation of the slowly varying wind direction
        [rad] (0 disables it).  The direction has NO gust component.
    tau_dir : time constant of the slow direction variation [s].
    dir_limit : hard bound on the direction excursion |beta(t) - beta| [rad];
        the project requires at most 5 degrees.
    seed : random seed for all stochastic components, so runs are
        reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, gust: bool = False,
                 gust_params: Dict | None = None, sigma_dir: float = 0.0,
                 tau_dir: float = 300.0, dir_limit: float = np.deg2rad(5.0),
                 seed: int | None = None):
        # TODO: Store and use the parameters above in step().
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)

        self.gust = bool(gust)
        self.gust_params = dict(gust_params or {})
        
        self.sigma_dir = float(sigma_dir)
        self.tau_dir = float(tau_dir)
        self.dir_limit = float(dir_limit)
        
        self.seed = seed

        if self.semantics not in ("from", "towards"):
            raise ValueError("semantics must be 'from' or 'towards'")
        
        if self.mean_speed < 0.0:
            raise ValueError("mean_speed must be nonnegative")
        
        if self.sigma_slow < 0.0 or self.sigma_dir < 0.0:
            raise ValueError("standard deviations must be nonnegative")
        
        if self.tau_slow <= 0.0 or self.tau_dir <=0:
            raise ValueError("time constants must be positive")
        
        if not 0.0 <= self.dir_limit <= np.deg2rad(5.0):
            raise ValueError("dir_limit must be between 0 and 5 degrees")
        #creating the reproducible rng for the object 
        seeds = np.random.SeedSequence(self.seed).spawn(3)

        self._rng_slow = np.random.default_rng(seeds[0])
        self._rng_dir = np.random.default_rng(seeds[1]) 
        self._rng_gust = np.random.default_rng(seeds[2])

        self._slow_speed = 0.0
        self._slow_direction = 0.0

        self._alpha_deg, self._coefficients = load_wind_coefficients()
        # using reference hight at 10m as in the compendium
        self._gust_height = float(self.gust_params.get("height", 10.0))
        # number of frequency components in the sprectrum
        self._gust_n_components = int(self.gust_params.get("n_components",100))
        # Creating the fail safes
        if self._gust_height <= 0.0:
            raise ValueError("gust height must be positive")

        if self._gust_n_components <2:
            raise ValueError("n_components must be at least 2")

        # choosing frequency range for gusts in Hz
        self._gust_f_min = float(self.gust_params.get("f_min", 0.0001))
        self._gust_f_max = float(self.gust_params.get("f_max", 1.0))

        if not 0.0 < self._gust_f_min < self._gust_f_max:
            raise ValueError("require 0 < f_min < f_max")

        # dividing the frequency range into intervals
        frequency_edges = np.geomspace(self._gust_f_min, self._gust_f_max, self._gust_n_components + 1,)
        self._gust_frequencies = np.sqrt(frequency_edges[:-1] * frequency_edges[1:])
        self._gust_df = np.diff(frequency_edges)

        # calculating the NORSOK spectrum at our frequencies, n is empirically determined and constant for the Norsok
        if self.gust and self.mean_speed > 0.0:
            u10 = self.mean_speed
            z = self._gust_height
            f = self._gust_frequencies
            n = 0.468

            x = (172.0*f * (z/10.0)**(2.0/3.0) * (u10/10.0)**(-3.0/4.0))

            spectrum = (320.0 * (u10/10.0)**2 * (z/10.0)**0.45 /(1.0+x**n)**(5.0/(3.0*n)))

            self._gust_amplitudes = np.sqrt(2.0 * spectrum * self._gust_df)
        else:
            self._gust_amplitudes = np.zeros(self._gust_n_components)

        self._gust_phases = self._rng_gust.uniform(0.0, 2.0 * np.pi, self._gust_n_components,)



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

            self._slow_speed = (decay * self._slow_speed + scale * self._rng_slow.standard_normal())
        if self.sigma_dir > 0.0 and dt > 0.0:
            decay_dir = np.exp(-dt/self.tau_dir)

            scale_dir = self.sigma_dir * np.sqrt(-np.expm1(-2.0 * dt / self.tau_dir))

            self._slow_direction = (decay_dir * self._slow_direction + scale_dir * self._rng_dir.standard_normal())

            self._slow_direction = float(np.clip(self._slow_direction, -self.dir_limit, self.dir_limit,))

        gust_speed = float(np.sum(self._gust_amplitudes * np.cos(2.0 *np.pi * self._gust_frequencies * t + self._gust_phases)))
        mean_speed = self.mean_speed + self._slow_speed
        speed = max(0.0, mean_speed + gust_speed)

        beta_ned = self.beta + self._slow_direction

        if self.semantics == "from":
            beta_ned += np.pi

        beta_ned = beta_ned % (2.0 * np.pi)
        angle = beta_ned - eta[5]

        relative_u = speed * np.cos(angle) - nu[0]
        relative_v = speed * np.sin(angle) - nu[1]

        speed_squared = relative_u**2 + relative_v**2

        alpha_body = np.arctan2(relative_v, relative_u)
        alpha_body = alpha_body % (2.0 * np.pi)

        alpha_deg = np.degrees(alpha_body)

        index = np.searchsorted(self._alpha_deg, alpha_deg, side = "right") - 1

        index = np.clip(index, 0, len(self._alpha_deg) - 2)

        weight = ((alpha_deg - self._alpha_deg[index]) / (self._alpha_deg[index + 1] - self._alpha_deg[index]))

        coefficients = ((1.0 - weight) * self._coefficients[index] + weight * self._coefficients[index + 1])

        tau_w6 = speed_squared * coefficients

        info = {
            "U": speed,
            "beta_ned": beta_ned,
            "alpha_body": alpha_body,
            "U_mean": mean_speed,
            "U_gust": gust_speed,
        }

        return tau_w6, info


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    wind = Wind(
        mean_speed=10.0,
        beta=np.deg2rad(45.0),
        semantics="from",
        sigma_slow=0.5,
        tau_slow=120.0,
        gust=True,
        sigma_dir=np.deg2rad(2.0),
        tau_dir=300.0,
        seed=123,
    )
    dt = 0.1
    duration = 10000.0
    time  = np.arange(0.0, duration, dt)

    eta = np.zeros(6)
    nu = np.zeros(6)

    wind_speed = np.zeros(len(time))
    mean_speed_history = np.zeros(len(time))
    gust_history = np.zeros(len(time))
    direction_history = np.zeros(len(time))
    loads = np.zeros((len(time),6))

    for i, t in enumerate(time):
        step_dt = 0.0 if i == 0 else dt
        tau_w6, info = wind.step(t, step_dt, eta, nu)

        wind_speed[i] = info["U"]
        mean_speed_history[i] = info["U_mean"]
        gust_history[i] = info["U_gust"]
        direction_history[i] = info["beta_ned"]
        loads[i, :] = tau_w6

    mean_direction = wind.beta
    if wind.semantics == "from":
        mean_direction += np.pi
    
    direction_deviation = np.degrees((direction_history - mean_direction + np.pi) % (2.0 * np.pi) - np.pi)

    fig, axes = plt.subplots(3, 1, figsize = (10,8), sharex = True)
    
    
    axes[0].plot(time, wind_speed, label="Total wind", linewidth=0.8)
    axes[0].plot(
        time, mean_speed_history,
        label="Mean + slow variation", linewidth=1.2,
    )
    axes[0].set_ylabel("Wind speed [m/s]")
    axes[0].legend()

    axes[1].plot(time, gust_history, linewidth=0.8)
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].set_ylabel("Gust speed [m/s]")

    axes[2].plot(time, direction_deviation, linewidth=0.8)
    limit_deg = np.degrees(wind.dir_limit)
    axes[2].axhline(limit_deg, color="red", linestyle="--")
    axes[2].axhline(-limit_deg, color="red", linestyle="--")
    axes[2].set_ylabel("Direction deviation [deg]")
    axes[2].set_xlabel("Time [s]")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_loads, axes_loads = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True
    )

    indices = [0, 1, 5]
    labels = [
        "Surge force [kN]",
        "Sway force [kN]",
        "Yaw moment [kNm]",
    ]

    for ax, index, label in zip(axes_loads, indices, labels):
        ax.plot(time, loads[:, index] / 1000.0, linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes_loads[-1].set_xlabel("Time [s]")
    fig_loads.tight_layout()

    wind_test = Wind(
        mean_speed=10.0,
        beta=0.0,
        semantics="from",
        sigma_slow=0.0,
        gust=False,
        sigma_dir=0.0,
    )

    eta_test = np.zeros(6)
    nu_test = np.zeros(6)

    tau_test, info_test = wind_test.step(
        t=0.0,
        dt=0.0,
        eta=eta_test,
        nu=nu_test,
    )

    print("\nConstant wind from north:")
    print("Relative angle [deg]:", np.degrees(info_test["alpha_body"]))
    print("Surge [kN]:", tau_test[0] / 1000.0)
    print("Sway [kN]:", tau_test[1] / 1000.0)
    print("Yaw [kNm]:", tau_test[5] / 1000.0)
    eta_test[5] = np.deg2rad(90.0)

    tau_test, info_test = wind_test.step(
        t=0.0,
        dt=0.0,
        eta=eta_test,
        nu=nu_test,
    )

    print("\nWind from north, vessel heading east:")
    print("Relative angle [deg]:", np.degrees(info_test["alpha_body"]))
    print("Surge [kN]:", tau_test[0] / 1000.0)
    print("Sway [kN]:", tau_test[1] / 1000.0)
    print("Yaw [kNm]:", tau_test[5] / 1000.0)
    eta_test[5] = 0.0
    nu_test[0] = 2.0

    tau_test, info_test = wind_test.step(
        t=0.0,
        dt=0.0,
        eta=eta_test,
        nu=nu_test,
    )

    print("\nHeadwind, vessel moving forward at 2 m/s:")
    print("Surge [kN]:", tau_test[0] / 1000.0)
    settings = {
        "mean_speed": 10.0,
        "beta": np.deg2rad(45.0),
        "sigma_slow": 0.5,
        "gust": True,
        "sigma_dir": np.deg2rad(2.0),
        "seed": 123,
    }

    wind_a = Wind(**settings)
    wind_b = Wind(**settings)

    eta_check = np.zeros(6)
    nu_check = np.zeros(6)
    identical = True

    for i, t in enumerate(time):
        step_dt = 0.0 if i == 0 else dt

        tau_a, info_a = wind_a.step(t, step_dt, eta_check, nu_check)
        tau_b, info_b = wind_b.step(t, step_dt, eta_check, nu_check)

        if not np.array_equal(tau_a, tau_b) or info_a != info_b:
            identical = False
            break

    print("\nSame seed gives identical results:", identical)
    wind_on = Wind(**settings)
    wind_off = Wind(**{**settings, "gust": False})

    slow_identical = True
    gust_disabled = True

    for i, t in enumerate(time):
        step_dt = 0.0 if i == 0 else dt

        _, info_on = wind_on.step(t, step_dt, eta_check, nu_check)
        _, info_off = wind_off.step(t, step_dt, eta_check, nu_check)

        if (
            info_on["U_mean"] != info_off["U_mean"]
            or info_on["beta_ned"] != info_off["beta_ned"]
        ):
            slow_identical = False

        if info_off["U_gust"] != 0.0:
            gust_disabled = False

    print("Slow processes unchanged:", slow_identical)
    print("Gust disabled:", gust_disabled)
    expected_variance = 0.5 * np.sum(wind._gust_amplitudes ** 2)
    measured_variance = np.var(gust_history)

    print("\nGust mean [m/s]:", np.mean(gust_history))
    print("Expected gust variance [(m/s)^2]:", expected_variance)
    print("Measured gust variance [(m/s)^2]:", measured_variance)

    from scipy.signal import welch

    frequencies, estimated_spectrum = welch(
        gust_history,
        fs=1.0 / dt,
        window="hann",
        nperseg=8192,
        detrend="constant",
        scaling="density",
    )

    target_spectrum = (
        wind._gust_amplitudes ** 2
        / (2.0 * wind._gust_df)
    )

    fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 5))

    ax_spectrum.loglog(
        frequencies[1:],
        estimated_spectrum[1:],
        label="Estimated from time series",
    )
    ax_spectrum.loglog(
        wind._gust_frequencies,
        target_spectrum,
        label="Discretized NORSOK",
    )

    ax_spectrum.set_xlim(wind._gust_f_min, wind._gust_f_max)
    ax_spectrum.set_xlabel("Frequency [Hz]")
    ax_spectrum.set_ylabel(r"PSD [(m/s)$^2$/Hz]")
    ax_spectrum.grid(True, which="both", alpha=0.3)
    ax_spectrum.legend()
    fig_spectrum.tight_layout()

    band_edges = [0.0001, 0.01, 0.1, 1.0]
    welch_df = frequencies[1] - frequencies[0]

    print("\nVariance per frequency band [(m/s)^2]:")

    for lower, upper in zip(band_edges[:-1], band_edges[1:]):
        target_mask = (
            (wind._gust_frequencies >= lower)
            & (wind._gust_frequencies < upper)
        )
        estimated_mask = (
            (frequencies >= lower)
            & (frequencies < upper)
        )

        target_variance = 0.5 * np.sum(
            wind._gust_amplitudes[target_mask] ** 2
        )
        estimated_variance = (
            np.sum(estimated_spectrum[estimated_mask]) * welch_df
        )

        print(
            f"{lower:.3f}–{upper:.3f} Hz: "
            f"target={target_variance:.4f}, "
            f"estimated={estimated_variance:.4f}"
        )

    print("\nSpectrum discretization check:")

    for n_components in [50, 100, 200, 400]:
        wind_resolution = Wind(
            mean_speed=10.0,
            gust=True,
            gust_params={
                "n_components": n_components,
                "f_min": 0.0001,
                "f_max": 1.0,
            },
            seed=123,
        )

        variance = 0.5 * np.sum(
            wind_resolution._gust_amplitudes ** 2
        )

        print(
            f"N={n_components:3d}: "
            f"variance={variance:.6f} (m/s)^2"
        )

    print("\nFrequency range check:")

    frequency_ranges = [
        (0.001, 1.0),
        (0.0001, 1.0),
        (0.001, 10.0),
        (0.0001, 10.0),
    ]

    for f_min, f_max in frequency_ranges:
        wind_range = Wind(
            mean_speed=10.0,
            gust=True,
            gust_params={
                "n_components": 1000,
                "f_min": f_min,
                "f_max": f_max,
            },
            seed=123,
        )

        variance = 0.5 * np.sum(
            wind_range._gust_amplitudes ** 2
        )

        print(
            f"{f_min:.4f}–{f_max:.1f} Hz: "
            f"variance={variance:.6f} (m/s)^2"
        )
    plt.show()

