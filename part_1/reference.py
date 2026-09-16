"""
Reference template

Students should filter or shape commanded setpoints before they are sent to
the controller. The simulator calls, once per step:

    ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; leave the rest zero.

Inputs:
    t       : current simulation time [s]
    dt      : time step [s]
    eta_cmd : (6,) commanded setpoint
              (use N_cmd = eta_cmd[0], E_cmd = eta_cmd[1], psi_cmd = eta_cmd[5])

Outputs (all NED-frame, (6,) each):
    eta_ref : filtered reference
              (fill in N_ref = [0], E_ref = [1], psi_ref = [5])
    nu_ref  : reference velocities
              (fill in Ndot_ref = [0], Edot_ref = [1], psidot_ref = [5])
    acc_ref : reference accelerations
              (fill in Nddot_ref = [0], Eddot_ref = [1], psiddot_ref = [5])

The simulator forwards all three to the controller, so a smooth reference
model here directly enables velocity/acceleration feedforward there.
"""
from typing import Tuple
import numpy as np

# Per-axis tuning parameters live with the rest of the Part 1 configuration.
from part_1.config import RefAxisConfig, default_ref_psi, default_ref_xy

def _wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ReferenceModel:
    """
    Critically damped second-order reference filter, one per controlled axis.

    The position axes (N, E) and the heading axis are tuned separately:
    position is limited by sway thrust, heading is not, so heading runs at
    twice the position bandwidth. See ``default_ref_xy()`` /
    ``default_ref_psi()`` in part_1/config.py for the values and the
    reasoning behind them.

    The defaults are taken from those factories rather than from bare
    ``RefAxisConfig()``, so the automated checks — which construct
    ``ReferenceModel(dt)`` with no further arguments — exercise the same
    tuning as the simulations.
    """

    def __init__(
        self,
        dt: float,
        cfg_xy: RefAxisConfig | None = None,
        cfg_psi: RefAxisConfig | None = None,
    ):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else default_ref_xy()
        self.cfg_psi = cfg_psi if cfg_psi is not None else default_ref_psi()
        self.eta_ref = np.zeros(6)
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def reset(self, eta0: np.ndarray) -> None:
        """Initialize the reference at the vessel's current (6,) state."""
        self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)



    def step_axis(self,
                  index: int,
                  cmd: float,
                  dt: float,
                  cfg: RefAxisConfig,
                  is_angle: bool = False) -> None:

        wn = cfg.wn
        zeta = cfg.zeta
        rate_limit = cfg.rate_limit

        error = cmd - self.eta_ref[index]
        if is_angle:
            error = _wrap_pi(error)          # FIX 1: always turn the short way

        vel_old = self.nu_ref[index]

        acc = wn**2 * error - 2 * zeta * wn * vel_old
        vel_new = vel_old + acc * dt

        if rate_limit is not None and abs(vel_new) > rate_limit:
            vel_new = np.sign(vel_new) * rate_limit

        self.acc_ref[index] = (vel_new - vel_old) / dt   # FIX 2: new minus old
        self.nu_ref[index] = vel_new
        self.eta_ref[index] += vel_new * dt
        if is_angle:
            self.eta_ref[index] = _wrap_pi(self.eta_ref[index])


        
    def step(
        self, t: float, dt: float, eta_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO: Replace this pass-through placeholder with your reference model.

        cmd = np.asarray(eta_cmd, dtype=float).reshape(6)
        self.step_axis(0, cmd[0], dt, self.cfg_xy)  # N
        self.step_axis(1, cmd[1], dt, self.cfg_xy)  # E
        self.step_axis(5, cmd[5], dt, self.cfg_psi, is_angle=True)  # psi
        return self.eta_ref, self.nu_ref, self.acc_ref
