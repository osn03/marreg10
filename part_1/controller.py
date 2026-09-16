"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""
import numpy as np


class DPController:
    """
    Template for student DP controller.

    Students may implement any type of controller (PID, LQR, backstepping,
    ...). Only compute() is required; everything else is optional.
    """

    def __init__(self, *args, **kwargs):

        self.Kp_pos = np.array([1502.0, 1767.0])
        self.Kd_pos = np.array([60070.0, 70670.0])

        self.Kp_psi = 136400.0
        self.Kd_psi = 5.456e6

        self.Ki_pos = np.array([10.0, 11.8])
        self.Ki_psi = 909

        # Integral states
        self.int_ned = np.zeros(2)
        self.int_psi = 0.0

        #Tracking Time constants
        self.Tt_pos = 50.0
        self.Tt_psi = 50.0

        self.last_tau_requested = np.zeros(6)

    def reset(self) -> None:
        """Optional: reset internal states (integrators, filters) before a run."""
        self.int_ned = np.zeros(2)
        self.int_psi = 0.0
        self.last_tau_requested = np.zeros(6)

    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,
        acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        # TODO: Replace this placeholder with your DP controller.
        # Return the (6,) desired BODY wrench — fill in tau_d[0] = Fx,
        # tau_d[1] = Fy, tau_d[5] = Mz and leave the rest zero.


        # Actual vessel state
        N = eta[0]
        E = eta[1]
        psi = eta[5]

        u = nu[0]
        v = nu[1]
        r = nu[5]


        # Desired/reference state
        N_d = eta_ref[0]
        E_d = eta_ref[1]
        psi_d = eta_ref[5]

        if nu_ref is None:
            Ndot_d = 0.0
            Edot_d = 0.0
            psidot_d = 0.0
        else:
            Ndot_d = nu_ref[0]
            Edot_d = nu_ref[1]
            psidot_d = nu_ref[5]


        # Position errors
        e_pos_ned = np.array([
            N_d - N,
            E_d - E
        ])

        e_psi = np.arctan2(
            np.sin(psi_d - psi),
            np.cos(psi_d - psi)
        )

        # Rotation BODY -> NED
        c = np.cos(psi)
        s = np.sin(psi)

        J = np.array([
            [c, -s],
            [s,  c]
        ])

        vel_ned = J @ np.array([u, v])


        # Velocity error
        vel_ref_ned = np.array([
            Ndot_d,
            Edot_d
        ])

        e_vel_ned = vel_ref_ned - vel_ned

        # Integrators
        self.int_ned += e_pos_ned * dt
        self.int_psi += e_psi * dt

        # North/east PID regulation in NED
        P_ned = self.Kp_pos * e_pos_ned
        I_ned = self.Ki_pos * self.int_ned
        D_ned = self.Kd_pos * e_vel_ned

        force_ned = P_ned + I_ned + D_ned

        # NED force -> BODY force
        force_body = J.T @ force_ned

        X = force_body[0]
        Y = force_body[1]

        # Yaw PID regulation in BODY
        e_r = psidot_d - r

        P_psi = self.Kp_psi * e_psi
        I_psi = self.Ki_psi * self.int_psi
        D_psi = self.Kd_psi * e_r

        Mz = P_psi + I_psi + D_psi

        # Desired BODY forces
        tau_d = np.zeros(6)

        tau_d[0] = X   # Surge force 
        tau_d[1] = Y   # Sway force 
        tau_d[5] = Mz  # Yaw moment

        # For anti wind up
        self.last_tau_requested = tau_d.copy()

        return tau_d

    def apply_external_aw(
        self,
        tau_applied: np.ndarray,
        psi: float,
        dt: float
    ) -> None:

        delta_tau_body = tau_applied - self.last_tau_requested

        c = np.cos(psi)
        s = np.sin(psi)

        J = np.array([
            [c, -s],
            [s,  c]
        ])

        delta_force_ned = J @ delta_tau_body[:2]

        for i in range(2):
            if self.Ki_pos[i] > 0.0:
                self.int_ned[i] += (
                    delta_force_ned[i]
                    / (self.Ki_pos[i] * self.Tt_pos)
                ) * dt

        if self.Ki_psi > 0.0:
            self.int_psi += (
                delta_tau_body[5]
                / (self.Ki_psi * self.Tt_psi)
            ) * dt