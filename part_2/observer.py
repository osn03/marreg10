"""Student observer templates and their common simulator interface.

The simulator calls, once per step (pose measurement only, no velocity):

    observer.step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)

    eta_measured : (6,) measured NED pose [N, E, z, phi, theta, psi]
    tau_est      : (6,) desired controller wrench (before thruster dynamics)
                   of the PREVIOUS step (zero at the first step)
    eta, nu      : (6,) low-frequency NED pose and BODY velocity estimates
    bias         : (6,) slowly varying bias estimate (NED force) — may be zeros

A plain ``(eta, nu, bias)`` tuple is accepted as well.  Before every run the
engine calls ``reset(eta0)`` with the TRUE initial pose, so an observer can
start from the vessel's actual position.  Select an implementation with
``Part2SimConfig(observer_type=...)`` (Simulations 4-7 use the
``SELECTED_OBSERVER`` constant of ``run_case_part_2.py``); extra constructor
arguments go through ``Part2SimConfig(observer_kwargs=...)``.  Both classes
intentionally start as pass-through placeholders; students must implement and
tune them, keeping the tuned parameters as constructor defaults because the
checks call ``select_observer(kind)`` without arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ObserverEstimate:
    eta: np.ndarray
    nu: np.ndarray
    bias: np.ndarray

    """
    For both estimators:
    
    Internal state vector (15 states):

        x = [xi, eta, nu, b]

        xi  = [Nw, Nw_dot, Ew, Ew_dot, psi_w, psi_w_dot]
               WF motion in NED (6 states)

        eta = [N, E, psi]
               LF position and heading in NED (3 states)

        nu  = [u, v, r]
               Body-fixed velocities (3 states)

        b   = [b_N, b_E, b_psi]
               Slowly varying environmental bias in NED (3 states)

    Input:
        Measured pose y and previous commanded wrench tau.

    Output:
        Estimated LF pose, body velocity and environmental bias.
    """

# M: Rigid-body mass + added mass.
M = np.array([
    [6.007e5,  0.0,      0.0],
    [0.0,      7.067e5, -4.733e5],
    [0.0,     -5.712e5,  5.456e7]
])

# D: Linear hydrodynamic damping.
D = np.diag([1118.0, 22290.0, 1.950e6])



# Initial design values, to be tuned later:

# omega_w: Natural frequency of the WF oscillator [rad/s].
omega_w = np.full(3, 2.0 * np.pi / 8.0)

# zeta_w: Wave damping ratio.
zeta_w = np.full(3, 0.1)

# T_b: Environmental-bias time constants [s].
T_b = np.full(3, 1000.0)


# ROTATION MATRIX
def J(psi):
    """
    Rotation from BODY velocity to NED velocity.
    """

    c = np.cos(psi)
    s = np.sin(psi)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])


# WAVE MODEL
def wave_matrices():
    """
    Second-order WF oscillator for North, East and yaw.

    For each direction:

        q_ddot + 2*zeta_w*omega_w*q_dot + omega_w**2*q = w_w

    State-space form:

        xi_dot = Aw @ xi + Ew @ w_w
        eta_w  = Cw @ xi

    xi = [N_w, N_w_dot,
          E_w, E_w_dot,
          psi_w, psi_w_dot]

    eta_w = [N_w, E_w, psi_w]

    Aw: WF dynamics (6x6)
    Ew: Wave-excitation input matrix (6x3)
    Cw: Extracts WF position and heading (3x6)
    """

    Aw = np.zeros((6, 6))
    Ew = np.zeros((6, 3))
    Cw = np.zeros((3, 6))

    for i in range(3):

        j = 2 * i

        # Second-order oscillator for direction i
        Aw[j:j+2, j:j+2] = [
            [0.0, 1.0],
            [-omega_w[i]**2, -2.0*zeta_w[i]*omega_w[i]]
        ]

        # Excitation enters the oscillator acceleration
        Ew[j+1, i] = 1.0

        # Output is the WF displacement/angle
        Cw[i, j] = 1.0

    return Aw, Ew, Cw

Aw, Ew, Cw = wave_matrices()


def model_derivative(x, tau, mass_matrix=M):
    """
    Common nonlinear model for NPO and EKF.

    Model equations:

        xi_dot  = Aw @ xi

        eta_dot = J(psi) @ nu

        nu_dot  = inv(M) @ (
                      -D @ nu
                      + J(psi).T @ b
                      + tau
                  )

        b_dot   = -inv(Tb) @ b

    Stochastic excitation is omitted from this
    deterministic state prediction.
    """

    # Extract states
    xi  = x[0:6]
    eta = x[6:9]
    nu  = x[9:12]
    b   = x[12:15]

    psi = eta[2]
    R = J(psi)

    # 1. Wave-frequency dynamics
    xi_dot = Aw @ xi

    # 2. Low-frequency kinematics (BODY -> NED)
    eta_dot = R @ nu

    # 3. Vessel dynamics (forces in BODY)
    nu_dot = np.linalg.solve(
        mass_matrix,
        -D @ nu + R.T @ b + tau
    )

    # 4. Slowly varying environmental bias
    b_dot = -b / T_b

    return np.concatenate([
        xi_dot,
        eta_dot,
        nu_dot,
        b_dot
    ])


def predicted_measurement(x):
    """
    y_hat = eta + Cw @ xi
    eta:     Low-frequency position/heading in NED
    Cw @ xi: Wave-frequency position/heading in NED
    """
    xi = x[0:6]
    eta = x[6:9]
    return eta + Cw @ xi


def measurement_jacobian():
    """
    Measurement model:
        y = eta + Cw @ xi + v

    x = [xi(6), eta(3), nu(3), b(3)]
    H = dh/dx = [Cw, I3, 0, 0]

    H has shape (3, 15), since we measure only North, East and heading.
    """

    H = np.zeros((3, 15))

    # WF displacement and heading contribute to y.
    H[:, 0:6] = Cw

    # LF position and heading also contribute to y.
    H[:, 6:9] = np.eye(3)

    return H


class Observer:

    name = "observer"

    def reset(self, eta0: np.ndarray | None = None):
        pass

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )


def dynamics_jacobian(x):
    """
    Jacobian of the continuous vessel model:
        A = df/dx

    A has shape (15, 15).
    """

    psi = x[8]
    nu = x[9:12]
    b = x[12:15]

    c = np.cos(psi)
    s = np.sin(psi)

    # Rotation matrix and its derivative w.r.t. heading
    Rot = J(psi)

    dRot = np.array([
        [-s, -c, 0.0],
        [ c, -s, 0.0],
        [0.0, 0.0, 0.0]
    ])

    A = np.zeros((15, 15))

    # Wave dynamics: xi_dot = Aw @ xi
    A[0:6, 0:6] = Aw

    # LF kinematics: eta_dot = J(psi) @ nu
    A[6:9, 9:12] = Rot
    A[6:9, 8] = dRot @ nu

    # Vessel dynamics:
    # nu_dot = M^-1(-D nu + J.T b + tau)
    A[9:12, 9:12] = -np.linalg.solve(M, D)

    A[9:12, 12:15] = np.linalg.solve(
        M, Rot.T
    )

    A[9:12, 8] = np.linalg.solve(
        M, dRot.T @ b
    )

    # Bias dynamics: b_dot = -Tb^-1 b
    A[12:15, 12:15] = -np.diag(1.0 / T_b)

    return A

    

class NonlinearPassiveObserver(Observer):

    name = "nonlinear_passive"

    def __init__(
        self,
        k=1.5,
        zeta_d=1.0,
        k4_xy=4000.0,
        k4_psi=3e5,
        beta=0.02,
    ):
        """
        Tuning parameters:

        k:
            Frequency factor: omega_c = k * omega_w.

        zeta_d:
            Desired wave-filter damping.

        k4_xy:
            Correction gain for translational dynamics [N/m].

        k4_psi:
            Correction gain for yaw dynamics [Nm/rad].

        beta:
            Bias correction ratio [1/s].
            K3 = beta * K4.
        """

        # Symmetric approximation of the mass matrix
        self.Mp = 0.5 * (M + M.T)

        # Basic tuning-parameter checks
        if k <= 1.0:
            raise ValueError("k must be greater than 1")
        if np.any(zeta_d <= zeta_w):
            raise ValueError(
                "Desired damping must exceed wave-model damping"
            )
        if not (0.0 < beta < np.min(omega_w)):
            raise ValueError("Invalid bias gain ratio")

        # K1: Wave-frequency correction (6x3)
        # xi_hat_dot = Aw @ xi_hat + K1 @ e
        self.K1 = np.zeros((6, 3))
        for axis in range(3):

            wave_frequency = omega_w[axis]
            wave_damping = zeta_w[axis]

            damping_difference = zeta_d - wave_damping

            position_index = 2 * axis
            velocity_index = position_index + 1

            # Correction of WF displacement
            self.K1[position_index, axis] = (
                2.0 * damping_difference * wave_frequency
            )

            # Correction of internal WF velocity
            self.K1[velocity_index, axis] = (
                2.0 * damping_difference
                * wave_frequency**2
                * (k - 2.0 * wave_damping)
            )

        # K2: LF position and heading correction (3x3)
        # eta_hat_dot = J(psi_hat) @ nu_hat + K2 @ e
        self.K2 = np.diag(k * omega_w)

        # K4: Velocity-dynamics correction (3x3)
        # Adds an equivalent force/moment based on e.
        self.K4 = np.diag([
            k4_xy,
            k4_xy,
            k4_psi
        ])

        # K3: Environmental-bias correction (3x3)
        # b_hat_dot = -Tb^-1 @ b_hat + K3 @ e
        self.K3 = beta * self.K4

        # Initialize all observer states
        self.reset()

    def reset(self, eta0=None):
        """
        Initialize the observer.

        xi_hat(0) = 0
        eta_hat(0) = eta0, if supplied
        nu_hat(0) = 0
        b_hat(0) = 0
        """

        self.x = np.zeros(15)

        if eta0 is not None:

            eta0 = np.asarray(
                eta0, dtype=float
            ).reshape(-1)

            # Convert 6-DOF pose to active 3-DOF pose
            if eta0.size == 6:
                eta0 = eta0[[0, 1, 5]]

            self.x[6:9] = eta0

            # Wrap initial heading
            self.x[8] = np.arctan2(
                np.sin(self.x[8]),
                np.cos(self.x[8])
            )

    def step(self, t, dt, eta_measured, tau_est):
        """
        Perform one observer update.

        1. Read measurements and previous control wrench.
        2. Calculate measurement innovation.
        3. Predict the state derivative using the common model.
        4. Add observer corrections.
        5. Integrate the estimated states.
        6. Return estimates to the DP controller.
        """

        # STEP 1: Read inputs

        y = eta_measured
        tau = tau_est

        # Extract [N, E, psi] from the 6-DOF measurement
        if y.size == 6:
            y = y[[0, 1, 5]]

        # Extract [Fx, Fy, Mz] from the 6-DOF wrench
        if tau.size == 6:
            tau = tau[[0, 1, 5]]


        # STEP 2: Measurement innovation

        # y_hat = eta_hat + Cw @ xi_hat
        y_hat = predicted_measurement(self.x)

        e = y - y_hat

        # Heading error must be wrapped
        e[2] = np.arctan2(
            np.sin(e[2]),
            np.cos(e[2])
        )


        # STEP 3: Predict state derivatives
        
        # x_hat_dot = f(x_hat, tau)
        dx = model_derivative(
            self.x,
            tau,
            mass_matrix=self.Mp
        )


        # STEP 4: Add measurement corrections
        # x_hat_dot = f(x_hat, tau) + K(x_hat) @ e

        # Wave-frequency correction
        dx[0:6] += self.K1 @ e

        # Low-frequency pose correction
        dx[6:9] += self.K2 @ e

        # Body-velocity correction
        R = J(self.x[8])

        dx[9:12] += np.linalg.solve(
            self.Mp,
            R.T @ (self.K4 @ e)
        )

        # Slowly varying bias correction
        dx[12:15] += self.K3 @ e


        # STEP 5: Forward Euler integration
        # x_hat[k+1] = x_hat[k] + dt * x_hat_dot[k]

        self.x += dt * dx

        # Wrap estimated heading
        self.x[8] = np.arctan2(
            np.sin(self.x[8]),
            np.cos(self.x[8])
        )


        # STEP 6: Return the estimated states

        # Simulator uses 6-DOF arrays.
        # Only indices [0, 1, 5] are active.

        eta6 = np.zeros(6)
        nu6 = np.zeros(6)
        bias6 = np.zeros(6)

        eta6[[0, 1, 5]] = self.x[6:9]
        nu6[[0, 1, 5]] = self.x[9:12]
        bias6[[0, 1, 5]] = self.x[12:15]

        return ObserverEstimate(
            eta=eta6,
            nu=nu6,
            bias=bias6
        )



class KalmanFilterObserver(Observer):

    name = "kalman"

    def __init__(self):

        self.H = measurement_jacobian()


        # CONTINUOUS PROCESS COVARIANCE Qc
        # These are initial tuning values:

        sigma_wave = np.array([
            0.20,
            0.20,
            np.deg2rad(0.5)
        ])

        sigma_nu = np.array([
            0.003,
            0.003,
            np.deg2rad(0.01)
        ])

        sigma_bias = np.array([
            400.0,
            400.0,
            15000.0
        ])

        self.Qc = np.zeros((15, 15))

        # Wave uncertainty enters through Ew.
        self.Qc[0:6, 0:6] = (
            Ew @ np.diag(sigma_wave**2) @ Ew.T
        )

        # Velocity-model uncertainty
        self.Qc[9:12, 9:12] = np.diag(
            sigma_nu**2
        )

        # Environmental-bias uncertainty
        self.Qc[12:15, 12:15] = np.diag(
            sigma_bias**2
        )


        # MEASUREMENT COVARIANCE R
        # Assumed measurement standard deviations:
        # North/East: 0.20 m
        # Heading:    0.20 degrees

        self.Rm = np.diag([
            0.20**2,
            0.20**2,
            np.deg2rad(0.20)**2
        ])

        #Initial estimation uncertainty
        self.initial_std = np.array([
            0.5, 0.5,                    # WF North
            0.5, 0.5,                    # WF East
            np.deg2rad(3),              # WF yaw
            np.deg2rad(2),              # WF yaw rate

            1.0, 1.0, np.deg2rad(5),     # LF pose

            0.2, 0.2, np.deg2rad(1),     # BODY velocity

            5000.0, 5000.0, 1e5         # Environmental bias
        ])

        self.reset()

    def reset(self, eta0=None):

        # Initial state estimate
        self.x = np.zeros(15)

        # Initial covariance
        self.P = np.diag(self.initial_std**2)

        if eta0 is not None:

            if eta0.size == 6:
                eta0 = eta0[[0, 1, 5]]

            self.x[6:9] = eta0

            self.x[8] = np.arctan2(
                np.sin(self.x[8]),
                np.cos(self.x[8])
            )

    def step(self, t, dt, eta_measured, tau_est):
        """
        One EKF iteration:

        1. Predict state
        2. Predict covariance
        3. Calculate innovation
        4. Calculate Kalman gain
        5. Correct state and covariance
        6. Return estimates
        """

        # STEP 1: READ INPUTS

        y = eta_measured
        tau = tau_est

        if y.size == 6:
            y = y[[0, 1, 5]]

        if tau.size == 6:
            tau = tau[[0, 1, 5]]


        # STEP 2: PREDICT STATE
        # x_pred = x + dt*f(x, tau)

        x_pred = self.x + dt * model_derivative(self.x, tau)

        # Wrap predicted LF heading
        x_pred[8] = np.arctan2(
            np.sin(x_pred[8]),
            np.cos(x_pred[8])
        )


        # STEP 3: PREDICT COVARIANCE
        # P_pred = F P F.T + Qk
        # Qk = dt * Qc

        A = dynamics_jacobian(self.x)
        F = np.eye(15) + dt * A
        
        P_pred = (
            F @ self.P @ F.T
            + dt * self.Qc
        )

        P_pred = 0.5 * (P_pred + P_pred.T)


        # STEP 4: MEASUREMENT INNOVATION

        y_pred = predicted_measurement(x_pred)
        e = y - y_pred

        # Heading innovation must be wrapped
        e[2] = np.arctan2(
            np.sin(e[2]),
            np.cos(e[2])
        )

        # STEP 5: KALMAN GAIN
        # S = H P_pred H.T + R
        # K = P_pred H.T S^-1

        S = (
            self.H @ P_pred @ self.H.T + self.Rm
        )

        K = np.linalg.solve(
            S,
            self.H @ P_pred
        ).T

        # STEP 6: CORRECT STATE
        # x_new = x_pred + K e

        self.x = x_pred + K @ e

        self.x[8] = np.arctan2(
            np.sin(self.x[8]),
            np.cos(self.x[8])
        )

        # STEP 7: CORRECT COVARIANCE
        # Joseph covariance update:
        # P = (I-KH)P_pred(I-KH).T + K R K.T

        I = np.eye(15)
        L = I - K @ self.H
        self.P = (
            L @ P_pred @ L.T
            + K @ self.Rm @ K.T
        )
        self.P = 0.5 * (self.P + self.P.T)


        # STEP 8: RETURN 6-DOF ESTIMATES

        eta6 = np.zeros(6)
        nu6 = np.zeros(6)
        bias6 = np.zeros(6)

        eta6[[0, 1, 5]] = self.x[6:9]
        nu6[[0, 1, 5]] = self.x[9:12]
        bias6[[0, 1, 5]] = self.x[12:15]

        return ObserverEstimate(
            eta=eta6,
            nu=nu6,
            bias=bias6
        )


def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc

