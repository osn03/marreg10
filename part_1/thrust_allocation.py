"""
Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.
"""
from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig


class ThrustAllocator:
    """Template for student thrust allocation."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        n = len(self.thrusters)

        # Use only surge, sway and yaw
        tau =  tau_d[[0, 1, 5]]

        # Extended thrust configuration matrix
        Be = np.array([
            [0.0,  1.0,   0.0,  1.0,   0.0],
            [1.0,  0.0,   1.0,  0.0,   1.0],
            [12.0, -3.0, -13.0,  3.0, -13.0]
        ])
        # Pseudoinverse allocation
        z = np.linalg.pinv(Be) @ tau
        # Current thrust magnitudes from the pseudoinverse solution
        uT = z[0]

        u1 = np.hypot(z[1], z[2])
        u2 = np.hypot(z[3], z[4])

        # Check how much each thruster exceeds its limit
        r = max(
            abs(uT) / 32000.0,
            u1 / 80000.0,
            u2 / 80000.0
        )

        # If any thruster exceeds its limit, scale the whole solution down
        if r > 1.0:
            z = z / r

        uT = z[0]

        Fx1 = z[1]
        Fy1 = z[2]

        Fx2 = z[3]
        Fy2 = z[4]

        # Convert azimuth forces to thrust magnitude and angle
        u1 = np.hypot(Fx1, Fy1)
        alpha1 = np.arctan2(Fy1, Fx1)

        u2 = np.hypot(Fx2, Fy2)
        alpha2 = np.arctan2(Fy2, Fx2)

        
        u_cmd = np.array([uT, u1, u2])
        alpha_cmd = np.array([np.pi/2, alpha1, alpha2])

        return u_cmd, alpha_cmd

