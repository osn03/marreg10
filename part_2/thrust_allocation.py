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

Contract with the checks, exactly as the public checks test it: ``allocate()``
must never command more thrust than a thruster can deliver, so for an
infeasible demand (the check uses 400 kN of sway) every |u_cmd| <= u_max.
*How* you handle an infeasible demand beyond that limit is your design
decision, not a checked criterion — state your rule and justify it in the
report.  The check also calls ``allocate()`` repeatedly from the initial
actuator state, applies each command through IDEAL actuators and feeds the
resulting actuator state back as ``u_now``/``alpha_now``; after 30 s of such
calls the achieved wrench must equal the requested one.  A steady-state
allocator therefore passes on its first call; a rate-aware allocator
(optional extension, and optional in the project text too) passes once it has
converged.  In the simulation the transient (thrust ramps, 2 rpm azimuth
slews) is shaped by the actuator model in ``models/thruster_dynamics.py``
whatever you command, so rate limiting your own commands is optional, not
required.  ``u_now``/``alpha_now`` are passed in because a thruster produces
the same force as (u, alpha) or as (-u, alpha + pi): which of the two you
command, and on what basis, is part of your allocation design.
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

        # Extract the desired 3-DOF wrench
        tau = tau_d[[0, 1, 5]]

        # Extended configuration matrix
        Be = np.array([
            [0.0,  1.0,   0.0,  1.0,   0.0],
            [1.0,  0.0,   1.0,  0.0,   1.0],
            [12.0, -3.0, -13.0,  3.0, -13.0]
        ])

        # Minimum-norm thrust allocation
        z = np.linalg.pinv(Be) @ tau

        # Calculate the thrust required from each thruster
        uT = abs(z[0])
        u1 = np.hypot(z[1], z[2])
        u2 = np.hypot(z[3], z[4])

        # Find the largest ratio between required and maximum thrust
        r = max(
            1.0,
            uT / self.thrusters[0].u_max,
            u1 / self.thrusters[1].u_max,
            u2 / self.thrusters[2].u_max
        )

        # Scale all allocated forces if a limit is exceeded
        z = z / r

        # Convert allocated forces to thruster commands
        u_cmd = np.array([
            z[0],
            np.hypot(z[1], z[2]),
            np.hypot(z[3], z[4])
        ])

        alpha_cmd = np.array([
            np.pi / 2,
            np.arctan2(z[2], z[1]),
            np.arctan2(z[4], z[3])
        ])
        # Select the equivalent command requiring less azimuth rotation
        if alpha_now is not None:
            for i in [1, 2]:

                # Keep current angle when almost no thrust is required
                if abs(u_cmd[i]) < 1.0:
                    alpha_cmd[i] = alpha_now[i]
                    continue

                # Shortest angular difference between desired and current angle
                delta = np.arctan2(
                    np.sin(alpha_cmd[i] - alpha_now[i]),
                    np.cos(alpha_cmd[i] - alpha_now[i])
                )

                # Reverse thrust instead of rotating more than 90 degrees
                if abs(delta) > np.pi / 2:
                    u_cmd[i] = -u_cmd[i]
                    alpha_cmd[i] = np.arctan2(
                        np.sin(alpha_cmd[i] + np.pi),
                        np.cos(alpha_cmd[i] + np.pi)
                    )

        return u_cmd, alpha_cmd
