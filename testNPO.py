
import numpy as np
import matplotlib.pyplot as plt
from part_2.observer import NonlinearPassiveObserver

# Simulation parameters
dt = 0.1
T = 100.0
time = np.arange(0, T, dt)

N_true = np.minimum(0.02 * time, 2.0)

# Wave-induced position: 1 m amplitude, 8 s period
N_wave = np.sin(2 * np.pi * time / 8.0)

# Initialize observer with 3 m position error
obs = NonlinearPassiveObserver()

eta0 = np.zeros(6)
eta0[0] = 3.0
obs.reset(eta0)

# No control forces in this synthetic test
tau = np.zeros(6)

# Store results
N_measured = []
N_estimated = []
N_wave_estimated = []
u_estimated = []

for i, t in enumerate(time):

    # Measurement = true LF position + wave motion
    y = np.zeros(6)
    y[0] = N_true[i] + N_wave[i]

    # Observer update
    estimate = obs.step(t, dt, y, tau)

    N_measured.append(y[0])
    N_estimated.append(estimate.eta[0])

    # First internal WF state = estimated wave displacement
    N_wave_estimated.append(obs.x[0])

    # Estimated LF surge velocity
    u_estimated.append(estimate.nu[0])


# Plot 1: Measured vs estimated LF position
plt.figure()
plt.plot(time, N_true, "--", label="True LF position")
plt.plot(time, N_measured, label="Measured position")
plt.plot(time, N_estimated, label="Estimated LF position")
plt.xlabel("Time [s]")
plt.ylabel("North [m]")
plt.legend()
plt.grid()
plt.tight_layout()


# Plot 2: Estimated wave motion
plt.figure()
plt.plot(time, N_wave, label="True wave motion")
plt.plot(time, N_wave_estimated,
         label="Estimated wave motion")
plt.xlabel("Time [s]")
plt.ylabel("Wave-induced North [m]")
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()
