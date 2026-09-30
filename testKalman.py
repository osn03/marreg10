
import numpy as np
import matplotlib.pyplot as plt

from part_2.observer import KalmanFilterObserver

dt = 0.1
T = 100.0
time = np.arange(0, T, dt)

# True LF position is constant.
N_true = np.zeros_like(time)

# Wave-induced motion: 8-second period.
N_wave = np.sin(2 * np.pi * time / 8.0)

obs = KalmanFilterObserver()

# Start with a 3 m LF estimation error.
eta0 = np.zeros(6)
eta0[0] = 3.0
obs.reset(eta0)

tau = np.zeros(6)

N_estimated = []
N_wave_estimated = []

for i, t in enumerate(time):

    y = np.zeros(6)
    y[0] = N_true[i] + N_wave[i]

    estimate = obs.step(t, dt, y, tau)

    N_estimated.append(estimate.eta[0])
    N_wave_estimated.append(obs.x[0])

plt.figure()
plt.plot(time, N_true, "--", label="True LF")
plt.plot(time, N_estimated, label="EKF LF estimate")
plt.plot(time, N_wave, alpha=0.4, label="Measured WF")
plt.xlabel("Time [s]")
plt.ylabel("North [m]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
