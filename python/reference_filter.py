# reference_filter.py
import numpy as np
from numpy.linalg import solve
from filterpy.kalman import KalmanFilter as FilterPyKF


class KalmanFilter:
    def __init__(self, state_dim, obs_dim, F, H, Q, R, x0, P0):
        self.x = x0.reshape(-1, 1)
        self.P = P0.copy()
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.state_dim = state_dim

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = z.reshape(-1, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = solve(S, self.P @ self.H.T).T
        # K = (solve(S.T, (self.P @ self.H.T).T)).T
        self.x = self.x + K @ y
        # Joseph form — numerically stable
        I = np.eye(self.state_dim)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T


def matrices_close(A, B, rtol=1e-5, atol=1e-6):
    return np.allclose(A, B, rtol=rtol, atol=atol)


def matrices_close_strict(A, B, rtol=1e-10, atol=1e-12):
    return np.allclose(A, B, rtol=rtol, atol=atol)


def generate_experiment_data(N, T, state_dim, seed=42):
    rng = np.random.default_rng(seed)  # use new-style Generator, not global seed

    # Pre-generate all initial conditions
    x0s = rng.normal(0, 100, (N, state_dim))

    # Pre-generate all measurements — shape (N, T, state_dim)
    noise = rng.normal(0, 5, (N, T, state_dim))
    measurements = x0s[:, np.newaxis, :] + noise  # broadcast x0 across timesteps

    return x0s, measurements


def run_experiment(N, T, state_dim, x0s, measurements):
    """
    N = number of filter instances
    T = number of timesteps
    state_dim = size of state vector (and matrices)
    x0s =
    measurements =
    """
    # Fixed matrices shared across all instances
    F = np.eye(state_dim)  # transition
    H = np.eye(state_dim)  # observe full state
    Q = np.eye(state_dim) * 0.01  # process noise
    R = np.eye(state_dim) * 5.0  # measurement noise

    results = np.zeros((N, T, state_dim))

    for i in range(N):
        # Each instance gets slightly different initial conditions
        x0 = x0s[i]
        P0 = np.eye(state_dim) * 500.0

        kf = KalmanFilter(state_dim, state_dim, F, H, Q, R, x0, P0)

        for t in range(T):
            kf.predict()
            kf.update(measurements[i, t])

            results[i, t] = kf.x.flatten()

    return results


def run_filterpy_experiment(N, T, state_dim, x0s, measurements):
    results = np.zeros((N, T, state_dim))

    for i in range(N):
        kf = FilterPyKF(dim_x=state_dim, dim_z=state_dim)
        kf.F = np.eye(state_dim)
        kf.H = np.eye(state_dim)
        kf.Q = np.eye(state_dim) * 0.01
        kf.R = np.eye(state_dim) * 5.0
        kf.x = x0s[i].reshape(-1, 1)
        kf.P = np.eye(state_dim) * 500.0

        for t in range(T):
            kf.predict()
            kf.update(measurements[i, t])
            results[i, t] = kf.x.flatten()

    return results


if __name__ == "__main__":
    N, T, state_dim = 64, 50, 64
    x0s, measurements = generate_experiment_data(N, T, state_dim)
    np.save("initial_states.npy", x0s)
    np.save("measurements.npy", measurements)

    results = run_experiment(N, T, state_dim, x0s=x0s, measurements=measurements)
    lib_results = run_filterpy_experiment(
        N, T, state_dim, x0s=x0s, measurements=measurements
    )
    np.save("reference_outputs.npy", results)
    print("Saved reference_outputs.npy")
    print("Shape:", results.shape)  # (64, 50, 64)
    print("First instance final state:", results[0, -1])

    if matrices_close(results, lib_results):
        print(f"PASS — max diff: {np.abs(results - lib_results).max():.2e}")
    else:
        diff = np.abs(results - lib_results)
        worst = np.unravel_index(diff.argmax(), diff.shape)
        print(f"FAIL — max diff: {diff.max():.2e} at index {worst}")
        print(f"  instance {worst[0]}, timestep {worst[1]}, state element {worst[2]}")
