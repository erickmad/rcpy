import numpy as np

def tanh(x):
    return np.tanh(x)

def linear_regression(R, trajectory, beta=1e-4):
    Rt = R.T
    inverse_part = np.linalg.inv(R @ Rt + beta * np.identity(R.shape[0]))
    return (trajectory.T @ Rt) @ inverse_part

