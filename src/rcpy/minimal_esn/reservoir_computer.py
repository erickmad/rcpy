import numpy as np
from .adjacency_matrix import generate_adjacency_matrix
from .utils import tanh, linear_regression

class ReservoirComputer:
    def __init__(self, dim_system, dim_reservoir, rho, input_scaling, connectivity, leak_rate=1.0, seed=None):
        self.dim_system = dim_system
        self.dim_reservoir = dim_reservoir
        self.r_state = np.zeros(dim_reservoir)
        self.rng = np.random.default_rng(seed)

        self.A = generate_adjacency_matrix(dim_reservoir, rho, connectivity, self.rng)
        self.W_in = 2 * input_scaling * (self.rng.random((dim_reservoir, dim_system)) - 0.5)
        self.W_out = np.zeros((dim_system, dim_reservoir))
        self.leak_rate = leak_rate
    
    def advance_r_state(self, u):
        pre_activation = np.dot(self.A, self.r_state) + np.dot(self.W_in, u)
        updated_state = tanh(pre_activation)
        self.r_state = (1 - self.leak_rate) * self.r_state + self.leak_rate * updated_state
        return self.r_state
    
    def v(self):
        return np.dot(self.W_out, self.r_state)
    
    def run(self, inputs, initial_state=None):
        """
        Advance the reservoir for a given sequence of inputs and 
        return the matrix of internal states.

        Parameters
        ----------
        inputs : ndarray of shape (T, dim_system)
            Sequence of input vectors to feed into the reservoir.
        initial_state : ndarray of shape (dim_reservoir,), optional
            Initial reservoir state. If None, starts at zero.

        Returns
        -------
        states : ndarray of shape (T, dim_reservoir)
            Reservoir states for each timestep.
        """
        T = inputs.shape[0]

        # Set initial state
        if initial_state is not None:
            if initial_state.shape != (self.dim_reservoir,):
                raise ValueError(f"initial_state must have shape ({self.dim_reservoir},), got {initial_state.shape}")
            self.r_state = initial_state.copy()
        else:
            self.r_state = np.zeros(self.dim_reservoir)

        states = np.zeros((T, self.dim_reservoir))
        for t in range(T):
            states[t] = self.advance_r_state(inputs[t])
        return states

    
    def train(self, trajectory):
        R = np.zeros((self.dim_reservoir, trajectory.shape[0]))
        for i in range(trajectory.shape[0]):
            self.advance_r_state(trajectory[i])
            R[:, i] = self.r_state
        self.W_out = linear_regression(R, trajectory)
    
    def predict(self, steps):
        prediction = np.zeros((steps, self.dim_system))
        for i in range(steps):
            v = self.v()
            prediction[i] = v
            self.advance_r_state(prediction[i])
        return prediction

