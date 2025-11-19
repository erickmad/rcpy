import numpy as np

def generate_adjacency_matrix(dim_reservoir, rho, connectivity, rng):
    """
    Generate a sparse reservoir adjacency matrix.
    
    Parameters
    ----------
    dim_reservoir : int
        Number of reservoir neurons.
    rho : float
        Desired spectral radius (max absolute eigenvalue after scaling).
    connectivity : float
        Fraction of non-zero connections (0 < connectivity <= 1).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    A : ndarray of shape (dim_reservoir, dim_reservoir)
        Reservoir adjacency matrix.
    """
    # Start with zeros
    A = np.zeros((dim_reservoir, dim_reservoir))

    mask = rng.random((dim_reservoir, dim_reservoir)) < connectivity
    
    # Fill non-zero entries from Uniform[-1, 1]
    A[mask] = rng.uniform(-1, 1, size=np.count_nonzero(mask))

    # Scale to have spectral radius rho
    eigenvalues = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigenvalues))
    if max_eig > 0:
        A *= rho / max_eig
    
    return A

