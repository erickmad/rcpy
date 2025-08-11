import numpy as np
from reservoirpy.nodes import Reservoir

def compute_distances(params, num_states, input_data, test_length):

    seed = params.get('seed', None)
    if seed is None:
        seed = np.random.randint(0, 1_000_000)

    reservoir = Reservoir(
        units=params['reservoir_units'],
        sr=params['spectral_radius'],
        rc_connectivity=params['p'],
        lr=params['leak_rate'],
        input_scaling=params['input_scaling'],
        seed=seed,
        input_bias=False
    )
    
    states_reference = reservoir.run(
        input_data[:test_length], reset=True,
        from_state=np.zeros(params['reservoir_units'])
    )

    distances = []
    for _ in range(num_states):
        state_random = np.random.uniform(-1, 1, params['reservoir_units'])

        states_from_random = reservoir.run(
            input_data[:test_length], reset=True,
            from_state=state_random
        )

        d = np.linalg.norm(states_reference - states_from_random, axis=1)
        distances.append(d)

    distances = np.array(distances)              # Shape: (runs, T)

    return distances
    
def compute_transient_times(distances, error_threshold):
    """
    Computes the transient times for each run based on the distances.
    """
    transient_times = []
    for i in range(distances.shape[0]):
        d = distances[i]
        idx = np.where(d < error_threshold)[0]
        transient_times.append(idx[0] if idx.size > 0 else np.nan)
    return np.array(transient_times)

def compute_mean_transient_times(hyperparams, experiment_params, data, seeds=None):
    """
    Compute mean transient times across multiple reservoir realizations.

    Parameters:
        hyperparams (dict): Reservoir hyperparameters.
        experiment_params (dict): Keys: num_reservoirs, num_states, test_length, error_threshold.
        data (np.ndarray): Input data for the reservoirs.
        seeds (list or None): Optional list of seeds to use for each reservoir. Must match num_reservoirs if given.

    Returns:
        np.ndarray: Array of mean transient times (length = num_reservoirs).
    """
    num_reservoirs = experiment_params["num_reservoirs"]
    mean_transient_times = []

    for i in range(num_reservoirs):
        if seeds is not None:
            hyperparams["seed"] = seeds[i]
        else:
            hyperparams["seed"] = None  # Will generate random seed inside compute_distances

        distances = compute_distances(
            params=hyperparams,
            num_states=experiment_params["num_states"],
            input_data=data[:experiment_params["test_length"]],
            test_length=experiment_params["test_length"]
        )

        transient_times = compute_transient_times(
            distances=distances,
            error_threshold=experiment_params["error_threshold"]
        )

        mean_transient_time = np.mean(transient_times[~np.isnan(transient_times)])
        mean_transient_times.append(mean_transient_time)

    return np.array(mean_transient_times)
