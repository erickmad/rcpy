import numpy as np
from reservoirpy.nodes import Reservoir, Ridge

def create_model(model_config):
    reservoir = Reservoir(
        units=model_config['reservoir_units'],
        sr=model_config['spectral_radius'],
        rc_connectivity=model_config['p'],
        lr=model_config['leak_rate'],
        input_scaling=model_config['input_scaling'],
        seed=model_config['seed'],
        input_bias=False
    )
    readout = Ridge(ridge=model_config['alpha'], output_dim=1)

    return reservoir >> readout