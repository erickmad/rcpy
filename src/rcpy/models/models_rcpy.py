import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
import os, pickle

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


def save_trained_model(model, out_dir, system, reservoir_units, seed, offset, loss_function):
    """
    Save a trained reservoir model to disk.
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{system}_N{reservoir_units}_S{seed}_T_{offset}_{loss_function}_rcpy_trained_model.pkl"
    filepath = os.path.join(out_dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Saved trained model to {filepath}")
    return filepath

def load_trained_model(filepath):
    """
    Load a previously saved trained reservoir model.
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model