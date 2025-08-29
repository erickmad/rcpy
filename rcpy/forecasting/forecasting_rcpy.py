import numpy as np
from rcpy.models import create_model
import random, json

def forecast_rcpy(model, warmup_data, forecast_length):

    dim = 1
    # Warm up the model
    warmup_y = model.run(warmup_data, reset=False)

    Y_pred = np.empty((forecast_length, dim))
    x = warmup_y[-1].reshape(1, -1)

    for i in range(forecast_length):
        x = model(x)
        Y_pred[i] = x

    return Y_pred


def multiple_forecasts_rcpy(
    data: dict,
    model_config: dict,
    forecast_length: int,
    seeds: list[int] = None,
    num_realizations: int = 10,
    system: str = None,  # Required when using seeds with JSON configs
    reservoir_units: int = None,
    params_path: str = None
) -> list:
    """
    Generate multiple forecasts using different reservoir seeds.

    Parameters
    ----------
    model_config : dict
        Shared hyperparameters (used only when seeds=None).
    data : dict
        Dictionary with 'train_data', 'train_target', and 'warmup_data'.
    forecast_length : int
        Number of forecast steps.
    seeds : list of int, optional
        List of seeds to use. If given, model_config will be loaded per seed from a JSON file.
    num_realizations : int
        Number of random forecasts to generate if seeds is None.
    system : str
        System name used to construct JSON filenames (only required if seeds is provided).
    reservoir_units : int
        Number of reservoir units used to construct JSON filenames.

    Returns
    -------
    forecasts : list of np.ndarray
        Forecasts from each model instance.
    """
    forecasts = []

    if seeds is None:
        # Use same model_config and generate random seeds
        for seed in random.sample(range(1_000_000), num_realizations):
            config = model_config.copy()
            config["seed"] = seed

            model = create_model(model_config=config)
            model.fit(data["train_data"], data["train_target"], warmup=240)

            pred = forecast_rcpy(
                model=model,
                warmup_data=data["warmup_data"],
                forecast_length=forecast_length,
            )
            forecasts.append(pred)

    else:
        assert system is not None and reservoir_units is not None, \
            "When using a list of seeds, 'system' and 'reservoir_units' must be provided."

        for seed in seeds:
            if params_path is not None:
                filename = f"{params_path}/{system}_N{reservoir_units}_S{seed}_params.json"
            else:
                filename = f"{system}_N{reservoir_units}_S{seed}_params.json"
            with open(filename, "r") as file:
                params = json.load(file)

            config = {
                "reservoir_units": reservoir_units,
                "p": params["p"],
                "leak_rate": params["leak_rate"],
                "spectral_radius": params["spectral_radius"],
                "input_scaling": params["input_scaling"],
                "alpha": params["alpha"],
                "seed": seed
            }

            model = create_model(model_config=config)
            model.fit(data["train_data"], data["train_target"], warmup=240)

            pred = forecast_rcpy(
                model=model,
                warmup_data=data["warmup_data"],
                forecast_length=forecast_length,
            )
            forecasts.append(pred)

    return forecasts
