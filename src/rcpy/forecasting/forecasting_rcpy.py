import numpy as np
import random, json, os
import pandas as pd

from rcpy.models import create_model, load_trained_model
from rcpy.training import train_model


import numpy as np

def forecast_rcpy(
    model,
    warmup_data,
    forecast_length,
    mode="autonomous",
    forecast_data=None,
    reset=True,
):
    """
    Reservoir forecasting with shared warmup.

    Parameters
    ----------
    model : trained reservoir model

    warmup_data : array (Tw, dim)
        Data used to synchronize reservoir state.

    forecast_data : array (Tf, dim)
        Ground-truth sequence used AFTER warmup.
        Required for teacher forcing.
        Used only for initialization in autonomous mode.

    forecast_length : int
        Number of forecast steps.

    mode : str
        "autonomous" or "teacher_forced"

    reset : bool
        Reset reservoir state before warmup.

    Returns
    -------
    Y_pred : array (forecast_length, dim)
    """

    if reset:
        model.reset()

    # -------------------------------
    # 1. WARMUP (shared)
    # -------------------------------
    warmup_y = model.run(warmup_data)

    dim = warmup_y.shape[1] if warmup_y.ndim > 1 else 1
    Y_pred = np.empty((forecast_length, dim))

    # last synchronized output/state
    x = warmup_y[-1]

    # -------------------------------
    # 2. FORECASTING
    # -------------------------------
    if mode == "autonomous":

        # first forecast sample = last warmup output
        Y_pred[0] = x

        # closed-loop prediction
        for i in range(1, forecast_length):
            x = model(x)
            Y_pred[i] = x

    elif mode == "teacher_forced":

        # driven prediction
        for i in range(forecast_length):
            u = forecast_data[i]      # true input y_t
            x = model(u)
            Y_pred[i] = x

    else:
        raise ValueError(
            "mode must be 'autonomous' or 'teacher_forced'"
        )

    return Y_pred

""" def forecast_rcpy(model, warmup_data, forecast_length, reset=True):

    dim = warmup_data.shape[1] if len(warmup_data.shape) > 1 else 1
    # Warm up the model
    if reset:
        model.reset()
    warmup_y = model.run(warmup_data)

    Y_pred = np.empty((forecast_length, dim))
    #x = warmup_y[-1].reshape(1, -1)
    x = warmup_y[-1]

    for i in range(forecast_length):
        x = model(x)
        Y_pred[i] = x

    return Y_pred """


def multiple_forecasts_rcpy(
    train_data: np.ndarray,
    warmup_data: np.ndarray,
    hyperparams: dict | None = None,
    seeds: list[int] | None = None,
    seed_hyperparams: dict[int, dict] | None = None,
    discard_training: int = 500,
    forecast_length: int = 500,
    num_reservoirs: int = 10,
) -> tuple[np.ndarray, np.ndarray]:

    forecasts = []

    # ------------------------------------------------------
    # Case 1: seeds=None → use global hyperparameters
    # ------------------------------------------------------
    if seeds is None:
        assert hyperparams is not None, "`hyperparams` must be provided when seeds=None."

        generated_seeds = random.sample(range(1_000_000), num_reservoirs)

        for seed in generated_seeds:
            model_params = hyperparams.copy()
            model_params["seed"] = seed

            model = create_model(
                hyperparams=model_params, 
                output_dim=train_data.shape[1]
            )

            model = train_model(
                model=model, 
                train_data=train_data, 
                forecasting_step=1, 
                washout_training=discard_training
            )

            pred = forecast_rcpy(model, warmup_data, forecast_length)
            forecasts.append(pred)

        return np.array(forecasts), np.array(generated_seeds)

    # ------------------------------------------------------
    # Case 2: seeds provided → use per-seed hyperparameters
    # ------------------------------------------------------
    else:
        assert seed_hyperparams is not None, (
            "`seed_hyperparams` must be provided when seeds is not None.\n"
            "It should map each seed to a hyperparameter dict."
        )

        for seed in seeds:
            assert seed in seed_hyperparams, f"No hyperparameters provided for seed={seed}."

            model_params = seed_hyperparams[seed].copy()
            model_params["seed"] = seed

            model = create_model(
                hyperparams=model_params,
                output_dim=train_data.shape[1]
            )

            model = train_model(
                model=model, 
                train_data=train_data, 
                forecasting_step=1, 
                washout_training=discard_training
            )

            pred = forecast_rcpy(model, warmup_data, forecast_length)
            forecasts.append(pred)

        return np.array(forecasts), np.array(seeds)