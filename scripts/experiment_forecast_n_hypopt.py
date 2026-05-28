#!/usr/bin/env python3
"""
Optimizing hyperparameters and forecasting multiple models
"""

from pathlib import Path
import os, json, time, sys, gc
import numpy as np
from rcpy.analysis import compute_skill
from rcpy.forecasting import forecast_rcpy
from rcpy.setup_experiments import collect_metadata, parse_args, load_config, apply_cli_overrides
from rcpy.data import load_data_rcpy, preprocess_data_rcpy, denormalize_data_rcpy
from rcpy.hypopt import build_objective, run_optimization, convert_numpy, save_best_params, load_seed_hyperparams
from rcpy.models import create_model, save_trained_model
from rcpy.training import train_model
from rcpy.forecasting import multiple_forecasts_rcpy

# -------------------------
# 1. Experiment logic
# -------------------------
def experiment_logic(config):
    # -------------------------
    # 1. Read config parameters
    # -------------------------
    system = config["system"]["name"]
    out_dir = config["results"]["output_dir"]

    # -------------------------
    # 2. Data generation + preprocessing
    # -------------------------
    config_preprocess = config["preprocessing"]

    print(f"\n▶ Generating/Loading and preprocessing data: system={system}")
    
    data_dir = "."
    try: 
        data_raw = np.load(f"{data_dir}/ts_{system}.npy")
    except FileNotFoundError:
        data_raw = np.loadtxt(f"{data_dir}/ts_{system}.csv")

    if "data_length" in config_preprocess and config_preprocess["data_length"] is not None:
        data_length = config_preprocess["data_length"]
    else:
        offset = config_preprocess["offset"] if "offset" in config_preprocess else 0
        data_length = len(data_raw) - offset
    print(f"   Total data length: {data_length}")
    

    train_length = config_preprocess["train_length"]
    val_length = config_preprocess["val_length"]
    warmup_length = config["forecasting"]["warmup_length"]
    offset = config_preprocess["offset"]

    # Preprocess data
    data = preprocess_data_rcpy(
        data=data_raw,
        init_discard=offset,
        train_length=train_length,
        val_length=val_length,
        normalize=config_preprocess["normalize"]
    )

    del data_raw

    # -------------------------
    # 3. Hyperparameter optimization
    # -------------------------
    
    config_hypopt = config["optimization"]
    config_train = config["training"]
    config_forecast = config["forecasting"]

    seeds = np.loadtxt(f'seeds_all.csv', delimiter=',', dtype=int)[:config["forecasting"]["num_reservoirs"]]
    #seeds = np.random.randint(0, 2**32 - 1, size=config_forecast["num_reservoirs"])

    print("DEBUG reservoir units in config:", config["reservoir"].get("units"))

    reservoir_units = int(config["reservoir"]["units"])
    print(f"\n▶ Starting hyperparameter optimization for reservoirs with: {reservoir_units} units")

    best_losses = np.empty(len(seeds), dtype=np.float64)
    for i, seed in enumerate(seeds):
        print(f'\n▶ Optimizing hyperparameters: {config_hypopt["config"]["trials"]} trials | seed={seed} ({i+1}/{len(seeds)})')
                
        if "p" not in config_hypopt["search_space"]:
            avg_degree = config["reservoir"]["avg_degree"]

            config_hypopt["search_space"]["p"] = {
                "fixed": avg_degree / reservoir_units
            }
        
        #loss_function = config_hypopt["loss"]["function"]

        study_name_base = f"{system}_N{reservoir_units}"
        study_name = f"{study_name_base}_S{seed}"
        
        if config_hypopt["config"]["save_db"] == True:
            print(f"   Saving Optuna database to: {out_dir}/hypopt/{study_name}.db")

            db_file = f"{out_dir}/hypopt/{study_name}.db"
            db_path = Path(db_file).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            db_file = None

        objective_func = build_objective(
            data=data,
            reservoir_units=reservoir_units,
            seed=seed,
            hypopt_config=config_hypopt,
            train_config=config_train,
            forecast_config=config_forecast,
        )

        study = run_optimization(
            study_name=study_name,
            db_file=db_file,
            objective_func=objective_func,
            hypopt_config= config_hypopt["config"]
        )

        # Get Optuna's best sampled params
        best_params = study.best_params.copy()
        best_value = study.best_value
        del study
        del objective_func

        # Add fixed params from YAML
        search_space = config_hypopt["search_space"]
        for name, settings in search_space.items():
            if isinstance(settings, dict) and "fixed" in settings:
                best_params[name] = settings["fixed"]

        best_params["reservoir_units"] = reservoir_units
        best_params["p"] = config_hypopt["search_space"]["p"]["fixed"] if "p" in config_hypopt["search_space"] else avg_degree/(reservoir_units)
        best_params["seed"] = int(seed)

        print(f"For seed {seed}:")
        print("Best value:", best_value)
        print("Best parameters:", best_params)

        save_best_params(best_params, filename=f"{out_dir}/hypopt/{study_name}_params.json")

        best_model = create_model(hyperparams=best_params, output_dim=data['train_data'].shape[1])
        best_model_trained = train_model(model=best_model, train_data=data['train_data'], forecasting_step=config_forecast["forecasting_step"], washout_training=config["training"]["washout_training"])

        # Validate best model on validation set
        Y_pred = forecast_rcpy(
            model=best_model_trained,
            warmup_data=data['train_data'][-warmup_length:],
            forecast_length=val_length
        )
        if config_hypopt["loss"]["function"] == "soft_horizon":
            best_losses[i] = compute_skill(data["val_data"], Y_pred, method="efh", threshold=config_hypopt["loss"]["threshold"], softness=config_hypopt["loss"]["softness"])
            print(f"Validation skill with best hyperparameters (SFH): {best_losses[i]}")
        elif config_hypopt["loss"]["function"] == "rmse":
            best_losses[i] = compute_skill(data["val_data"], Y_pred, method="error", metric='rmse')
            print(f"Validation skill with best hyperparameters (RMSE): {best_losses[i]}")

    filename_seeds = f"{out_dir}/seeds/{study_name_base}_seeds.csv"
    os.makedirs(os.path.dirname(filename_seeds), exist_ok=True)
    np.savetxt(filename_seeds, np.array(seeds, dtype=int), fmt='%d', delimiter=',')

    filename = f"{out_dir}/skills/{study_name_base}_losses.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, np.array(best_losses), delimiter=',')

    del Y_pred
    del best_model
    del best_model_trained
    gc.collect()

    # -------------------------
    # 4. Forecast with best model (trained)
    # -------------------------
    
    seed_hyperparams = load_seed_hyperparams(
        seeds=seeds,
        out_dir=out_dir,
        filename_template="{system}_N{reservoir_units}_S{seed}_params.json",
        system=system,
        reservoir_units=reservoir_units,
    )
    
    Y_preds, _ = multiple_forecasts_rcpy(
        train_data=data['train_data'],
        warmup_data=data['val_data'][-warmup_length:],
        hyperparams=None,
        seeds=seeds,
        seed_hyperparams=seed_hyperparams,
        discard_training=config_train["washout_training"],
        forecast_length=config_forecast["forecast_length"]
    )

    # Initialize an array with the same shape as Y_preds
    #Y_preds_denorm = np.empty_like(Y_preds)
    #for i in range(Y_preds.shape[0]):
    #    Y_preds_denorm[i] = denormalize_data_rcpy(Y_preds[i], data['train_min'], data['train_max'])
    #Y_preds = Y_preds_denorm

    filename = f"{out_dir}/forecasts/{study_name_base}_all_forecasts.npy"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, Y_preds)

    skills_efh = []
    skills_rmse = []
    for forecast in Y_preds:
        val_skill_efh = compute_skill(data["test_data"][:forecast.shape[0]], forecast, method="efh", threshold=config_hypopt["loss"]["threshold"], softness=config_hypopt["loss"]["softness"])
        skills_efh.append(val_skill_efh)
        val_skill_nrmse = compute_skill(data["test_data"][:forecast.shape[0]], forecast, method="error", metric='nrmse')
        skills_rmse.append(val_skill_nrmse)
    skills_efh = np.array(skills_efh)
    skills_rmse = np.array(skills_rmse)

    np.savetxt(f"{out_dir}/skills/{study_name_base}_scores_efh.csv", skills_efh, delimiter=',')
    np.savetxt(f"{out_dir}/skills/{study_name_base}_scores_nrmse.csv", skills_rmse, delimiter=',')

# -------------------------
# 5. Main
# -------------------------
def main():
    args = parse_args()
    config = load_config(args.config)

    print(f"🚀 Running experiment with config: {args.config}")
    apply_cli_overrides(args, config)

    start_time = time.time()

    experiment_logic(config)

    elapsed_time = time.time() - start_time
    print(f"⏱ Experiment completed in {elapsed_time:.2f} seconds")

    extra_metadata = {
        #"notes": "offset used: " + str(config["preprocessing"]["offset"]),
        "runtime_seconds": f"{elapsed_time:.2f}"
    }
    metadata_path = f"{config['results']['output_dir']}/metadata/{config['system']['name']}_N{config['reservoir']['units']}_metadata.json"
    collect_metadata(config, args.config, extra_metadata=extra_metadata, output_path=metadata_path)

if __name__ == "__main__":
    main()
