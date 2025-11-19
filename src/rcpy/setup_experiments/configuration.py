import argparse
from html import parser
import yaml
from dataclasses import dataclass
from typing import Optional

# -------------------------
# 2. Config dataclasses
# -------------------------
# rcpy/configs.py


@dataclass
class SystemConfig:
    name: str

@dataclass
class PreprocessingConfig:
    warmup_length: int
    train_length: int
    val_length: int
    offset: int = 0
    data_length: Optional[int] = None  # optional, CLI can override

@dataclass
class ReservoirConfig:
    p: float
    leak_rate: float
    input_scaling: float
    spectral_radius: Optional[float] = None
    reservoir_units: Optional[int] = 200
    alpha: Optional[float] = None
    seed: Optional[int] = None

@dataclass
class TrainingConfig:
    discard_training: int

@dataclass
class OptimizationConfig:
    loss_function: str
    trials: int
    timeout_hours: int

@dataclass
class ForecastingConfig:
    forecast_length: int
    num_reservoirs: int

@dataclass
class ResultsConfig:
    output_dir: str

@dataclass
class Config:
    system: SystemConfig
    preprocessing: PreprocessingConfig
    reservoir: ReservoirConfig
    training: TrainingConfig
    optimization: OptimizationConfig
    forecasting: ForecastingConfig
    results: ResultsConfig


# -------------------------
# 2. Parse arguments
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run an experiment from a config file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. experiments/configs/exp1.yaml)"
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help="Optional preprocessing offset"
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Name of the dynamical system (e.g., lorenz, henon, ikeda). Overrides config if provided."
    )
    parser.add_argument(
        "--data_length",
        type=int,
        default=None,
        help="Length of the total data. Overrides config if provided."
    )
    parser.add_argument(
        "--reservoir_units",
        type=int,
        default=None,
        help="Number of reservoir units. Overrides config if provided."
    )
    return parser.parse_args()

# -------------------------
# 2. Load YAML config
# -------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

