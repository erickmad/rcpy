import argparse
import yaml

# -------------------------
# 1. Parse arguments
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
        type=int,
        default=None,
        help="Optional preprocessing offset"
    )
    return parser.parse_args()

# -------------------------
# 2. Load YAML config
# -------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

