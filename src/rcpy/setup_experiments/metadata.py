from pathlib import Path
import shutil
import json
import sys
import platform
import subprocess
from datetime import datetime

def collect_metadata(config, config_path, extra_metadata=None, experiment_id=None):
    """
    Save experiment metadata in a JSON file and copy the config file for record-keeping.
    
    Parameters:
    - config: dict, configuration of the experiment
    - config_path: str/Path, path to the config file
    - out_dir: str/Path, directory to save metadata.json
    - extra_metadata: dict, optional, any additional experiment info
    - experiment_id: str, optional, explicit experiment name
    """

    out_dir = Path(config["results"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy config file
    if config_path:
        shutil.copy(Path(config_path), out_dir / "config.yaml")
        print(f"✅ Config copied to {out_dir / 'config.yaml'}")

    # Git commit
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        commit_hash = "unknown"

    metadata = {
        "experiment_id": experiment_id or out_dir.name,
        "timestamp": datetime.now().isoformat(),
        "git_commit": commit_hash,
        "config_file": str(Path(config_path).resolve()) if config_path else None,
        "config": config,
        #"metrics": metrics or {},
        "system": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    # Save metadata
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"📝 Metadata saved to {metadata_path}")

