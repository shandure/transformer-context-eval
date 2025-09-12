# src/data_path.py
from pathlib import Path
import os

def get_dataset_dir(cli_arg: str | None = None) -> Path:
    """
    Priority: explicit CLI arg > DATA_DIR env var > ./data/raw/function_dataset
    """
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()
    env = os.getenv("DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "data" / "raw" / "function_dataset"
