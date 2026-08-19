from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_EXPORT_DIR = PROJECT_ROOT / "data" / "raw_export"   # raw MQL5 CSV dumps
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"                 # parsed Parquet
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"     # post feature engineering

MODELS_CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "checkpoints"
ONNX_EXPORT_DIR = PROJECT_ROOT / "onnx_export"

QUANTILES_CONFIG_PATH = PROJECT_ROOT / "config" / "quantiles.yaml"


def load_quantiles_config() -> dict:
    with open(QUANTILES_CONFIG_PATH) as f:
        return yaml.safe_load(f)
