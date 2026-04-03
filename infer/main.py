import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from infer.config_schema import InferConfig
from infer.pipeline import run_generate_pipeline
from utils.log_utils import setup_logging

logger = setup_logging(__name__, "app.log")

load_dotenv()


def load_config(path: str) -> InferConfig:
    yaml.add_constructor(
        "!env",
        lambda loader, node: os.environ.get(loader.construct_scalar(node)),
        Loader=yaml.SafeLoader,
    )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = InferConfig.model_validate(raw)
    cfg.data.records_file = str(Path(cfg.data.out_dir) / cfg.data.records_file)
    return cfg


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="infer/configs/example.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Loaded infer config: %s", cfg)
    run_generate_pipeline(cfg)


if __name__ == "__main__":
    cli()
