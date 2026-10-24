from experiments.data_creation.create_concat import create_concat
from experiments.data_creation.plots_and_analysis.features.calculate_concat_feats_train import (
    calculate_concat_train_feats,
)
from experiments.data_creation.create_diverse_sources import create_diverse_sources

from experiments.transfer_learning.run_tl import run_tl
import hydra
import subprocess
import os


def run_in_data_prep_env(cfg):
    # Run create_sources_and_targets in the data-prep environment

    python_executable = os.path.join(cfg.params.data_prep_env, "bin", "python")

    subprocess.run(
        [
            python_executable,
            "-c",
            f"import sys; sys.path.append('{cfg.params.run_dir}'); from data_creation.create_sources_and_targets import create_sources_and_targets; create_sources_and_targets({cfg})",
        ]
    )


@hydra.main(config_path="hydra_configs", config_name="config", version_base="1.2")
def main(cfg):
    """
    Runs one run configured in config.yaml

    Args:
        cfg: hydra configuration
    """
    # create a folder with subfolders in data for the run
    os.makedirs(cfg.params.path_to_data, exist_ok=True)
    os.makedirs(cfg.params.path_to_data + "cluster_centers", exist_ok=True)
    os.makedirs(cfg.params.path_to_data + "original_sources", exist_ok=True)
    os.makedirs(cfg.params.path_to_data + "outputs", exist_ok=True)
    os.makedirs(cfg.params.interim_path, exist_ok=True)
    os.makedirs(cfg.params.path_to_outputs_forecasts, exist_ok=True)
    os.makedirs(cfg.params.path_to_outputs_train, exist_ok=True)
    create_concat(cfg)
    create_diverse_sources(cfg)
    run_in_data_prep_env(cfg)
    run_tl(cfg)
    calculate_concat_train_feats(cfg)


if __name__ == "__main__":
    main()
