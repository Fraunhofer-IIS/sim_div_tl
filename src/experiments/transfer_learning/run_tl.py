import yaml
from .main import main
from datetime import datetime
from hydra.core.global_hydra import GlobalHydra


def run_tl(cfg):
    """
    Train an N-BEATS ensemble on the source for all original source data sets.
    Zero-shot forecast on all target data sets.

    Args:
        cfg: hydra config
    """
    print("Running transfer learning.")
    run = cfg["params"]["run"]
    path_to_config = cfg["params"]["path_to_config"]
    path_to_nbeats_config = cfg["params"]["path_to_nbeats_config"]
    path_to_configs_datasets = cfg["params"]["path_to_configs_datasets"]
    path_to_data_interim = cfg["params"]["interim_path"]
    path_to_outputs_train = cfg["params"]["path_to_outputs_train"]
    path_to_outputs_forecasts = cfg["params"]["path_to_outputs_forecasts"]

    print(f"Starting transfer learning run, date: {datetime.now()}")
    for percentile in [10, 5, 0]:

        # pretain for source div10 to div0
        with open(path_to_config, "r") as file:
            config = yaml.safe_load(file)

            # modify config.yaml
        for item in config["defaults"]:
            if isinstance(item, dict) and "datasets" in item:
                item["datasets"] = "source_div" + str(percentile) + run
                break
        config["hydra"]["run"]["dir"] = (
            path_to_outputs_train + "pretrain_nbeats/pretrain_${datasets.dataset_name}"
        )
        config["params"]["mode"] = "train"
        config["params"]["model_path"] = (
            path_to_data_interim
            + "source_div"
            + str(percentile)
            + run
            + "/models/nbeats/ensemble/weekly/horizon_15/"
        )

        # Write the updated config back to the file
        with open(path_to_config, "w") as file:
            yaml.dump(config, file)

            # modify NBEATSEnsemble.yaml
        with open(path_to_nbeats_config, "r") as file:
            config = yaml.safe_load(file)

        config["model_params"]["epochs"] = 800
        with open(path_to_nbeats_config, "w") as file:
            yaml.dump(config, file)

        # modify dataset.yaml
        with open(
            path_to_configs_datasets + "source_div" + str(percentile) + run + ".yaml",
            "r",
        ) as file:
            config = yaml.safe_load(file)

        config["trunc_date"] = "2023-09-17"
        with open(
            path_to_configs_datasets + "source_div" + str(percentile) + run + ".yaml",
            "w",
        ) as file:
            yaml.dump(config, file)

        print(
            "Updated configs for pretraining source",
            "_div",
            str(percentile),
            run,
        )
        GlobalHydra.instance().clear()
        main()

        # scratch forecasts for sources
        with open(path_to_config, "r") as file:
            config = yaml.safe_load(file)

            # modify config.yaml
        for item in config["defaults"]:
            if isinstance(item, dict) and "datasets" in item:
                item["datasets"] = "source_div" + str(percentile) + run
                break
        config["hydra"]["run"]["dir"] = (
            path_to_outputs_forecasts + "${datasets.dataset_name}"
        )
        config["params"]["mode"] = "prediction"
        config["params"]["model_path"] = (
            path_to_data_interim
            + "source_div"
            + str(percentile)
            + run
            + "/models/nbeats/ensemble/weekly/horizon_15/"
        )

        # Write the updated config back to the file
        with open(path_to_config, "w") as file:
            yaml.dump(config, file)

        # modify NBEATSEnsemble.yaml
        with open(path_to_nbeats_config, "r") as file:
            config = yaml.safe_load(file)

        config["model_params"]["epochs"] = 0
        with open(path_to_nbeats_config, "w") as file:
            yaml.dump(config, file)

        # modify dataset.yaml
        with open(
            path_to_configs_datasets + "source_div" + str(percentile) + run + ".yaml",
            "r",
        ) as file:
            config = yaml.safe_load(file)

        config["trunc_date"] = "2024-01-14"
        with open(
            path_to_configs_datasets + "source_div" + str(percentile) + run + ".yaml",
            "w",
        ) as file:
            yaml.dump(config, file)

        print(
            "Updated configs for scratch forecasting ",
            "source_div",
            str(percentile),
            run,
        )
        GlobalHydra.instance().clear()
        main()

        for pot_target_size in [5000, 12000, 25000, 50000, 100000, "all"]:
            # tl for targets sim0 to sim100
            with open(path_to_config, "r") as file:
                config = yaml.safe_load(file)

            # modify config.yaml
            for item in config["defaults"]:
                if isinstance(item, dict) and "datasets" in item:
                    item["datasets"] = (
                        "target_div"
                        + str(percentile)
                        + "_sim"
                        + str(pot_target_size)
                        + run
                    )
                    break
            config["hydra"]["run"]["dir"] = (
                path_to_outputs_forecasts + "${datasets.dataset_name}"
            )
            config["params"]["mode"] = "prediction"
            config["params"]["model_path"] = (
                path_to_data_interim
                + "source_div"
                + str(percentile)
                + run
                + "/models/nbeats/ensemble/weekly/horizon_15/"
            )

            # Write the updated config back to the file
            with open(path_to_config, "w") as file:
                yaml.dump(config, file)

            # modify NBEATSEnsemble.yaml
            with open(path_to_nbeats_config, "r") as file:
                config = yaml.safe_load(file)

            config["model_params"]["epochs"] = 0
            with open(path_to_nbeats_config, "w") as file:
                yaml.dump(config, file)

            print(
                "Updated configs for tl forecasting ",
                "target_div",
                str(percentile),
                "_sim",
                str(pot_target_size),
                run,
            )
            GlobalHydra.instance().clear()
            main()

    print("Done.")
