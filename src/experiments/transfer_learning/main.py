import warnings
import hydra
import pandas as pd

from .models.gluonts_extensions import (
    paste_inputs_labels,
)


warnings.filterwarnings(
    "ignore",
    message="Timestamp.freq is deprecated and will be removed in a future version.",
)


@hydra.main(config_path="../hydra_configs", config_name="config", version_base="1.2")
def main(cfg):
    """
    Either train on a data set or predict one.

    Args:
        cfg: hydra configs
    """

    if cfg.params.mode == "prediction":
        test_horizon = pd.date_range(
            start=cfg.datasets.split_date_test,
            end=cfg.datasets.trunc_date,
            freq="W-SUN",
        )

        if len(test_horizon) - (cfg.params.forecast_horizon + 1) <= 0:
            print(
                "There is no full horizon forecast possible, change split_date_test or trunc_date."
            )

        else:
            dataset = hydra.utils.instantiate(
                cfg.datasets,
                interim_path=cfg.params.interim_path,
                forecast_horizon=cfg.params.forecast_horizon,
                context_length=cfg.params.context_length,
                mode=cfg.params.mode,
                epoch=cfg.models.model_params.epochs,
            )

            training_dataset, test_pairs = dataset.get_data_split()

            print("Instantiating the model.")
            model = hydra.utils.instantiate(
                cfg.models.model_params,
                forecast_horizon=cfg.params.forecast_horizon,
                freq=dataset.freq,
            )

            print("Finetune the source model.")
            model.finetune_model(training_dataset, cfg.params.model_path)

            print("Creating target forecasts.")
            forecasts = model.predict(
                test_pairs.input, num_samples=cfg.params.num_samples
            )

            model.save_forecast(forecasts)

            tst_concat_series, tst_concat_list = paste_inputs_labels(
                test_pairs.input, test_pairs.label
            )

            print("Evaluating target forecasts.")
            num_series = len(test_pairs.label)
            model.evaluate(
                test_data=tst_concat_list, forecast=forecasts, num_series=num_series
            )

            print("Plotting target forecasts.")
            model.plot_prob_forecasts(
                ts_entry=tst_concat_series,
                forecast_entry=forecasts,
                n_plots=min(cfg.params.n_plots, len(forecasts)),
            )

            print("End " + cfg.params.mode)

    else:

        dataset = hydra.utils.instantiate(
            cfg.datasets,
            interim_path=cfg.params.interim_path,
            forecast_horizon=cfg.params.forecast_horizon,
            context_length=cfg.params.context_length,
            mode=cfg.params.mode,
        )

        training_dataset, test_pairs = dataset.get_data_split()

        print("Instantiating the model.")
        model = hydra.utils.instantiate(
            cfg.models.model_params,
            forecast_horizon=cfg.params.forecast_horizon,
            freq=dataset.freq,
        )

        print("Training and saving the source model.")

        model.train(training_dataset, cfg.datasets.save_model, cfg.params.model_path)

        print("End pretraining")


if __name__ == "__main__":
    main()
