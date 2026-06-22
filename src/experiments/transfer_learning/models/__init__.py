from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys, os

    sys.path.append(os.path.abspath("..."))
    sys.path.append(os.path.abspath(".."))
    sys.path.append(os.path.abspath("."))
    from .gluonts_extensions import (
        evaluate_per_forecast_step,
    )
else:
    from .gluonts_extensions import (
        evaluate_per_forecast_step,
        forecast_to_df,
    )
import os
from typing import Optional, Tuple, List
import random


class BaseModel(ABC):
    """
    Base class for univariate ML/DL models to perform predictions.
    Univariate means that the model does not apply the GluonTS MultivariateGrouper on the dataset
    """

    @abstractmethod
    def __init__(self, name, forecast_horizon):
        self.name = name
        self.forecast_horizon = forecast_horizon
        print("Model training on : ", name)

    def evaluate(
        self,
        test_data,
        forecast,
        num_workers: int = 0,
        num_series: Optional[int] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        For all time steps between 1 and prediction_length compute accuracy metrics by
        comparing actual data to the forecasts.

        Parameters
        ----------
        ts_iterator
            iterator containing true target on the predicted range
        fcst_list
            list of forecasts on the predicted range
        num_series
            number of series of the iterator
            (optional, only used for displaying progress)
        quantiles
            list of quantiles to get metrics for

        Returns
        -------
        pd.DataFrame
            DataFrame of aggregated metrics for all forecast steps
        pd.DataFrame
            DataFrame containing per-time-series metrics for all forecast steps
        """
        freq = test_data[0]["start"].freq
        fcst_data_start = str(forecast[0].start_date)[:10]
        test_df = [
            pd.DataFrame(
                {
                    "start": pd.period_range(
                        (data["start"]).to_timestamp(),
                        periods=len(data["target"]),
                        freq=freq,
                    ).to_timestamp(),
                    "target": data["target"],
                }
            )
            for data in test_data
        ]

        [df.set_index("start", inplace=True) for df in test_df]
        test_list = [df.set_index(df.index.to_period(freq=freq)) for df in test_df]

        agg_metrics, item_metrics = evaluate_per_forecast_step(
            test_list,
            forecast,
            self.forecast_horizon,
            num_workers,
            num_series,
            quantiles,
        )

        self.save_log_metrics(agg_metrics, item_metrics, fcst_data_start)
        return agg_metrics, item_metrics

    def save_log_metrics(self, agg_metrics, item_metrics, fcst_data_start):
        agg_metrics.columns = agg_metrics.columns.str.replace(
            "[", "_", regex=True
        ).str.replace("]", "", regex=True)
        item_metrics.columns = item_metrics.columns.str.replace(
            "[", "_", regex=True
        ).str.replace("]", "", regex=True)

        agg_metrics.to_parquet(
            self.name + "_" + fcst_data_start + "_agg_metrics.parquet"
        )
        item_metrics.to_parquet(
            self.name + "_" + fcst_data_start + "_item_metrics.parquet"
        )

        print(agg_metrics.tail(1))

    def plot_prob_forecasts(self, ts_entry, forecast_entry, n_plots=1):
        random.seed(9001)
        random_plotids = sorted(random.sample(range(0, len(ts_entry)), n_plots))
        pred_label, observations_label = ["predictions", "observations"]

        for i in random_plotids:
            plot_length = 50
            prediction_intervals = (50.0, 90.0)

            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            forecast_entry[i].plot(
                prediction_intervals=prediction_intervals, color="g", label=pred_label
            )
            ts_entry[i][-plot_length:].plot(ax=ax, color="b", label=observations_label)
            plt.grid(which="both")
            plt.legend(loc="upper left")
            plt.title(forecast_entry[i].item_id)
            plt.savefig(
                self.name
                + "_"
                + str(forecast_entry[0].start_date)[:10]
                + "_forecast_plot"
                + str(i)
                + ".png"
            )

    def save_forecast(self, forecast):
        df_list = []
        for fcst in forecast:
            df = forecast_to_df(fcst)
            df_list.append(df)
        df_all_forecasts = pd.concat(df_list)
        df_all_forecasts.fcst_step_date = df_all_forecasts.fcst_step_date.astype(
            str
        ).str[:10]
        df_all_forecasts.start_date = df_all_forecasts.start_date.astype(str).str[:10]
        df_all_forecasts.to_parquet(
            self.name
            + "_"
            + str(df_all_forecasts.start_date.iloc[0])[:10]
            + "_forecast.parquet"
        )
