import copy
import pandas as pd
import numpy as np
from gluonts.evaluation import aggregate_no_nan
from typing import Optional, Iterable, Union, Tuple, List, Mapping, Dict, cast, Callable
from gluonts.model.forecast import Forecast
from gluonts.evaluation import Evaluator, MultivariateEvaluator
from gluonts.torch.model.forecast import Forecast
import logging
from gluonts.time_feature import get_seasonality
from gluonts.exceptions import GluonTSUserError
import multiprocessing
from gluonts.evaluation.metrics import (
    abs_error,
    abs_target_mean,
    abs_target_sum,
    coverage,
    mape,
    mase,
    mse,
    msis,
    quantile_loss,
    smape,
    calculate_seasonal_error,
)
from gluonts.model.forecast import Quantile


class Quantile_gluonts11(Quantile):
    @property
    def loss_name(self):
        return f"QuantileLoss[{self.name}]"

    @property
    def weighted_loss_name(self):
        return f"wQuantileLoss[{self.name}]"

    @property
    def coverage_name(self):
        return f"Coverage[{self.name}]"

    @classmethod
    def checked(cls, value: float, name: str) -> "Quantile":
        if not 0 <= value <= 1:
            raise GluonTSUserError(
                f"quantile value should be in [0, 1] but found {value}"
            )

        return Quantile(value=value, name=name)


class Evaluator_with_NRMSE(Evaluator):
    default_quantiles = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

    def __init__(
        self,
        quantiles: Iterable[Union[float, str]] = default_quantiles,
        seasonality: Optional[int] = None,
        alpha: float = 0.05,
        calculate_owa: bool = False,
        custom_eval_fn: Optional[Dict] = None,
        num_workers: Optional[int] = multiprocessing.cpu_count(),
        chunk_size: int = 32,
        aggregation_strategy: Callable = aggregate_no_nan,
        ignore_invalid_values: bool = True,
        allow_nan_forecast: bool = False,
    ) -> None:
        self.quantiles = tuple(map(Quantile_gluonts11.parse, quantiles))
        self.seasonality = seasonality
        self.alpha = alpha
        self.calculate_owa = calculate_owa
        self.custom_eval_fn = custom_eval_fn
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.aggregation_strategy = aggregation_strategy
        self.ignore_invalid_values = ignore_invalid_values
        self.allow_nan_forecast = allow_nan_forecast

    def calculate_seasonal_rmse(
        self,
        past_data: np.ndarray,
        freq: Optional[str] = None,
        seasonality: Optional[int] = None,
    ):
        r"""
        .. math::

        seasonal\_error = mean(|Y[t] - Y[t-m]|)

        where m is the seasonal frequency. See [HA21]_ for more details.
        """
        # Check if the length of the time series is larger than the seasonal
        # frequency
        if not seasonality:
            assert freq is not None, "Either freq or seasonality must be provided"
            seasonality = get_seasonality(freq)

        if seasonality < len(past_data):
            forecast_freq = seasonality
        else:
            # edge case: the seasonal freq is larger than the length of ts
            # revert to freq=1

            # logging.info('The seasonal frequency is larger than the length of the
            # time series. Reverting to freq=1.')
            forecast_freq = 1

        y_t = past_data[:-forecast_freq]
        y_tm = past_data[forecast_freq:]

        return np.sqrt(
            np.mean(np.square(y_t - y_tm))
        )  # RMSE of the seasonal naive forecast

    def calculate_seasonal_mae(
        self,
        past_data: np.ndarray,
        freq: Optional[str] = None,
        seasonality: Optional[int] = None,
    ):

        return calculate_seasonal_error(
            past_data=past_data, freq=freq, seasonality=seasonality
        )

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Mapping[str, Union[float, str, None, np.ma.core.MaskedConstant]]:
        pred_target = np.array(self.extract_pred_target(time_series, forecast))
        past_data = np.array(self.extract_past_data(time_series, forecast))

        if self.ignore_invalid_values:
            past_data = np.ma.masked_invalid(past_data)
            pred_target = np.ma.masked_invalid(pred_target)

        try:
            mean_fcst = getattr(forecast, "mean", None)
        except NotImplementedError:
            mean_fcst = None

        median_fcst = forecast.quantile(0.5)
        seasonal_rmse = self.calculate_seasonal_rmse(
            past_data, forecast.start_date.freqstr, self.seasonality
        )
        seasonal_mae = self.calculate_seasonal_mae(
            past_data, forecast.start_date.freqstr, self.seasonality
        )

        metrics: Dict[str, Union[float, str, None]] = {
            "item_id": forecast.item_id,
            "forecast_start": forecast.start_date,
            "MSE": mse(pred_target, mean_fcst) if mean_fcst is not None else None,
            "abs_error": abs_error(pred_target, median_fcst),
            "abs_target_sum": abs_target_sum(pred_target),
            "abs_target_mean": abs_target_mean(pred_target),
            "seasonal_mae": seasonal_mae,
            "seasonal_rmse": seasonal_rmse,
            "MASE": mase(pred_target, median_fcst, seasonal_mae),
            "MAPE": mape(pred_target, median_fcst),
            "sMAPE": smape(pred_target, median_fcst),
        }
        metrics["ND"] = cast(float, metrics["abs_error"]) / cast(
            float, metrics["abs_target_sum"]
        )
        metrics["RMSE"] = np.sqrt(metrics["MSE"])
        metrics["NRMSE"] = (
            metrics["RMSE"] / metrics["seasonal_rmse"]
        )  # RMSE/RMSE(seasonal naive)

        if self.custom_eval_fn is not None:
            for k, (eval_fn, _, fcst_type) in self.custom_eval_fn.items():
                if fcst_type == "mean":
                    if mean_fcst is not None:
                        target_fcst = mean_fcst
                    else:
                        logging.warning(
                            "mean_fcst is None, therefore median_fcst is used."
                        )
                        target_fcst = median_fcst
                else:
                    target_fcst = median_fcst

                try:
                    val = {
                        k: eval_fn(
                            pred_target,
                            target_fcst,
                        )
                    }
                except Exception:
                    logging.warning(f"Error occured when evaluating {k}.")
                    val = {k: np.nan}

                metrics.update(val)

        try:
            metrics["MSIS"] = msis(
                pred_target,
                forecast.quantile(self.alpha / 2),
                forecast.quantile(1.0 - self.alpha / 2),
                seasonal_mae,
                self.alpha,
            )
        except Exception:
            logging.warning("Could not calculate MSIS metric.")
            metrics["MSIS"] = np.nan

        for quantile in self.quantiles:
            forecast_quantile = forecast.quantile(quantile.value)

            metrics[quantile.loss_name] = quantile_loss(
                pred_target, forecast_quantile, quantile.value
            )
            metrics[quantile.coverage_name] = coverage(pred_target, forecast_quantile)

        return metrics

    def get_aggregate_metrics(
        self, metric_per_ts: pd.DataFrame
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        # Define how to aggregate metrics
        agg_funs = {
            "MSE": "mean",
            "abs_error": "sum",
            "abs_target_sum": "sum",
            "abs_target_mean": "mean",
            "seasonal_mae": "mean",
            "MAPE": "mean",
            "sMAPE": "mean",
        }

        for quantile in self.quantiles:
            agg_funs[quantile.loss_name] = "sum"
            agg_funs[quantile.coverage_name] = "mean"

        if self.custom_eval_fn is not None:
            for k, (_, agg_type, _) in self.custom_eval_fn.items():
                agg_funs.update({k: agg_type})

        assert (
            set(metric_per_ts.columns) >= agg_funs.keys()
        ), "Some of the requested item metrics are missing."

        # Compute the aggregation
        totals = self.aggregation_strategy(
            metric_per_ts=metric_per_ts, agg_funs=agg_funs
        )

        # Compute derived metrics
        totals["RMSE"] = np.sqrt(totals["MSE"])
        totals["NRMSE"] = np.mean(
            metric_per_ts["NRMSE"].loc[metric_per_ts["NRMSE"] != np.inf]
        )
        totals["MSIS"] = np.mean(
            metric_per_ts["MSIS"].loc[metric_per_ts["MSIS"] != np.inf]
        )
        totals["MASE"] = np.mean(
            metric_per_ts["MASE"].loc[metric_per_ts["MASE"] != np.inf]
        )
        totals["ND"] = totals["abs_error"] / totals["abs_target_sum"]

        for quantile in self.quantiles:
            totals[quantile.weighted_loss_name] = (
                totals[quantile.loss_name] / totals["abs_target_sum"]
            )

        totals["mean_absolute_QuantileLoss"] = np.array(
            [totals[quantile.loss_name] for quantile in self.quantiles]
        ).mean()

        totals["mean_wQuantileLoss"] = np.array(
            [totals[quantile.weighted_loss_name] for quantile in self.quantiles]
        ).mean()

        totals["MAE_Coverage"] = np.mean(
            [
                np.abs(totals[q.coverage_name] - np.array([q.value]))
                for q in self.quantiles
            ]
        )

        return totals, metric_per_ts


def forecast_to_df(forecast):
    """
    Creates pandas df from gluonts SampleForecast.
    Columns are item_id, start_date, fcst_step and one column per sample (sample_fcst1, ..., sample_fcstns).
    As a result, the dimensions are (h, 3+ns).
    """
    samples = forecast.samples
    if len(samples.shape) > 2:
        samples = samples.squeeze(1)
    ns, h = samples.shape
    id = [forecast.item_id] * h
    start_date = [forecast.start_date.end_time.normalize()] * h
    fcst_step = [i + 1 for i in range(h)]
    fcst_step_date = [
        forecast.start_date.end_time.normalize() + (step - 1) * forecast.start_date.freq
        for step in fcst_step
    ]

    df_samples = pd.DataFrame(samples.swapaxes(0, 1))
    df_samples.columns = ["sample_fcst" + str(i) for i in range(ns)]

    df_ids = pd.DataFrame(
        {
            "item_id": id,
            "start_date": start_date,
            "fcst_step": fcst_step,
            "fcst_step_date": fcst_step_date,
        }
    )

    df_samples = pd.concat([df_ids, df_samples], axis=1)

    return df_samples


def mean_error(x, y):
    return np.mean(x - y)


def evaluate_per_forecast_step(
    ts_list: Iterable[Union[pd.DataFrame, pd.Series]],
    fcst_list: List[Forecast],
    prediction_length: int,
    num_workers: int,
    num_series: Optional[int] = None,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    univariate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute accuracy metrics by
    comparing actual data to the forecasts.

    Parameters
    ----------
    ts_list
        list containing true target on the predicted range
    fcst_list
        list of forecasts on the predicted range
    ts_list_cumsummed
        list containing cumsummed true target on the predicted range
    fcst_list_cumsummed
        list of cumsummed forecasts on the predicted range
    num_series
        number of series of the iterator
        (optional, only used for displaying progress)
    quantiles
        The quantiles that should be evaluated
    univariate
        Specifies if the ts_iterator is a multivariate forecast or univariate

    Returns
    -------
    pd.DataFrame
        DataFrame of aggregated metrics for all forecast steps
    pd.DataFrame
        DataFrame containing per-time-series metrics for all forecast steps
    """
    error = {
        "error_median": [mean_error, "mean", "median"],
        "error_mean": [mean_error, "mean", "mean"],
    }
    multivariate_evaluator = MultivariateEvaluator(
        quantiles=quantiles, custom_eval_fn=error, num_workers=num_workers
    )
    univariate_evaluator = Evaluator_with_NRMSE(
        quantiles=quantiles, custom_eval_fn=error, num_workers=num_workers
    )
    agg_metrics_list = []
    item_metrics_list = []

    fcst_list_copy = copy.deepcopy(fcst_list)
    for f1, f2 in zip(fcst_list, fcst_list_copy):
        f2.samples = f1.samples[:, :prediction_length]
    if univariate or prediction_length == 1:
        agg_metrics, item_metrics = univariate_evaluator(
            iter(ts_list), iter(fcst_list_copy), num_series=num_series
        )
    else:
        agg_metrics, item_metrics = multivariate_evaluator(
            iter(ts_list), iter(fcst_list_copy), num_series=num_series
        )
    agg_metrics_list.append(agg_metrics)
    item_metrics_list.append(item_metrics)

    # Format results
    agg_metrics = pd.DataFrame(
        agg_metrics_list, index=range(prediction_length, prediction_length + 1)
    )
    agg_metrics.index.name = "forecast_horizon"
    fcst_step = np.hstack([np.repeat(prediction_length, num_series)])
    item_metrics = pd.concat(item_metrics_list)
    item_metrics["start_date"] = item_metrics["forecast_start"].astype(str).str[:10]
    item_metrics = item_metrics.drop(["forecast_start"], axis=1)
    date = item_metrics.pop("start_date")
    item_metrics.insert(1, "start_date", date)
    item_metrics["forecast_horizon"] = fcst_step
    step = item_metrics.pop("forecast_horizon")
    item_metrics.insert(2, "forecast_horizon", step)
    return agg_metrics, item_metrics


def paste_inputs_labels(inputs, labels):
    """
    Concat test_pairs.label to test_pairs.input at the end ('target' values only).
    Then turn the array into a pd.Series object. Make a list of these.
    And pass the list to the plot_prob_forecats.
    """
    tst_concat_series = []
    tst_concat_list = []
    for i, j in zip(inputs, labels):
        start_date = i["start"].to_timestamp()
        freq = i["start"].freq
        concat_val = list(np.concatenate((i["target"], j["target"])))
        index = pd.period_range(
            start_date, periods=len(concat_val), freq=freq
        ).to_timestamp()
        tst_concat_series.append(pd.Series(concat_val, index=index))
        tst_concat_list.append(
            {
                "start": i["start"],
                "target": np.concatenate((i["target"], j["target"])),
                "item_id": i["item_id"],
            }
        )
    return tst_concat_series, tst_concat_list
