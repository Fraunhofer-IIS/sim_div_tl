import pandas as pd
from tqdm import tqdm
from tsfresh.feature_extraction import feature_calculators


def calculate_cv2(ts):
    ts = pd.Series(ts)

    ts_non_zero = ts[ts != 0]

    return ts_non_zero.var(ddof=0) / ts_non_zero.mean() ** 2


def calculate_p(ts):
    ts = pd.Series(ts)

    return len(ts) / sum(ts != 0)


def calculate_ts_metric(ts, id_ts, trim_zeros):

    index_non_zero = ts.nonzero()[0]
    number_of_non_zero = len(index_non_zero)

    # if there are less than 4 observations unequal zero, don't bother, skip to the next
    if number_of_non_zero < 4:
        # fill up dataframe
        return_dict = {
            "id_ts": id_ts,
            "erraticness": None,
            "intermittency": None,
            "abs_energy": None,
            "agg_autocorrelation_max": None,
            "kurtosis": None,
            "mean": None,
            "median": None,
            "skewness": None,
            "standard_deviation": None,
            "agg_linear_trend_slope": None,
        }

    if trim_zeros & number_of_non_zero >= 4:
        ts = ts.iloc[min(index_non_zero) : (max(index_non_zero) + 1)]

    if number_of_non_zero >= 4:

        # calculate metrics:
        abs_energy = feature_calculators.abs_energy(ts)
        agg_autocorrelation_max = feature_calculators.agg_autocorrelation(
            ts, param=[{"f_agg": "max", "maxlag": 5}]
        )[0][1]

        agg_linear_trend_slope = feature_calculators.linear_trend(
            ts, param=[{"attr": "slope"}]
        )[0][1]

        # cv2 (gives an indication about the variation of the demand)
        erraticness_cv2 = calculate_cv2(ts)
        intermittency_p = calculate_p(ts)

        kurtosis = feature_calculators.kurtosis(ts)

        mean = feature_calculators.mean(ts)
        median = feature_calculators.median(ts)

        skewness = feature_calculators.skewness(ts)
        standard_deviation = feature_calculators.standard_deviation(ts)

        return_dict = {
            "id_ts": id_ts,
            "erraticness": erraticness_cv2,
            "intermittency": intermittency_p,
            "abs_energy": abs_energy,
            "agg_autocorrelation_max": agg_autocorrelation_max,
            "agg_linear_trend_slope": agg_linear_trend_slope,
            "kurtosis": kurtosis,
            "mean": mean,
            "median": median,
            "skewness": skewness,
            "standard_deviation": standard_deviation,
        }
    return return_dict


def wrapper_calculate_ts_metrics(ts_dict, trim_zeros, id_string):
    metrics_list = []

    for i in tqdm(ts_dict):
        iter_ts = i["target"]
        metrics_list.append(
            calculate_ts_metric(ts=iter_ts, id_ts=i[id_string], trim_zeros=trim_zeros)
        )

    return pd.DataFrame(metrics_list)
