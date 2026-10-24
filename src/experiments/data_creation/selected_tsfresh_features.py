import pandas as pd
import numpy as np
from tqdm import tqdm
from tsfresh.feature_extraction import feature_calculators
from statsmodels.tsa.seasonal import STL
from hurst import compute_Hc
from pycatch22 import catch22_all

# please install hurst and pycatch22 yourself if you want to use these features


def calculate_cv2(ts):
    ts = pd.Series(ts)

    ts_non_zero = ts[ts != 0]

    return ts_non_zero.var(ddof=0) / ts_non_zero.mean() ** 2


def calculate_p(ts):
    ts = pd.Series(ts)

    return len(ts) / sum(ts != 0)


def seasonal_strength(x, period):
    """
    Compute seasonal strength as defined by Hyndman.

    Parameters
    ----------
    x : array-like or pandas.Series
        Time series data (univariate).
    period : int
        Seasonal period (e.g., 12 for monthly data with yearly seasonality,
        7 for daily data with weekly seasonality).

    Returns
    -------
    float
        Seasonal strength in [0, 1], where higher values indicate stronger seasonality.
    """
    # Ensure we have a float Series
    x = pd.Series(x).astype(float)

    # STL decomposition (additive model)
    stl = STL(x, period=period, robust=True)
    res = stl.fit()

    seasonal = res.seasonal
    remainder = res.resid

    # Variance of remainder and of seasonal + remainder
    var_r = np.var(remainder, ddof=1)
    var_s_r = np.var(seasonal + remainder, ddof=1)

    # Avoid division by zero
    if var_s_r == 0:
        return 0.0

    # Hyndman's seasonal strength formula (additive)
    F_s = max(0.0, 1.0 - var_r / var_s_r)
    return F_s


def calculate_fourier_entropy(ts, bins=10):

    try:
        entropy = feature_calculators.fourier_entropy(ts, bins=bins)
        return entropy
    except Exception:
        return np.nan


def hurst_exponent(x):
    """
    Compute the Hurst exponent of a time series.

    Parameters
    ----------
    x : array-like
        Time series data (univariate).

    Returns
    -------
    float
        Hurst exponent H.
    """
    x = np.asarray(x, dtype=float)
    try:
        H, c, data = compute_Hc(x, kind="random_walk", simplified=True)
        return H
    except Exception:
        return np.nan


def catch22_features(x):
    """
    Compute all 22 catch22 features for a univariate time series.

    Parameters
    ----------
    x : array-like
        Time series data (1D).

    Returns
    -------
    dict
        Dictionary mapping feature names to their values.
    """
    x = np.asarray(x, dtype=float)
    res = catch22_all(x)
    return dict(zip(res["names"], res["values"]))


def calculate_ts_metric(ts, id_ts, trim_zeros):

    index_non_zero = ts.nonzero()[0]
    number_of_non_zero = len(index_non_zero)
    CATCH22_NAMES = catch22_all(np.zeros(100))["names"]

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
            # uncomment for extended features:
            # "entropy": None,
            # "seasonal_strength": None,
            # "hurst_exponent": None,
            # **{name: None for name in CATCH22_NAMES},
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

        # uncomment for extended features:
        # entropy = calculate_fourier_entropy(ts, bins=10)

        # seasonal_strength_value = seasonal_strength(ts, period=52)

        # hurst_exponent_value = hurst_exponent(ts)

        # catch22_dict = catch22_features(ts)

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
            # uncomment for extended features:
            # "entropy": entropy,
            # "seasonal_strength": seasonal_strength_value,
            # "hurst_exponent": hurst_exponent_value,
            # **catch22_dict,
        }
    return return_dict


def wrapper_calculate_ts_metrics(ts_dict, trim_zeros, id_string, on_train):
    metrics_list = []

    for i in tqdm(ts_dict):
        iter_ts = i["target"]
        if on_train:
            iter_ts = iter_ts[:-15]
        metrics_list.append(
            calculate_ts_metric(ts=iter_ts, id_ts=i[id_string], trim_zeros=trim_zeros)
        )

    return pd.DataFrame(metrics_list)
