import numpy as np
import pandas as pd
import os


def create_diverse_sources(cfg):
    """
    Creates three original source data sets from CONCAT with the lowest, median and highest diversity.
    This is done by randomly sampling 80% of the series from CONCAT 500 times, calculating the realtive feature diversity (relative to CONCAT)
    and saving those with the lowest, median and highest.

    Args:
        cfg: hydra configuration

    Returns:
        None, just saves the files.
    """

    print("Creating diverse original source data.")
    path_to_concat = cfg["params"]["path_to_concat"]
    path_to_data = cfg["params"]["path_to_data"]
    features = pd.read_parquet(path_to_concat + "CONCAT_feat.parquet")[
        [
            "id_ts",
            "abs_energy",
            "intermittency",
            "mean",
            "median",
            "kurtosis",
            "skewness",
            "standard_deviation",
            "agg_autocorrelation_max",
            "erraticness",
            "agg_linear_trend_slope",
        ]
    ]

    def choose_ids_randomly(df, size):

        cluster_ids = np.random.choice(a=df.id_ts.unique(), size=size, replace=False)

        return list(cluster_ids)

    # choose ids for the original source randomly a couple of times
    runs = 500
    div = pd.DataFrame(
        columns=features.columns[1:]
    )  # Initialize div with the same columns as features
    source_data = {}  # Dictionary to hold source data

    for i in range(runs):
        cluster_ids = choose_ids_randomly(
            df=features, size=round(0.8 * features.id_ts.nunique())
        )
        source_data[i] = features[features.id_ts.isin(cluster_ids)]
        variance_ratio = source_data[i].var(numeric_only=True) / features.var(
            numeric_only=True
        )
        source_div = pd.DataFrame(
            variance_ratio.values.reshape(1, -1), columns=features.columns[1:]
        )
        div = pd.concat([div, source_div], ignore_index=True)

    div["sum"] = div.sum(axis=1)
    print(div)
    print(div["sum"].describe())

    # save the most and least diverse source dataset for now
    most_div_source = source_data[div[div["sum"] == div["sum"].max()].index.values[0]]
    least_div_source = source_data[div[div["sum"] == div["sum"].min()].index.values[0]]
    median_value = div["sum"].median()
    nearest_value = (
        div["sum"].iloc[(div["sum"] - median_value).abs().argsort()[:1]].values[0]
    )
    median_div_source = source_data[div[div["sum"] == nearest_value].index.values[0]]

    os.makedirs(
        path_to_data + "original_sources",
        exist_ok=True,
    )
    most_div_source.to_parquet(
        path_to_data + "original_sources" + "/percentile_" + str(10) + ".parquet"
    )

    median_div_source.to_parquet(
        path_to_data + "original_sources" + "/percentile_" + str(5) + ".parquet"
    )

    least_div_source.to_parquet(
        path_to_data + "original_sources" + "/percentile_" + str(0) + ".parquet"
    )
    print("Done.")
