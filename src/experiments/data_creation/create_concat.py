import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from .selected_tsfresh_features import wrapper_calculate_ts_metrics
from tqdm import tqdm
import pandas as pd
import numpy as np
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset import DatasetCollection
import os

# Preparation:
# Get the M5 data set sales_train_validation.csv from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data and put them in the folder data/M5.


def create_concat(cfg):
    """
    Create the data basis CONCAT from multiple weekly data sets.
    Then extract the ten features for each series.
    This has to be done only once.

    Args:
        cfg: hydra configuration

    Returns:
        Nothing, just saves CONCAT and CONCAT_feat in parquet files if not already existing
    """

    # check if concat has already been built in previous runs and load it in that case
    if os.path.exists(cfg["params"]["path_to_concat"] + "CONCAT.parquet"):
        return

    # Get all weekly datasets from the Monash Time Series Repository through GluonTS
    def gluonts_dataset_to_pandas_dataframe(dataset: DatasetCollection):
        pandas_df = pd.DataFrame(columns=["Date", "Target", "item_id"])
        for i in tqdm(dataset):
            len_ts = len(i["target"])
            freq = i["start"].freqstr

            date = pd.date_range(start=i["start"].start_time, periods=len_ts, freq=freq)
            target = i["target"]
            item_id = np.repeat(i["item_id"], len_ts)
            data = {"Date": date, "Target": target, "item_id": item_id}
            df = pd.DataFrame(data)
            pandas_df = pd.concat([pandas_df, df])
            pandas_df["item_id"] = pandas_df["item_id"].astype(str)

        if freq != "W-SUN":
            pandas_df.set_index("Date", inplace=True)
            pandas_df = pandas_df.groupby("item_id").resample("W-SUN").sum()
            pandas_df.reset_index(inplace=True)
        return pandas_df

    dataset = get_dataset("electricity")
    electricity = gluonts_dataset_to_pandas_dataframe(dataset.test)

    dataset = get_dataset("m4_weekly")
    m4 = gluonts_dataset_to_pandas_dataframe(dataset.test)

    dataset = get_dataset("nn5_weekly")
    nn5 = gluonts_dataset_to_pandas_dataframe(dataset.test)

    dataset = get_dataset("solar-energy")
    solar = gluonts_dataset_to_pandas_dataframe(dataset.test)

    dataset = get_dataset("kaggle_web_traffic_weekly")
    kaggle = gluonts_dataset_to_pandas_dataframe(dataset.test)

    dataset = pd.read_csv(cfg["params"]["M5_path"] + "sales_train_validation.csv")
    m5_dates = pd.date_range(
        start="2011-01-29", periods=1913, freq="D"
    )  # 2011-01-29 is the first date in the calendar data from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
    dictionary = dict(zip(dataset.iloc[:, 6:].columns, m5_dates))
    dataset.rename(columns=dictionary, inplace=True)

    m5 = pd.melt(
        dataset,
        value_vars=dataset.iloc[:, 6:].columns,
        var_name="Date",
        value_name="Target",
        ignore_index=False,
        id_vars=dataset.iloc[:, :6].columns,
    )

    m5.set_index("Date", inplace=True)
    m5 = m5.groupby("id").resample("W-SUN").sum()
    m5.reset_index(inplace=True)
    m5.rename(columns={"id": "item_id"}, inplace=True)

    dataset = get_dataset("traffic")
    traffic = gluonts_dataset_to_pandas_dataframe(dataset.test)

    # Define the desired end date
    desired_end_date = "2023-12-31"

    def remove_trailing_zeros(group):
        # Find the last non-zero index
        if (group["Target"] == 0).all():  # Check if the entire series is zero
            return group
        else:
            last_non_zero_index = group["Target"][group["Target"] != 0].index[-1]
            return group.loc[:last_non_zero_index]

    def align_to_end_date(group):
        current_end_date = group["Date"].max()
        shift_days = (pd.to_datetime(desired_end_date) - current_end_date).days
        if shift_days < 0:
            return group[
                group["Date"]
                >= (pd.to_datetime(desired_end_date) + pd.DateOffset(days=shift_days))
            ]
        else:
            group["Date"] = group["Date"] + pd.DateOffset(days=shift_days)
            return group

    for data, name in zip(
        [m4, m5, nn5, electricity, kaggle, solar, traffic],
        ["m4", "m5", "nn5", "electricity", "kaggle", "solar", "traffic"],
    ):
        if "store_id" in data.columns:
            data["id_ts"] = (
                (data.item_id).astype(str)
                + "_"
                + (data.store_id).astype(str)
                + "_"
                + name
            )
        else:
            data["id_ts"] = (data.item_id).astype(str) + "_" + name
        data["data"] = name

    concat = pd.concat([m4, m5, nn5, electricity, kaggle, solar, traffic])
    concat = concat[["id_ts", "Date", "Target", "data"]]
    concat = concat.groupby("id_ts", group_keys=False).apply(remove_trailing_zeros)
    concat = concat.groupby("id_ts").apply(align_to_end_date)
    concat = concat.reset_index(drop=True)
    concat.set_index("Date", inplace=True)

    ts_dict = PandasDataset.from_long_dataframe(
        concat, target="Target", item_id="id_ts"
    )
    extracted_features = wrapper_calculate_ts_metrics(
        ts_dict, trim_zeros=False, id_string="item_id"
    )
    extracted_features.dropna(axis=0, inplace=True)
    extracted_features["data"] = extracted_features["id_ts"].str.split("_").str[-1]

    extracted_features.to_parquet(
        cfg["params"]["path_to_concat"] + "/CONCAT_feat.parquet"
    )

    # drop the ids not in the features
    concat = concat[concat["id_ts"].isin(list(extracted_features.id_ts))]
    concat = concat.reset_index()

    concat.to_parquet(cfg["params"]["path_to_concat"] + "/CONCAT.parquet")
    print("Done.")
