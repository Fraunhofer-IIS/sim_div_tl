import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
import numpy as np
from datetime import datetime
import torch.utils.data


if __name__ == "__main__":

    import sys, os

    sys.path.append(os.path.abspath("..."))
    sys.path.append(os.path.abspath(".."))
    sys.path.append(os.path.abspath("."))


class gluon_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        frequency: str,
        forecast_horizon: int,
        context_length: int,
        split_date_test: str,
        trunc_date: str,
        level: str,
        timestamp_var: str,
        target_var: str,
        subsample_size: int,
        interim_path: str,
        step_size: int = 1,
        save_model: bool = True,
        mode: str = "train",
        epoch: int = 0,
    ):
        """
        Initialize Dataset object.

        Parameters
        ----------
        dataset_name: str
            string gives the dataset name to work with.
        frequency: str
            frequency string.
        forecast_horizon: int
            how long to forecast into the future.
        context_length: int
            how long to look into the past the create a forecast.
        split_date_test: str
            The timestamp to which the train-test-split is performed on the time axis.
        trunc_date: str
            The timestamp to which the data set is cut on the time axis.
        level: str
            A string giving the aggregation level
        timestamp_var: str
            The column giving the timestamp to use for aggregating.
        target_var: str
            The column giving the target.
        subsample_size: int
            shrink the test set to this number of time series for computational efficiency.
        interim_path: str
            Path of the interim data relative to project directory.
        step_size: int
            The gap between the start of two sliding windows
        save_model:
            whether to save a trained model to the model path.
        mode: str
            train or predict.
        epoch:
            how many epochs to train the model.

        Returns
        ----------
        A gluonts training and test set.
        """

        self.dataset_name = dataset_name
        self.forecast_horizon = forecast_horizon
        self.context_length = context_length
        self.level = level
        self.timestamp_var = timestamp_var
        self.target_var = target_var
        self.freq = frequency
        print("Granularity: ", frequency)
        self.interim_path = interim_path + self.dataset_name + "/"
        self.step_size = step_size
        self.save_model = save_model
        self.split_date_test = split_date_test
        self.trunc_date = trunc_date
        if mode == "train":
            self.split_date = trunc_date
            self.truncate_date = None
        else:
            self.split_date = split_date_test
            self.truncate_date = trunc_date

        self.mode = mode
        self.subsample_size = subsample_size

        self.columns = [
            self.level,
            self.timestamp_var,
            self.target_var,
        ]

    def apply_train_test_split(self, ds: PandasDataset, df: pd.DataFrame):
        # train/val or train+val/test split according to truncate and split_date
        # Calculate number of windows
        steps_after_split = (
            df.loc[df[self.level] == df[self.level].iloc[0], self.timestamp_var]
            > self.split_date
        ).sum()
        windows = ((steps_after_split - self.forecast_horizon) // self.step_size) + 1

        training_dataset, test_template = split(
            ds, date=pd.Period(self.split_date, self.freq)
        )

        test_pairs = test_template.generate_instances(
            prediction_length=self.forecast_horizon,
            windows=windows,
            distance=self.step_size,
        )

        return training_dataset, test_pairs

    def filter_long_ts(self, df, min_len=4):
        df_ts_len = df[[self.level, self.target_var]].groupby(by=[self.level]).count()
        df_ts_len = df_ts_len[df_ts_len[self.target_var] >= min_len]
        i1 = df_ts_len.index
        i2 = df.set_index([self.level]).index
        df = df[i2.isin(i1)]

        return df

    def prepare_df(self) -> pd.DataFrame:

        # Get data
        df = pd.read_parquet(
            self.interim_path + self.dataset_name + ".parquet",
            engine="fastparquet",
        )

        if df.index.name == "Date":
            df = df.reset_index()

        if self.truncate_date:
            df = df.loc[
                df[self.timestamp_var]
                < pd.to_datetime(self.truncate_date, format="%Y-%m-%d")
            ]

        if self.mode == "train":
            min_len = (
                int(
                    (
                        datetime.strptime(self.trunc_date, "%Y-%m-%d")
                        - datetime.strptime(self.split_date_test, "%Y-%m-%d")
                    ).days
                    / 7
                )
                + 1
            )
        else:
            min_len = 2 * self.forecast_horizon

        df = self.filter_long_ts(df, min_len)

        df = df[self.columns]

        return df

    def get_data_split(self):

        df = self.prepare_df()

        if self.mode == "prediction":

            df_all = df.copy()
            all_ids = df[self.level].unique()
            # if forecast subsamples, draw random ids
            np.random.seed(42)
            sampled_ids = np.random.choice(
                all_ids, replace=False, size=min(self.subsample_size, len(all_ids))
            )

            df = df.loc[df[self.level].isin(sampled_ids)]

            ds_subset = PandasDataset.from_long_dataframe(
                dataframe=df.copy(),
                target=self.target_var,
                item_id=self.level,
                timestamp=self.timestamp_var,
                freq=self.freq,
            )
            print("Applying train-test-split.")
            training_dataset_subset, test_pairs_subset = self.apply_train_test_split(
                ds_subset, df
            )

            ds_all = PandasDataset.from_long_dataframe(
                dataframe=df_all.copy(),
                target=self.target_var,
                item_id=self.level,
                timestamp=self.timestamp_var,
                freq=self.freq,
            )
            training_dataset_all, test_pairs_all = self.apply_train_test_split(
                ds_all, df_all
            )
            training_dataset = training_dataset_all
            test_pairs = test_pairs_subset
            print("Model trained training dataset of len", len(training_dataset))
            print("Model tested on test dataset of len", len(test_pairs))

        else:
            print(
                "number of time series used for training:"
                + str(df[self.level].nunique())
            )
            print("Converting to dataset.")
            ds = PandasDataset.from_long_dataframe(
                dataframe=df.copy(),
                target=self.target_var,
                item_id=self.level,
                timestamp=self.timestamp_var,
                freq=self.freq,
            )
            print("Applying train-test-split.")
            training_dataset, test_pairs = self.apply_train_test_split(ds, df)

        return training_dataset, test_pairs
