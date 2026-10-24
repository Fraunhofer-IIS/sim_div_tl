import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from ...selected_tsfresh_features import wrapper_calculate_ts_metrics
import pandas as pd


def calculate_concat_train_feats(cfg):
    """
    Recalculate the features of CONCAT on the training set.
    This has to be done only once.

    Args:
        cfg: hydra configuration

    Returns:
        Nothing, just saves CONCAT_feat_train in parquet files if not already existing
    """

    concat = pd.read_parquet(cfg["params"]["path_to_concat"])
    ts_dict = PandasDataset.from_long_dataframe(
        concat, target="Target", item_id="id_ts"
    )
    # extract features on the training set
    extracted_features = wrapper_calculate_ts_metrics(
        ts_dict, trim_zeros=False, id_string="item_id", on_train=True
    )

    extracted_features.dropna(axis=0, inplace=True)
    extracted_features["data"] = extracted_features["id_ts"].str.split("_").str[-1]

    extracted_features.to_parquet(cfg["params"]["path_to_concat_feat"])
    print("Done.")
