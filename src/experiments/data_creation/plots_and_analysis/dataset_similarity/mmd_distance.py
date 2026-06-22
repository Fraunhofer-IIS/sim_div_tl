import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import rbf_kernel

# Please install pyarrow yourself if you want to calculate the mmd distance
import pyarrow as pa
import pyarrow.parquet as pq

path_to_evaluation = "/home/dev/projects/data/evaluation/"
path_to_data = "/home/dev/projects/data"

metrics = ["NRMSE", "MASE", "sMAPE"]
p_list = ["5000", "12000", "25000", "50000", "100000", "all"]
feats = [
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


def mmd(X, Y, gamma=1.0):
    Kxx = rbf_kernel(X, X, gamma=gamma).mean()
    Kyy = rbf_kernel(Y, Y, gamma=gamma).mean()
    Kxy = rbf_kernel(X, Y, gamma=gamma).mean()
    return Kxx + Kyy - 2 * Kxy


output_dir = Path(config.path_to_evaluation) / "mmd_distances"
output_dir.mkdir(parents=True, exist_ok=True)
counter = 0


for run in [
    "run1",
    "run2",
    "run3",
    "run4",
    "run5",
]:
    path_to_forecasts = f"{config.path_to_data}/{run}/outputs/forecasts/"
    for div in [10, 5, 0]:
        print(div)
        targets_feat = []
        targets_ids = []
        for p in p_list:
            concat = pd.read_parquet(
                f"{config.path_to_data}/concat/CONCAT_feat.parquet",
            )
            target = pd.read_parquet(
                f"{config.path_to_data}/{run}/interim/target_div{div}_sim{p}{run}/target_div{div}_sim{p}{run}.parquet"
            )
            target_ids = target.id_ts.unique().tolist()
            target_feat = concat[concat["id_ts"].isin(target_ids)][feats + ["id_ts"]]
            targets_feat.append(target_feat)
            targets_ids.append(target_ids)
        targets_feat = pd.concat(targets_feat)
        source = pd.read_parquet(
            f"{config.path_to_data}/{run}/interim/source_div{div}{run}/source_div{div}{run}.parquet"
        )
        source_ids = source.id_ts.unique().tolist()
        source_feat = concat[concat["id_ts"].isin(source_ids)][feats + ["id_ts"]]
        all_feat = pd.concat([targets_feat, source_feat], axis=0)
        feat_std = all_feat.copy()
        feat_std[feats] = (all_feat[feats] - all_feat[feats].mean()) / all_feat[
            feats
        ].std()
        for i, p in zip(range(len(p_list)), p_list):
            print(p)
            target_feat_std = feat_std[feat_std["id_ts"].isin(targets_ids[i])]
            source_feat_std = feat_std[feat_std["id_ts"].isin(source_ids)]
            distance = mmd(
                target_feat_std.drop(columns=["id_ts"]).to_numpy(),
                source_feat_std.drop(columns=["id_ts"]).to_numpy(),
            )
            row = pd.DataFrame([{"run": run, "div": div, "p": p, "mmd": distance}])
            table = pa.Table.from_pandas(row, preserve_index=False)
            pq.write_table(
                table,
                output_dir / f"mmd_{run}_{div}_{p}.parquet",
            )
            counter += 1
