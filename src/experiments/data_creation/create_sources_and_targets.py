target_size = 5000


def create_sources_and_targets(cfg):
    """
    For each original source data set, build a source and multiple target data sets, which are increasingly similar to the source, regarding the features.
    This is done by drawing the target series from an increasingly large cluster.
    The similarity between them is calculated and saved.
    The source and target data sets with their data set configs are saved in the end.

    Args:
        cfg: hydra config

    Returns:
        Nothing, saves the files.
    """
    print("Creating source and target data sets.")
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    import os
    import yaml
    from k_means_constrained import KMeansConstrained

    run = cfg["params"]["run"]
    concat = pd.read_parquet(cfg["params"]["path_to_concat"] + "CONCAT.parquet")

    for percentile in [10, 5, 0]:
        org_source_feat = pd.read_parquet(
            cfg["params"]["path_to_data"]
            + "original_sources/"
            + "percentile_"
            + str(percentile)
            + ".parquet"
        )[["id_ts"] + cfg["params"]["cluster_features"]]
        org_source_ids = list(org_source_feat.id_ts.unique())

        def reserve_target_ids(df, pot_target_size):
            
            if pot_target_size == "all":
                reserve_for_target = (
                        np.random.choice(
                            a=df.id_ts.unique(),
                            size=target_size,
                            replace=False,
                        )
                    ).tolist()

            else:

                if pot_target_size > 0.5 * df.id_ts.nunique():
                        clf = KMeansConstrained(n_clusters=2, size_max=pot_target_size)
                else:
                        clf = KMeansConstrained(n_clusters=2, size_min=pot_target_size)

                clf.fit_predict(df.loc[:, df.columns != "id_ts"])
                print(clf.cluster_centers_)
                print(clf.labels_)
                print(
                        "Cluster centers for potential target size:"
                        + str(pot_target_size)
                )
                print(pd.DataFrame(clf.cluster_centers_, columns=df.columns[1:]))
                centers = pd.DataFrame(clf.cluster_centers_, columns=df.columns[1:])
                centers.to_parquet(
                    cfg["params"]["path_to_data"]
                    + f"cluster_centers/centers_{pot_target_size}.parquet"
                )

                pot_target_ids = list(df[clf.labels_ == 0]["id_ts"])

                reserve_for_target = (
                        np.random.choice(
                            a=pot_target_ids,
                            size=target_size,
                            replace=False,
                        )
                    ).tolist()

            return reserve_for_target

        reserve_for_target = {}
        reserve_for_target_lst = []

        pot_target_sizes = [5000, 12000, 25000, 50000, 100000, "all"]

        for pot_target_size in pot_target_sizes:
            pot_target_ids = reserve_target_ids(
                df=org_source_feat, pot_target_size=pot_target_size
            )
            reserve_for_target[f"similarity_{pot_target_size}"] = pot_target_ids

            reserve_for_target_lst.append(pot_target_ids)

        source_ids = list(
            set(org_source_ids)
            - {item for sublist in reserve_for_target_lst for item in sublist}
        )
        source_feat = org_source_feat[org_source_feat.id_ts.isin(source_ids)]
        source = concat[concat.id_ts.isin(source_ids)]
        similarity_df = pd.DataFrame(
            columns=["pot target size", "similarity", "intermittency", "erraticness"]
        )
        similarity_df["pot target size"] = [str(size) for size in pot_target_sizes] + [
            "source"
        ]

        similarity_df.loc[
            similarity_df["pot target size"] == "source", "similarity"
        ] = None
        similarity_df.loc[
            similarity_df["pot target size"] == "source", "intermittency"
        ] = source_feat["intermittency"].mean()
        similarity_df.loc[
            similarity_df["pot target size"] == "source", "erraticness"
        ] = source_feat["erraticness"].mean()

        for pot_target_size in pot_target_sizes:
            similarity = []

            print(f"Potential target size: {pot_target_size}")
            target_ids = reserve_for_target[f"similarity_{pot_target_size}"]
            target_feat = org_source_feat[org_source_feat.id_ts.isin(target_ids)]
            target = concat[concat.id_ts.isin(target_ids)]

            for feature in cfg["params"]["cluster_features"]:
                ks_statistic, p_value = stats.ks_2samp(
                    target_feat[feature], source_feat[feature]
                )
                similarity.append(1 - ks_statistic)
                print(
                    f"1-KS Statistic for feature {feature}: {1-ks_statistic}, P-value: {p_value}"
                )
            print(f"Sum of 1-ksstats (higher means more similarity): {sum(similarity)}")

            os.makedirs(
                cfg["params"]["path_to_data"]
                + "interim/"
                + "source_div"
                + str(percentile)
                + cfg["params"]["run"]
                + "/",
                exist_ok=True,
            )

            similarity_df.loc[
                similarity_df["pot target size"] == str(pot_target_size), "similarity"
            ] = sum(similarity)
            similarity_df.loc[
                similarity_df["pot target size"] == str(pot_target_size),
                "intermittency",
            ] = target_feat["intermittency"].mean()
            similarity_df.loc[
                similarity_df["pot target size"] == str(pot_target_size), "erraticness"
            ] = target_feat["erraticness"].mean()

            source.to_parquet(
                cfg["params"]["path_to_data"]
                + "interim/"
                + "source_div"
                + str(percentile)
                + cfg["params"]["run"]
                + "/"
                + "source_div"
                + str(percentile)
                + cfg["params"]["run"]
                + ".parquet"
            )

            yaml_data = {
                "_target_": "transfer_learning.preprocessing.gluon_dataset",
                "dataset_name": "source_div" + str(percentile) + cfg["params"]["run"],
                "frequency": "W-SUN",
                "split_date_test": "2023-09-17",
                "trunc_date": "2024-01-14",
                "level": "id_ts",
                "timestamp_var": "Date",
                "target_var": "Target",
                "save_model": True,
                "subsample_size": 5000,
            }

            with open(
                cfg["params"]["path_to_configs_datasets"]
                + f"source_div{percentile}{run}.yaml",
                "w",
            ) as yaml_file:
                yaml.dump(yaml_data, yaml_file, default_flow_style=False)

            os.makedirs(
                cfg["params"]["path_to_data"]
                + "interim/"
                + "target_div"
                + str(percentile)
                + "_sim"
                + str(pot_target_size)
                + run
                + "/",
                exist_ok=True,
            )
            target.to_parquet(
                cfg["params"]["path_to_data"]
                + "interim/"
                + "target_div"
                + str(percentile)
                + "_sim"
                + str(pot_target_size)
                + run
                + "/"
                + "target_div"
                + str(percentile)
                + "_sim"
                + str(pot_target_size)
                + run
                + ".parquet"
            )

            yaml_data = {
                "_target_": "transfer_learning.preprocessing.gluon_dataset",
                "dataset_name": "target_div"
                + str(percentile)
                + "_sim"
                + str(pot_target_size)
                + run,
                "frequency": "W-SUN",
                "split_date_test": "2023-09-17",
                "trunc_date": "2024-01-14",
                "level": "id_ts",
                "timestamp_var": "Date",
                "target_var": "Target",
                "save_model": True,
                "subsample_size": 5000,
            }

            with open(
                cfg["params"]["path_to_configs_datasets"]
                + f"target_div{percentile}_sim{pot_target_size}{run}.yaml",
                "w",
            ) as yaml_file:
                yaml.dump(yaml_data, yaml_file, default_flow_style=False)

        similarity_df.to_parquet(
            cfg["params"]["path_to_data"]
            + "interim/"
            + "source_div"
            + str(percentile)
            + run
            + "/"
            + "similarity"
            + ".parquet"
        )

    print("Done.")
