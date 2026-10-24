import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    GroupKFold,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score


from scipy.stats import linregress

# adjust paths in /data_creation/plots_and_analysis/config.py
path_to_evaluation = "/home/dev/projects/data/evaluation/"
path_to_data = "/home/dev/projects/data"


param_grid = {
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_leaf": [2, 3, 5, 10],
}


all_metrics_df = pd.read_parquet(
    f"{path_to_evaluation}results_metrics_features_train_mean.parquet"
)

all_metrics_df["group"] = all_metrics_df["div"] + "_" + all_metrics_df["run"]


# features = [
#     "trend",
#     # "erraticness",
# ]  # for feature importance analysis, use less correlated features
features = [
    "abs_energy",
    "intermittency",
    "mean",
    "median",
    "kurtosis",
    "skewness",
    "sd",
    "autocorr",
    "erraticness",
    "trend",
]

# for extended features:
# features = list(
#     set(all_metrics_df.columns)
#     - set(
#         [
#             "RMSSE",
#             "MASE",
#             "sMAPE",
#             "p",
#             "similarity",
#             "div",
#             "run",
#             "group",
#         ]
#     )
# )

# LOOSD: leave one OSD out cross validation
logo = LeaveOneGroupOut()
groups = all_metrics_df["group"]


# --------------------------------------------------
# DATA
# --------------------------------------------------

div = "aggregated"

targets = ["MASE", "RMSSE", "sMAPE"]

all_columns = features + targets


if div == "aggregated":
    complete_data = all_metrics_df[all_columns].dropna()
else:
    complete_data = all_metrics_df.loc[
        all_metrics_df["div"] == div,
        all_columns,
    ].dropna()


X = complete_data[features]


prediction_results = {}
results_dict = {}


# --------------------------------------------------
# TARGET LOOP
# --------------------------------------------------

for target in targets:

    print(f"\nProcessing {target}")

    y = complete_data[target].values

    y_true_all = []
    y_pred_all = []
    group_all = []

    fold_importances = []
    fold_coefficients = []
    fold_pvalues = []

    fold_train_r2 = []

    for train_idx, test_idx in logo.split(X, y, groups):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()

        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X.columns,
            index=X_train.index,
        )

        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X.columns,
            index=X_test.index,
        )

        # --------------------------------------
        # INNER CV (Nested CV)
        # --------------------------------------

        inner_groups = groups[train_idx]

        grid = GridSearchCV(
            estimator=DecisionTreeRegressor(random_state=42),
            param_grid=param_grid,
            scoring="r2",
            cv=GroupKFold(n_splits=5),
            n_jobs=-1,
        )

        grid.fit(
            X_train_scaled,
            y_train,
            groups=inner_groups,
        )

        best_model = grid.best_estimator_

        # --------------------------------------
        # Training R²
        # --------------------------------------

        y_train_pred = best_model.predict(X_train_scaled)

        train_r2 = r2_score(
            y_train,
            y_train_pred,
        )

        fold_train_r2.append(train_r2)

        # --------------------------------------
        # OOS prediction
        # --------------------------------------

        y_pred = best_model.predict(X_test_scaled)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        group_all.extend(np.asarray(groups)[test_idx])

        # --------------------------------------
        # Permutation Importance
        # --------------------------------------

        perm = permutation_importance(
            best_model,
            X_test_scaled,
            y_test,
            scoring="r2",
            n_repeats=20,
            random_state=42,
            n_jobs=-1,
        )

        fold_importances.append(perm.importances.T)

    # ------------------------------------------------
    # Overall CV R²
    # ------------------------------------------------

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    cv_r2 = r2_score(
        y_true_all,
        y_pred_all,
    )

    mean_train_r2 = np.mean(fold_train_r2)

    print("CV R²:", cv_r2)
    print("mean training R²:", mean_train_r2)

    prediction_results[target] = {
        "y_true": y_true_all.copy(),
        "y_pred": y_pred_all.copy(),
        "cv_r2": cv_r2,
        "mean_train_r2": mean_train_r2,
    }

    # ------------------------------------------------
    # Importance statistics
    # ------------------------------------------------

    fold_importances = pd.DataFrame(
        np.vstack(fold_importances),
        columns=features,
    )

    fold_importances.to_parquet(
        path_to_evaluation + f"LOOSD_dt_feature_importances_{target}.parquet"
    )

    mean_importance = fold_importances.mean(axis=0)

    std_importance = fold_importances.std(axis=0)

    ci_lower = np.percentile(
        fold_importances,
        2.5,
        axis=0,
    )

    ci_upper = np.percentile(
        fold_importances,
        97.5,
        axis=0,
    )

    results_dict[target] = {
        "cv_r2": cv_r2,
        "mean_train_r2": mean_train_r2,
        "importance_mean": mean_importance,
        "importance_std": std_importance,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# --------------------------------------------------
# SUMMARY TABLE
# --------------------------------------------------

rows = []

for i, feature in enumerate(features):

    row = {"Feature": feature}

    for target in targets:

        row[f"{target}_Importance"] = round(
            results_dict[target]["importance_mean"][i],
            4,
        )

        row[f"{target}_CI_Lower"] = round(
            results_dict[target]["ci_lower"][i],
            4,
        )

        row[f"{target}_CI_Upper"] = round(
            results_dict[target]["ci_upper"][i],
            4,
        )

        row[f"{target}_CV_R2"] = round(
            results_dict[target]["cv_r2"],
            4,
        )

        row[f"{target}_Mean_Train_R2"] = round(
            results_dict[target]["mean_train_r2"],
            4,
        )

    rows.append(row)


final_df = pd.DataFrame(rows)


print("\nCross-validated R²")

for target in targets:

    print(f"{target}: " f"{results_dict[target]['cv_r2']:.4f}")


print("\nMean Training R²")

for target in targets:

    print(f"{target}: " f"{results_dict[target]['mean_train_r2']:.4f}")


print("\nFeature Importances")
print(final_df)


final_df.to_csv(
    path_to_evaluation + f"tree_nested_cv_importance_div{div}_loosd.csv",
    index=False,
)


# --------------------------------------------------
# PREDICTION PLOTS
# --------------------------------------------------

targets = ["RMSSE", "MASE", "sMAPE"]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(24, 8),
)

# Global font settings
plt.rcParams.update(
    {
        "font.size": 22,
        "axes.labelsize": 27,
        "axes.titlesize": 32,
        "xtick.labelsize": 23,
        "ytick.labelsize": 23,
        "legend.fontsize": 20,
    }
)


for ax, target in zip(axes, targets):

    y_true = prediction_results[target]["y_true"]
    y_pred = prediction_results[target]["y_pred"]

    # ------------------------------------------
    # Scatter
    # ------------------------------------------

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.80,
        s=150,
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
    )

    # ------------------------------------------
    # Set bounds
    # ------------------------------------------

    min_val = min(
        np.min(y_true),
        np.min(y_pred),
    )

    max_val = max(
        np.max(y_true),
        np.max(y_pred),
    )

    padding = 0.07 * (max_val - min_val)

    min_val -= padding
    max_val += padding

    # ------------------------------------------
    # Perfect Prediction
    # ------------------------------------------

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=4.5,
        label="Perfect prediction",
        zorder=2,
    )

    # ------------------------------------------
    # Regression Line
    # ------------------------------------------

    slope, intercept, r, p, stderr = linregress(
        y_true,
        y_pred,
    )

    x_line = np.linspace(
        min_val,
        max_val,
        200,
    )

    y_line = intercept + slope * x_line

    ax.plot(
        x_line,
        y_line,
        linewidth=4.5,
        label=f"Fit (slope = {slope:.2f})",
        zorder=4,
    )

    # ------------------------------------------
    # Labels
    # ------------------------------------------

    ax.set_xlabel(
        f"Observed {target}",
        fontsize=28,
        labelpad=16,
        fontweight="bold",
    )

    ax.set_ylabel(
        f"Predicted {target}",
        fontsize=28,
        labelpad=16,
        fontweight="bold",
    )

    # ------------------------------------------
    # Tick labels
    # ------------------------------------------

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=23,
        width=2.8,
        length=11,
        pad=9,
    )

    # ------------------------------------------
    # Axes
    # ------------------------------------------

    ax.set_xlim(
        min_val,
        max_val,
    )

    ax.set_ylim(
        min_val,
        max_val,
    )

    # ------------------------------------------
    # Grid
    # ------------------------------------------

    ax.grid(
        alpha=0.30,
        linewidth=1.5,
        zorder=0,
    )

    # ------------------------------------------
    # Spine thickness
    # ------------------------------------------

    for spine in ax.spines.values():
        spine.set_linewidth(2.8)


# ------------------------------------------
# Layout
# ------------------------------------------

plt.tight_layout(
    pad=3.5,
)


# Vector PDF for paper
plt.savefig(
    path_to_evaluation + f"decision_tree_prediction_vs_observed_div{div}_loosd.pdf",
    format="pdf",
    bbox_inches="tight",
)


# High-resolution PNG
plt.savefig(
    path_to_evaluation + f"decision_tree_prediction_vs_observed_div{div}_loosd.png",
    format="png",
    dpi=600,
    bbox_inches="tight",
)


plt.show()
