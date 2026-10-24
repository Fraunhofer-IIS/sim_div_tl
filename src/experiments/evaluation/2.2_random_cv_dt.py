"""
Random 10-fold cross-validation of the decision-tree performance predictor.

Counterpart to the LOOSD evaluation (2_accuracy_features_dt_grouped.py), used for the
limitations check: are the R² values similar when folds ignore the OSD structure?

- Outer CV: random KFold(10), shuffled
- Inner CV (hyperparameter tuning): random KFold(5), shuffled
- Repeated over several seeds, because random CV results depend on the split
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# N-BEATS: ".../evaluation/"; transformer: ".../evaluation/transformer/"
path_to_evaluation = "/home/dev/projects/data/evaluation/"

param_grid = {
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_leaf": [2, 3, 5, 10],
}

N_OUTER = 10
N_INNER = 5
SEEDS = [0, 1, 2, 3, 42]  # repetitions with different random splits

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
targets = ["MASE", "RMSSE", "sMAPE"]

div = "aggregated"

# --------------------------------------------------
# DATA
# --------------------------------------------------

all_metrics_df = pd.read_parquet(
    f"{path_to_evaluation}results_metrics_features_train_mean.parquet"
)

all_columns = features + targets

if div == "aggregated":
    complete_data = all_metrics_df[all_columns].dropna()
else:
    complete_data = all_metrics_df.loc[
        all_metrics_df["div"] == div, all_columns
    ].dropna()

print(f"Number of datasets: {len(complete_data)} (expected 105)")

X = complete_data[features].reset_index(drop=True)

# --------------------------------------------------
# RANDOM 10-FOLD CV
# --------------------------------------------------

rows = []

for target in targets:

    y = complete_data[target].to_numpy()

    for seed in SEEDS:

        outer_cv = KFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
        inner_cv = KFold(n_splits=N_INNER, shuffle=True, random_state=seed)

        y_pred_all = np.empty_like(y, dtype=float)
        fold_train_r2 = []

        for train_idx, test_idx in outer_cv.split(X):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y[train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            grid = GridSearchCV(
                estimator=DecisionTreeRegressor(random_state=42),
                param_grid=param_grid,
                scoring="r2",
                cv=inner_cv,
                n_jobs=-1,
            )
            grid.fit(X_train_scaled, y_train)
            best_model = grid.best_estimator_

            fold_train_r2.append(r2_score(y_train, best_model.predict(X_train_scaled)))

            # store predictions at their original positions
            y_pred_all[test_idx] = best_model.predict(X_test_scaled)

        rows.append(
            {
                "target": target,
                "seed": seed,
                "cv_r2": r2_score(y, y_pred_all),  # pooled out-of-sample R²
                "mean_train_r2": np.mean(fold_train_r2),
            }
        )

results = pd.DataFrame(rows)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

summary = results.groupby("target").agg(
    cv_r2_mean=("cv_r2", "mean"),
    cv_r2_min=("cv_r2", "min"),
    cv_r2_max=("cv_r2", "max"),
    train_r2_mean=("mean_train_r2", "mean"),
)

print("\nRandom 10-fold CV, pooled out-of-sample R² across seeds")
print(summary.round(4))

results.to_csv(path_to_evaluation + f"tree_random10fold_cv_div{div}.csv", index=False)
summary.to_csv(path_to_evaluation + f"tree_random10fold_cv_summary_div{div}.csv")
