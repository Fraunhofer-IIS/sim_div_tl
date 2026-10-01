import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import LeaveOneGroupOut

# adjust paths in /data_creation/plots_and_analysis/config.py
path_to_evaluation = "/home/dev/projects/data/evaluation/"
path_to_data = "/home/dev/projects/data"


all_metrics_df = pd.read_parquet(
    f"{path_to_evaluation}/results_metrics_features.parquet"
)
all_metrics_df["group"] = all_metrics_df["div"] + "_" + all_metrics_df["run"]

# features = ["intermittency"]  # test single feature linear regression
features = [
    "autocorr",
    "mean",
    "trend",
    "skewness",
]  # use less correlated features for linear regression


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

    fold_importances = []
    fold_coefficients = []
    fold_pvalues = []

    for train_idx, test_idx in logo.split(X, y, groups):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # --------------------------------------
        # Linear Regression (Prediction)
        # --------------------------------------

        model = Pipeline(
            [("scaler", StandardScaler()), ("regressor", LinearRegression())]
        )

        model.fit(X_train, y_train)

        # --------------------------------------
        # OLS for coefficients + p-values
        # --------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        ols = sm.OLS(y_train, sm.add_constant(X_train_scaled)).fit()

        fold_coefficients.append(ols.params[1:])  # remove intercept

        fold_pvalues.append(ols.pvalues[1:])  # remove intercept

        # --------------------------------------
        # OOS prediction
        # --------------------------------------

        y_pred = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        # --------------------------------------
        # Permutation Importance
        # --------------------------------------

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="r2",
            n_repeats=20,
            random_state=42,
            n_jobs=-1,
        )

        fold_importances.append(perm.importances_mean)

    # ------------------------------------------------
    # Overall CV R²
    # ------------------------------------------------

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    cv_r2 = r2_score(y_true_all, y_pred_all)

    prediction_results[target] = {
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "cv_r2": cv_r2,
    }

    # ------------------------------------------------
    # Importances
    # ------------------------------------------------

    fold_importances = np.array(fold_importances)

    mean_importance = fold_importances.mean(axis=0)

    ci_lower = np.percentile(fold_importances, 2.5, axis=0)

    ci_upper = np.percentile(fold_importances, 97.5, axis=0)

    # ------------------------------------------------
    # Coefficients
    # ------------------------------------------------

    fold_coefficients = np.array(fold_coefficients)

    coef_mean = fold_coefficients.mean(axis=0)

    coef_ci_lower = np.percentile(fold_coefficients, 2.5, axis=0)

    coef_ci_upper = np.percentile(fold_coefficients, 97.5, axis=0)

    # ------------------------------------------------
    # p-values / significance
    # ------------------------------------------------

    fold_pvalues = np.array(fold_pvalues)

    median_pvalue = np.median(fold_pvalues, axis=0)

    significance_rate = (fold_pvalues < 0.05).mean(axis=0)

    # ------------------------------------------------
    # Store results
    # ------------------------------------------------

    results_dict[target] = {
        "cv_r2": cv_r2,
        "importance_mean": mean_importance,
        "importance_ci_lower": ci_lower,
        "importance_ci_upper": ci_upper,
        "coef_mean": coef_mean,
        "coef_ci_lower": coef_ci_lower,
        "coef_ci_upper": coef_ci_upper,
        "median_pvalue": median_pvalue,
        "significance_rate": significance_rate,
    }

# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------

rows = []

for i, feature in enumerate(features):

    row = {"Feature": feature}

    for target in targets:

        row[f"{target}_Coef"] = round(results_dict[target]["coef_mean"][i], 4)

        row[f"{target}_Coef_CI_Lower"] = round(
            results_dict[target]["coef_ci_lower"][i], 4
        )

        row[f"{target}_Coef_CI_Upper"] = round(
            results_dict[target]["coef_ci_upper"][i], 4
        )

        row[f"{target}_Importance"] = round(
            results_dict[target]["importance_mean"][i], 4
        )

        row[f"{target}_MedianP"] = round(results_dict[target]["median_pvalue"][i], 6)

        row[f"{target}_SignifRate"] = round(
            results_dict[target]["significance_rate"][i], 3
        )

    rows.append(row)

final_df = pd.DataFrame(rows)
final_df.to_csv(
    path_to_evaluation + f"lr_cv_importance_div{div}_oos_osd.csv",
    index=False,
)

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\nLOOSD Cross-validated R²")

for target in targets:
    print(f"{target}: " f"{results_dict[target]['cv_r2']:.4f}")

print(final_df)
