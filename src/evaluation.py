""" Evaluation """

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


""" Saving outputs into the results folder """

def ensure_results_folders(results_dir="results"):
    fig_dir = os.path.join(results_dir, "figures")
    metrics_dir = os.path.join(results_dir, "metrics")
    appendix_dir = os.path.join(results_dir, "appendix")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(appendix_dir, exist_ok=True)

    return fig_dir, metrics_dir, appendix_dir


""" Evaluation Metrics (MSE, RMSE, MAE) """

def compute_metrics(y_train, y_test, is_pred, oos_pred):

    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()
    
    # In sample (IS) predictions
    is_pred = np.asarray(is_pred).ravel()
    # Out of sample (OOS) predictions
    oos_pred = np.asarray(oos_pred).ravel()

    metrics = {
        "mse_is": float(mean_squared_error(y_train, is_pred)),
        "rmse_is": float(np.sqrt(mean_squared_error(y_train, is_pred))),
        "mae_is": float(mean_absolute_error(y_train, is_pred)),

        "mse_oos": float(mean_squared_error(y_test, oos_pred)),
        "rmse_oos": float(np.sqrt(mean_squared_error(y_test, oos_pred))),
        "mae_oos": float(mean_absolute_error(y_test, oos_pred)),
    }
    return metrics

    
""" Performance Evaluation Metrics for Report Table (OOS only) """

def save_oos_metrics_csv(metrics_rows, out_path):

    df = pd.DataFrame(metrics_rows)

    # Keeping only OOS metrics
    df = df[[
        "feature_set",
        "model",
        "mse_oos",
        "rmse_oos",
        "mae_oos",
    ]].copy()

    # Renaming columns for the report
    df = df.rename(columns={
        "feature_set": "Feature set",
        "model": "Model",
        "mse_oos": "MSE",
        "rmse_oos": "RMSE",
        "mae_oos": "MAE",
    })

    # Renaming rows for models
    def make_label(row):
        if row["Model"] == "naive":
            return (
                "Naive Forecast (Benchmark)"
            )
        if row["Model"] in ("random_forest", "rf"):
            return (
                "Random Forest Regression"
            )
        if row["Model"] == "lstm":
            return (
                "Long Short-Term Memory"
            )

        return {row['Model']}

    df["Model"] = df.apply(make_label, axis=1)

    # Keeping final columns only
    df = df[["Model", "MSE", "RMSE", "MAE"]]

    # Rounding values to two decimal places
    df[["MSE", "RMSE", "MAE"]] = df[["MSE", "RMSE", "MAE"]].round(2)

    # Reordering rows for clarity
    order = [
        "Naive Forecast (Benchmark)",
        "Random Forest Regression",
        "Long Short-Term Memory",
    ]
    df["Model"] = pd.Categorical(df["Model"], categories=order, ordered=True)
    df = df.sort_values("Model").reset_index(drop=True)

    df.to_csv(out_path, index=False)
    return df


""" Plotting True vs Prediction Values Graph Figures """

def plot_true_vs_pred(dates_arr, y_true_all, is_dates, oos_dates, is_pred, oos_pred,
                      title, ylabel, out_path):

    # Saving plots that compare true vs gold prices in and out of sample
    y_true_all = np.asarray(y_true_all).ravel()
    is_pred = np.asarray(is_pred).ravel()
    oos_pred = np.asarray(oos_pred).ravel()

    plt.figure(figsize=(15, 7))

    plt.plot(dates_arr, y_true_all, label="True",
             color="#444444", linewidth=2)

    plt.plot(is_dates, is_pred, label="In-Sample Prediction",
             color="dodgerblue", linewidth=2)

    plt.plot(oos_dates, oos_pred, label="Out-of-Sample Prediction",
             color="darkorange", linewidth=2)

    plt.title(title, fontsize=15)
    plt.xlabel("Date")
    plt.ylabel(ylabel)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gcf().autofmt_xdate()

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


""" Correlation Matrix between gold and financial indicators (Appendix) """

def correlation_matrix(Cleaned_Data_Set, out_path):
    
    corr_matrix = Cleaned_Data_Set[
        ["XAUUSD", "DXY", "SP500", "VIX", "WTI"]
    ].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar=True
    )

    plt.title(
        "Exploratory Correlation Matrix: Gold Price (XAUUSD) and Financial Predictors",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
