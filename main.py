""" Main Script """

from pathlib import Path
import os
import random
import numpy as np
import tensorflow as tf

from src.data_loader import (
    load_and_clean,
    add_returns_features,
    build_lstm_sequences,
    train_test_split_time,
    flatten_for_rf,
)

from src.models import (
    run_naive,
    run_random_forest,
    tune_rf_random_search,
    run_lstm,
    tune_lstm_grid_search,
)

from src.evaluation import (
    ensure_results_folders,
    correlation_matrix,
    compute_metrics,
    save_oos_metrics_csv,
    plot_true_vs_pred,
)

fig_dir, metrics_dir, appendix_dir = ensure_results_folders()


""" Main """

def main():

    print("=" * 80)
    print("Gold Price Forecasting using Random Forest Regressor and Long Short Term Memory")
    print("=" * 80)
    
    # Reproducibility by setting a random.seed = 42
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # Setting parameters
    time_steps = 63
    horizon = 63
    target_col = "XAUUSD"

    # Loading and Cleaning data
    print("\n1. Loading and clearning data...")
    Cleaned_Data_Set = load_and_clean("data/raw")

    # Correlation matrix of gold prices with financial indicators (Appendix)
    correlation_path = os.path.join(
        appendix_dir,
        "Exploratory_Correlation_Matrix.png"
    )
    correlation_matrix(Cleaned_Data_Set, correlation_path)
    
    # Gold with transformed financial indicators feature set for the models
    df_ret = add_returns_features(Cleaned_Data_Set, use_ret=True,)

    feature_set_name = "Gold_with_indicators"
    df_used = df_ret
    features = ["XAUUSD", "SP500_ret", "VIX_ret", "WTI_ret", "DXY_ret"]
    print("\n" + "=" * 100)
    print(f"Feature Set: {feature_set_name}, features={features}")
    print("=" * 100)

    # Building LSTM sequence
    X_seq, y_seq, dates_arr = build_lstm_sequences(
        df=df_used,
        features=features,
        target_col=target_col,
        time_steps=time_steps,
        horizon=horizon
    )

    # Temporally splitting the data (70/30)
    print(f"\n2. Temporally splitting the data (70/30) using {feature_set_name}...")
    x_train, x_test, y_train, y_test, is_dates, oos_dates = train_test_split_time(
        X_seq, y_seq, dates_arr, train_frac=0.7
    )

    # For creating metrics table
    metrics_rows = []
    # For saving and plotting results later
    y_true_all = y_seq

    """ Naive Forecast (Baseline Model) """
    print(f"\n3. Running Naive Forecast Model (Baseline) using {feature_set_name}...")
    y_true_naive, y_pred_naive = run_naive(
        df=df_used,
        time_steps=time_steps,
        horizon=horizon,
        target_col=target_col
    )
    # Aligning and training the Naive to the LSTM/RF splits
    tr = int(np.ceil(X_seq.shape[0] * 0.7))
    y_train_naive = y_true_naive[:tr]
    y_test_naive = y_true_naive[tr:]
    is_pred_naive = y_pred_naive[:tr].ravel()
    oos_pred_naive = y_pred_naive[tr:].ravel()         
    naive_metrics = compute_metrics(y_train_naive, y_test_naive, is_pred_naive, oos_pred_naive)
    metrics_rows.append({
        "feature_set": feature_set_name,
        "model": "naive",
        **naive_metrics
    })
    naive_fig_path = os.path.join(fig_dir, "Naive_Model_Predictions_Plot.png")
    plot_true_vs_pred(dates_arr, y_true_all, is_dates, oos_dates,
                      is_pred_naive, oos_pred_naive,
                      title=f"Naive — {feature_set_name} (63-day ahead)",
                      ylabel="Gold Price (USD per ounce)",
                      out_path=naive_fig_path)
    print(f"\n Naïve Forecast Performance using {feature_set_name}\n")
    print(f"  MSE : {naive_metrics['mse_oos']:.2f}")
    print(f"  RMSE: {naive_metrics['rmse_oos']:.2f}")
    print(f"  MAE : {naive_metrics['mae_oos']:.2f}")

    """ Random Forest Model """
    print(f"\n4. Training Random Forest (Hyperparameter tuning using Random Search) using {feature_set_name}...")
    # Flattening lagged variables from LSTM sequences for RF
    x_train_rf = flatten_for_rf(x_train)
    x_test_rf = flatten_for_rf(x_test)
    best_params_rf, best_val_mse_rf, trials_df = tune_rf_random_search(x_train_rf, y_train, x_test_rf)
    # Saving top 5 hyperparameter combinations tested for Appendix
    rf_top5 = trials_df.head(5).copy()
    rf_top5 = rf_top5.round(2)
    rf_top5.to_csv(
        os.path.join(appendix_dir, "rf_hyperparameter_trials_top5.csv"),
        index=False
    )
    print("\n Best hyperparameters for RF:")
    print(f"  n_estimators : {best_params_rf['n_estimators']}")
    print(f"  max_depth : {best_params_rf['max_depth']}")
    print(f"  min_samples_split : {best_params_rf['min_samples_split']}")
    print(f"  min_samples_leaf : {best_params_rf['min_samples_leaf']}")
    print()
    best_params_rf.pop("n_jobs", None)
    # Running RF with best hyperparameter combination
    is_pred_rf, oos_pred_rf, rf_model = run_random_forest(
        x_train_rf=x_train_rf,
        y_train=y_train,
        x_test_rf=x_test_rf,
        **best_params_rf
    )
    rf_metrics = compute_metrics(y_train, y_test, is_pred_rf, oos_pred_rf)
    metrics_rows.append({
        "feature_set": feature_set_name,
        "model": "random_forest",
        **rf_metrics
    })
    rf_fig_path = os.path.join(fig_dir, "RF_Model_Predictions_Plot.png")
    plot_true_vs_pred(dates_arr, y_true_all, is_dates, oos_dates,
                      is_pred_rf, oos_pred_rf,
                      title=f"Random Forest — {feature_set_name} (63-day ahead)",
                      ylabel="Gold Price (USD per ounce)",
                      out_path=rf_fig_path)
    print(f"\n Random Forest Regressor Performance using {feature_set_name}\n")
    print(f"  MSE : {rf_metrics['mse_oos']:.2f}")
    print(f"  RMSE: {rf_metrics['rmse_oos']:.2f}")
    print(f"  MAE : {rf_metrics['mae_oos']:.2f}")

    """ LSTM Model """
    print(f"\n5. Training LSTM (Hyperparameter tuning using Grid Search) using {feature_set_name}...")
    best_params_lstm, best_val_mse_lstm, lstm_trials_df = tune_lstm_grid_search(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test
    )
    # Saving Top 5 hyperparameter combinations
    lstm_trials_df.head(5)[
        ["Rank", "lstm_units", "batch_size", "best_epoch", "best_val_mse"]
    ].round(2).to_csv(
        os.path.join(appendix_dir, "lstm_hyperparameter_trials_top5.csv"),
        index=False
    )
    print("\n Best hyperparameters for LSTM:")
    print(f"  Units : {best_params_lstm['lstm_units']}")
    print(f"  Batch size : {best_params_lstm['batch_size']}")
    print(f"  Best epoch : {best_params_lstm['best_epoch']}")
    print()
    # Running the LSTM with best hyperparameter combination (excluding epochs from best_params as we run early stopping again)
    params_lstm = best_params_lstm.copy()
    params_lstm.pop("best_epoch", None)
    lstm_res = run_lstm(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        **params_lstm
    )    
    is_pred_lstm = lstm_res["is_pred"]
    oos_pred_lstm = lstm_res["oos_pred"]
    lstm_metrics = compute_metrics(y_train, y_test, is_pred_lstm, oos_pred_lstm)
    metrics_rows.append({
        "feature_set": feature_set_name,
        "model": "lstm",
        **lstm_metrics
    })        
    lstm_fig_path = os.path.join(fig_dir, "LSTM_Model_Predictions_Plot.png")
    plot_true_vs_pred(dates_arr, y_true_all, is_dates, oos_dates,
                      is_pred_lstm, oos_pred_lstm,
                      title=f"LSTM — {feature_set_name} (63-day ahead)",
                      ylabel="Gold Price (USD per ounce)",
                      out_path=lstm_fig_path)


    print(f"\n Long Short Term Memory Performance using {feature_set_name}\n")
    print(f"  MSE : {lstm_metrics['mse_oos']:.2f}")
    print(f"  RMSE: {lstm_metrics['rmse_oos']:.2f}")
    print(f"  MAE : {lstm_metrics['mae_oos']:.2f}")


    # Saving all metrics, predictions and figures
    print("\n6. Saving all Metrics and Figures...")
    print("\n Saved all figures to:", fig_dir)
    # Performance Evaluation Metrics (OOS only)
    oos_metrics_csv_path = os.path.join(metrics_dir, "Model_Performance_Metrics_Report_Results.csv")
    df_oos = save_oos_metrics_csv(metrics_rows, oos_metrics_csv_path)  
    print("\n Saved OOS Final Report Evaluation metrics to:")
    print(oos_metrics_csv_path)

    # Conclusion: Winner per OOS metric
    print("\n7. Conclusion...")
    metrics = ["MSE", "RMSE", "MAE"]   
    print("\n" + "=" * 60)
    print("Winners by Metric")
    print("=" * 60)    
    for m in metrics:
        winner_row = df_oos.loc[df_oos[m].idxmin()]
        print(f"{m} winner: {winner_row['Model']} ({winner_row[m]:.2f})")    
    print("=" * 60)

if __name__ == "__main__":
    main()
