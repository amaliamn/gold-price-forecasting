""" Models """

import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import randint

""" Naive Forecast Model (Baseline) """

def run_naive(df, time_steps=63, horizon=63, target_col="XAUUSD"):

    df = df.copy().sort_values("Date").reset_index(drop=True)
    y = df[target_col].astype(float).values

    # Naive persistence forecast considering the lookback window (time_steps) and a forecast horizon (horizon)
    y_true = []
    y_pred = []
    for t in range(len(df) - time_steps - horizon + 1):
        idx_y = t + time_steps + horizon - 1
        idx_today = t + time_steps - 1

        y_true.append(y[idx_y])
        y_pred.append(y[idx_today])

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)

    return y_true, y_pred


""" Random Forest Regressor Model """

# Using default bootstrap sampling (automatically has bootstrap = True)
def run_random_forest(x_train_rf, y_train, x_test_rf,
                      n_estimators=600,
                      max_depth=None,
                      min_samples_leaf=10,
                      min_samples_split=20,
                      max_features="sqrt",
                      random_state=42):

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=random_state,
        verbose = 1,
        n_jobs=-1
    )

    rf.fit(x_train_rf, y_train.ravel())

    is_pred = rf.predict(x_train_rf)
    oos_pred = rf.predict(x_test_rf)

    return is_pred, oos_pred, rf


""" Random Forest Hyperparameter tuning (Random Search) """

def tune_rf_random_search(
    x_train_rf, y_train, x_test_rf=None,
    n_iter=20,
    n_splits=5,
    random_state=42,
    verbose=1
):
    y_train_1d = np.asarray(y_train).ravel()

    param_dist = {
        "n_estimators": randint(200, 800),
        "max_depth": randint(10, 30),
        "min_samples_split": randint(20, 80),
        "min_samples_leaf": randint(10, 80),
        "max_features": ["sqrt"],
    }

    base_rf = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1
    )

    cv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
        verbose=verbose,
        refit=False
    )

    search.fit(x_train_rf, y_train_1d)

    # Recording best parameters
    best_params = search.best_params_
    best_params["random_state"] = random_state
    best_params["max_features"] = "sqrt"
    # Making MSE positive
    best_val_mse = -float(search.best_score_)

    # For the Appendix table to show top 5 combinations
    cvres = pd.DataFrame(search.cv_results_)
    params_df = pd.DataFrame(cvres["params"].tolist())

    trials_df = pd.DataFrame({
        "Rank": cvres["rank_test_score"].astype(int),
        "n_estimators": params_df["n_estimators"].astype(int),
        "max_depth": params_df["max_depth"].astype(int),
        "min_samples_leaf": params_df["min_samples_leaf"].astype(int),
        "min_samples_split": params_df["min_samples_split"].astype(int),
        "mean_val_MSE": (-cvres["mean_test_score"]).astype(float),
    }).sort_values("Rank").reset_index(drop=True)

    return best_params, best_val_mse, trials_df


""" Long Short Term Memory Model """

def run_lstm(
    x_train, y_train, x_test,
    max_epochs=200,
    lstm_units=64,
    dropout=0.2,
    patience=10,
    lr=1e-3,
    batch_size=32,
    val_frac=0.10,
    seed=42,
    verbose=1
):
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    # Only standardizing the training data set (Standardisation: mean = 0, std = 1)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Reshaping to 2D so scaling is applied for each feature across all timesteps
    x_train_s = scaler_X.fit_transform(
        x_train.reshape(-1, x_train.shape[2])
    ).reshape(x_train.shape)

    x_test_s = scaler_X.transform(
        x_test.reshape(-1, x_test.shape[2])
    ).reshape(x_test.shape)

    y_train_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    y_train_s = scaler_y.fit_transform(y_train_2d)

    # LSTM Structure: 1 hidden layer model
    model = tf.keras.Sequential([
        # Input layer: Input sequence (63 days)
        tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        # Hidden layer 1: Reads the input sequence (63 days) and outputs the final hiddden state)
        tf.keras.layers.LSTM(lstm_units, dropout=dropout),
        # Output layer: Predicted (next 63rd day ahead) gold price
        tf.keras.layers.Dense(1)
    ])

    # Using the Adam optimiser with the default learning rate = 0.001 through lr and the Loss function = MSE
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse"
    )

    # Using Early stopping to prevent overfitting
    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        x_train_s,
        y_train_s,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_split=val_frac,
        shuffle=False,
        callbacks=[cb],
        verbose=verbose
    )

    # Identifing the epoch with lowest validation loss function (MSE) in scaled units
    val_losses = history.history["val_loss"]
    best_val_loss_scaled = float(np.min(val_losses))
    best_epoch = int(np.argmin(val_losses) + 1)

    # Converting scaled MSE to the original gold units
    y_std = float(scaler_y.scale_[0])
    best_val_loss = best_val_loss_scaled * (y_std ** 2)

    # Predictions (In sample (IS) and Out of sample (OOS))
    is_pred = scaler_y.inverse_transform(
        model.predict(x_train_s, verbose=0)
    ).ravel()

    oos_pred = scaler_y.inverse_transform(
        model.predict(x_test_s, verbose=0)
    ).ravel()

    return {
        "is_pred": is_pred,
        "oos_pred": oos_pred,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch
    }


""" LSTM Hyperparameter tuning (Grid Search) """

def tune_lstm_grid_search(
    x_train, y_train, x_test,
    batch_sizes=(16, 32, 64, 128),
    neurons_list=(32, 64, 128),
    lr=1e-3,
    patience=10,
    max_epochs=200,
    val_frac=0.10,
    dropout=0.2,
    seed=42,
    verbose=1
):
    best_params = None
    best_val_mse = float("inf")

    # To store all the trials
    trials = []

    for units in neurons_list:
        for bs in batch_sizes:
            res = run_lstm(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                max_epochs=max_epochs,
                lstm_units=units,
                dropout=dropout,
                patience=patience,
                lr=lr,
                batch_size=bs,
                val_frac=val_frac,
                seed=seed,
                verbose=0
            )

            # Saving the hyperparameter combination
            trials.append({
                "lstm_units": units,
                "batch_size": bs,
                "best_epoch": res["best_epoch"],
                "best_val_mse": res["best_val_loss"]
            })

            # To show the process in the main.py
            if verbose:
                print(f"[LSTM] units={units}, batch={bs}, val_MSE={res['best_val_loss']:.4f}")

            # Keeping the combination with the smallest validation loss function (MSE)
            if res["best_val_loss"] < best_val_mse:
                best_val_mse = res["best_val_loss"]
                best_params = {
                    "max_epochs": max_epochs,
                    "lstm_units": units,
                    "dropout": dropout,
                    "patience": patience,
                    "lr": lr,
                    "batch_size": bs,
                    "val_frac": val_frac,
                    "seed": seed,
                    "verbose": 1,
                    "best_epoch": res["best_epoch"]
                }
    trials_df = pd.DataFrame(trials).sort_values("best_val_mse").reset_index(drop=True)
    trials_df["Rank"] = trials_df.index + 1

    return best_params, best_val_mse, trials_df
