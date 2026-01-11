""" Data Loading and Cleaning """

from pathlib import Path

import os
import numpy as np
import pandas as pd


""" Loading and Cleaning Data """

def load_and_clean(data_folder):    
    # To find the data/raw path
    data_folder = Path(data_folder)
    
    #######################
    # Data Loading
    #######################
    # Loading Data and Renaming Columns (All raw datasets cover January 1 2014 to January 1 2024)
    
    # Data below downloaded from Investing.com:
    
    # Dates in the MM/DD/YYYY format to be transformed into the YYYY-MM-DD format (ISO 8601)
    
    # Daily Gold Prices (XAUUSD)
    XAUUSD = pd.read_csv(data_folder/"XAUUSD.csv", parse_dates=["Date"],
                         date_format="%m/%d/%Y")
    # Keeping only the Date and Price (Closing Price) columns
    XAUUSD = XAUUSD[["Date", "Price"]].rename(columns={
                                    "Price": "XAUUSD"})
    # Cleaning commas from numbers and converting values to numeric
    XAUUSD["XAUUSD"] = (XAUUSD["XAUUSD"].astype(str).str
        .replace(",", "", regex=False))
    XAUUSD["XAUUSD"] = pd.to_numeric(XAUUSD["XAUUSD"], errors="coerce")

    # Daily Dollar Index (DXY)
    DXY = pd.read_csv(data_folder/"DXY.csv", parse_dates=["Date"],
                         date_format="%m/%d/%Y")
    # Keeping only the Date and Price (Closing Price) columns
    DXY = DXY[["Date", "Price"]].rename(columns={
                                "Price": "DXY"})

    # Data below downloaded from FRED.com:

    # Dates in the DD/MM/YYYY format to be transformed into the YYYY-MM-DD format (ISO 8601)

    # Daily Stock Market Index (SP500)
    SP500 = pd.read_csv(data_folder/"SP500.csv", parse_dates=["observation_date"],
                         date_format="%d/%m/%Y"
                         ).rename(columns={
                                    "observation_date": "Date"})
    
    # Dates already in the YYYY-DD-MM format (ISO 8601)

    # Daily Volatility Index (VIX)
    VIX = pd.read_csv(data_folder/"VIX.csv", parse_dates=["observation_date"]
                          ).rename(columns={
                                "VIXCLS": "VIX", 
                                "observation_date": "Date"})
    
    # Daily Crude Oil Futures Prices (WTI)
    WTI = pd.read_csv(data_folder/"WTI.csv", parse_dates=["observation_date"]
                       ).rename(columns={
                            "DCOILWTICO": "WTI",
                            "observation_date": "Date"})

    # Merging all datasets and sorting by date
    Merged_Raw_Data_Sets = (XAUUSD.merge(DXY, on = "Date", how = "outer")
                        .merge(SP500, on = "Date", how = "outer")
                        .merge(VIX, on = "Date", how = "outer")
                        .merge(WTI, on = "Date", how = "outer")
                       .sort_values("Date"))
      
    #######################
    # Data Cleaning
    #######################
    # Forward-fill all financial indicators (Keeping Gold prices and Dates untouched):
    #(To account for different holidays and missing values)
    cols_to_fill = [col for col in Merged_Raw_Data_Sets.columns
                     if col not in ["Date", "XAUUSD"]]
    Merged_Raw_Data_Sets[cols_to_fill] = Merged_Raw_Data_Sets[cols_to_fill].ffill()
    
    # Dropping rows with NaN values and creating a fresh index
    Cleaned_Data_Set = Merged_Raw_Data_Sets.dropna().reset_index(drop=True)
    
    return Cleaned_Data_Set


""" Transforming Indicators into returns (Feature Engineering) """

def add_returns_features(df, use_ret=True,):

    df = df.copy().sort_values("Date").reset_index(drop=True)

    if use_ret:
        df["SP500_ret"] = df["SP500"].pct_change()
        df["VIX_ret"]   = df["VIX"].pct_change()
        df["WTI_ret"]   = df["WTI"].pct_change()
        df["DXY_ret"]   = df["DXY"].pct_change()

    df = df.dropna().reset_index(drop=True)
    return df


""" Building LSTM Sequences """

def build_lstm_sequences(df, features, target_col="XAUUSD", time_steps=63, horizon=63):

    df = df.copy().sort_values("Date").reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)

    X_raw = df[features].astype(float).values
    y_raw = df[[target_col]].astype(float).values
    dates_full = df["Date"].values

    X_seq, y_seq, dates = [], [], []
    for t in range(len(df) - time_steps - horizon + 1):
        X_seq.append(X_raw[t:t+time_steps, :])
        y_seq.append(y_raw[t + time_steps + horizon - 1, 0])
        dates.append(dates_full[t + time_steps + horizon - 1])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq).reshape(-1, 1)
    dates_arr = np.array(dates)

    return X_seq, y_seq, dates_arr


""" Temporal Splitting (70/30) """

def train_test_split_time(X_seq, y_seq, dates_arr, train_frac=0.7):

    tr = int(np.ceil(X_seq.shape[0] * train_frac))

    x_train = X_seq[:tr]
    x_test  = X_seq[tr:]
    y_train = y_seq[:tr]
    y_test  = y_seq[tr:]

    is_dates = dates_arr[:tr]
    oos_dates = dates_arr[tr:]

    return x_train, x_test, y_train, y_test, is_dates, oos_dates


""" Flattening Random Forest Data Set for Lagged features (Feature Engineering) """
def flatten_for_rf(X_seq):

    # Converts the lagged 3D windows from LSTM sequences into tabular lag features for RF
    # (From X_seq shape = (N, time_steps, n_features) to X_rf shape = (N, time_steps * n_features))
    # Hence, RF uses the same information as the LSTM 63-step window through lagged features (Feature Engineering)

    return X_seq.reshape(X_seq.shape[0], -1)