import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
AAPL_PATH = DATA_DIR / "AAPL.csv"

def ComputeDataAAPL(window):
    df = pd.read_csv(AAPL_PATH)

    # convert to datetime

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.reset_index(drop=True)

    #use only this columns

    df = df[["date", "open", "high", "low", "adj_close", "volume"]]
    df = df.dropna()

    # create features

    df["log_ret"] = np.log(df["adj_close"]).diff()
    df = df.dropna(subset=["log_ret"])

    df["future_ret"] = df["log_ret"].rolling(window).sum().shift(-window)
    df["target"] = (df["future_ret"] > 0).astype(int)
    df = df.dropna(subset=["future_ret"])

    df["log_volume"] = np.log(df["volume"] + 1)
    df = df.dropna(subset=["log_volume"])

    # debug
    assert df.isna().sum().sum() == 0

    model_df = df[["open", "high", "low", "adj_close", "log_ret", "log_volume", "target"]].copy()
    model_df = model_df.reset_index(drop=True)

    return df, model_df

def create_sequences(model_df, seq_len):
    X = []
    y = []

    features = model_df[["log_ret", "log_volume", "open", "high", "low", "adj_close"]].values
    targets = model_df["target"].values

    for i in range(seq_len - 1, len(model_df)):
        X.append(features[i-seq_len+1 : i+1])
        y.append(targets[i])

    return np.array(X), np.array(y)

def split_sets(X, y):
    split_idx = int(0.8 * len(X))

    X_train = X[:split_idx]
    y_train = y[:split_idx]

    X_val = X[split_idx:]
    y_val = y[split_idx:]

    return X_train, y_train, X_val, y_val

if __name__ == "__main__":
    df, model_df = ComputeDataAAPL(window=5)
    X, y = create_sequences(model_df, 20)
    X_train, y_train, X_val, y_val = split_sets(X, y)

    # print(df["date"].diff().value_counts().head())

    print(X.shape, y.shape)
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)