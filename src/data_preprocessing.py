import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def select_top_features(master_df, target_col="Close", num_features=20):
    """
    Automatically choose the best predictive features based on correlation 
    with future log returns (same target used in training).
    Returns a list of column names to be used as manual_features.
    """

    df = master_df.copy()
    eps = 1e-8
    
    # Compute future log-return target
    df["target"] = df.groupby("Ticker")[target_col].transform(
        lambda x: np.log((x.shift(-1) + eps) / (x + eps))
    )
    
    df = df.dropna(subset=["target"])
    
    # Exclude columns that should never be used as model inputs
    exclude_cols = {"Ticker", "Report Date", target_col, "target"}
    candidate_cols = [
        col for col in df.columns 
        if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)
    ]
    
    if len(candidate_cols) == 0:
        raise ValueError("No numeric candidate columns found for feature selection.")
    
    # Compute correlations with target
    correlations = df[candidate_cols].corrwith(df["target"]).abs().sort_values(ascending=False)
    
    # Pick top N
    selected_features = correlations.head(num_features).index.tolist()
    
    return selected_features


def prepare_financial_data_timesplit(df, manual_features, target_col="Close", 
                                      train_ratio=0.8, dtype=torch.float32):
    """Time-based train/test split per ticker"""
    df = df.copy()
    df = df.sort_values(["Ticker", "Report Date"])
    
    eps = 1e-8
    df["target"] = df.groupby("Ticker")[target_col].transform(
        lambda x: np.log((x.shift(-1) + eps) / (x + eps))
    )
    df = df.dropna(subset=["target"])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Time-based split per ticker
    train_list = []
    test_list = []
    
    for ticker in df["Ticker"].unique():
        ticker_data = df[df["Ticker"] == ticker].copy()
        n = len(ticker_data)
        train_size = int(n * train_ratio)
        
        train_list.append(ticker_data.iloc[:train_size])
        test_list.append(ticker_data.iloc[train_size:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    # Prepare tensors
    X_train = train_df[manual_features].values.astype(np.float32)
    y_train = train_df["target"].values.astype(np.float32).reshape(-1, 1)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    X_test = test_df[manual_features].values.astype(np.float32)
    X_test = scaler.transform(X_test)
    y_test = test_df["target"].values.astype(np.float32).reshape(-1, 1)
    
    X_train_tensor = torch.tensor(X_train, dtype=dtype)
    y_train_tensor = torch.tensor(y_train, dtype=dtype)
    X_test_tensor = torch.tensor(X_test, dtype=dtype)
    y_test_tensor = torch.tensor(y_test, dtype=dtype)
    
    train_df_clean = train_df[["Ticker", "Report Date"] + manual_features + [target_col, "target"]].copy()
    test_df_clean = test_df[["Ticker", "Report Date"] + manual_features + [target_col, "target"]].copy()
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            manual_features, scaler, train_df_clean, test_df_clean)