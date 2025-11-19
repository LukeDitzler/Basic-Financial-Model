import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ticker_predictions(ticker, train_df, test_df, test_predictions, target_col="Close"):
    """Plot actual vs predicted prices for a single ticker"""
    train_ticker = train_df[train_df["Ticker"] == ticker].copy()
    test_ticker = test_df[test_df["Ticker"] == ticker].copy()
    
    if len(train_ticker) == 0 or len(test_ticker) == 0:
        print(f"Skipping {ticker}: insufficient data")
        return
    
    test_mask = test_df["Ticker"] == ticker
    ticker_predictions = test_predictions[test_mask.values]
    
    start_price = train_ticker[target_col].iloc[0]
    
    # Reconstruct prices
    train_prices = [start_price]
    for log_return in train_ticker["target"].values:
        train_prices.append(train_prices[-1] * np.exp(log_return))
    
    test_actual_prices = [train_prices[-1]]
    for log_return in test_ticker["target"].values:
        test_actual_prices.append(test_actual_prices[-1] * np.exp(log_return))
    
    test_pred_prices = [train_prices[-1]]
    for log_return in ticker_predictions:
        test_pred_prices.append(test_pred_prices[-1] * np.exp(log_return))
    
    # Build plot data
    data_for_plot = []
    for i in range(len(train_prices)):
        data_for_plot.append({"Quarter": i, "Price": train_prices[i], "Type": "Train Actual"})
    
    for i in range(1, len(test_actual_prices)):
        data_for_plot.append({"Quarter": len(train_prices) - 1 + i, "Price": test_actual_prices[i], "Type": "Test Actual"})
    
    for i in range(1, len(test_pred_prices)):
        data_for_plot.append({"Quarter": len(train_prices) - 1 + i, "Price": test_pred_prices[i], "Type": "Test Predicted"})
    
    df_plot = pd.DataFrame(data_for_plot)
    
    # Plot
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df_plot, x="Quarter", y="Price", hue="Type", style="Type", markers=True)
    plt.axvline(x=len(train_prices)-1, color='gray', linestyle='--', linewidth=2, label="Train/Test Split")
    plt.title(f"{ticker} - Price Reconstruction from Log Returns\n(Train: first 80% quarters, Test: last 20% quarters)")
    plt.xlabel("Quarter Index")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()