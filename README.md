# Financial Markets Price Prediction Model

## Project Overview

This project builds and evaluates neural network models to predict stock price movements using quarterly fundamental financial data. The goal is to learn how different model architectures and feature sets impact predictive performance for financial forecasting.

## Project Structure

```
Financial_Markets_Model/
├── data/
│   └── historical_finance_data.csv          # Raw financial data (input)
├── outputs/
│   └── model_results.csv                    # Model performance tracking (grows with each run)
├── model_evaluation.py                      # Evaluation metrics and performance tracking
├── train.py                                 # Main training script
└── README.md                                # This file
```

## Data

**File:** `historical_finance_data.csv`

**Structure:**
- Quarterly fundamental data for multiple stocks
- Columns include: Ticker, Report Date, Price Date, Close, Revenue, Gross Profit, Operating Income, Total Assets, Net Cash from Operating Activities

**Data Preparation:**
1. Tickers with fewer than 20 quarters (5 years) of data are removed
2. Data is sorted by ticker and report date
3. Log returns are computed as targets: `log(Close_t+1 / Close_t)`
4. Features are normalized using StandardScaler
5. Time-based split: First 80% of each ticker's data for training, last 20% for testing

## Models Implemented

### 1. Feed-Forward NN (Baseline)
**Architecture:** Simple 3-layer feedforward neural network
- Input → 128 units (ReLU) → 64 units (ReLU) → Output
- Dropout: 0.2
- Basic architecture for baseline comparison

**Best for:** Quick baseline and understanding data characteristics

**Typical Performance:** 55-56% directional accuracy

### 2. Deep NN with BatchNorm
**Architecture:** Deeper network with batch normalization
- Input → 256 units (ReLU + BatchNorm) → Dropout(0.3)
- 256 → 128 units (ReLU + BatchNorm) → Dropout(0.3)
- 128 → 128 units (ReLU + BatchNorm) → Dropout(0.3) [with residual connection]
- 128 → 64 units (ReLU + BatchNorm) → Dropout(0.2)
- 64 → Output

**Features:**
- Batch Normalization: Stabilizes training and allows higher learning rates
- Residual Connections: Helps gradient flow through deeper networks
- Higher Dropout: Reduces overfitting on small financial datasets

**Best for:** Improved generalization and better training dynamics

**Typical Performance:** 55-57% directional accuracy (slight improvement over baseline)

### 3. LSTM (Long Short-Term Memory) - **Not Implemented**
**Why it failed:** LSTM expects 3D sequential data (batch, sequence_length, features) but your data is 2D (batch, features). 

**Would require:** Complete data restructuring to create rolling windows of historical quarters. This adds complexity with limited benefit given sparse quarterly data.

**Better alternative for temporal patterns:** Explicit temporal features (lagged values, moving averages) work better with tabular quarterly data.

## Evaluation Metrics

All metrics computed on the test set (last 20% of each ticker's data).

### Primary Metrics

1. **Directional Accuracy (%)** - Most Important
   - Percentage of predictions where sign is correct (up/down)
   - Random baseline: 50%
   - Good: > 55%
   - Excellent: > 58%
   - Financial constraint: Difficult to exceed 60% with public fundamental data

2. **Correlation**
   - Pearson correlation between predicted and actual returns
   - Range: -1 to 1
   - Good: > 0.1
   - Excellent: > 0.3
   - Financial markets are noisy; correlations often low

3. **R² (Variance Explained)**
   - Proportion of variance in returns explained by model
   - Range: -∞ to 1 (negative means worse than predicting mean)
   - Good: > 0.05
   - Excellent: > 0.1
   - Often negative in finance due to market efficiency

### Secondary Metrics

- **RMSE (Root Mean Squared Error):** Average magnitude of prediction error
- **MAE (Mean Absolute Error):** Average absolute error in prediction
- **N_Predictions:** Number of test samples

## Performance Tracking

**File:** `outputs/model_results.csv`

Each time you train a model, a new row is added to this CSV with:
- Model name and architecture description
- Number of features used
- All evaluation metrics
- Hyperparameters and notes

**To view results:**
```bash
# Open the CSV file directly
outputs/model_results.csv
```

**Columns:**
- Model_Name: Architecture name
- Num_Features: Number of input features
- Directional_Accuracy_%: Primary performance metric
- Correlation: Correlation with actuals
- R²: Variance explained
- RMSE: Prediction error magnitude
- MAE: Mean absolute error
- N_Predictions: Test set size
- Notes: Hyperparameters and configuration details

## Usage

### Basic Training and Evaluation

```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model_evaluation import quick_eval, ModelPerformanceTracker

# Initialize performance tracker
tracker = ModelPerformanceTracker(csv_file="model_results.csv")

# After training your model:
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test).flatten().numpy()

# Quick evaluation (prints to console)
metrics = quick_eval(y_test, y_test_pred)
# Output:
# Directional Accuracy: 55.62%
# Correlation:          -0.0319
# R² (Variance):        -0.0028

# Add to tracking CSV
tracker.add_result(
    model_name="Deep NN w/ BatchNorm",
    num_features=5,
    metrics_dict=metrics,
    notes="256-128-128-64-1, ReLU, 100 epochs, lr=0.001"
)

# View comparison table
tracker.print_summary()
```

### Key Functions

**model_evaluation.py:**
- `evaluate_model(y_true, y_pred)` → Returns dict of all metrics
- `quick_eval(y_test, y_pred)` → Prints key metrics and returns dict
- `ModelPerformanceTracker` → Class to manage CSV tracking

## Hyperparameters to Experiment With

### Model Architecture
- Hidden layer sizes (wider vs deeper networks)
- Dropout rates (0.2 to 0.5)
- Batch normalization (on/off)
- Residual connections (on/off)

### Training
- Learning rate (0.0001 to 0.01)
- Batch size (32, 64, 128)
- Number of epochs (50 to 200)
- Optimizer (Adam, SGD with momentum)
- Loss function (MSE for regression)

### Data
- Number of features (5 to 50+)
- Feature engineering (ratios, growth rates, momentum)
- Target variable (absolute returns vs relative returns)
- Data normalization method

## Current Results Summary

| Model | Features | Directional Accuracy | Correlation | R² | Notes |
|-------|----------|---------------------|-------------|----|----|
| Feed-Forward NN | 5 | 55.62% | -0.0319 | -0.0028 | Basic 128-64-1 baseline |
| Deep NN w/ BatchNorm | 5 | 55.67% | 0.0951 | -0.0107 | 256-128-128-64-1, BatchNorm, Residual |

**Interpretation:**
- Both models perform similarly (~55.6% directional accuracy)
- Slight positive correlation (0.095) in Deep NN suggests some predictive signal
- Negative R² indicates models don't explain variance well
- Performance barely exceeds random (50%)
- More/better features needed for significant improvement

## Recommendations for Improvement

### 1. Add More Informative Features (Highest Priority)
Your current 5 raw features are insufficient. Consider:

**Financial Ratios:**
- P/E Ratio, P/B Ratio, P/S Ratio
- ROE (Return on Equity), ROA (Return on Assets)
- Profit margins, debt-to-equity ratio

**Growth Metrics:**
- Quarter-over-quarter revenue growth
- Year-over-year revenue/earnings growth
- Revenue acceleration (is growth rate increasing?)

**Quality Metrics:**
- Asset turnover, current ratio
- Cash flow to revenue ratio
- Working capital metrics

**Market/Momentum:**
- Stock momentum (past returns)
- Relative to market performance
- Volatility of earnings

### 2. Change Target Variable
Instead of predicting absolute returns:
- Predict **relative performance** (vs market benchmark)
- Predict **direction only** (binary classification)
- These are easier to learn and more practical

### 3. Model Architecture Improvements
- Try ensemble of multiple simpler models
- Wider, shallower networks often work better for tabular data
- Experiment with different activation functions (Tanh, GELU)
- Add learning rate scheduling

### 4. Training Strategy
- Add early stopping (stop when validation loss plateaus)
- Use validation set (not in current implementation)
- Try different optimizers (SGD with momentum, AdamW)
- Experiment with loss function weighting

### 5. Temporal Approaches
Rather than LSTM, try:
- Explicit lagged features (previous quarter values)
- Moving averages of metrics
- Trend indicators
- These are simpler and often work better with sparse quarterly data

## Challenges and Limitations

### Fundamental Challenges
1. **Market Efficiency:** Public fundamental data is quickly priced in
2. **Information Lag:** Quarterly reports are backward-looking
3. **Sparse Data:** Only 20 quarters per stock = limited training samples
4. **Missing Data:** Model lacks real-time market sentiment, news, technical indicators
5. **Non-Stationarity:** Market regimes change over time

### Data Limitations
- Only 5 basic features (need 20-50+ for better results)
- Quarterly granularity (daily/weekly would be better)
- No external data (market indices, sector performance, macro indicators)
- Limited to ~20 years of history per company

### Model Limitations
- Neural networks may overfit on small financial datasets
- Difficult to interpret why model makes predictions (black box)
- Random seed and initialization affect results
- May not generalize to different market periods

## Realistic Expectations

**What's achievable with this data:**
- 55-60% directional accuracy = Good but not trading-ready
- Correlation 0.05-0.2 = Weak but meaningful signal
- R² above 0.05 = Decent for financial data

**What's NOT achievable:**
- Perfect stock picking (no one achieves this)
- Consistent 70%+ accuracy with just fundamentals
- Beating market without additional information/alpha sources
- Predicting large price movements from data that's already public

**Professional quant funds typically:**
- Use 100+ features across multiple categories
- Combine multiple data sources and models
- Focus on relative performance, not absolute returns
- Accept that edge is often small (53-55% accuracy)

## Future Work

### Short Term (Quick Wins)
1. Add 20+ engineered features
2. Switch to relative return prediction
3. Try ensemble of 3-5 models

### Medium Term
1. Implement proper validation set
2. Add learning rate scheduling
3. Experiment with different architectures
4. Try gradient boosting models (XGBoost, LightGBM)

### Long Term
1. Add external data (market indicators, sentiment)
2. Implement proper backtesting framework
3. Build portfolio optimization layer
4. Create ensemble with multiple prediction targets
5. Explore attention mechanisms

## Getting Started

1. **Load data:**
   ```python
   master_df = pd.read_csv('data/historical_finance_data.csv', 
                            parse_dates=["Report Date", "Price Date"])
   ```

2. **Clean data (remove tickers with <20 quarters):**
   ```python
   ticker_counts = master_df.groupby("Ticker").size()
   master_df_clean = master_df[master_df["Ticker"].isin(
       ticker_counts[ticker_counts >= 20].index
   )]
   ```

3. **Prepare features and train:**
   ```python
   X_train, y_train, X_test, y_test, features, scaler, train_df, test_df = \
       prepare_financial_data_timesplit(master_df_clean, manual_features)
   ```

4. **Train model and evaluate:**
   ```python
   model = YourModel(input_dim)
   # ... training loop ...
   metrics = quick_eval(y_test, y_test_pred)
   tracker.add_result(model_name, num_features, metrics, notes)
   ```

## References and Further Reading

**Academic Foundations:**
- Fama-French Factor Model (fundamental factors that predict returns)
- Efficient Market Hypothesis (why prediction is hard)
- Momentum and Value anomalies

**Practical Resources:**
- Quantopian research (quant trading research)
- Academic papers on financial machine learning
- QuantInsti courses on machine learning in finance

**Key Insight:**
Even if your model achieves only 2-3% higher accuracy than random, that translates to significant alpha over thousands of trades. The goal isn't perfection; it's finding a small, consistent edge.

## Notes

- All monetary values and prices are as provided in the original dataset
- Model predictions are for educational purposes only
- This is not financial advice; do not use for actual trading decisions
- Financial market prediction is inherently uncertain and risky

---

**Last Updated:** November 2025  
**Status:** Active Development  
**Next Steps:** Feature engineering and model experimentation
