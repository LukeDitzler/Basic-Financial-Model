# Data Documentation

## Overview
This document describes the data collection and processing pipeline for the Financial Markets Model project. The dataset combines historical price data with quarterly fundamental metrics for S&P 500 stocks.

## Data Source
- **Provider**: Yahoo Finance (via `yfinance` Python library)
- **Date Range**: 2000-01-01 to 2025-11-01
- **Tickers**: 503 S&P 500 component stocks
- **Update Frequency**: Quarterly for fundamentals, daily for price data

## Dataset Structure

### Master Dataset: `master_stock_data.csv`

**Shape**: [To be filled after generation]
- Rows: ~XXX,XXX (quarterly observations across all tickers)
- Columns: ~XXX features + metadata

**Index**: 
- `date`: Quarterly date (end of quarter)
- `ticker`: Stock ticker symbol

### Column Categories

1. **Metadata**
   - `ticker`: Stock symbol
   - `date`: Observation date (quarterly)

2. **Price & Volume Data**
   - `open`, `high`, `low`, `close`: OHLC prices
   - `volume`: Trading volume
   - `adj_close`: Adjusted closing price

3. **Fundamental Metrics** (Quarterly)
   - Income statement items
   - Balance sheet items
   - Cash flow statement items
   - Financial ratios
   - Company info metrics

## Data Collection Process

### Pipeline Steps

1. **Ticker Fetching** (`build_quarterly_ticker_df`)
   - Fetch historical price data (daily)
   - Resample to quarterly frequency
   - Fetch quarterly fundamentals
   - Fetch company info/metrics
   - Apply minimum data quality threshold (20% non-null per column)

2. **Column Standardization** (`standardize_columns`)
   - Analyze column presence across all tickers
   - Keep only columns present in ≥80% of tickers
   - Ensure rectangular dataset structure
   - Fill missing values with NaN

3. **Dataset Assembly** (`build_master_dataset`)
   - Combine all ticker dataframes
   - Sort by ticker and date
   - Save to CSV

### Data Quality Thresholds

- **Per-ticker column filtering**: Columns must have ≥20% non-null values
- **Cross-ticker column filtering**: Columns must be present in ≥80% of tickers
- **Rate limiting**: 0.5 second delay between ticker requests

## Missing Data

Missing data (NaN values) can occur for several reasons:
1. **Company-specific**: Not all companies report all metrics
2. **Time-based**: Companies may start/stop reporting certain metrics
3. **Standardization**: Columns not present for a specific ticker after standardization
4. **Data availability**: Historical data limitations from Yahoo Finance

## Usage Example

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/master_stock_data.csv', parse_dates=['date'])

# Set multi-index
df = df.set_index(['ticker', 'date'])

# Get data for a specific ticker
aapl_data = df.xs('AAPL', level='ticker')

# Get data for a specific date
q4_2023 = df.xs('2023-12-31', level='date')

# Check data coverage
print(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
print(f"Number of tickers: {df.index.get_level_values('ticker').nunique()}")
print(f"Total observations: {len(df)}")
print(f"Null percentage by column:\n{df.isnull().mean().sort_values(ascending=False).head(10)}")
```

## File Specifications

- **Format**: CSV (comma-separated values)
- **Encoding**: UTF-8
- **Size**: ~XXX MB (to be filled)
- **Location**: `data/master_stock_data.csv`

## Processing Statistics

**Collection Run: [Date]**
- Total tickers attempted: 503
- Successfully processed: XXX
- Failed tickers: XXX
- Unique columns before standardization: XXX
- Columns after standardization: XXX
- Columns dropped: XXX

### Failed Tickers
[To be populated with list of tickers that failed to process and reasons]

## Known Issues & Limitations

1. **Survivorship Bias**: Dataset only includes current S&P 500 constituents, not historical members
2. **Restatements**: Historical fundamentals may not reflect subsequent restatements
3. **Corporate Actions**: Stock splits, mergers may affect historical comparability
4. **Data Quality**: Yahoo Finance data quality varies by ticker and time period
5. **Frequency Mismatch**: Price data is available daily but fundamentals are quarterly

## Future Enhancements

- [ ] Add support for delisted companies
- [ ] Include technical indicators (RSI, MACD, etc.)
- [ ] Add macroeconomic indicators
- [ ] Implement automated daily updates
- [ ] Add data validation checks
- [ ] Include earnings call sentiment data

## Maintenance

**Last Updated**: [Date]  
**Next Scheduled Update**: [Date]  
**Maintainer**: Luke Ditzler

## References

- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [Yahoo Finance](https://finance.yahoo.com)
- [S&P 500 Components](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)

---

For questions or issues with the dataset, please open an issue in the project repository.