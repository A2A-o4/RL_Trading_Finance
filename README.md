# Stock Data Analysis Project

This project downloads historical stock data for Apple Inc. (AAPL), processes it, adds technical indicators using FinRL, creates visualizations, and splits it into training and testing datasets.

## Features

- Data download using yfinance
- Missing data handling
- Feature scaling
- Technical indicators calculation using stockstats:
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
- Data visualization
- Dataset splitting for training and testing

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the analysis script:

```bash
python stock_analysis.py
```

## Outputs

- `train_data.csv`: Training dataset (2013-2023)
- `test_data.csv`: Testing dataset (2023-2025)
- `plots/`: Directory containing visualizations:
  - `price_volume.png`: Stock price and volume over time
  - `macd.png`: MACD indicator
  - `rsi.png`: RSI indicator
  - `bollinger_bands.png`: Bollinger Bands
