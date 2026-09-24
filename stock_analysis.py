import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import stockstats for technical indicators
from stockstats import StockDataFrame

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="darkgrid")

# Download historical stock data for AAPL
print("Downloading AAPL stock data from 2013 to 2025...")
ticker = "AAPL"
start_date = "2013-01-01"
end_date = "2025-01-01"  # Note: Will only download data up to current date

# Download data
data = yf.download(ticker, start=start_date, end=end_date)
print(f"Downloaded {len(data)} records")

# Inspect the data's column structure
print(f"Column structure: {type(data.columns)}")
print(f"Columns: {data.columns}")

# Handle MultiIndex columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    # Flatten the multi-index columns to a single level
    data.columns = [col[0] for col in data.columns]
    print("Flattened MultiIndex columns to a single level")

# Data Investigation
print("\nData Overview:")
print(data.head())
print("\nData Information:")
print(data.info())

print("\nChecking for missing values:")
missing_values = data.isnull().sum()
print(missing_values)

# Handle missing data - Forward fill
if missing_values.sum() > 0:
    print("Handling missing values with forward fill method...")
    data = data.fillna(method='ffill')
    # For any remaining NaN values at the beginning, fill with backfill
    data = data.fillna(method='bfill')
    print("Missing values after handling:")
    print(data.isnull().sum())

# Feature Scaling
print("\nApplying feature scaling...")
scaler = MinMaxScaler()
# Get all columns for scaling
cols_to_scale = list(data.columns)
data_scaled = data.copy()
data_scaled.loc[:, cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

# Add Technical Indicators using stockstats
print("\nAdding technical indicators using stockstats...")

# Convert the dataframe to lowercase column names to work with stockstats
data_for_indicators = data.copy()
# Make sure column names are lowercase for stockstats
data_for_indicators.columns = [x.lower() for x in data_for_indicators.columns]

# Convert to StockDataFrame
stock_df = StockDataFrame(data_for_indicators)

# Calculate MACD
print("Calculating MACD...")
macd = stock_df['macd']  # MACD line
macds = stock_df['macds']  # MACD signal line
macdh = stock_df['macdh']  # MACD histogram

# Calculate RSI
print("Calculating RSI...")
rsi = stock_df['rsi_14']  # 14-day RSI

# Calculate Bollinger Bands
print("Calculating Bollinger Bands...")
boll = stock_df['boll']  # Middle band
boll_ub = stock_df['boll_ub']  # Upper band
boll_lb = stock_df['boll_lb']  # Lower band

# Add these indicators to the original dataframe
data['macd'] = macd
data['macds'] = macds
data['macdh'] = macdh
data['rsi_14'] = rsi
data['boll'] = boll
data['boll_ub'] = boll_ub
data['boll_lb'] = boll_lb

# Drop rows with NaN values after adding indicators (usually the first few rows)
data.dropna(inplace=True)
print(f"Data shape after adding technical indicators: {data.shape}")

# Data Visualization
print("\nCreating visualizations...")

# Create folder for plots if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Plot 1: Price and Volume
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
ax1.set_title(f'{ticker} Stock Price (2013-2025)')
ax1.plot(data.index, data['Close'], color='blue', label='Close Price')
ax1.set_ylabel('Price ($)')
ax1.grid(True)
ax1.legend()

# Volume subplot
ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
ax2.set_ylabel('Volume')
ax2.grid(True)
plt.tight_layout()
plt.savefig('plots/price_volume.png')
plt.close()

# Plot 2: MACD
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['macd'], label='MACD', color='blue')
plt.plot(data.index, data['macds'], label='Signal Line', color='red')
plt.bar(data.index, data['macdh'], label='Histogram', color='green', alpha=0.5)
plt.title(f'{ticker} MACD Indicator')
plt.legend()
plt.grid(True)
plt.savefig('plots/macd.png')
plt.close()

# Plot 3: RSI
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['rsi_14'], color='purple')
plt.axhline(y=70, color='r', linestyle='-', alpha=0.5)
plt.axhline(y=30, color='g', linestyle='-', alpha=0.5)
plt.title(f'{ticker} RSI Indicator')
plt.fill_between(data.index, data['rsi_14'], 70, where=(data['rsi_14'] >= 70), color='red', alpha=0.3, interpolate=True)
plt.fill_between(data.index, data['rsi_14'], 30, where=(data['rsi_14'] <= 30), color='green', alpha=0.3, interpolate=True)
plt.grid(True)
plt.savefig('plots/rsi.png')
plt.close()

# Plot 4: Bollinger Bands
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.5)
plt.plot(data.index, data['boll_ub'], label='Upper Band', color='orange', alpha=0.7)
plt.plot(data.index, data['boll'], label='Middle Band', color='red', alpha=0.7)
plt.plot(data.index, data['boll_lb'], label='Lower Band', color='orange', alpha=0.7)
plt.fill_between(data.index, data['boll_ub'], data['boll_lb'], alpha=0.1, color='gray')
plt.title(f'{ticker} Bollinger Bands')
plt.legend()
plt.grid(True)
plt.savefig('plots/bollinger_bands.png')
plt.close()

# Data Splitting
print("\nSplitting data into training and testing sets...")

# Define the split date
split_date = '2023-01-01'
train = data[data.index < split_date]
test = data[data.index >= split_date]

print(f"Training data shape: {train.shape} (from {train.index.min().date()} to {train.index.max().date()})")
print(f"Testing data shape: {test.shape} (from {test.index.min().date()} to {test.index.max().date()})")

# Save the data to CSV files
train.to_csv('train_data.csv')
test.to_csv('test_data.csv')

print("\nAnalysis complete. Data has been processed and saved.")
print("Visualizations have been saved to the 'plots' directory.") 