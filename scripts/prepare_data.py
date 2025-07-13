import pandas as pd
import numpy as np 
# Load Sensex data
stock = pd.read_excel("data/sensex_2007_2011.csv.xlsx", skiprows=2, header=None)
stock.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
stock['Date'] = pd.to_datetime(stock['Date'])


# Create T+1 binary label
stock['Close_t+1'] = stock['Close'].shift(-1)
stock['label'] = (stock['Close_t+1'] > stock['Close']).astype(int)
stock = stock.dropna(subset=['label'])

# Load precomputed news features
news = pd.read_csv("data/daily_news_features.csv")
news['Date'] = pd.to_datetime(news['Date'])

# Merge
merged = pd.merge(stock, news, on='Date', how='inner')
merged = merged.dropna().reset_index(drop=True)

# Save
merged.to_csv("data/processed_dataset_labeled.csv", index=False)
