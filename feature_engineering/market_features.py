import pandas as pd
import numpy as np
import ta

class MarketFeatureEngineer:
    def __init__(self):
        pass

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values('Date')
        
        # 1. Return Based
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rolling_mean_7'] = df['log_return'].rolling(window=7).mean()
        df['rolling_mean_30'] = df['log_return'].rolling(window=30).mean()
        
        # 2. Volatility Based
        df['rolling_std_7'] = df['log_return'].rolling(window=7).std()
        df['rolling_std_30'] = df['log_return'].rolling(window=30).std()
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(
            high=df['High'], low=df['Low'], close=df['Close'], window=14
        )
        
        # 3. Momentum
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # 4. Risk Indicators
        df['rolling_max'] = df['Close'].rolling(window=252, min_periods=1).max()
        df['drawdown'] = (df['Close'] - df['rolling_max']) / df['rolling_max']
        
        # Lag features to prevent leakage
        feature_cols = [c for c in df.columns if c not in ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        for col in feature_cols:
            df[f'{col}_lag1'] = df[col].shift(1)
            
        return df.dropna()
