import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import logging

logger = logging.getLogger(__name__)

class VolatilityForecastModel:
    def __init__(self):
        self.model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        self.is_fitted = False

    def prepare_target(self, df: pd.DataFrame, horizon=7):
        df['future_vol'] = df['log_return'].rolling(window=horizon).std().shift(-horizon)
        return df

    def fit(self, df: pd.DataFrame):
        df = self.prepare_target(df)
        df = df.dropna()
        
        feature_cols = ['rolling_std_7', 'rolling_std_30', 'log_return_lag1', 'Volume']
        # Ensure all features exist
        available_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[available_cols]
        y = df['future_vol']
        
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Volatility Model fitted")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise Exception("Model not fitted")
            
        feature_cols = ['rolling_std_7', 'rolling_std_30', 'log_return_lag1', 'Volume']
        available_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[available_cols].fillna(method='ffill').fillna(0)
        preds = self.model.predict(X)
        return pd.Series(preds, index=df.index, name='forecasted_volatility')
