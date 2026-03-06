import pandas as pd
import numpy as np

class RiskScoringEngine:
    def __init__(self, weights: dict):
        self.weights = weights

    def compute_score(self, df: pd.DataFrame) -> pd.Series:
        # Normalize inputs to 0-1 scale
        
        # 1. Volatility
        vol_min = df['forecasted_volatility'].min()
        vol_max = df['forecasted_volatility'].max()
        vol_norm = (df['forecasted_volatility'] - vol_min) / (vol_max - vol_min + 1e-9)
        
        # 2. Regime (Assume higher state number is safer)
        regime_max = df['regime_label'].max()
        regime_norm = 1 - (df['regime_label'] / (regime_max + 1e-9))
        
        # 3. Sentiment (-1 to 1 mapped to 0-1 risk)
        sent_norm = (1 - df['sentiment_index']) / 2 
        
        # 4. Drawdown
        dd_abs = df['drawdown'].abs()
        dd_min = dd_abs.min()
        dd_max = dd_abs.max()
        dd_norm = (dd_abs - dd_min) / (dd_max - dd_min + 1e-9)
        
        score = (self.weights['volatility'] * vol_norm +
                 self.weights['regime'] * regime_norm +
                 self.weights['sentiment'] * sent_norm +
                 self.weights['drawdown'] * dd_norm)
        
        return score * 100
