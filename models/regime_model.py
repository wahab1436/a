import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import logging

logger = logging.getLogger(__name__)

class RegimeDetectionModel:
    def __init__(self, n_states=3):
        self.n_states = n_states
        # Using GMM for stability and free dependency (sklearn)
        self.model = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        features = df[['log_return', 'rolling_std_7', 'Volume']].dropna()
        
        if features.empty:
            raise ValueError("Insufficient data for regime fitting")
            
        self.model.fit(features)
        self.is_fitted = True
        logger.info("Regime Model fitted successfully")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise Exception("Model not fitted")
            
        features = df[['log_return', 'rolling_std_7', 'Volume']].fillna(method='ffill').fillna(method='bfill')
        states = self.model.predict(features)
        return pd.Series(states, index=df.index, name='regime_label')
