import unittest
import pandas as pd
import numpy as np
from feature_engineering.market_features import MarketFeatureEngineer

class TestFeatureEngineering(unittest.TestCase):
    def test_no_leakage(self):
        dates = pd.date_range(start='2023-01-01', periods=100)
        df = pd.DataFrame({
            'Date': dates,
            'Close': np.random.rand(100) * 100,
            'High': np.random.rand(100) * 100,
            'Low': np.random.rand(100) * 100,
            'Volume': np.random.rand(100) * 1000
        })
        
        engine = MarketFeatureEngineer()
        result = engine.compute_features(df)
        
        self.assertTrue('log_return_lag1' in result.columns)
        self.assertEqual(result['log_return_lag1'].iloc[1], result['log_return'].iloc[0])

if __name__ == '__main__':
    unittest.main()
