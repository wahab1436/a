import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    @staticmethod
    def validate(df: pd.DataFrame, ticker: str) -> bool:
        if df.empty:
            logger.error(f"Validation Failed: Empty dataframe for {ticker}")
            return False

        issues = []
        
        if 'Date' not in df.columns:
            issues.append("Missing Date column")
            return False
            
        if df['Date'].isnull().any():
            issues.append("Missing timestamps")
        
        if df['Date'].duplicated().any():
            issues.append("Duplicate rows detected")
            df = df.drop_duplicates(subset=['Date'])
        
        critical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in critical_cols:
            if col in df and df[col].isnull().any():
                issues.append(f"Null values in {col}")
        
        if issues:
            logger.warning(f"Validation issues for {ticker}: {issues}")
        
        return True
