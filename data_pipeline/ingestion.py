import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MarketDataIngestor:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def fetch_data(self, ticker: str, start_date: str) -> pd.DataFrame:
        logger.info(f"Fetching market data for {ticker}")
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Handle MultiIndex columns from newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure Date is a column
            if 'Date' not in df.columns:
                df.reset_index(inplace=True)
            
            df['Ticker'] = ticker
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def save_raw(self, df: pd.DataFrame, ticker: str):
        if df.empty:
            return
        
        filename = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.storage_path, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved raw data to {filepath}")
