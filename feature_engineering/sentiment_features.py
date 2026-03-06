import pandas as pd
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class SentimentFeatureEngineer:
    def __init__(self):
        # Load FinBERT for sentiment analysis (Local, Free)
        self.classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def analyze_news(self, headlines: list) -> pd.DataFrame:
        results = []
        # Process in batches to avoid overload
        for headline in headlines[:50]: # Limit for demo speed
            try:
                if not headline or not isinstance(headline, str):
                    continue
                res = self.classifier(headline)[0]
                results.append({
                    'headline': headline,
                    'label': res['label'],
                    'score': res['score']
                })
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
        
        return pd.DataFrame(results)

    def aggregate_sentiment(self, df_sentiment: pd.DataFrame) -> float:
        if df_sentiment.empty:
            return 0.0
        
        mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
        df_sentiment['score_numeric'] = df_sentiment['label'].map(mapping)
        
        weighted_sum = (df_sentiment['score_numeric'] * df_sentiment['score']).sum()
        total_confidence = df_sentiment['score'].sum()
        
        if total_confidence == 0:
            return 0.0
            
        return weighted_sum / total_confidence
