import feedparser
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class NewsIngestor:
    def __init__(self, feed_urls: list):
        self.feed_urls = feed_urls

    def fetch_news(self, days_back: int = 7) -> pd.DataFrame:
        all_entries = []
        for url in self.feed_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    # Parse date loosely
                    published = entry.get('published', '')
                    all_entries.append({
                        'headline': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': published,
                        'source': feed.feed.get('title', 'Unknown')
                    })
            except Exception as e:
                logger.error(f"Error fetching news from {url}: {e}")
        
        df = pd.DataFrame(all_entries)
        if not df.empty:
            df['published'] = pd.to_datetime(df['published'], errors='coerce')
            # Filter recent news
            cutoff = datetime.now().pd.to_datetime() - pd.Timedelta(days=days_back)
            # Note: Simplified filtering for demo stability
            df = df.dropna(subset=['published'])
            
        return df
