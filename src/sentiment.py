"""
Sentiment Analysis
Analyze sentiment from social media (Twitter) for stocks.
"""

import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment from text data."""
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self._textblob_available = False
        self._transformers_available = False
        
        # Try to import textblob
        try:
            from textblob import TextBlob
            self.TextBlob = TextBlob
            self._textblob_available = True
        except ImportError:
            logger.warning("textblob not installed. Run: pip install textblob")
        
        # Try to import transformers
        try:
            from transformers import pipeline
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            self._transformers_available = True
        except ImportError:
            logger.warning("transformers not installed. Run: pip install transformers torch")
    
    def analyze_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with polarity and subjectivity
        """
        if not self._textblob_available:
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}
        
        try:
            blob = self.TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment
            }
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}
    
    def analyze_huggingface(self, text: str) -> Dict:
        """
        Analyze sentiment using Hugging Face transformers.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with label and score
        """
        if not self._transformers_available:
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        try:
            # Truncate text if too long
            text = text[:512]
            result = self.sentiment_pipeline(text)[0]
            return result
        except Exception as e:
            logger.error(f"Error in HuggingFace analysis: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def analyze_batch_textblob(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with sentiment results
        """
        results = []
        
        for text in texts:
            result = self.analyze_textblob(text)
            result['text'] = text[:100]  # Truncate for display
            results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_batch_huggingface(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for a batch using transformers.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with sentiment results
        """
        results = []
        
        for text in texts:
            result = self.analyze_huggingface(text)
            result['text'] = text[:100]
            results.append(result)
        
        return pd.DataFrame(results)
    
    def aggregate_sentiment(self, df: pd.DataFrame, method: str = 'textblob') -> Dict:
        """
        Aggregate sentiment from a DataFrame.
        
        Args:
            df: DataFrame with sentiment analysis results
            method: 'textblob' or 'huggingface'
            
        Returns:
            Dictionary with aggregate statistics
        """
        if method == 'textblob':
            return {
                'avg_polarity': df['polarity'].mean(),
                'avg_subjectivity': df['subjectivity'].mean(),
                'positive_ratio': (df['sentiment'] == 'positive').sum() / len(df),
                'negative_ratio': (df['sentiment'] == 'negative').sum() / len(df),
                'neutral_ratio': (df['sentiment'] == 'neutral').sum() / len(df),
            }
        elif method == 'huggingface':
            positive = (df['label'] == 'POSITIVE').sum() / len(df)
            return {
                'positive_ratio': positive,
                'negative_ratio': 1 - positive,
                'avg_score': df['score'].mean()
            }
        
        return {}


class StockSentimentFetcher:
    """Fetch tweets and analyze stock sentiment."""
    
    def __init__(self):
        """Initialize stock sentiment fetcher."""
        self.analyzer = SentimentAnalyzer()
        self._twitter_available = False
        
        # Try to import tweepy
        try:
            import tweepy
            self.tweepy = tweepy
            self._twitter_available = True
        except ImportError:
            logger.warning("tweepy not installed. Twitter fetching will not work.")
    
    def set_twitter_credentials(
        self,
        api_key: str,
        api_secret: str,
        access_token: str,
        access_token_secret: str
    ):
        """
        Set Twitter API credentials.
        
        Args:
            api_key: Twitter API key
            api_secret: Twitter API secret
            access_token: Access token
            access_token_secret: Access token secret
        """
        if not self._twitter_available:
            logger.error("tweepy not available")
            return
        
        try:
            auth = tweepy.OAuthHandler(api_key, api_secret)
            auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(auth)
            logger.info("Twitter API authenticated successfully")
        except Exception as e:
            logger.error(f"Error authenticating Twitter API: {e}")
    
    def search_tweets(
        self,
        query: str,
        count: int = 100,
        lang: str = 'en'
    ) -> List[Dict]:
        """
        Search for tweets containing a query.
        
        Args:
            query: Search query (e.g., stock ticker)
            count: Number of tweets to fetch
            lang: Language filter
            
        Returns:
            List of tweet dictionaries
        """
        if not self._twitter_available or not hasattr(self, 'api'):
            logger.error("Twitter API not configured")
            return []
        
        try:
            tweets = self.api.search_tweets(
                q=query,
                count=count,
                lang=lang,
                tweet_mode='extended'
            )
            
            results = []
            for tweet in tweets:
                results.append({
                    'id': tweet.id,
                    'text': tweet.full_text,
                    'created_at': tweet.created_at,
                    'user': tweet.user.screen_name,
                    'retweets': tweet.retweet_count,
                    'likes': tweet.favorite_count
                })
            
            logger.info(f"Fetched {len(results)} tweets for '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []
    
    def analyze_stock_sentiment(self, ticker: str, count: int = 100) -> pd.DataFrame:
        """
        Fetch and analyze sentiment for a stock ticker.
        
        Args:
            ticker: Stock ticker symbol
            count: Number of tweets to analyze
            
        Returns:
            DataFrame with tweets and sentiment
        """
        # Search for tweets
        tweets = self.search_tweets(f"${ticker}", count=count)
        
        if not tweets:
            logger.warning(f"No tweets found for {ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(tweets)
        
        # Analyze sentiment
        texts = df['text'].tolist()
        
        # Use whichever analyzer is available
        if self.analyzer._transformers_available:
            sentiment_df = self.analyzer.analyze_batch_huggingface(texts)
        elif self.analyzer._textblob_available:
            sentiment_df = self.analyzer.analyze_batch_textblob(texts)
        else:
            logger.error("No sentiment analyzer available")
            return df
        
        # Combine results
        df = pd.concat([df, sentiment_df], axis=1)
        
        return df


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Test with sample texts
    texts = [
        "Stock market is crashing today!",
        "Great earnings report from Apple today.",
        "The market is stable today."
    ]
    
    print("TextBlob Analysis:")
    df = analyzer.analyze_batch_textblob(texts)
    print(df)
    
    print("\nAggregate:")
    print(analyzer.aggregate_sentiment(df, 'textblob'))
