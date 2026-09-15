"""
Sentiment Analysis
==================

Two backends:

* :class:`TextBlobSentiment` — lightweight, lexicon-based. Default.
* :class:`HuggingFaceSentiment` — neural, more accurate, heavier.

And :class:`StockSentimentFetcher`, a thin Twitter client wrapper that
uses the **Twitter API v2** (the v1.1 search endpoints were retired in
2023 — the original implementation in this file referenced them).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

POSITIVE_THRESHOLD = 0.1
NEGATIVE_THRESHOLD = -0.1


def _classify_polarity(polarity: float) -> str:
    if polarity > POSITIVE_THRESHOLD:
        return "positive"
    if polarity < NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


class TextBlobSentiment:
    """Lexicon-based sentiment using TextBlob.

    Falls back to a neutral result if ``textblob`` is not installed.
    """

    def __init__(self) -> None:
        self._available = False
        try:
            from textblob import TextBlob  # type: ignore

            self._TextBlob = TextBlob
            self._available = True
        except ImportError:
            logger.warning("textblob not installed. Run: pip install textblob")

    @property
    def available(self) -> bool:
        return self._available

    def analyze(self, text: str) -> Dict[str, float | str]:
        """Score a single text. Always returns a dict."""
        if not self._available:
            return {"polarity": 0.0, "subjectivity": 0.0, "sentiment": "neutral"}

        try:
            blob = self._TextBlob(text)
        except Exception as exc:  # textblob can raise on garbage input
            logger.error("Error in TextBlob analysis: %s", exc)
            return {"polarity": 0.0, "subjectivity": 0.0, "sentiment": "neutral"}

        polarity = float(blob.sentiment.polarity)
        subjectivity = float(blob.sentiment.subjectivity)
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": _classify_polarity(polarity),
        }

    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """Score a list of texts and return a DataFrame."""
        rows = []
        for text in texts:
            row = self.analyze(text)
            row["text"] = text[:100]
            rows.append(row)
        return pd.DataFrame(rows)


class HuggingFaceSentiment:
    """Neural sentiment using a Hugging Face ``pipeline``.

    Args:
        model: Hugging Face model id. Defaults to a small English
            sentiment model. Pass ``"ProsusAI/finbert"`` for finance.
        device: Device passed straight through to ``pipeline`` (e.g.
            ``-1`` for CPU, ``0`` for first GPU).
    """

    def __init__(self, model: str = "distilbert-base-uncased-finetuned-sst-2-english", device: int = -1) -> None:
        self._available = False
        self._pipeline = None
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline("sentiment-analysis", model=model, device=device)
            self._available = True
        except ImportError:
            logger.warning(
                "transformers not installed. Run: pip install transformers torch"
            )
        except Exception as exc:
            logger.error("Could not load HuggingFace model %r: %s", model, exc)

    @property
    def available(self) -> bool:
        return self._available

    def analyze(self, text: str) -> Dict[str, float | str]:
        """Score a single text. Always returns a dict."""
        if not self._available or self._pipeline is None:
            return {"label": "NEUTRAL", "score": 0.5}
        try:
            result = self._pipeline(text[:512])[0]
        except Exception as exc:
            logger.error("Error in HuggingFace analysis: %s", exc)
            return {"label": "NEUTRAL", "score": 0.5}
        return {"label": result["label"], "score": float(result["score"])}

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> pd.DataFrame:
        """Score a list of texts in vectorized batches.

        ``batch_size`` controls throughput vs. memory; tune for your
        model and hardware.
        """
        if not self._available or self._pipeline is None:
            rows = [
                {"label": "NEUTRAL", "score": 0.5, "text": t[:100]}
                for t in texts
            ]
            return pd.DataFrame(rows)

        try:
            truncated = [t[:512] for t in texts]
            raw = self._pipeline(truncated, batch_size=batch_size)
        except Exception as exc:
            logger.error("Error in HuggingFace batch analysis: %s", exc)
            return pd.DataFrame(
                [
                    {"label": "NEUTRAL", "score": 0.5, "text": t[:100]}
                    for t in texts
                ]
            )

        rows = []
        for text, result in zip(texts, raw):
            rows.append(
                {
                    "label": result["label"],
                    "score": float(result["score"]),
                    "text": text[:100],
                }
            )
        return pd.DataFrame(rows)


class SentimentAnalyzer:
    """High-level analyzer that picks the best available backend.

    Preference order: Hugging Face (more accurate) -> TextBlob (lighter).
    """

    def __init__(self, hf_model: Optional[str] = None) -> None:
        self.textblob = TextBlobSentiment()
        self.huggingface = HuggingFaceSentiment(model=hf_model) if hf_model else HuggingFaceSentiment()

    @property
    def backend(self) -> str:
        if self.huggingface.available:
            return "huggingface"
        if self.textblob.available:
            return "textblob"
        return "none"

    def analyze_textblob(self, text: str) -> Dict:
        return self.textblob.analyze(text)

    def analyze_huggingface(self, text: str) -> Dict:
        return self.huggingface.analyze(text)

    def analyze_batch_textblob(self, texts: List[str]) -> pd.DataFrame:
        return self.textblob.analyze_batch(texts)

    def analyze_batch_huggingface(self, texts: List[str]) -> pd.DataFrame:
        return self.huggingface.analyze_batch(texts)

    @staticmethod
    def aggregate_sentiment(df: pd.DataFrame, method: str = "textblob") -> Dict:
        """Aggregate a per-text sentiment DataFrame into summary stats.

        Handles empty input and missing columns defensively.
        """
        if df is None or df.empty:
            return {
                "count": 0,
                "avg_polarity": 0.0,
                "avg_subjectivity": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
            }

        if method == "textblob":
            return {
                "count": len(df),
                "avg_polarity": float(df["polarity"].mean()),
                "avg_subjectivity": float(df["subjectivity"].mean()),
                "positive_ratio": float((df["sentiment"] == "positive").mean()),
                "negative_ratio": float((df["sentiment"] == "negative").mean()),
                "neutral_ratio": float((df["sentiment"] == "neutral").mean()),
            }
        if method == "huggingface":
            labels = df["label"].astype(str).str.upper()
            positive = float((labels == "POSITIVE").mean())
            return {
                "count": len(df),
                "positive_ratio": positive,
                "negative_ratio": float((labels == "NEGATIVE").mean()),
                "avg_score": float(df["score"].mean()),
            }
        raise ValueError(f"Unknown method: {method!r}; expected 'textblob' or 'huggingface'")


class StockSentimentFetcher:
    """Fetch tweets via Twitter API v2 and run sentiment on them.

    Twitter API v1.1 search was retired by Twitter in 2023. The original
    implementation in this file used those endpoints; this version uses
    ``tweepy.Client`` against v2.

    Args:
        bearer_token: Twitter API v2 bearer token. Can also be loaded
            from ``TWEET_BEARER_TOKEN`` env var.
    """

    def __init__(self, bearer_token: Optional[str] = None) -> None:
        import os

        self.analyzer = SentimentAnalyzer()
        self._client = None
        self._tweepy = None

        try:
            import tweepy  # type: ignore

            self._tweepy = tweepy
        except ImportError:
            logger.warning(
                "tweepy not installed. Twitter fetching will not work. "
                "Install with: pip install tweepy"
            )
            return

        token = bearer_token or os.environ.get("TWEET_BEARER_TOKEN")
        if not token:
            logger.info(
                "No Twitter bearer token provided. Set TWEET_BEARER_TOKEN env "
                "var or call set_twitter_credentials()."
            )
            return

        try:
            self._client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)
            logger.info("Twitter API v2 client initialized")
        except Exception as exc:
            logger.error("Error initializing Twitter client: %s", exc)

    def set_twitter_credentials(self, bearer_token: str) -> None:
        """Replace the bearer token (e.g. after loading from secrets)."""
        if self._tweepy is None:
            logger.error("tweepy not available")
            return
        try:
            self._client = self._tweepy.Client(
                bearer_token=bearer_token, wait_on_rate_limit=True
            )
            logger.info("Twitter API v2 client (re)initialized")
        except Exception as exc:
            logger.error("Error initializing Twitter client: %s", exc)

    def search_tweets(
        self,
        query: str,
        count: int = 100,
        lang: str = "en",
    ) -> List[Dict]:
        """Search recent tweets matching ``query``.

        Returns an empty list (and logs an error) if no credentials are
        configured.
        """
        if self._client is None:
            logger.error("Twitter API not configured")
            return []

        try:
            response = self._client.search_recent_tweets(
                query=query,
                max_results=min(count, 100),  # v2 caps at 100 per request
                tweet_fields=["created_at", "public_metrics", "author_id"],
                expansions=["author_id"],
                user_fields=["username"],
            )
        except Exception as exc:
            logger.error("Error searching tweets: %s", exc)
            return []

        if response.data is None:
            return []

        users = {u.id: u.username for u in (response.includes or {}).get("users", [])}

        results: List[Dict] = []
        for tweet in response.data:
            metrics = tweet.public_metrics or {}
            results.append(
                {
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "user": users.get(tweet.author_id, str(tweet.author_id)),
                    "retweets": metrics.get("retweet_count", 0),
                    "likes": metrics.get("like_count", 0),
                }
            )
        logger.info("Fetched %d tweets for %r", len(results), query)
        return results

    def analyze_stock_sentiment(
        self, ticker: str, count: int = 100
    ) -> pd.DataFrame:
        """Fetch tweets about ``ticker`` (``$TICKER`` query) and score them.

        Returns a DataFrame combining tweet metadata with a sentiment
        column (``label`` or ``sentiment`` depending on backend).
        """
        tweets = self.search_tweets(f"${ticker}", count=count)
        if not tweets:
            logger.warning("No tweets found for %s", ticker)
            return pd.DataFrame()

        df = pd.DataFrame(tweets)
        texts = df["text"].tolist()

        if self.analyzer.huggingface.available:
            sentiment_df = self.analyzer.analyze_batch_huggingface(texts)
        elif self.analyzer.textblob.available:
            sentiment_df = self.analyzer.analyze_batch_textblob(texts)
        else:
            logger.error("No sentiment analyzer available")
            return df

        return pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample = [
        "Stock market is crashing today!",
        "Great earnings report from Apple today.",
        "The market is stable today.",
    ]
    print("TextBlob Analysis:")
    print(analyzer.analyze_batch_textblob(sample))
    print("\nAggregate:")
    print(analyzer.aggregate_sentiment(analyzer.analyze_batch_textblob(sample), "textblob"))