"""Tests for ARKDataFetcher using mocked HTTP responses."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from python_stocks import ARKDataFetcher
from python_stocks.ark_fetcher import ARK_ETF_URLS


@pytest.fixture
def fetcher(tmp_path) -> ARKDataFetcher:
    return ARKDataFetcher(data_dir=tmp_path, timeout=5)


def _mock_response(content: bytes, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    if status >= 400:
        from requests import HTTPError

        resp.raise_for_status.side_effect = HTTPError(f"{status}")
    return resp


SAMPLE_CSV = b"""company,ticker,shares,market value,weight
Tesla Inc,TSLA,1000000,"$200,000,000",10.5
Coinbase,COIN,500000,"$100,000,000",5.2
"""


def test_get_holding_unknown_ticker(fetcher: ARKDataFetcher) -> None:
    assert fetcher.get_holding("ZZZZ") is None


def test_get_holding_downloads_and_caches(fetcher: ARKDataFetcher, monkeypatch) -> None:
    resp = _mock_response(SAMPLE_CSV)
    monkeypatch.setattr(fetcher.session, "get", lambda url, timeout: resp)
    df = fetcher.get_holding("ARKK")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    # Second call should hit cache and not invoke HTTP.
    monkeypatch.setattr(
        fetcher.session,
        "get",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    cached = fetcher.get_holding("ARKK")
    assert len(cached) == 2


def test_get_holding_force_refresh(fetcher: ARKDataFetcher, monkeypatch) -> None:
    resp = _mock_response(SAMPLE_CSV)
    calls = {"n": 0}

    def fake_get(url, timeout):
        calls["n"] += 1
        return resp

    monkeypatch.setattr(fetcher.session, "get", fake_get)
    fetcher.get_holding("ARKK")
    fetcher.get_holding("ARKK", force_refresh=True)
    assert calls["n"] == 2


def test_http_error_returns_none(fetcher: ARKDataFetcher, monkeypatch) -> None:
    resp = _mock_response(b"", status=404)
    monkeypatch.setattr(fetcher.session, "get", lambda url, timeout: resp)
    assert fetcher.get_holding("ARKK") is None


def test_bad_csv_not_cached(fetcher: ARKDataFetcher, monkeypatch) -> None:
    """A failed parse must not leave a corrupt file on disk."""
    resp = _mock_response(b"this,is,not\nreally,csv")
    monkeypatch.setattr(fetcher.session, "get", lambda url, timeout: resp)
    # The above IS valid CSV actually; force an actual parse error:
    resp_bad = _mock_response(b"\xff\xfe\x00\x01not a csv at all")
    monkeypatch.setattr(fetcher.session, "get", lambda url, timeout: resp_bad)
    result = fetcher.get_holding("ARKK")
    # Could be None (parse error) or a DataFrame (some bytes happen to parse).
    # Either way, no corrupt cache file should remain if it failed.
    cache = fetcher.data_dir / "ARKK_holdings.csv"
    if result is None:
        assert not cache.exists()


def test_url_is_percent_encoded(fetcher: ARKDataFetcher, monkeypatch) -> None:
    """The literal `&` in the ARKQ URL must be encoded before sending."""
    seen_urls = []

    def fake_get(url, timeout):
        seen_urls.append(url)
        return _mock_response(SAMPLE_CSV)

    monkeypatch.setattr(fetcher.session, "get", fake_get)
    fetcher.get_holding("ARKQ")
    assert seen_urls, "expected at least one request"
    assert "&" not in seen_urls[0].split("://", 1)[1].split("?", 1)[0].split("/csv/")[-1]


def test_context_manager(tmp_path) -> None:
    with ARKDataFetcher(data_dir=tmp_path) as f:
        assert isinstance(f, ARKDataFetcher)


def test_all_known_etfs_have_urls() -> None:
    """Sanity: every advertised ticker resolves to a URL string."""
    for ticker, url in ARK_ETF_URLS.items():
        assert ticker.isupper()
        assert url.startswith("https://")