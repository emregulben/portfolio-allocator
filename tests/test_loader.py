import numpy as np
import pytest
import pandas as pd
from unittest.mock import patch

from simulator.loader import MarketDataLoader

def test_loader_initialization():
    """Validates that parameters are correctly assigned to the instance."""
    loader = MarketDataLoader(
        tickers=["AAPL", "MSFT"],
        start_date="2016-01-01",
        end_date="2026-08-01"
    )
    
    assert loader.start_date == "2016-01-01"
    assert loader.end_date == "2026-08-01"
    assert "AAPL" in loader.tickers


@patch("simulator.loader.yf.download")
def test_fetch_data_logic(mock_download):
    """
    Validates the mathematical conversion from raw prices to returns,
    and ensures the first NaN row is correctly dropped.
    """
    # 1. Create a mock dataframe
    # Day 1 to Day 2: 100 -> 105 (5% return)
    # Day 2 to Day 3: 105 -> 102.9 (-2% return)
    mock_df = pd.DataFrame({
        "Adj Close": [100.0, 105.0, 102.9],
        "Volume": [1000, 1100, 900]
    })
    
    mock_download.return_value = mock_df
    
    loader = MarketDataLoader(tickers=["AAPL"], start_date="2023-01-01", end_date="2023-01-03")
    
    # 2. Call the method
    returns = loader.fetch_data()
    
    # 3. Assert: Check the logic
    # It should have 2 rows (3 original days minus 1 dropped NaN row)
    assert len(returns) == 2
    
    assert returns["AAPL"].iloc[0] == pytest.approx(0.05)
    assert returns["AAPL"].iloc[1] == pytest.approx(-0.02)


@patch("simulator.loader.yf.download")
def test_fetch_data_missing_column(mock_download):
    """Validates that the correct error is raised if 'Adj Close' is missing."""
    # Create a dataframe missing the required column
    mock_df = pd.DataFrame({
        "Close": [100.0, 105.0],
        "Volume": [1000, 1100]
    })
    mock_download.return_value = mock_df
    
    loader = MarketDataLoader(tickers=["AAPL"], start_date="2023-01-01", end_date="2023-01-03")
    
    with pytest.raises(ValueError, match="Missing 'Adj Close' column in data."):
        loader.fetch_data()


@patch("simulator.loader.yf.download")
def test_fetch_data_empty_response(mock_download):
    """Validates that an empty API response raises an error."""
    mock_download.return_value = pd.DataFrame()
    
    loader = MarketDataLoader(tickers=["AAPL"], start_date="2023-01-01", end_date="2023-01-03")
    
    with pytest.raises(ValueError, match="No data fetched from Yahoo Finance."):
        loader.fetch_data()


@patch("simulator.loader.yf.download")
def test_fetch_data_unexpected_nan(mock_download):
    # 1. Simulate a missing price on day 3
    mock_df = pd.DataFrame({
        "Adj Close": [100.0, 105.0, np.nan, 102.0],
        "Volume": [1000, 1100, 900, 1050]
    })
    mock_download.return_value = mock_df
    
    loader = MarketDataLoader(
        tickers=["AAPL"], 
        start_date="2023-01-01", 
        end_date="2023-01-04"
    )
    
    with pytest.raises(ValueError, match="Unexpected missing values found in the time series."):
        loader.fetch_data()

def _fake_yahoo_prices():
    """Shaped like a real multi-ticker yf.download result: MultiIndex columns, Date index."""
    dates = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"])
    cols = pd.MultiIndex.from_product([["Adj Close", "Volume"], ["AAPL", "SPY"]], names=["Price", "Ticker"])
    data = np.array([
        [100.0, 380.0, 1, 1],
        [105.0, 383.3, 1, 1],
        [102.9, 379.123456789, 1, 1],
        [104.1, 385.987654321, 1, 1],
    ])
    df = pd.DataFrame(data, index=dates, columns=cols)
    df.index.name = "Date"
    return df


@patch("simulator.loader.yf.download")
def test_first_fetch_downloads_and_saves_snapshot(mock_download, tmp_path):
    """No snapshot yet: download once, save the returns to the snapshot file."""
    mock_download.return_value = _fake_yahoo_prices()
    path = tmp_path / "returns.csv"

    loader = MarketDataLoader(["AAPL", "SPY"], "2023-01-01", "2023-01-07", cache_path=str(path))
    returns = loader.fetch_data()

    assert mock_download.call_count == 1
    assert path.exists()
    assert list(returns.columns) == ["AAPL", "SPY"]
    assert len(returns) == 3


@patch("simulator.loader.yf.download")
def test_later_fetch_reads_snapshot_without_download(mock_download, tmp_path):
    """Snapshot exists: never call Yahoo, and return exactly the same numbers."""
    mock_download.return_value = _fake_yahoo_prices()
    path = tmp_path / "returns.csv"
    first = MarketDataLoader(["AAPL", "SPY"], "2023-01-01", "2023-01-07", cache_path=str(path)).fetch_data()

    mock_download.reset_mock()
    mock_download.side_effect = AssertionError("must not download when a snapshot exists")
    second = MarketDataLoader(["AAPL", "SPY"], "2023-01-01", "2023-01-07", cache_path=str(path)).fetch_data()

    mock_download.assert_not_called()
    pd.testing.assert_frame_equal(first, second, check_exact=True)


@patch("simulator.loader.yf.download")
def test_snapshot_with_wrong_tickers_raises(mock_download, tmp_path):
    """A snapshot made for other tickers must not be used silently."""
    mock_download.return_value = _fake_yahoo_prices()
    path = tmp_path / "returns.csv"
    MarketDataLoader(["AAPL", "SPY"], "2023-01-01", "2023-01-07", cache_path=str(path)).fetch_data()

    with pytest.raises(ValueError, match="snapshot"):
        MarketDataLoader(["AAPL", "MSFT"], "2023-01-01", "2023-01-07", cache_path=str(path)).fetch_data()


@patch("simulator.loader.yf.download")
def test_snapshot_accepts_tickers_in_any_order(mock_download, tmp_path):
    """Yahoo sorts tickers alphabetically; the config lists SPY first. Both must work."""
    mock_download.return_value = _fake_yahoo_prices()
    path = tmp_path / "returns.csv"

    returns = MarketDataLoader(["SPY", "AAPL"], "2023-01-01", "2023-01-07", cache_path=str(path)).fetch_data()
    assert list(returns.columns) == ["SPY", "AAPL"]
