import numpy as np
import pytest
import pandas as pd
from simulator.loader import MarketDataLoader
from unittest.mock import patch


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