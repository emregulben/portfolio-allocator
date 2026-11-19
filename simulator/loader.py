import os

import pandas as pd
import yfinance as yf

class MarketDataLoader:
    """
    Downloads market data from Yahoo Finance and calculates daily returns.

    Yahoo recomputes adjusted prices for all past dates whenever a stock pays a new
    dividend, so a fresh download can differ from an earlier one. Pass `cache_path`
    to freeze the data: the first call downloads and saves the returns there, and
    every later call reads that file instead of downloading.
    """
    
    def __init__(self, tickers: list[str], start_date: str, end_date: str, cache_path: str | None = None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.cache_path = cache_path

    def fetch_data(self) -> pd.DataFrame:
        """
        Returns daily percentage returns, from the snapshot file if one exists.

        Raises:
            ValueError: If the snapshot holds other tickers, or the download is
                empty, misses columns or contains unexpected NaNs.

        Returns:
            pd.DataFrame: Daily percentage returns (columns = tickers).
        """
        if self.cache_path is None:
            return self._download_returns()

        if not os.path.exists(self.cache_path):
            folder = os.path.dirname(self.cache_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            self._download_returns().to_csv(self.cache_path)

        # Always return what is on disk, so the first run and later runs are identical.
        returns = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
        if sorted(returns.columns) != sorted(self.tickers):
            raise ValueError(
                f"Data snapshot {self.cache_path} has tickers {list(returns.columns)}, "
                f"expected {list(self.tickers)}. Delete it or use another cache_path."
            )
        return returns[list(self.tickers)]

    def _download_returns(self) -> pd.DataFrame:
        """Downloads adjusted close prices from Yahoo and computes daily returns."""
        raw_data = yf.download(
            tickers=self.tickers, 
            start=self.start_date, 
            end=self.end_date, 
            auto_adjust=False,
            progress=False
        )

        if raw_data is None:
            raise ValueError("Yahoo Finance returned None instead of a DataFrame.")

        if raw_data.empty:
            raise ValueError("No data fetched from Yahoo Finance.")
            
        if 'Adj Close' not in raw_data:
            raise ValueError("Missing 'Adj Close' column in data.")
            
        adj_close = raw_data['Adj Close']
        
        # Calculate daily returns
        returns = adj_close.pct_change(fill_method=None)
        
        # Drop only the first row (which is NaN due to pct_change)
        returns = returns.iloc[1:]
        
        # Enforce DataFrame type for single-ticker downloads
        if isinstance(returns, pd.Series):
            returns = returns.to_frame(name=self.tickers[0])

        # Guard check: prevent silent data deletion if mid-series NaNs exist
        if returns.isna().any().any():
            raise ValueError("Unexpected missing values found in the time series.")
        
        return returns


if __name__ == "__main__":
    loader = MarketDataLoader(
        tickers=["AAPL", "MSFT", "TSLA"], 
        start_date="2014-01-01", 
        end_date="2024-01-01"
    )
    df_returns = loader.fetch_data()
    print(df_returns.head())