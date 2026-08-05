import pandas as pd
import yfinance as yf


class MarketDataLoader:
    """
    Downloads market data from Yahoo Finance and calculates daily returns.
    """
    
    def __init__(self, tickers: list[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches adjusted close prices and computes daily percentage returns.
        
        Raises:
            ValueError: If data is empty, missing columns, or contains unexpected NaNs.
            
        Returns:
            pd.DataFrame: Daily percentage returns.
        """
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
        returns = adj_close.pct_change()
        
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