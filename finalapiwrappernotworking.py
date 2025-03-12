import requests
import logging
import pandas as pd
import simfin as sf
from simfin.names import *

class SimFinAPIError(Exception):
    """Custom exception for SimFin API errors."""
    pass

class PySimFin:
    def __init__(self, api_key: str):
        """Initialize the API wrapper with the given API key."""
        self.api_key = api_key
        sf.set_api_key(self.api_key)
        sf.set_data_dir('~/simfin_data/')
        
        # Configure logging
        logging.basicConfig(
            filename='simfin_api.log', 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("SimFin API Wrapper initialized.")
    
    def get_share_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Retrieve historical stock price data for the given ticker and date range."""
        logging.info(f"Fetching historical share prices for {ticker} from {start} to {end}.")
        try:
            prices = sf.load_shareprices(variant='daily', market='us')  # Load daily historical data
            available_tickers = prices.index.get_level_values(0).unique()
            
            if ticker not in available_tickers and f"{ticker}_US" in available_tickers:
                ticker = f"{ticker}_US"
            
            if ticker not in available_tickers:
                raise SimFinAPIError(f"Ticker {ticker} not found in historical share prices dataset.")
            
            prices = prices.loc[ticker]
            prices.reset_index(inplace=True)
            
            print("Earliest available date:", prices['Date'].min(), "Latest available date:", prices['Date'].max())
            prices_filtered = prices[(prices['Date'] >= start) & (prices['Date'] <= end)]
            
            if prices_filtered.empty:
                logging.warning(f"No historical share price data available for {ticker} in the given period.")
                return pd.DataFrame({"Error": ["No data available for selected dates."]})
            return prices_filtered
        except Exception as e:
            logging.error(f"Error fetching historical share prices: {e}")
            raise SimFinAPIError(f"Error fetching historical share prices: {e}")
    
    def get_financial_statement(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Retrieve financial statement data for the given ticker and date range."""
        logging.info(f"Fetching financial statements for {ticker} from {start} to {end}.")
        try:
            financials = sf.load_income(variant='annual', market='us')
            if ticker not in financials.index:
                raise SimFinAPIError(f"Ticker {ticker} not found in financial statements dataset.")
            
            financials = financials.loc[ticker]
            financials.reset_index(inplace=True)
            financials_filtered = financials[(financials['Fiscal Year'] >= int(start[:4])) & (financials['Fiscal Year'] <= int(end[:4]))]
            
            if financials_filtered.empty:
                logging.warning(f"No financial data available for {ticker} in the given period.")
            return financials_filtered
        except Exception as e:
            logging.error(f"Error fetching financial statements: {e}")
            raise SimFinAPIError(f"Error fetching financial statements: {e}")

# Example Usage (Replace 'your_api_key' with an actual key)
if __name__ == "__main__":
    api_key = "e1c75cc5-3bca-4b0c-b847-6447bd4ed901"  # Replace with your actual SimFin API key
    simfin = PySimFin(api_key)
    
    # Fetch share prices
    prices_df = simfin.get_share_prices("AAPL", "2023-01-01", "2024-03-01")
    print(prices_df.head())
    
    # Fetch financial statements
    financials_df = simfin.get_financial_statement("AAPL", "2023-01-01", "2024-03-01")
    print(financials_df.head())
