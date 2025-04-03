import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
from typing import List, Dict, Optional

companies_df = pd.read_csv(os.path.join('input_data', 'market_cap_topcompanies.csv'))
# Clean up column names and data
companies_df.columns = [col.lower().strip() for col in companies_df.columns]
included_countries = ['United States', 
                              'China', 
                              'Japan', 
                              'Germany',
                              'United Kingdom',
                              'France',
                              'Italy',
                              'Spain',
                              'Netherlands',
                              'Switzerland',
                              "Austria",
                              "Australia",
                              "Canada",
                              "Sweden",
                              "Norway",
                              "Denmark",
                              "Finland",
                              "South Korea",
                              "Taiwan", 
                              "Hong Kong",
                              "Belgium"
                              ]
        
        # Filter to keep only included countries
companies_df = companies_df[companies_df['country'].isin(included_countries)]
        
        # Sort by market cap
companies_df = companies_df.sort_values('marketcap', ascending=False)

companies_df = companies_df[['symbol', 'name', 'country', 'marketcap']]

print(f"Loaded {len(companies_df)} companies available on etoro")

symbol = companies_df['symbol'].iloc[0]

def get_stock_data(symbol: str, period: str = "3y") -> pd.DataFrame:
        """Fetch stock data for a given symbol using yfinance."""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

aapl = get_stock_data(symbol)

def get_earnings_data(symbol: str) -> List[Dict]:
        """Fetch earnings data for a given symbol."""
        try:
            stock = yf.Ticker(symbol)
            earnings = stock.earnings_dates
            if earnings is None:
                return []
            
            earnings_data = []
            for date, row in earnings.iterrows():
                earnings_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'reported_eps': row.get('Reported EPS', None),
                    'estimated_eps': row.get('EPS Estimate', None),
                    'surprise': row.get('Surprise(%)', None),
                })
            return earnings_data
        except Exception as e:
            print(f"Error fetching earnings data for {symbol}: {str(e)}")
            return []

def process_company_data(self, symbol: str) -> Dict:
        """Process all data for a given company symbol."""
        stock_data = self.get_stock_data(symbol)
        earnings_data = self.get_earnings_data(symbol)
        
        return {
            'symbol': symbol,
            'stock_data': stock_data,
            'earnings_data': earnings_data
        }
    
def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get company information from the CSV file."""
        company_info = self.companies_df[self.companies_df['symbol'] == symbol]
        if not company_info.empty:
            return company_info.iloc[0].to_dict()
        return None

def save_data(data: Dict, output_dir: str):
    """Save processed data to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol, company_data in data.items():
        # Save stock data
        if not company_data['stock_data'].empty:
            company_data['stock_data'].to_csv(os.path.join(output_dir, f"{symbol}_stock_data.csv"))
        
        # Save earnings data
        if company_data['earnings_data']:
            pd.DataFrame(company_data['earnings_data']).to_csv(
                os.path.join(output_dir, f"{symbol}_earnings_data.csv"),
                index=False
            ) 
