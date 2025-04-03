import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
from typing import List, Dict, Optional

class StockDataProcessor:
    def __init__(self, companies_csv_path: str = 'companies.csv'):
        """Initialize the processor with the path to the companies CSV file."""
        try:
            # Read the CSV file
            self.companies_df = pd.read_csv(companies_csv_path)
            
            # Print the actual columns to help debug
            print("Available columns:", self.companies_df.columns.tolist())
            
            # Clean up column names and data
            self.companies_df.columns = [col.lower().strip() for col in self.companies_df.columns]
            
            # Clean up market cap - handle different possible formats
            if 'marketcap' in self.companies_df.columns:
                # Convert market cap to numeric, handling different formats
                if self.companies_df['marketcap'].dtype == object:  # if it's string
                    self.companies_df['marketcap'] = (self.companies_df['marketcap']
                                                    .replace({',': ''}, regex=True)
                                                    .astype(float))
                # If it's already numeric, leave it as is
            
            # Rename columns if they have different names
            column_mapping = {
                'Country': 'country',
                'Symbol': 'symbol',
                'Name': 'name',
                'Market Cap': 'marketcap'
                # Add any other column mappings that might be needed
            }
            
            self.companies_df = self.companies_df.rename(columns=column_mapping)
            
            # Define countries to include
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
            
            # Check if required columns exist
            required_columns = ['symbol', 'name', 'country', 'marketcap']
            missing_columns = [col for col in required_columns if col not in self.companies_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {self.companies_df.columns.tolist()}")
            
            # Filter to keep only included countries
            self.companies_df = self.companies_df[self.companies_df['country'].isin(included_countries)]
            
            # Sort by market cap
            self.companies_df = self.companies_df.sort_values('marketcap', ascending=False)
            
            print(f"Loaded {len(self.companies_df)} companies")
            
        except Exception as e:
            print(f"Error loading companies data: {str(e)}")
            # Initialize with empty DataFrame if file loading fails
            self.companies_df = pd.DataFrame(columns=['symbol', 'name', 'country', 'marketcap'])

    def filter_companies(self, countries: Optional[List[str]] = "Germany", top_n: Optional[int] = 20) -> pd.DataFrame:
        """Filter companies by countries and/or top N by market cap.
        
        Args:
            countries: Optional list of country names to filter by
            top_n: Optional number of top companies by market cap to return
            
        Returns:
            Filtered DataFrame of companies
        """
        filtered_df = self.companies_df.copy()
        
        if countries:
            filtered_df = filtered_df[filtered_df['country'].isin(countries)]
            
        if top_n:
            filtered_df = filtered_df.head(top_n)
            
        return filtered_df
    
    def get_stock_data(self, symbol: str, period: str = "3y") -> pd.DataFrame:
        """Fetch stock data for a given symbol using yfinance."""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_earnings_and_price_changes(self, symbol: str) -> Dict:
        """Get earnings data and associated price changes in one efficient call."""
        try:
            stock = yf.Ticker(symbol)
            earnings = stock.earnings_dates
            if earnings is None:
                return []
            
            earnings_data = []
            # Get historical data once for the entire period
            stock_data = stock.history(period="1y")  # Adjust period as needed
            
            for date, row in earnings.iterrows():
                # Find the index positions for the earnings date
                date_str = date.strftime('%Y-%m-%d')
                try:
                    # Get the closing prices around earnings date
                    date_idx = stock_data.index.get_indexer([date], method='nearest')[0]
                    
                    # Calculate price changes if we have enough data
                    one_day_change = None
                    five_day_change = None
                    
                    if date_idx > 0 and date_idx < len(stock_data) - 1:
                        one_day_change = ((stock_data['Close'].iloc[date_idx + 1] / 
                                         stock_data['Close'].iloc[date_idx] - 1) * 100)
                    
                    if date_idx > 0 and date_idx < len(stock_data) - 5:
                        five_day_change = ((stock_data['Close'].iloc[date_idx + 5] / 
                                          stock_data['Close'].iloc[date_idx] - 1) * 100)
                    
                    earnings_data.append({
                        'date': date_str,
                        'reported_eps': row.get('Reported EPS', None),
                        'estimated_eps': row.get('EPS Estimate', None),
                        'surprise': row.get('Surprise(%)', None),
                        'price_change_1d': one_day_change,
                        'price_change_5d': five_day_change
                    })
                except Exception as e:
                    print(f"Error processing earnings date {date_str}: {str(e)}")
                    continue
                
            return earnings_data
        except Exception as e:
            print(f"Error fetching earnings data for {symbol}: {str(e)}")
            return []
    
    def process_company_data(self, symbol: str) -> Dict:
        """Process all data for a given company symbol."""
        stock_data = self.get_stock_data(symbol)
        earnings_data = self.get_earnings_and_price_changes(symbol)
        premarket_data = self.get_premarket_data(symbol)
        
        return {
            'symbol': symbol,
            'stock_data': stock_data,
            'earnings_data': earnings_data,
            'premarket_data': premarket_data
        }
    
    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed company information from yfinance and CSV file."""
        try:
            # Get basic info from CSV
            csv_info = self.companies_df[self.companies_df['symbol'] == symbol]
            
            # Get additional info from yfinance
            stock = yf.Ticker(symbol)
            info = stock.info
            
            company_info = {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'country': info.get('country', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'marketcap': info.get('marketCap', 'N/A'),
                'analyst_rating': info.get('recommendationKey', 'N/A'),
                'target_low': info.get('targetLowPrice', 'N/A'),
                'target_median': info.get('targetMedianPrice', 'N/A'),
                'target_high': info.get('targetHighPrice', 'N/A'),
                'recommendation_mean': info.get('recommendationMean', 'N/A'),
                'number_of_analysts': info.get('numberOfAnalystOpinions', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A')
            }
            
            return company_info
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {str(e)}")
            return None

    def get_price_changes(self, symbol: str) -> Dict:
        """Get price changes for different time periods using calendar days."""
        try:
            # Get stock data for 2 years
            stock_data = self.get_stock_data(symbol, period="2y")
            if stock_data.empty:
                return {}
            
            # Get the most recent price
            current_price = stock_data['Close'].iloc[-1]
            current_date = stock_data.index[-1]
            
            def get_safe_change(data, days):
                try:
                    # Get the date 'days' days ago
                    target_date = current_date - pd.Timedelta(days=days)
                    # Find the closest trading day before that date
                    historical_price = data['Close'][data.index <= target_date].iloc[-1]
                    return ((current_price / historical_price - 1) * 100)
                except:
                    return None
            
            # Calculate price changes using calendar days
            changes = {
                'current_price': current_price,
                '1d_change': get_safe_change(stock_data, 1),      # 1 calendar day
                '1w_change': get_safe_change(stock_data, 7),      # 7 calendar days
                '1m_change': get_safe_change(stock_data, 30),     # 30 calendar days
                '1y_change': get_safe_change(stock_data, 365),    # 365 calendar days
                '2y_change': get_safe_change(stock_data, 730)     # 730 calendar days
            }
            
            return changes
        except Exception as e:
            print(f"Error calculating price changes for {symbol}: {str(e)}")
            return {}

    def get_premarket_data(self, symbol: str) -> Dict:
        """Get pre-market data including after-hours and pre-market changes."""
        try:
            stock = yf.Ticker(symbol)
            
            # Get pre-market quote
            quote = stock.info
            
            # Get regular market price (previous close)
            regular_market_price = quote.get('regularMarketPrice', None)
            previous_close = quote.get('previousClose', None)
            
            # Get pre-market price
            pre_market_price = quote.get('preMarketPrice', None)
            pre_market_change = quote.get('preMarketChange', None)
            pre_market_change_percent = quote.get('preMarketChangePercent', None)
            
            # Get after hours price
            post_market_price = quote.get('postMarketPrice', None)
            post_market_change = quote.get('postMarketChange', None)
            post_market_change_percent = quote.get('postMarketChangePercent', None)
            
            return {
                'regular_market_price': regular_market_price,
                'previous_close': previous_close,
                'pre_market_price': pre_market_price,
                'pre_market_change': pre_market_change,
                'pre_market_change_percent': pre_market_change_percent,
                'post_market_price': post_market_price,
                'post_market_change': post_market_change,
                'post_market_change_percent': post_market_change_percent
            }
        except Exception as e:
            print(f"Error fetching pre-market data for {symbol}: {str(e)}")
            return {}

def save_data(data: Dict, output_dir: str = 'output'):
    """Save processed data to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol, company_data in data.items():
        # Save stock data
        if not company_data['stock_data'].empty:
            company_data['stock_data'].to_csv(f"{symbol}_stock_data.csv")
        
        # Save earnings data
        if company_data['earnings_data']:
            pd.DataFrame(company_data['earnings_data']).to_csv(
                f"{symbol}_earnings_data.csv",
                index=False
            ) 