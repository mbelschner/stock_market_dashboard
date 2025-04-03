import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
from typing import List, Dict, Optional

# Assume 'companies.xlsx' is in 'input_data' relative to the script
# If not, adjust the path accordingly.
# data = pd.read_csv(os.path.join('input_data', 'market_cap_topcompanies.csv')) # This line seems unused, removed for clarity

class StockDataProcessor:
    def __init__(self, companies_excel_path: str = os.path.join('input_data', 'companies.xlsx')):
        """Initialize the processor with the path to the companies Excel file."""
        try:
            self.companies_df = pd.read_excel(companies_excel_path)
        except FileNotFoundError:
            print(f"Error: Companies file not found at {companies_excel_path}")
            # Initialize an empty DataFrame to avoid errors later if file is missing
            self.companies_df = pd.DataFrame(columns=['symbol', 'name', 'country', 'marketcap', 'industry', 'sector']) # Add expected columns

        # Clean up column names and data only if DataFrame is not empty
        if not self.companies_df.empty:
            self.companies_df.columns = [col.lower().strip() for col in self.companies_df.columns]

            # Ensure 'marketcap' column exists and handle potential errors
            if 'marketcap' in self.companies_df.columns:
                 # Convert marketcap to numeric, coercing errors to NaN
                self.companies_df['marketcap'] = pd.to_numeric(self.companies_df['marketcap'], errors='coerce')
                # Drop rows where marketcap couldn't be converted
                self.companies_df.dropna(subset=['marketcap'], inplace=True)
                # Sort by market cap
                self.companies_df = self.companies_df.sort_values('marketcap', ascending=False)
            else:
                print("Warning: 'marketcap' column not found in companies file. Sorting by market cap is skipped.")

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

            # Filter to keep only included countries if 'country' column exists
            if 'country' in self.companies_df.columns:
                self.companies_df = self.companies_df[self.companies_df['country'].isin(included_countries)]
            else:
                 print("Warning: 'country' column not found in companies file. Filtering by country is skipped.")

        print(f"Initialized with {len(self.companies_df)} companies from specified countries (if available).")


    def filter_companies(self, countries: Optional[List[str]] = ["Germany"], top_n: Optional[int] = 20) -> pd.DataFrame:
        """Filter companies by countries and/or top N by market cap.

        Args:
            countries: Optional list of country names to filter by. Pass None to skip country filter.
            top_n: Optional number of top companies by market cap to return. Pass None to skip top N filter.

        Returns:
            Filtered DataFrame of companies
        """
        filtered_df = self.companies_df.copy()

        if countries and 'country' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['country'].isin(countries)]
        elif countries:
             print("Warning: Cannot filter by country, 'country' column missing.")

        if top_n and 'marketcap' in filtered_df.columns:
            filtered_df = filtered_df.head(top_n)
        elif top_n:
             print("Warning: Cannot filter by top_n, 'marketcap' column missing.")

        return filtered_df

    def get_stock_data(self, symbol: str, period: str = "3y", start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch stock data for a given symbol using yfinance.
           Allows specifying period or start/end dates.
        """
        try:
            stock = yf.Ticker(symbol)
            if start_date and end_date:
                 df = stock.history(start=start_date, end=end_date)
            else:
                 df = stock.history(period=period)

            # Ensure index is DatetimeIndex and timezone-naive for consistency
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            else: # If index is not DatetimeIndex, try to convert it
                 try:
                     df.index = pd.to_datetime(df.index).tz_localize(None)
                 except Exception:
                     print(f"Warning: Could not convert index to DatetimeIndex for {symbol}")

            return df
        except Exception as e:
            print(f"Error fetching history data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_earnings_data(self, symbol: str) -> List[Dict]:
        """Fetch earnings data for a given symbol."""
        try:
            stock = yf.Ticker(symbol)
            earnings = stock.earnings_dates
            if earnings is None or earnings.empty:
                print(f"No earnings dates found for {symbol}")
                return []

            earnings_data = []
            # Ensure index is DatetimeIndex and timezone-naive
            if earnings.index.tz is not None:
                earnings.index = earnings.index.tz_localize(None)
            # Reset index to make date a column for easier access
            earnings = earnings.reset_index()
            # Rename the date column if necessary (it's often 'Earnings Date')
            date_col = earnings.columns[0] # Assume first column is the date
            earnings.rename(columns={date_col: 'Date'}, inplace=True)
            earnings['Date'] = pd.to_datetime(earnings['Date']) # Ensure it's datetime

            for _, row in earnings.iterrows():
                earnings_data.append({
                    'date': row['Date'].strftime('%Y-%m-%d'), # Format date string
                    'reported_eps': row.get('Reported EPS', None),
                    'estimated_eps': row.get('EPS Estimate', None),
                    'surprise': row.get('Surprise(%)', None),
                })
            # Sort by date descending (most recent first)
            earnings_data.sort(key=lambda x: x['date'], reverse=True)
            return earnings_data
        except Exception as e:
            print(f"Error fetching earnings data for {symbol}: {str(e)}")
            return []

    # --- NEW METHOD ---
    def get_earnings_call_price_change(
        self,
        symbol: str,
        earnings_call_date_str: str,
        days_after: int = 1
    ) -> Optional[Dict]:
        """
        Calculates the stock price change after a specific earnings call date.

        Compares the closing price on the trading day *before* the earnings call
        to the closing price *days_after* trading days *after* the earnings call.

        Args:
            symbol: The stock ticker symbol.
            earnings_call_date_str: The date of the earnings call (YYYY-MM-DD).
                                    This is the date the earnings were *announced*.
            days_after: How many trading days after the earnings call announcement date
                        to measure the price change to. (e.g., 1 for next day close,
                        5 for one week close).

        Returns:
            A dictionary containing before/after dates, prices, and percentage change,
            or None if data is insufficient or an error occurs.
            Example: {'symbol': 'AAPL', 'earnings_call_date': Timestamp('2023-08-03'),
                      'before_date': Timestamp('2023-08-02'), 'before_price': 192.58,
                      'after_date': Timestamp('2023-08-04'), 'after_price': 181.99,
                      'days_after_call': 1, 'percentage_change': -5.499}
        """
        try:
            # 1. Validate and Parse Date
            try:
                # Normalize to midnight, ignore time component if present
                earnings_call_date = pd.Timestamp(earnings_call_date_str).normalize()
            except ValueError:
                print(f"Error: Invalid date format for {earnings_call_date_str}. Use YYYY-MM-DD.")
                return None

            if days_after < 1:
                 print(f"Error: days_after must be at least 1.")
                 return None

            # 2. Define Date Range for Fetching Stock Data
            # Fetch data starting well before the earnings date and ending well after
            # to ensure we capture the necessary trading days around the event.
            # Buffer helps handle weekends/holidays near the target dates.
            buffer_days = 15 # Days before/after target dates to fetch
            start_fetch_date = earnings_call_date - timedelta(days=buffer_days)
            end_fetch_date = earnings_call_date + timedelta(days=days_after + buffer_days)

            # 3. Fetch Historical Stock Data
            hist = self.get_stock_data(symbol, start_date=start_fetch_date.strftime('%Y-%m-%d'), end_date=end_fetch_date.strftime('%Y-%m-%d'))

            if hist.empty or 'Close' not in hist.columns:
                print(f"Error: No historical data or 'Close' price found for {symbol} in the required date range.")
                return None

            # 4. Find the 'Before' Price (Closing price on trading day *before* the earnings call date)
            hist_before_call = hist[hist.index < earnings_call_date]
            if hist_before_call.empty:
                print(f"Error: Could not find trading day data strictly *before* {earnings_call_date.date()} for {symbol}.")
                return None
            # Get the last trading day's data before the call date
            before_data = hist_before_call.iloc[-1]
            actual_before_date = before_data.name # This is the index (date)
            before_price = before_data['Close']

            # 5. Find the 'After' Price (Closing price N trading days *after* the earnings call date)
            # Filter history to include only trading days *after* the earnings call date
            hist_after_call = hist[hist.index > earnings_call_date]

            if len(hist_after_call) < days_after:
                 print(f"Error: Not enough trading days ({len(hist_after_call)}) found after {earnings_call_date.date()} to calculate {days_after}-day change for {symbol}.")
                 return None

            # Get the data for the Nth trading day after the call (index is days_after - 1)
            after_data = hist_after_call.iloc[days_after - 1]
            actual_after_date = after_data.name # This is the index (date)
            after_price = after_data['Close']

            # 6. Calculate Percentage Change
            if pd.isna(before_price) or pd.isna(after_price) or before_price == 0:
                print(f"Error: Missing price data or zero 'before' price for {symbol} around {earnings_call_date.date()}. Before: {before_price}, After: {after_price}")
                return None

            percentage_change = ((after_price / before_price) - 1) * 100

            return {
                'symbol': symbol,
                'earnings_call_date': earnings_call_date,
                'before_date': actual_before_date,
                'before_price': before_price,
                'after_date': actual_after_date,
                'after_price': after_price,
                'days_after_call': days_after, # Include this for clarity
                'percentage_change': percentage_change
            }

        except IndexError:
             # This can happen if iloc[-1] or iloc[days_after - 1] fails
             print(f"Error: Index out of bounds. Not enough trading days found around {earnings_call_date_str} for {symbol} to get prices.")
             return None
        except KeyError as e:
             print(f"Error: Missing expected data column ('Close'?) or date index issue for {symbol} around {earnings_call_date_str}: {e}")
             return None
        except Exception as e:
            print(f"An unexpected error occurred calculating price change for {symbol} around {earnings_call_date_str}: {str(e)}")
            return None

    # --- End of NEW METHOD ---


    def process_company_data(self, symbol: str) -> Dict:
        """Process all data for a given company symbol."""
        stock_data = self.get_stock_data(symbol) # Default period 3y
        earnings_data = self.get_earnings_data(symbol)

        return {
            'symbol': symbol,
            'stock_data': stock_data, # This is a DataFrame
            'earnings_data': earnings_data # This is a list of Dicts
        }

    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed company information from yfinance and potentially CSV file (if loaded)."""
        try:
            # Get basic info from the loaded DataFrame (if available and symbol found)
            company_row = None
            if not self.companies_df.empty and 'symbol' in self.companies_df.columns:
                 matching_rows = self.companies_df[self.companies_df['symbol'] == symbol]
                 if not matching_rows.empty:
                     company_row = matching_rows.iloc[0] # Take the first match

            # Get additional info from yfinance
            stock = yf.Ticker(symbol)
            info = stock.info # Fetch info dict

            # Combine information, prioritizing yfinance for most fields
            company_info = {
                'symbol': symbol,
                'name': info.get('longName', company_row.get('name', 'N/A') if company_row is not None else 'N/A'),
                'country': info.get('country', company_row.get('country', 'N/A') if company_row is not None else 'N/A'),
                'industry': info.get('industry', company_row.get('industry', 'N/A') if company_row is not None else 'N/A'),
                'sector': info.get('sector', company_row.get('sector', 'N/A') if company_row is not None else 'N/A'),
                'marketcap': info.get('marketCap', company_row.get('marketcap', 'N/A') if company_row is not None else 'N/A'),
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
            # Catch potential errors from yfinance (e.g., symbol not found, network issues)
            print(f"Error fetching company info for {symbol}: {str(e)}")
            # Still return basic info if available from the DataFrame
            if company_row is not None:
                 return {
                    'symbol': symbol,
                    'name': company_row.get('name', 'N/A'),
                    'country': company_row.get('country', 'N/A'),
                    'industry': company_row.get('industry', 'N/A'),
                    'sector': company_row.get('sector', 'N/A'),
                    'marketcap': company_row.get('marketcap', 'N/A'),
                    # Add other fields as 'N/A' since yfinance failed
                    'analyst_rating': 'N/A',
                    'target_low': 'N/A',
                    'target_median': 'N/A',
                    'target_high': 'N/A',
                    'recommendation_mean': 'N/A',
                    'number_of_analysts': 'N/A',
                    'description': 'N/A (yfinance fetch failed)'
                 }
            return None # Return None if no info could be gathered at all


    def get_price_changes(self, symbol: str) -> Dict:
        """Get price changes for different time periods ending on the latest data point."""
        try:
            # Get stock data for 2 years to cover all periods
            # Use the internal method to fetch data
            stock_data = self.get_stock_data(symbol, period="2y")
            if stock_data.empty or 'Close' not in stock_data.columns:
                 print(f"Warning: No stock data or 'Close' column for {symbol} to calculate price changes.")
                 return {}

            # Get the most recent price
            if len(stock_data) < 1:
                 print(f"Warning: Not enough data points (0) for {symbol} to calculate price changes.")
                 return {}
            current_price = stock_data['Close'].iloc[-1]
            current_date = stock_data.index[-1] # Get the date of the current price

            changes = {'current_price': current_price, 'current_date': current_date.strftime('%Y-%m-%d')}

            # Define periods in approximate trading days
            periods = {
                '1d': 1,
                '1w': 5,
                '1m': 21, # Approx 21 trading days in a month
                '3m': 63,
                '6m': 126,
                '1y': 252, # Approx 252 trading days in a year
                '2y': 504
            }

            for label, days in periods.items():
                # Check if enough data exists (need current day + 'days' previous days)
                if len(stock_data) > days:
                    try:
                        # iloc[-1] is current, iloc[-2] is prev day, ..., iloc[-(days+1)] is 'days' trading days ago
                        old_price = stock_data['Close'].iloc[-(days + 1)]
                        if pd.notna(old_price) and old_price != 0:
                            change_pct = ((current_price / old_price) - 1) * 100
                            changes[f'{label}_change_pct'] = change_pct
                        else:
                            changes[f'{label}_change_pct'] = None # Handle NaN or zero old price
                    except IndexError:
                        # Should not happen due to len check, but as safeguard
                        changes[f'{label}_change_pct'] = None
                    except Exception as e_inner:
                         print(f"Inner error calculating {label} change for {symbol}: {e_inner}")
                         changes[f'{label}_change_pct'] = None

                else:
                    changes[f'{label}_change_pct'] = None # Not enough data for this period

            return changes
        except Exception as e:
            print(f"Error calculating price changes for {symbol}: {str(e)}")
            return {} # Return empty dict on error

# Example Usage:

# 1. Make sure 'input_data/companies.xlsx' exists or adjust the path
processor = StockDataProcessor(companies_excel_path=os.path.join('input_data', 'companies.xlsx'))

# 2. Example: Find price change for Apple after its August 3, 2023 earnings
#    (You'd typically get this date from get_earnings_data first)
aapl_earnings_date = '2023-08-03'
aapl_change_1day = processor.get_earnings_call_price_change('AAPL', aapl_earnings_date, days_after=1)
aapl_change_5day = processor.get_earnings_call_price_change('AAPL', aapl_earnings_date, days_after=5) # 5 trading days (approx 1 week)

if aapl_change_1day:
    print(f"\n--- Apple ({aapl_change_1day['symbol']}) Price Change after {aapl_earnings_date} Earnings ---")
    print(f"  Closing Price Before ({aapl_change_1day['before_date'].strftime('%Y-%m-%d')}): ${aapl_change_1day['before_price']:.2f}")
    print(f"  Closing Price {aapl_change_1day['days_after_call']} Day(s) After ({aapl_change_1day['after_date'].strftime('%Y-%m-%d')}): ${aapl_change_1day['after_price']:.2f}")
    print(f"  Percentage Change: {aapl_change_1day['percentage_change']:.2f}%")

if aapl_change_5day:
    print(f"\n--- Apple ({aapl_change_5day['symbol']}) Price Change after {aapl_earnings_date} Earnings ---")
    print(f"  Closing Price Before ({aapl_change_5day['before_date'].strftime('%Y-%m-%d')}): ${aapl_change_5day['before_price']:.2f}")
    print(f"  Closing Price {aapl_change_5day['days_after_call']} Day(s) After ({aapl_change_5day['after_date'].strftime('%Y-%m-%d')}): ${aapl_change_5day['after_price']:.2f}")
    print(f"  Percentage Change: {aapl_change_5day['percentage_change']:.2f}%")

# Example: Microsoft July 25, 2023 earnings
msft_earnings_date = '2023-07-25'
msft_change_1day = processor.get_earnings_call_price_change('MSFT', msft_earnings_date, days_after=1)

if msft_change_1day:
    print(f"\n--- Microsoft ({msft_change_1day['symbol']}) Price Change after {msft_earnings_date} Earnings ---")
    print(f"  Closing Price Before ({msft_change_1day['before_date'].strftime('%Y-%m-%d')}): ${msft_change_1day['before_price']:.2f}")
    print(f"  Closing Price {msft_change_1day['days_after_call']} Day(s) After ({msft_change_1day['after_date'].strftime('%Y-%m-%d')}): ${msft_change_1day['after_price']:.2f}")
    print(f"  Percentage Change: {msft_change_1day['percentage_change']:.2f}%")

# Example: A date too recent where there might not be enough data after
# today = datetime.now().strftime('%Y-%m-%d')
# recent_change = processor.get_earnings_call_price_change('NVDA', today, days_after=5)
# if recent_change:
#      print("\nRecent change calculation succeeded (unlikely for today's date):", recent_change)
# else:
#      print(f"\nRecent change calculation likely failed as expected for date {today}.")


# --- You can keep your save_data function as is ---
output_dir = "C:/Users/maxib/OneDrive/Dokumente/Finance/Stock Market Dashboard/output" # Adjust as needed

def save_data(data: Dict, output_dir: str):
    """Save processed data (stock history, earnings) to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for symbol, company_data in data.items():
        # Save stock data DataFrame
        if 'stock_data' in company_data and isinstance(company_data['stock_data'], pd.DataFrame) and not company_data['stock_data'].empty:
            stock_df = company_data['stock_data']
            # Ensure index is saved if it's meaningful (like dates)
            save_index = isinstance(stock_df.index, pd.DatetimeIndex)
            stock_df.to_csv(os.path.join(output_dir, f"{symbol}_stock_data.csv"), index=save_index)
            print(f"Saved stock data for {symbol}")

        # Save earnings data (List of Dicts)
        if 'earnings_data' in company_data and isinstance(company_data['earnings_data'], list) and company_data['earnings_data']:
            try:
                pd.DataFrame(company_data['earnings_data']).to_csv(
                    os.path.join(output_dir, f"{symbol}_earnings_data.csv"),
                    index=False
                )
                print(f"Saved earnings data for {symbol}")
            except Exception as e:
                 print(f"Error saving earnings data for {symbol} to CSV: {e}")


# Example of processing and saving data for a few companies
companies_to_process = ['AAPL', 'MSFT', 'GOOGL']
all_processed_data = {}
for sym in companies_to_process:
     print(f"\nProcessing {sym}...")
     processed_data = processor.process_company_data(sym)
     all_processed_data[sym] = processed_data
     # Optionally add price change info
     price_changes = processor.get_price_changes(sym)
     all_processed_data[sym]['price_changes'] = price_changes
     # Optionally add company info
     company_info = processor.get_company_info(sym)
     all_processed_data[sym]['company_info'] = company_info


print("\nSaving processed data...")
save_data(all_processed_data, output_dir)
print("Done.")