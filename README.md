# Stock Market Dashboard

A real-time stock market dashboard built with Streamlit that displays company information, stock prices, and earnings data.

## Features
- Company information and financials
- Stock price history with technical indicators
- Earnings data and analysis
- Pre-market and after-hours trading data
- Multi-country and sector filtering

## Setup
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your market cap companies CSV file in the `data/` directory
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Data Structure
The `data/market_cap_topcompanies.csv` file should contain the following columns:
- symbol
- name
- country
- sector
- marketcap

## Project Structure

- `app.py`: Main Streamlit dashboard application
- `data_processor.py`: Data processing and fetching module
- `update_data.py`: Automated data update service
- `companies.csv`: Company information file
- `data/`: Directory for storing processed data
- `dashboard_update.log`: Log file for the update service

## Deployment

To deploy this dashboard on a server:

1. Set up a server with Python 3.8+
2. Install the requirements
3. Set up a process manager (e.g., Supervisor) to run `update_data.py`
4. Use a reverse proxy (e.g., Nginx) to serve the Streamlit app
5. Set up SSL certificates for secure access

## Data Sources

- Stock data: Yahoo Finance (via yfinance)
- Company information: Custom CSV file
- Earnings data: Yahoo Finance

## Notes

- The dashboard updates daily at market close (4 PM EST)
- All data is cached to improve performance
- The update service logs all activities to `dashboard_update.log` 