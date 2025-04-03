# Stock Market Dashboard

A real-time stock market dashboard that displays company information, stock prices, and earnings data. The dashboard updates automatically daily and provides interactive visualizations.

## Features

- Company information display
- 3-year stock price history with candlestick charts
- Earnings history with beat/miss indicators
- Daily automatic updates
- Interactive Streamlit interface

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `companies.csv` file with the following columns:
   - symbol (e.g., AAPL, MSFT)
   - name (company name)
   - sector (optional)
   - industry (optional)
   - description (optional)

   Example:
   ```csv
   symbol,name,sector,industry,description
   AAPL,Apple Inc.,Technology,Consumer Electronics,Technology company
   MSFT,Microsoft Corporation,Technology,Software,Software company
   ```

4. Start the data update service:
   ```bash
   python update_data.py
   ```

5. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

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