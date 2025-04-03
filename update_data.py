import schedule
import time
from data_processor import StockDataProcessor, save_data
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    filename='dashboard_update.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def update_data():
    """Update all company data."""
    try:
        logging.info("Starting data update")
        processor = StockDataProcessor('companies.csv')
        companies = processor.companies_df['symbol'].tolist()
        
        for symbol in companies:
            try:
                logging.info(f"Processing {symbol}")
                company_data = processor.process_company_data(symbol)
                save_data({symbol: company_data}, 'data')
                logging.info(f"Successfully updated {symbol}")
            except Exception as e:
                logging.error(f"Error processing {symbol}: {str(e)}")
        
        logging.info("Data update completed successfully")
    except Exception as e:
        logging.error(f"Error during data update: {str(e)}")

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Run initial update
    update_data()
    
    # Schedule daily update at market close (4 PM EST)
    schedule.every().day.at("16:00").do(update_data)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main() 