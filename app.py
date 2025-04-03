import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_processor import StockDataProcessor
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = None

# Load data processor
@st.cache_resource
def load_data_processor():
    return StockDataProcessor('companies.csv')

# Load company data
@st.cache_data
def load_company_data(symbol):
    processor = load_data_processor()
    return processor.process_company_data(symbol)

# Search function
def search_companies(search_term: str, companies_df: pd.DataFrame) -> pd.DataFrame:
    """Search companies by name or symbol"""
    return companies_df[
        companies_df['name'].str.contains(search_term, case=False, na=False) |
        companies_df['symbol'].str.contains(search_term, case=False, na=False)
    ]

# Modify the stock chart function to include earnings dates
def create_stock_chart(stock_data, earnings_data):
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='OHLC'
    ))
    
    # Add vertical lines for earnings dates
    if earnings_data:
        earnings_df = pd.DataFrame(earnings_data)
        earnings_df['date'] = pd.to_datetime(earnings_df['date'])
        
        # Add shapes for earnings dates
        fig.update_layout(
            shapes=[
                dict(
                    type='line',
                    xref='x',
                    yref='paper',
                    x0=date,
                    x1=date,
                    y0=0,
                    y1=1,
                    line=dict(
                        color='black',
                        width=1,
                        dash='dash'
                    )
                )
                for date in earnings_df['date']
            ]
        )
        
        # Add annotations for earnings dates
        fig.update_layout(
            annotations=[
                dict(
                    x=date,
                    y=1,
                    xref='x',
                    yref='paper',
                    text='Earnings',
                    showarrow=False,
                    textangle=-90,
                    font=dict(size=10)
                )
                for date in earnings_df['date']
            ]
        )
    
    fig.update_layout(
        title='Stock Price History',
        yaxis_title='Price',
        xaxis_title='Date',
        height=600
    )
    return fig

def create_earnings_chart(earnings_data):
    df = pd.DataFrame(earnings_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add EPS bars
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['reported_eps'],
        name='Reported EPS',
        marker_color=['green' if x > 0 else 'red' for x in df['surprise']]
    ))
    
    # Add price change markers
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price_change_1d'],
        mode='markers',
        name='1-Day Price Change',
        yaxis='y2',
        marker=dict(
            size=10,
            symbol='diamond'
        )
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price_change_5d'],
        mode='markers',
        name='5-Day Price Change',
        yaxis='y2',
        marker=dict(
            size=10,
            symbol='square'
        )
    ))
    
    fig.update_layout(
        title='Earnings Performance and Price Reactions',
        yaxis=dict(
            title='EPS',
            side='left'
        ),
        yaxis2=dict(
            title='Price Change %',
            side='right',
            overlaying='y'
        ),
        height=400,
        showlegend=True
    )
    return fig

# Main app
st.title("Stock Market Dashboard 📈")

# Sidebar
st.sidebar.header("Company Selection")
processor = load_data_processor()

# Add search box
search_term = st.sidebar.text_input("Search companies by name or symbol")
if search_term:
    filtered_companies = search_companies(search_term, processor.companies_df)
else:
    filtered_companies = processor.companies_df

# Update company selection to use filtered results
if not filtered_companies.empty:
    # Create a display format that combines symbol and name
    company_options = filtered_companies.apply(
        lambda x: f"{x['symbol']} - {x['name']}", 
        axis=1
    ).tolist()
    
    selected_option = st.sidebar.selectbox(
        "Select a company",
        company_options
    )
    
    # Extract symbol from the selected option
    selected_symbol = selected_option.split(' - ')[0] if selected_option else None
else:
    st.sidebar.warning("No companies found matching your search.")
    selected_symbol = None

if selected_symbol:
    st.session_state.selected_symbol = selected_symbol
    company_data = load_company_data(selected_symbol)
    company_info = processor.get_company_info(selected_symbol)
    
    # Company Info Section
    st.header(f"{company_info['name']} ({selected_symbol})")
    
    # Company description
    with st.expander("Company Description"):
        st.write(company_info['description'])
    
    # Create 4 columns now
    col1, col2, col3, col4 = st.columns(4)
    
    # First three columns remain the same
    with col1:
        st.subheader("Company Details")
        st.write(f"Country: {company_info['country']}")
        st.write(f"Industry: {company_info['industry']}")
        st.write(f"Sector: {company_info['sector']}")
    
    with col2:
        st.subheader("Analyst Ratings")
        st.write(f"Consensus: {company_info['analyst_rating'].title()}")
        st.write(f"Mean Rating: {company_info['recommendation_mean']:.2f}")
        st.write(f"Number of Analysts: {company_info['number_of_analysts']}")
    
    with col3:
        st.subheader("Price Targets")
        st.metric("High Target", f"${company_info['target_high']:.2f}")
        st.metric("Median Target", f"${company_info['target_median']:.2f}")
        st.metric("Low Target", f"${company_info['target_low']:.2f}")
    
    # Add new column for price changes
    with col4:
        st.subheader("Price Changes")
        price_changes = processor.get_price_changes(selected_symbol)
        if price_changes:
            if 'current_price' in price_changes:
                st.metric("Current Price", f"${price_changes['current_price']:.2f}")
            for period, label in [
                ('1d_change', '1 Day'),
                ('1w_change', '1 Week'),
                ('1m_change', '1 Month'),
                ('1y_change', '1 Year'),
                ('2y_change', '2 Years')
            ]:
                if period in price_changes and price_changes[period] is not None:
                    st.metric(label, f"{price_changes[period]:.2f}%")
                else:
                    st.metric(label, "N/A")
    
    # Stock Price Chart
    st.plotly_chart(
        create_stock_chart(
            company_data['stock_data'],
            company_data['earnings_data']
        ),
        use_container_width=True
    )
    
    # Earnings Section
    st.header("Earnings History")
    if company_data['earnings_data']:
        st.plotly_chart(create_earnings_chart(company_data['earnings_data']), use_container_width=True)
        
        # Create a DataFrame for the earnings table
        earnings_df = pd.DataFrame(company_data['earnings_data'])
        earnings_df = earnings_df.sort_values('date', ascending=False)
        
        # Format the price changes
        for col in ['price_change_1d', 'price_change_5d']:
            earnings_df[col] = earnings_df[col].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            )
        
        st.dataframe(earnings_df)
    else:
        st.info("No earnings data available for this company.")

    # Add this in the main section where you display company info
    if company_data['premarket_data']:
        premarket = company_data['premarket_data']
        
        st.subheader("Extended Hours Trading")
        cols = st.columns(3)
        
        # Regular Market Info
        with cols[0]:
            st.markdown("**Regular Market**")
            if premarket.get('regular_market_price'):
                st.metric(
                    "Current Price",
                    f"${premarket['regular_market_price']:.2f}",
                    f"{((premarket['regular_market_price'] / premarket['previous_close'] - 1) * 100):.2f}%" 
                    if premarket.get('previous_close') else None
                )
        
        # Pre-market Info
        with cols[1]:
            st.markdown("**Pre-market**")
            if premarket.get('pre_market_price'):
                st.metric(
                    "Pre-market Price",
                    f"${premarket['pre_market_price']:.2f}",
                    f"{premarket['pre_market_change_percent']:.2f}%" 
                    if premarket.get('pre_market_change_percent') else None
                )
            else:
                st.write("Pre-market closed")
        
        # After Hours Info
        with cols[2]:
            st.markdown("**After Hours**")
            if premarket.get('post_market_price'):
                st.metric(
                    "After Hours Price",
                    f"${premarket['post_market_price']:.2f}",
                    f"{premarket['post_market_change_percent']:.2f}%" 
                    if premarket.get('post_market_change_percent') else None
                )
            else:
                st.write("After hours closed")

# Footer
st.markdown("---")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 
st.markdown("Created by Maximilian Belschner")