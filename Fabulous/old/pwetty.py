import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pyfinviz.quote import Quote
import datetime

# Import the function to fetch stock quote
from pyfinviz.quote import Quote

def get_stock_quote(ticker):
    return Quote(ticker=ticker)

def create_line_plot(df, feature, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[feature], label=feature)
    plt.title(f"{feature} Over Time for {ticker}")
    plt.xlabel("Date")
    plt.ylabel(feature)
    plt.legend()
    st.pyplot()

def main():
    st.title("Stock Information Dashboard")

    # User input for stock ticker
    stock_ticker = st.text_input("Enter stock ticker:")
    min_date = datetime.date(2020, 1, 1) 
    max_date = datetime.date(2024, 1, 18)
    start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)


    if stock_ticker:
        # Fetch stock quote information
        stock_quote = get_stock_quote(ticker=stock_ticker)

        # Display relevant information
        st.subheader("Stock Quote Information")
        st.write(f"Ticker: {stock_quote.ticker}")

        st.write(f"Exchange: {stock_quote.exchange}")
        st.write(f"Company Name: {stock_quote.company_name}")
        st.write(f"Sectors: {', '.join(stock_quote.sectors)}")

        st.subheader("Fundamental Information")
        st.dataframe(stock_quote.fundamental_df)

        st.subheader("Outer Ratings")
        st.dataframe(stock_quote.outer_ratings_df)

        st.subheader("Outer News")
        st.dataframe(stock_quote.outer_news_df)

        st.subheader("Income Statement")
        st.dataframe(stock_quote.income_statement_df)

        st.subheader("Insider Trading")
        st.dataframe(stock_quote.insider_trading_df)

        # Allow users to select features for line plots
        selected_features = st.multiselect("Select features for line plots:", stock_quote.fundamental_df.columns)

        # Create line plots for selected features over time
        if selected_features:
            for feature in selected_features:
                create_line_plot(stock_quote.fundamental_df, feature, stock_ticker)
                st.text("Find how to add start/end dates to plot")

if __name__ == "__main__":
    main()





