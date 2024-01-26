import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pyfinviz.quote import Quote
import datetime
import yfinance as yf
# Import the function to fetch stock quote
from pyfinviz.quote import Quote
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import scipy
import seaborn as sns
from helper import NewsClassifier
from helper import save_model, load_model

st.title("Stock Information Dashboard")
with st.sidebar:
    st.title('Configuration')

    stock_ticker = st.selectbox('Choose your capitalist fighter!!', ["S&P 500","AMZN", "META", "GOOG", "AAPL", "MSFT"]) #'AOTA',
    max_date = datetime.date.today()
    min_date = max_date - datetime.timedelta(days=365*4)
    start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

if stock_ticker== "S&P 500":
    stock_ticker = "^GSPC"
    df = yf.download(stock_ticker, start_date, end_date)
else:
    df = yf.download(stock_ticker, start_date, end_date)
if stock_ticker == '^GSPC':
    # Create a candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    # Set the chart title and axis labels
    fig.update_layout(
        title= ' S&P 500 Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        plot_bgcolor='black')

    # Display the chart
    st.plotly_chart(fig)
else: 
    # Create a candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    # Set the chart title and axis labels
    fig.update_layout(
        title= stock_ticker + ' Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        plot_bgcolor='black')
    # Display the chart
    st.plotly_chart(fig)

        
    st.set_option('deprecation.showPyplotGlobalUse', False)
    def get_stock_quote(ticker):
        return Quote(ticker=ticker)

    def create_line_plot(df, feature, ticker, start_date, end_date):
        df.index = pd.to_datetime(df.index)
        filtered_df = df.loc[start_date:end_date]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(filtered_df.index, filtered_df[feature], label=feature)
        ax.set_title(f"{feature} Over Time for {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel(feature)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)  # Explicitly close the Matplotlib figure


    def main():

        # User input for stock ticker

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

            quote = Quote(ticker=stock_ticker)
            # Downloading and instantiating model
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            df_news = quote.outer_news_df
            # Function to perform sentiment analysis
            def get_sentiment(text):
                tokens = tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**tokens)
                probabilities = torch.softmax(outputs.logits, dim=1).detach().numpy()
                predicted_class = torch.argmax(outputs.logits, dim=1).item()
                sentiment_classes = ['negative', 'neutral', 'positive']
                predicted_sentiment = sentiment_classes[predicted_class]
                return predicted_sentiment, probabilities[0][0], probabilities[0][1], probabilities[0][2]
                return  NewsClassifier().sentiment_for(text)
            # Apply sentiment analysis to all headlines
            df_news[['Sentiment', 'Negative_Probability', 'Neutral_Probability', 'Positive_Probability']] = df_news['Headline'].apply(get_sentiment).apply(pd.Series)
            df_news = df_news[['Headline','From','Sentiment', 'Negative_Probability', 'Neutral_Probability', 'Positive_Probability']]

            # Display the resulting DataFrame with sentiment predictions
            st.title("Sentiment Analysis:")
            st.dataframe(df_news.head(10))

            st.subheader("Outer Ratings")
            st.dataframe(stock_quote.outer_ratings_df)

            st.subheader("Outer News")
            st.dataframe(stock_quote.outer_news_df)

            st.subheader("Income Statement")
            st.dataframe(stock_quote.income_statement_df)

            st.subheader("Insider Trading")
            st.dataframe(stock_quote.insider_trading_df)





            # Allow users to select features for line plots
           # selected_features = st.multiselect("Select features for line plots:", stock_quote.fundamental_df.columns)

            # Create line plots for selected features over time
        #  if selected_features:
        #     for feature in selected_features:
            #        create_line_plot(stock_quote.fundamental_df, feature, stock_ticker, start_date, end_date)
            #       st.text("Find how to add start/end dates to plot")

    if __name__ == "__main__":
        main()





