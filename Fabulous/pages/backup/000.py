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

    stock_ticker = st.selectbox('Choose your capitalist fighter!!', ["AAPL", "AMZN", "MSFT",  "META", "GOOG"]) 
    min_date = datetime.date(2019, 1, 1) # 4years only, this is a dictatorship, with full censorship!
    max_date = datetime.date(2024, 1, 21)
    start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)
df = yf.download(stock_ticker, start_date, end_date)
def get_stock_quote(ticker):
    return Quote(ticker=ticker)
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
    return pd.Series({
        'Sentiment': predicted_sentiment,
        'Negative_Probability': probabilities[0][0],
        'Neutral_Probability': probabilities[0][1],
        'Positive_Probability': probabilities[0][2]
    })
# Apply sentiment analysis to all headlines
df_news = df_news[['Sentiment', 'Negative_Probability', 'Neutral_Probability', 'Positive_Probability']]
df_news = df_news['Headline'].apply(get_sentiment)
save_model(df_news, f'{stock_ticker}.pickle')
#df_news = df_news[['Headline','From','Sentiment', 'Negative_Probability', 'Neutral_Probability', 'Positive_Probability']]

