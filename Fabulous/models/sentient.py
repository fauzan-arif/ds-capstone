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

st.title("Sentient")
with st.sidebar:
    st.title('Configuration')

    stock_ticker = st.selectbox('Choose your capitalist fighter!!', ["AMZN", "META", "GOOG", "AAPL", "MSFT"]) #'AOTA',

def get_stock_quote(ticker):
        return Quote(ticker=ticker)

stock_quote = get_stock_quote(ticker=stock_ticker)
quote = Quote(ticker=stock_ticker)
# Downloading and instantiating model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
df_news = quote.outer_news_df

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

save_model(df_news, f'{stock_ticker}_sent.pickle')