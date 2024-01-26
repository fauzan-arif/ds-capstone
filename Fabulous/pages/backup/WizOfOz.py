import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pyfinviz.quote import Quote
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
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

# Streamlit App
st.title("Financial Sentiment Analysis")

# Sidebar for user input
ticker = st.sidebar.text_input("Enter stock ticker:", "AAPL")
quote = Quote(ticker=ticker)
df_news = quote.outer_news_df

# Sentiment analysis and visualization
if not df_news.empty:
    # Apply sentiment analysis to all headlines
    df_news[['Sentiment', 'Negative_Probability', 'Neutral_Probability', 'Positive_Probability']] = df_news['Headline'].apply(get_sentiment).apply(pd.Series)

    # Display the resulting DataFrame with sentiment predictions
    st.subheader("Sentiment Analysis Results")
    st.dataframe(df_news.head(5))

    # Plot sentiment trend
    st.subheader("Sentiment Trend Over Time")
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Date', y='Positive_Probability', data=df_news, label='Positive Probability')
    sns.lineplot(x='Date', y='Negative_Probability', data=df_news, label='Negative Probability')
    plt.title('Sentiment Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.legend()
    st.pyplot(fig)
else:
    st.warning("No news data available for the entered stock ticker.")






