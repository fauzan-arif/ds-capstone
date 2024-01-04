from flask import Flask, render_template
from flask_babel import Babel
from urllib.parse import urlparse
import yfinance as yf
import pandas as pd

from news_classifier import NewsClassifier

app   = Flask(__name__)
babel = Babel(app, default_locale='en')

@app.route('/')
def index():
    tickers = sorted(pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol.to_list())
    return render_template('index.html', tickers=tickers)
    

@app.route('/<symbol>')
def ticker(symbol):
    ticker = yf.Ticker(symbol.upper())
    domain = urlparse(ticker.info['website']).hostname
    ticker_news = news_with_sentiment(ticker.news)
    
    return render_template('ticker.html', ticker=ticker, info=ticker.info, domain=domain, ticker_news=ticker_news)
    





def news_with_sentiment(news):
    nlp = NewsClassifier()
    for i, article in enumerate(news):
        news[i]['sentiment'] = nlp.sentiment_for(article['title'])

    return news
    
