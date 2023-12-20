from flask import Flask, render_template
import yfinance as yf
import pandas as pd
from urllib.parse import urlparse

app = Flask(__name__)

@app.route('/')
def index():
    tickers = sorted(pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol.to_list())
    return render_template('index.html', tickers=tickers)
    
@app.route('/<symbol>')
def ticker(symbol):
    ticker = yf.Ticker(symbol.upper())
    domain = urlparse(ticker.info['website']).hostname
    return render_template('ticker.html', ticker=ticker, info=ticker.info, domain=domain)