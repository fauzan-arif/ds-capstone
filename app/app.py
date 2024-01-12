# have a look at this chatbot:
# https://github.com/langchain-ai/opengpts/tree/main

# analyzing documents with langchain
#https://github.com/langchain-ai/langchain/blob/master/cookbook/analyze_document.ipynb

# Build a Chatbot in minutes with Chainlit, GPT-4 and LangChain
#https://medium.com/@cleancoder/build-a-chatbot-in-minutes-with-chainlit-gpt-4-and-langchain-7690968578f0

#https://data.nasdaq.com/tools/api

from flask import Flask, render_template, request, jsonify
from flask_babel import Babel
from flask_cors import CORS
from urllib.parse import urlparse
import yfinance as yf

app   = Flask(__name__)
babel = Babel(app, default_locale='en')
#CORS(app)

from tools.ticker_tools import TickerLists
from tools.news_classifier import NewsClassifier
from tools.chatbot import *

from flask import Response
import time
import json

class Custom:
    def chat(self, body):
        # Text messages are stored inside request body using the Deep Chat JSON format:
        # https://deepchat.dev/docs/connect
        #print(body)
        # Sends response back to Deep Chat using the Response format:
        # https://deepchat.dev/docs/connect/#Response
        return {"text": "This is a respone from a Flask server. Thankyou for your message!"}
        

@app.route('/')
def index():
    return render_template('index.html', tickers=TickerLists.all)
    
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')
    
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json['messages'][0]['text']
    response   = agent(user_input)
    return jsonify({"text": response })



@app.route('/<symbol>')
def ticker(symbol):
    ticker = yf.Ticker(symbol.upper())
    if 'website' in ticker.info:
        domain = urlparse(ticker.info['website']).hostname
    else:
        domain = ''
    ticker_news = news_with_sentiment(ticker.news)
    
    return render_template('ticker.html', ticker=ticker, info=ticker.info, domain=domain, ticker_news=ticker_news)


def news_with_sentiment(news):
    nlp = NewsClassifier()
    for i, article in enumerate(news):
        news[i]['sentiment'] = nlp.sentiment_for(article['title'])

    return news

