from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import yfinance as yf

load_dotenv()

def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period = "1d")
    return round(todays_data["Close"][0], 2)

#print(get_stock_price("AAPL"))

print(get_stock_price("GOOG"))
