from dotenv import find_dotenv, dotenv_values, load_dotenv
load_dotenv(find_dotenv())


from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, initialize_agent
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import (
    Tool, BaseTool, StructuredTool, 
    DuckDuckGoSearchRun,
)
from pydantic.v1 import BaseModel, Field
from typing import Optional



# TOOLS #################################################################
from datetime import datetime, timedelta
import yfinance as yf

def get_date_range(days_ago, duration=1):
    from_date = datetime.now() - timedelta(days=days_ago+1)
    till_date = from_date + timedelta(days=duration)
    return from_date, till_date

def get_stock_price(symbol, days_ago=0, price_type='Close'):
    dates = get_date_range(days_ago)
    facts = yf.Ticker(symbol).history(
        start = dates[0].strftime('%Y-%m-%d'), 
        end   = dates[1].strftime('%Y-%m-%d')
    )
    return round(facts.iloc[-1][price_type], 2) if not facts.empty else None

def get_stock_price_for_date_range(symbol, start_date, end_date):
    return yf.Ticker(symbol).history(
        start = start_date.strftime('%Y-%m-%d'), 
        end   = end_date.strftime('%Y-%m-%d')
    )

def get_dividends(symbol):
    return yf.Ticker(symbol).dividends

def get_stock_performance(symbol, days_ago=7):
    start_date, end_date = get_date_range(days_ago, days_ago)
    history = get_stock_price_for_date_range(symbol, start_date, end_date)
    old_price = history.iloc[0]['Close']
    new_price = history.iloc[-1]['Close']
    return round(((new_price - old_price) / old_price) * 100, 2)


def calculate_SMA(ticker,window):
  data = yf.Ticker(ticker).history(period='1y').Close
  return str(data.rolling(window=window).mean().iloc[-1])

def calculate_EMA(ticker,window):
  data = yf.Ticker(ticker).history(period='1y').Close
  return str(data.ewm(span=window,adjust=False).mean().iloc[-1])

def calculate_RSI(ticker):
  data = yf.Ticker(ticker).history(period='1y').Close
  delta = data.diff()
  up = delta.clip(lower=0)
  down= -1 *delta.clip(upper=0)
  ema_up=up.ewm(com=14-1, adjust= False).mean()
  ema_down=down.ewm(com=14-1, adjust= False).mean()
  rs = ema_up/ema_down
  return str(100-(100/(1+rs)).iloc[-1])

def calculate_MACD(ticker):
  data = yf.Ticker(ticker).history(period='1y').Close
  short_EMA=data.ewm(span=12,adjust=False).mean()
  long_EMA=data.ewm(span=26,adjust=False).mean()
  
  MACD=short_EMA-long_EMA
  signal =MACD.ewm(span=9,adjust=False).mean()
  
  MACD_histogram= MACD-signal
  
  return f'{MACD[-1]},{signal[-1]}, {MACD_histogram[-1]}'

def plot_stock_price(ticker):
  data = yf.Ticker(ticker).history(period='1y')
  plt.figure(figsize=(10,5))
  plt.plot(data.index,data.Close)
  plt.title(f'{ticker} Stock Price Over Last Year')
  plt.xlabel('Date')
  plt.ylabel('Stock Price ($)')
  plt.grid(True)
  plt.savefig('stock.png')
  plt.close()
  
    
search = DuckDuckGoSearchAPIWrapper()
search_tool = Tool.from_function(
    func=search.run,
    name="DuckDuckGo Search",
    description="Useful to browse information from the Internet."
)

class StockPriceSchema(BaseModel):
    """Input for Stock price check."""
    symbol:     str = Field(..., description="Ticker symbol for stock or index")
    days_ago:   Optional[int] = Field(default=0, description="Number of days ago to check the stock price for")
    price_type: Optional[str] = Field(default="Close", description="Which price to look for (Open, High, Low, Close, Volume)")

class StockPriceTool(BaseTool):
    name = "yFinance Stock Ticker Tool"
    description = "Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"
    args_schema = StockPriceSchema

    def _run(self, symbol: str, days_ago: Optional[int] = 0, price_type='Close'):
        return get_stock_price(symbol, days_ago, price_type)

stock_tool = StructuredTool.from_function(
    func=get_stock_price,
    name="Stock Price",
    description="Useful for when you are need to find a stock price",
    args_schema=StockPriceSchema,
)

stock_dividend_tool = Tool.from_function(
    func=get_dividends,
    name="Stock Price Dividends",
    description="Useful for when you are need to find the dividends of a stock",
)

stock_performance_tool = Tool.from_function(
    func=get_stock_performance,
    name="Stock Performance Tool",
    description="Useful for when you are need to find out how a stock performed i a certain timeframe defined by days_ago",
)


import yahoo_fin.stock_info as si

stock_daily_losers = Tool.from_function(
    func=si.get_day_losers,
    name="Daily Stock losers",
    description="Useful for when you are need to find out the 100 worst performing stocks for today",
)
stock_daily_gainers = Tool.from_function(
    func=si.get_day_gainers,
    name="Daily Stock gainers",
    description="Useful for when you are need to find out the 100 best performing stocks for today",
)

# currency_tool = Tool.from_function(
#     func=si.get_currencies,
#     name="Currency Lookup Tool",
#     description="Useful for when you are need to find currencies and conversion rates",
# )



tools = [StockPriceTool(), stock_dividend_tool, stock_performance_tool, stock_daily_losers, stock_daily_gainers, search_tool]
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
agent_type = "structured-chat-zero-shot-react-description"
agent = initialize_agent(
    tools, llm, agent=agent_type, verbose=True
)

#prompt = "What's the high, low, closing and opening prices as well the volume of Apple seven days ago?"
#prompt = "Calculate the annual dividend of apple in the year 2023"
#prompt = "How high was the last dividend paid by apple"
#prompt = "How many days since last monday?"
#prompt = "Whats Microsoft stocks price?"
#agent.run(prompt)