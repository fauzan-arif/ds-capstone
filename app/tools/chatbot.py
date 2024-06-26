from dotenv import find_dotenv, dotenv_values, load_dotenv
load_dotenv(find_dotenv())


from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, AgentExecutor, initialize_agent, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import (
    Tool, BaseTool, StructuredTool, 
    DuckDuckGoSearchRun,
)
from pydantic.v1 import BaseModel, Field
from typing import Optional



# TOOLS #################################################################
import yfinance as yf
from datetime import datetime, timedelta
from langchain_experimental.tools import PythonREPLTool

class StockInfoSchema(BaseModel):
    """Input for Stock information."""
    symbol: str = Field(..., description="Ticker symbol for stock or index")
    key: Optional[str] = Field(default="currentPrice", description="Which information to look for. Default is the current price.")

class StockInfoTool(BaseTool):
    args_schema = StockInfoSchema
    name = "Stock Information Tool"
    description = """Useful for when you need to find out informations about a stock. 
        These Keys are available: address1, city, state, zip, country, phone, website, industry, 
        industryKey, industryDisp, sector, sectorKey, sectorDisp, longBusinessSummary, 
        fullTimeEmployees, companyOfficers, auditRisk, boardRisk, compensationRisk, 
        shareHolderRightsRisk, overallRisk, governanceEpochDate, compensationAsOfEpochDate,
        maxAge, priceHint, previousClose, open, dayLow, dayHigh, regularMarketPreviousClose, 
        regularMarketOpen, regularMarketDayLow, regularMarketDayHigh, dividendRate, dividendYield, 
        exDividendDate, payoutRatio, fiveYearAvgDividendYield, beta, trailingPE, forwardPE, volume, 
        regularMarketVolume, averageVolume, averageVolume10days, averageDailyVolume10Day, bid, ask,
        bidSize, askSize, marketCap, fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months,
        fiftyDayAverage, twoHundredDayAverage, trailingAnnualDividendRate, trailingAnnualDividendYield, 
        currency, enterpriseValue, profitMargins, floatShares, sharesOutstanding, sharesShort, 
        sharesShortPriorMonth, sharesShortPreviousMonthDate, dateShortInterest, sharesPercentSharesOut, 
        heldPercentInsiders, heldPercentInstitutions, shortRatio, shortPercentOfFloat, 
        impliedSharesOutstanding, bookValue, priceToBook, lastFiscalYearEnd, nextFiscalYearEnd, 
        mostRecentQuarter, earningsQuarterlyGrowth, netIncomeToCommon, trailingEps, forwardEps, pegRatio, 
        lastSplitFactor, lastSplitDate, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange, 
        SandP52WeekChange, lastDividendValue, lastDividendDate, exchange, quoteType, symbol, 
        underlyingSymbol, shortName, longName, firstTradeDateEpochUtc, timeZoneFullName, 
        timeZoneShortName, uuid, messageBoardId, gmtOffSetMilliseconds, currentPrice, 
        targetHighPrice, targetLowPrice, targetMeanPrice, targetMedianPrice, recommendationMean, 
        recommendationKey, numberOfAnalystOpinions, totalCash, totalCashPerShare, ebitda, totalDebt, 
        quickRatio, currentRatio, totalRevenue, debtToEquity, revenuePerShare, returnOnAssets, 
        returnOnEquity, grossProfits, freeCashflow, operatingCashflow, earningsGrowth, revenueGrowth, 
        grossMargins, ebitdaMargins, operatingMargins, financialCurrency, trailingPegRatio
    """

    def _run(self, symbol: str, key: str = 'currentPrice'):
        ticker = yf.Ticker(symbol)
#        return ticker.info[key] if key in ticker.info else None
        return ticker.info[key]
    


class StockNewsSchema(BaseModel):
    symbol: str = Field(..., description="Ticker symbol for stock.")

class StockNewsTool(BaseTool):
    args_schema = StockNewsSchema
    name = "Stock News Tool"
    description = "Useful for when you need to find news about a company or stock"
    def _run(self, symbol: str):
        ticker = yf.Ticker(symbol)
        return ticker.news

from tools.news_classifier import NewsClassifier

class StockNewsSentimentSchema(BaseModel):
    text: str = Field(..., description="News text or headline to analyse")

class StockNewsSentimentTool(BaseTool):
    args_schema = StockNewsSentimentSchema
    name = "Financial News Sentiment Tool"
    description = "Useful for when you need to get a financial sentiment for a news headlines."
    def _run(self, text: str):
        return NewsClassifier().sentiment_for(text)

    

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
    start_date: str = Field(..., description="The exact in the past")
    price_type: Optional[str] = Field(default="Close", description="Which price to look for (Open, High, Low, Close, Volume)")

class StockPriceTool(BaseTool):
    name = "yFinance Stock Ticker Tool"
    description = "Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"
    args_schema = StockPriceSchema

    def _run(self, symbol: str, start_date: str, price_type='Close'):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = start_date + timedelta(days=1)
        data = yf.Ticker(symbol).history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        return round(data.iloc[0][price_type], 2) if not data.empty else "No data available for this date."


# class StockPriceSchema(BaseModel):
#     """Input for Stock price check."""
#     symbol:     str = Field(..., description="Ticker symbol for stock or index")
#     days_ago:   Optional[int] = Field(default=0, description="Number of days ago to check the stock price for")
#     price_type: Optional[str] = Field(default="Close", description="Which price to look for (Open, High, Low, Close, Volume)")
#
# class StockPriceTool(BaseTool):
#     name = "yFinance Stock Ticker Tool"
#     description = "Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"
#     args_schema = StockPriceSchema
#
#     def _run(self, symbol: str, days_ago: Optional[int] = 0, price_type='Close'):
#         return get_stock_price(symbol, days_ago, price_type)

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

from datetime import datetime
date_tool = Tool.from_function(
    func=lambda x: datetime.now().strftime("%A, %B %d, %Y"),
    name="Current Date",
    description="Useful for when you are need to find the current date and/or time",
)


#tools = [StockPriceTool(), stock_dividend_tool, stock_performance_tool, stock_daily_losers, stock_daily_gainers, search_tool]

tools = [PythonREPLTool(), StockInfoTool(), StockNewsTool(), StockNewsSentimentTool(), StockPriceTool(), stock_dividend_tool, search_tool, date_tool]

memory = ConversationBufferMemory(memory_key="chat_history")
readonlymemory = ReadOnlySharedMemory(memory=memory)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106", streaming=True)

agent_type = "structured-chat-zero-shot-react-description"

agent = initialize_agent(
    tools=tools, llm=llm, agent=agent_type, verbose=True, memory=readonlymemory
)

#agent_executor = AgentExecutor(agent=agent, tools=tools)

# agent.run("You are an unparalleled financial expert with an exceptional understanding of both the stock market and cryptocurrencies. Your knowledge extends to intricate details of companies and their financial landscapes. Your expertise allows you to seamlessly correlate the movement of stock prices with relevant news stories, providing a comprehensive understanding of market dynamics. Additionally, you possess the unique ability to articulate complex financial concepts in a way that is accessible and easily understandable for individuals with varying levels of financial knowledge.")
#prompt = "What's the high, low, closing and opening prices as well the volume of Apple seven days ago?"
#prompt = "Calculate the annual dividend of apple in the year 2023"
#prompt = "How high was the last dividend paid by apple"
#prompt = "How many days since last monday?"
#prompt = "Whats Microsoft stocks price?"
#agent.run(prompt)