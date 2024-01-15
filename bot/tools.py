from langchain.tools import Tool, BaseTool
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain
from pydantic.v1 import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta
from news_classifier import NewsClassifier
import yfinance as yf

#ll_match_chain = LLMMathChain.from_llm(llm=llm_model, verbose=True)

# calculator = Tool(
#     name="Calculator",
#     func=llm_math_chain.run,
#     description="useful for when you need to answer questions about math",
# )

date_tool = Tool(
    name="current_datetime",
    func=lambda x: datetime.now(),
    description="Useful for when you are need to find the current date and/or time. It returns a python datetime object for further querying. Call this before any other functions if you are unaware of the current date.",
)

class StockInfoSchema(BaseModel):
    """Input for Stock information."""
    symbol: str = Field(..., description="Ticker symbol for stock or index")
    key:    str = Field(default="currentPrice", description="Which information to look for. Default is the current price.")

class StockInfoTool(BaseTool):
    name = "Stock-Information-Tool"
    args_schema = StockInfoSchema
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
        return yf.Ticker(symbol).info[key]


class StockNewsSchema(BaseModel):
    symbol: str = Field(..., description="Ticker symbol for stock.")

class StockNewsTool(BaseTool):
    name = "Stock-News-Tool"
    args_schema = StockNewsSchema
    description = "Useful for when you need to find news about a company or stock"
    def _run(self, symbol: str):
        return yf.Ticker(symbol).news



class StockNewsSentimentSchema(BaseModel):
    text: str = Field(..., description="News text or headline to analyse")

class StockNewsSentimentTool(BaseTool):
    args_schema = StockNewsSentimentSchema
    name = "Financial-News-Sentiment-Tool"
    description = "Useful for when you need to get a financial sentiment for a news headlines."
    def _run(self, text: str):
        return NewsClassifier().sentiment_for(text)


stock_news_sentiment_tool = Tool(
    name = "Financial-News-Sentiment-Tool",
    func = lambda x: NewsClassifier().sentiment_for(x),
    description = "Useful for when you need to get a financial sentiment for a news headlines.",
)

stock_news_tool = Tool(
    name="Stock-News-Tool",
    func=lambda x: yf.Ticker(x).news,
    description="Useful for when you need to find news about a company or stock",
)

stock_dividend_tool = Tool(
    name="Stock-Dividend-Tool",
    func=lambda x: yf.Ticker(x).dividends,
    description="Useful for when you need to find dividends paid by a company. it returns a pandas dataframe which you can further analyse",
)

# Web Search Tool
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Web-Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on world events, issues, etc. Worth using for general topics. Use precise questions.",
)

# Wikipedia Tool
wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="A useful tool for searching the Internet to find information on world events, issues, etc. Worth using for general topics. Use precise questions.",
)

