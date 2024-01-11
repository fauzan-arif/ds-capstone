import pandas as pd
from cachetools import cached, TTLCache 
from datetime import timedelta

class TickerLists:
    @classmethod
    @property
    @cached(cache=TTLCache(maxsize=128, ttl=timedelta(days=1).total_seconds()))
    def nasdaq100(self):
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        return sorted(pd.read_html(url)[4].Ticker.to_list())

    @classmethod
    @property
    @cached(cache=TTLCache(maxsize=128, ttl=timedelta(days=1).total_seconds()))
    def sp500(self):
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        return sorted(pd.read_html(url)[0].Symbol.to_list())
        
    @classmethod
    @property
    @cached(cache=TTLCache(maxsize=128, ttl=timedelta(days=1).total_seconds()))
    def dow(self):
        url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        return sorted(pd.read_html(url)[1].Symbol.to_list())
        
    @classmethod
    @property
    @cached(cache=TTLCache(maxsize=128, ttl=timedelta(days=1).total_seconds()))
    def dow(self):
        url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        return sorted(pd.read_html(url)[1].Symbol.to_list())
        
    @classmethod
    @property
    @cached(cache=TTLCache(maxsize=128, ttl=timedelta(days=1).total_seconds()))
    def ftse100(self):
        url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
        return sorted(pd.read_html(url)[4].Ticker.to_list())

    @classmethod
    @property
    @cached(cache=TTLCache(maxsize=128, ttl=timedelta(days=1).total_seconds()))
    def nifty50(self):
        url = 'https://en.wikipedia.org/wiki/NIFTY_50'
        return sorted(pd.read_html(url)[3].Symbol.to_list())

    @classmethod
    @property
    @cached(cache=TTLCache(maxsize=128, ttl=timedelta(days=1).total_seconds()))
    def dax40(self):
        url = 'https://en.wikipedia.org/wiki/DAX'
        return sorted(pd.read_html(url)[4].Ticker.to_list())

    @classmethod
    @property
    @cached(cache=TTLCache(maxsize=128, ttl=timedelta(days=1).total_seconds()))
    def mdax50(self):
        url = 'https://en.wikipedia.org/wiki/MDAX'
        return sorted(pd.read_html(url)[2].Symbol.to_list())

    @classmethod
    @property
    @cached(cache=TTLCache(maxsize=128, ttl=timedelta(days=1).total_seconds()))
    def sdax70(self):
        url = 'https://en.wikipedia.org/wiki/SDAX'
        return sorted(pd.read_html(url)[2].Symbol.to_list())

    @classmethod
    @property
    def all(self):
        return {
            "Nasdaq 100": self.nasdaq100,
            "S&P 500":    self.sp500,
            "Dow Jones":  self.dow,
            "FTSE 100":   self.ftse100,
            "Nifty 50":   self.nifty50,
            "DAX 40":     self.dax40,
            "MDAX 50":    self.mdax50,
            "SDAX 70":    self.sdax70,
        }