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
    def all(self):
        return {
            "Nasdaq 100": self.nasdaq100,
            "S&P 500": self.sp500 
        }