from pydantic import BaseModel
from datetime import date, timedelta
from typing import List, Dict, Any, Tuple, Optional
import abc
from datasets import load_dataset, Dataset
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from tqdm.auto import tqdm
import pickle
import json
from pprint import pprint
import os
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum
import requests
from typing import Dict, Any
import time
from datetime import date
import holidays
import modal
import pickle
import os
# Select country
CACHE_VOLUME="/cache"
us_holidays = holidays.US()
"""
REMEMBER TO SET NEWSAPI_SECRET in your environment variables!
"""
OLDEST_POSSIBLE_DATE = "2023-06-30"
NEWEST_POSSIBLE_DATE = "2025-4-1"

def get_business_days(start_date="2023-06-30", end_date=None):
    """
    Generate a list of business days between start_date and end_date (inclusive).
    If end_date is not provided, uses current date.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format

    Returns:
        list: List of datetime objects representing business days
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Convert string dates to datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Generate business days using pandas
    business_days = pd.date_range(start=start, end=end, freq='B')

    return business_days.tolist()



class FinancialsT(BaseModel):
    year: date
    financials: str


class PriceT(BaseModel):
    open: float
    close: float
    price_date: date
    open_previous: float
    close_previous: float
    previous_date: date


class CompanyInfoT(BaseModel):
    name: str
    description: str


class NewsT(BaseModel):
    news_headlines: List[str]
    news_date : date

class CompanyInfoAtDate(BaseModel):
    ticker: str
    current_date: date

    company_info: CompanyInfoT
    news: NewsT
    financials: FinancialsT
    price: PriceT


class AbstractCompanyInfoCreator:
    @abc.abstractmethod
    def fetch_company_info(self, ticker: str, current_date: date) -> CompanyInfoAtDate:
        pass
def format_datetime(newsdate : date):
  return newsdate.strftime("%Y-%m-%d")


class NewsDatabase():
    def __init__(self, start_date: date, end_date: date):
        self.ds = None
        self.cache = {}
        self.cache_file = os.path.join(CACHE_VOLUME,"news_cache_7.pkl")
        self._load_cache(start_date, end_date)


    def _load_cache(self, start_date: date, end_date: date):
        """Load cache from disk if it exists"""
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
        except (FileNotFoundError):
            self.cache = {}
            self.preprocess_date_range(start_date, end_date)

    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def preprocess_date_range(self, start_date: date, end_date: date):
        """
        Preprocess and cache news headlines for all stocks between start_date and end_date.

        Args:
            start_date (date): Start date for preprocessing
            end_date (date): End date for preprocessing
        """
        print(os.listdir(CACHE_VOLUME), os.listdir("/"))
        self.ds = load_dataset(
            "chuotchuilacduong/FNSPID_IMPROVED", split="train"
        ).to_pandas()

        # Convert dates to string format for comparison with dataset
        start_str = format_datetime(start_date)
        end_str = format_datetime(end_date)

        # Filter dataset for date range
        date_filtered = self.ds[
            (self.ds["date"] >= start_str) &
            (self.ds["date"] <= end_str)
        ]

        # Group by date and stock
        grouped = date_filtered.groupby(["date", "stock"])["title"].apply(list).to_dict()

        # Update cache
        for (date_str, stock), headlines in tqdm(grouped.items()):
            date_str = (date_str[:10])
            # print(date_str, stock, headlines)
            if date_str not in self.cache:
                print("not in cache")
                self.cache[date_str] = {}
            self.cache[date_str][stock] = headlines
        # Save cache to disk
        self._save_cache()

    # Hàm lấy tin tức từ 7 ngày trước
    def fetch_news_for_date(self, newsdate: date, stock: str, company_info: CompanyInfoT) -> NewsT:
        """
        Fetch news for a given date and stock, using cache if available.

        Args:
            newsdate (date): Date to fetch news for
            stock (str): Stock symbol
            company_info (CompanyInfoT): Company information

        Returns:
            NewsT: News headlines and date
        """
        seven_days_ago = newsdate - timedelta(days=7)
        headlines = []

        # Try to get headlines from cache for the past 7 days
        current_date = seven_days_ago
        while current_date < newsdate:
            date_str = format_datetime(current_date)
            if date_str in self.cache and stock in self.cache[date_str]:
                headlines.extend(self.cache[date_str][stock])
            else:
                pass
                # # If not in cache, fetch from dataset
                # day_headlines = self.ds[
                #     (self.ds["date"] == date_str) &
                #     (self.ds["stock"] == stock)
                # ]["title"].tolist()
                # print(current_date, day_headlines)
                # # Update cache for this date and stock
                # if date_str not in self.cache:
                #     self.cache[date_str] = {}
                # self.cache[date_str][stock] = day_headlines
                # headlines.extend(day_headlines)
                # self._save_cache()

            current_date += timedelta(days=1)
        # print(headlines, newsdate, seven_days_ago)
        return NewsT(news_headlines=headlines, news_date=seven_days_ago)

    def summary(self) -> dict:
        return {
            "max_date": self.ds["date"].max(),
            "min_date": self.ds["date"].min(),
            "stock_count": self.ds["stock"].nunique(),
        }



class PriceOpenPriceCloseDatabase:
    CACHE_FILE = os.path.join(CACHE_VOLUME,"price_cache.pkl")

    def __init__(self):
        self.ds = None
        self.cache = self._load_or_create_cache()

    def _load_or_create_cache(self):
        if os.path.exists(self.CACHE_FILE):
            # Load existing cache
            with open(self.CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        else:
            # Create and save new cache
            cache = self._preprocess_data()
            with open(self.CACHE_FILE, 'wb') as f:
                pickle.dump(cache, f)
            return cache

    def _preprocess_data(self):
        from collections import defaultdict
        self.ds = load_dataset(
            "chuotchuilacduong/deepstock-stock-historical-prices-dataset-processed",
            split="train",
        ).to_pandas()
        print("Creating new cache...")
        cache = defaultdict(dict)

        for _, row in tqdm(self.ds.iterrows(), total=13900000):
            date_str = row['date']
            stock = row['stock']
            cache[date_str][stock] = {
                'open': row['open'],
                'close': row['close']
            }

        print("Cache creation complete")
        return cache

    # Lấy giá hiện tại và 7 ngày trước
    def fetch_open_close_for_date(self, price_date: date, stock: str) -> PriceT:
        seven_days_ago = price_date - timedelta(days=7)
        current_data = self.get_stock_price(stock, price_date)
        assert current_data is not None, f"Could not fetch data for {stock} on {price_date}"
        # Nếu không thấy lùi tới khi nào thấy
        while (seven_days_ago_data := self.get_stock_price(stock, seven_days_ago)) is None:
            seven_days_ago -= timedelta(days=1)
        # tạo 1 object Price để chứa giá hiện tại và 7 ngày trước
        return PriceT(
            open=current_data['open'],
            close=current_data['close'],
            price_date=price_date,
            open_previous=seven_days_ago_data['open'],
            close_previous=seven_days_ago_data['close'],
            previous_date=seven_days_ago
        )

    def get_stock_price(self, stock: str, pricedate: date) -> Optional[Dict[str, float]]:
        date_str = format_datetime(pricedate)
        try:
            return self.cache[date_str][stock]
        except KeyError:
            try:
                stock_data = yf.Ticker(stock).history(start=pricedate, end=pricedate+timedelta(days=1))
                print(f"Fetching {stock} data for {date_str}", stock_data)

                open_price = stock_data['Open'].iloc[0]
                close_price = stock_data['Close'].iloc[0]
                self.cache[date_str][stock] = {
                    'open': open_price,
                    'close': close_price
                }
                return self.cache[date_str][stock]
            except Exception as e:
                print(f"Error fetching {stock} data for {date_str}: {e}")
                return None


# lấy data từ từ Yahoo Finance
class FinancialsDatabase:
    def __init__(self):
        self.financials_cache = {}

    def fetch_financials_for_date(self, stock_date: date, stock: str) -> FinancialsT:
        if stock not in self.financials_cache:
            self.financials_cache[stock] = yf.Ticker(stock).financials
        dates = [date.date() for date in self.financials_cache[stock].columns]
        sorted_dates = sorted(dates)
        right_date = None
        for i in range(len(sorted_dates) - 1):
            ind = min(len(sorted_dates) - 1, i + 1)
            if sorted_dates[ind] > stock_date:
                right_date = sorted_dates[ind]
                break
        if right_date is None and stock_date > sorted_dates[-1]:
            right_date = sorted_dates[-1]
        return FinancialsT(
            financials=json.dumps(self.financials_cache[stock][
                right_date.strftime("%Y-%m-%d")
            ].to_dict()),
            year=right_date,
        )

# Lấy thông tin công ty từ Yahoo Finance
class CompanyInfoDatabase:
    def __init__(self):
        self.company_info_cache = {}

    def fetch_company_info(self, stock: str) -> CompanyInfoT:
        if stock not in self.company_info_cache:
            self.company_info_cache[stock] = yf.Ticker(stock).info
        return CompanyInfoT(
            name=self.company_info_cache[stock]["shortName"],
            description=self.company_info_cache[stock]["longBusinessSummary"],
        )


class CompanyInfoCreator(AbstractCompanyInfoCreator):
    def __init__(self, earliest_date: date, latest_date: date):
        self.news_db = NewsDatabase(earliest_date - timedelta(days=10), latest_date + timedelta(days=10))
        self.price_db = PriceOpenPriceCloseDatabase()
        self.financials_db = FinancialsDatabase()
        self.company_info_db = CompanyInfoDatabase()
        # print(self.news_db.summary())

    #Tổng hợp tất cả thông tin trong phương thức fetch_company_info
    def fetch_company_info(self, ticker: str, current_date: date) -> CompanyInfoAtDate:
        # start_time = time.time()
        company_info = self.company_info_db.fetch_company_info(ticker)
        # print(f"Fetched company info in {time.time() - start_time} seconds")
        # start_time = time.time()
        news = self.news_db.fetch_news_for_date(current_date, ticker, company_info)
        # print(f"Fetched news in {time.time() - start_time} seconds")
        # start_time = time.time()
        financials = self.financials_db.fetch_financials_for_date(current_date, ticker)
        # print(f"Fetched financials in {time.time() - start_time} seconds")
        # start_time = time.time()
        price = self.price_db.fetch_open_close_for_date(current_date, ticker)
        # print(f"Fetched price in {time.time() - start_time} seconds")
        return CompanyInfoAtDate(
            ticker=ticker,
            current_date=current_date,
            company_info=company_info,
            news=news,
            financials=financials,
            price=price,
        )
def get_sp500_tickers() -> List[str]:
    # Giả dạng làm một trình duyệt web thông thường
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # Dùng `requests` để tải nội dung HTML với header hợp lệ
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Báo lỗi nếu có vấn đề
    
    # Đưa nội dung HTML đã tải cho pandas xử lý
    tables = pd.read_html(response.text)
    return tables[0]["Symbol"].tolist()
def dump_company(company_info: Optional[CompanyInfoAtDate]) -> dict:
        if company_info is None:
            return None
        return company_info.model_dump()

def process_single_stock(data):
    cic = CompanyInfoCreator(datetime.strptime(OLDEST_POSSIBLE_DATE, "%Y-%m-%d").date(), datetime.strptime(NEWEST_POSSIBLE_DATE, "%Y-%m-%d").date())
    company_info : List[CompanyInfoAtDate] = []
    for ticker, day in zip(data['ticker'], data['day']):
        try:
            ci = cic.fetch_company_info(ticker, day.date())
            company_info.append(ci)
        except Exception as e:
            print(e)
            print(ticker, day)
            company_info.append(None)
            pass
    return {"company_info": [dump_company(ci) for ci in company_info]}

if __name__ == "__main__" and not (os.path.exists("price_cache.pkl") and os.path.exists("news_cache_7.pkl")):
    NewsDatabase(datetime.strptime(OLDEST_POSSIBLE_DATE, "%Y-%m-%d").date(), datetime.strptime(NEWEST_POSSIBLE_DATE, "%Y-%m-%d").date())
    PriceOpenPriceCloseDatabase()

#Tạo 1 image chứa env trên modal để đảm bảo env tính toán trên cloud
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.2.1")
    .pip_install([
        "datasets",
        "yfinance",
        "pandas",
        "requests",
        "pydantic",
        "tqdm",
        "holidays",
        "modal",
        "numpy",
        "transformers",
        "huggingface_hub",
    ])
    .run_commands("mkdir /cache")
    .add_local_file("price_cache.pkl", remote_path="/cache/price_cache.pkl")
    .add_local_file("news_cache_7.pkl", remote_path="/cache/news_cache_7.pkl")
)

app = modal.App(name="deepstock", image=image)

# End

@app.function(timeout=2000)
def get_company_info(ticker: str) -> Tuple[List[CompanyInfoAtDate], str]:
    cic = CompanyInfoCreator(datetime.strptime(OLDEST_POSSIBLE_DATE, "%Y-%m-%d").date(), datetime.strptime(NEWEST_POSSIBLE_DATE, "%Y-%m-%d").date())
    company_info : List[CompanyInfoAtDate] = []
    for day in get_business_days(OLDEST_POSSIBLE_DATE, NEWEST_POSSIBLE_DATE):
        try:
            ci = cic.fetch_company_info(ticker, day.date())
            company_info.append(ci)
        except Exception as e:
            print(e)
            print(ticker, day.date())
            company_info.append(None)
            pass
    return company_info, ticker

@app.local_entrypoint()
def main():
    tickers = get_sp500_tickers()
    tickers.remove("KVUE")
    tickers.remove("CEG")
    tickers.remove("VLTO")
    tickers.remove("GEHC")
    company_info_info = {}
    for result in get_company_info.map(tickers):
        company_info_dates, ticker = result
        company_info_info[ticker] = company_info_dates
    with open("company_info.pkl", "wb") as f:
        pickle.dump(company_info_info, f)
    dataset = []
    count_none = 0
    count_total = 0
    for ticker, company_info_dates in company_info_info.items():
        for company_info in company_info_dates:
            count_total += 1
            if company_info is None:
                count_none += 1
                continue
            dataset.append({
                "ticker": ticker,
                "company_info": company_info.model_dump()
            })
    print(f"Total number of data points: {count_total}")
    print(f"Number of missing data points: {count_none}")
    dataset = Dataset.from_list(dataset)
    dataset.push_to_hub("chuotchuilacduong/deepstock-sp500-companies-with-info")
