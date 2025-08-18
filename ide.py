# US Stock & Options Screener Notebook with Option Chains, Greeks & Combined Ranking
# ================================================================
# This final code merges all previous versions: fetches US stocks, TradingView ratings, Yahoo Finance data, option chains, computes simplified Greeks,
# enforces earnings filters, liquidity, IV scoring adjustment, and outputs top N ranked stocks/options to CSV with a plot.

# ------------------------------
# 1. Install required packages
# ------------------------------
#!pip install yfinance tradingview_ta pandas numpy tqdm requests beautifulsoup4 joblib scipy mibian matplotlib

# ------------------------------
# 2. Imports
# ------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
from tradingview_ta import TA_Handler, Interval, Exchange
from tqdm import tqdm  # Instead of tqdm.notebook
from joblib import Parallel, delayed
from scipy.stats import norm
import datetime
import time
import logging
import mibian
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------
# 3. Configuration
# ------------------------------
CONFIG = {
    'MAX_STOCK_PRICE_TO_OWN_100': 1000, # How much I want to spend to buy 100 shares if option fails
    'MAX_PREMIUM_PER_CONTRACT': 500, # Cost to buy 1 contract
    'EARNINGS_WITHIN_DAYS': 14, # Next quarterly earning within
    'REQUIRE_TV_BUY': True, # True = only BUY/STRONG_BUY from TradingView
    'EXPIRY_DAYS_MIN': 7, # Option minimum expiration day
    'EXPIRY_DAYS_MAX': 60, # Option maximum expiration day
    'MAX_WORKERS': 24, # leave None to auto-scale ≈ 4× CPU cores
    'TOP_N': 50, # Top n list
    'OUTPUT_CSV_PATH': 'us_stock_options_screened.csv',
    'SCORE_WEIGHTS': {'tv':0.3, 'upside':0.3, 'delta':0.2, 'iv':0.2},
    'MIN_OPTION_VOLUME': 100,
    'MIN_OPTION_OPEN_INTEREST': 500
}


# ------------------------------
# 4. Timer decorator
# ------------------------------
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time for {func.__name__}: {end - start:.2f} seconds")
        logging.info(f"Execution time for {func.__name__}: {end - start:.2f} seconds")
        return result
    return wrapper

# ------------------------------
# 5. Fetch all US tickers
# ------------------------------
def fetch_us_stock_list():
    urls = {
        'nasdaq': 'https://old.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download',
        'nyse': 'https://old.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download',
        'amex': 'https://old.nasdaq.com/screening/companies-by-industry.aspx?exchange=AMEX&render=download'
    }
    tickers = []
    for url in urls.values():
        df = pd.read_csv(url)
        tickers.extend(df['Symbol'].tolist())
    return sorted(list(set(tickers)))

# ------------------------------
# 6. Fetch TradingView rating
# ------------------------------
def get_tv_rating(symbol):
    try:
        handler = TA_Handler(symbol=symbol, screener='america', exchange='NASDAQ', interval=Interval.INTERVAL_1_DAY)
        analysis = handler.get_analysis()
        rating = analysis.summary['RECOMMEND']
        return {'rating': rating, 'score': {'STRONG_BUY':1.0,'BUY':0.8,'NEUTRAL':0.5,'SELL':0.2,'STRONG_SELL':0.0}.get(rating,0.5)}
    except:
        return {'rating': None, 'score': 0.5}

# ------------------------------
# 7. Fetch Yahoo Finance info and option chains with Greeks
# ------------------------------
def get_yf_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        current_price = info.get('regularMarketPrice', np.nan)
        target_price = info.get('targetMeanPrice', np.nan)
        earnings_date = info.get('earningsDate', [None])[0]
        options = stock.options
        option_data = []
        for exp in options[:5]:
            calls = stock.option_chain(exp).calls
            calls['expirationDate'] = exp
            calls['type'] = 'call'
            T = (pd.to_datetime(exp) - datetime.datetime.now()).days/365
            S = current_price
            for idx,row in calls.iterrows():
                K = row['strike']
                sigma = row['impliedVolatility'] if row['impliedVolatility']>0 else 0.3
                d1 = (np.log(S/K)+0.5*sigma**2*T)/(sigma*np.sqrt(T)) if T>0 else 0
                row['delta'] = norm.cdf(d1)
                row['gamma'] = norm.pdf(d1)/(S*sigma*np.sqrt(T)) if T>0 else 0
                row['theta'] = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) if T>0 else 0
                row['vega'] = S*norm.pdf(d1)*np.sqrt(T) if T>0 else 0
                calls.loc[idx]=row
            option_data.append(calls)
        options_df = pd.concat(option_data, ignore_index=True) if option_data else pd.DataFrame()
        return {
            'symbol': symbol,
            'price': current_price,
            'target_price': target_price,
            'earnings_date': earnings_date,
            'options': options_df
        }
    except:
        return None

# ------------------------------
# 8. Screener function with scoring and filters
# ------------------------------
def screen_ticker(symbol):
    tv = get_tv_rating(symbol)
    yfdata = get_yf_data(symbol)
    if yfdata is None:
        return None
    if CONFIG['REQUIRE_TV_BUY'] and tv['rating'] not in ['BUY','STRONG_BUY']:
        return None
    if yfdata['price'] is None or yfdata['price']*100>CONFIG['MAX_STOCK_PRICE_TO_OWN_100']:
        return None
    # enforce earnings filter
    if yfdata['earnings_date']:
        days_until_earnings = (pd.to_datetime(yfdata['earnings_date']) - datetime.datetime.now()).days
        if days_until_earnings > CONFIG['EARNINGS_WITHIN_DAYS']:
            return None
    top_call = yfdata['options'].sort_values(by='impliedVolatility', ascending=False).head(1) if not yfdata['options'].empty else pd.DataFrame()
    # chance of profit estimation
    cop = top_call['delta'].iloc[0]*100 if not top_call.empty else 50
    # IV adjustment
    iv_score = min(top_call['impliedVolatility'].iloc[0]/50, 0.7) if not top_call.empty else 0.3
    # liquidity check
    if not top_call.empty:
        if top_call['volume'].iloc[0]<CONFIG['MIN_OPTION_VOLUME'] or top_call['openInterest'].iloc[0]<CONFIG['MIN_OPTION_OPEN_INTEREST']:
            return None
    upside = yfdata['target_price']/yfdata['price'] if yfdata['target_price'] else 1
    delta_score = top_call['delta'].iloc[0] if not top_call.empty else 0.5
    score = CONFIG['SCORE_WEIGHTS']['tv']*tv['score'] + CONFIG['SCORE_WEIGHTS']['upside']*upside + CONFIG['SCORE_WEIGHTS']['delta']*delta_score + CONFIG['SCORE_WEIGHTS']['iv']*iv_score
    return {
        'symbol': symbol,
        'price': yfdata['price'],
        'target_price': yfdata['target_price'],
        'earnings_date': yfdata['earnings_date'],
        'tv_rating': tv['rating'],
        'top_call': top_call,
        'score': score,
        'chance_of_profit': cop
    }

# ------------------------------
# 9. Run screener with timer and plot
# ------------------------------
@timed
def run_screener():
    try:
        print("Starting screener...")  # Debug print
        tickers = fetch_us_stock_list()
        logging.info(f"Total tickers fetched: {len(tickers)}")
        results = Parallel(n_jobs=CONFIG['MAX_WORKERS'])(delayed(screen_ticker)(t) for t in tqdm(tickers))
        results = [r for r in results if r is not None]
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='score', ascending=False).head(CONFIG['TOP_N'])
        df_results.to_csv(CONFIG['OUTPUT_CSV_PATH'], index=False)
        logging.info(f"Screener results exported to {CONFIG['OUTPUT_CSV_PATH']}")

        plt.figure(figsize=(12,6))
        plt.scatter(df_results['score'], df_results['chance_of_profit'], c='blue', alpha=0.7)
        for i, row in df_results.iterrows():
            plt.text(row['score'], row['chance_of_profit'], row['symbol'], fontsize=8, alpha=0.7)
        plt.xlabel('Screener Score')
        plt.ylabel('Chance of Profit (%)')
        plt.title(f'Top {CONFIG["TOP_N"]} Stocks: Score vs. Chance of Profit')
        plt.grid(True)
        plt.show()

        return df_results
    except Exception as e:
        print(f"Error in run_screener: {str(e)}")  # Debug print
        logging.error(f"Error in run_screener: {str(e)}")
        raise

# ------------------------------
# 10. Manual ticker analysis helper
# ------------------------------
def analyze_ticker(symbol):
    """
    Analyze a single stock manually:
    - Fetch TradingView rating
    - Fetch Yahoo Finance data and option chains
    - Display top call options with Greeks and IV
    """
    print(f"Analyzing {symbol}...\n")
    tv = get_tv_rating(symbol)
    yfdata = get_yf_data(symbol)
    
    if yfdata is None:
        print("No data available for this symbol.")
        return
    
    print(f"TradingView Rating: {tv['rating']}")
    print(f"Current Price: {yfdata['price']}")
    print(f"Target Price: {yfdata['target_price']}")
    print(f"Earnings Date: {yfdata['earnings_date']}\n")
    
    if not yfdata['options'].empty:
        top_calls = yfdata['options'].sort_values(by='impliedVolatility', ascending=False).head(5)
        print("Top 5 Call Options by Implied Volatility:")
        print(top_calls[['contractSymbol','strike','lastPrice','impliedVolatility','delta','gamma','theta','vega']])
    else:
        print("No option data available for this symbol.")

if __name__ == "__main__":
    run_screener()