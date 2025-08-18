# US Stock Option Screener

This project is a Jupyter Notebook-based screener for US stocks and their options, featuring option chain analysis, Greeks calculation, TradingView ratings, Yahoo Finance data, and a combined ranking system. It helps identify top option opportunities based on configurable filters and scoring.

## Features

- Fetches all US stock tickers (NASDAQ, NYSE, AMEX)
- Retrieves TradingView technical ratings
- Downloads Yahoo Finance stock info and option chains
- Calculates simplified option Greeks (Delta, Gamma, Theta, Vega)
- Filters by earnings date, liquidity, and price constraints
- Scores and ranks options using configurable weights
- Outputs top N results to CSV and visualizes them
- Manual analysis helper for any ticker

## Requirements

- Python 3.7+
- Jupyter Notebook

### Python Packages

Install all dependencies with:

```sh
pip install yfinance tradingview_ta pandas numpy tqdm requests beautifulsoup4 joblib scipy mibian matplotlib
```

## Usage

1. Open `screener.ipynb` in Jupyter Notebook or VS Code.
2. Run the notebook cells in order.
3. To screen all stocks and export results, run:

```python
run_screener()
```

4. To analyze a single ticker manually, run:

```python
analyze_ticker("TICKER")
```

5. Adjust the `CONFIG` dictionary at the top of the notebook to change filters, scoring, and output options.

## Output

- Top N ranked stocks/options are saved to `us_stock_options_screened.csv`.
- A scatter plot of Screener Score vs. Chance of Profit is displayed.

## File Structure

- [`screener.ipynb`](screener.ipynb): Main notebook with all logic and usage examples.

## Notes

- Data is fetched live from Yahoo Finance and TradingView; internet connection required.
- Screening all US stocks may take significant time and resources.
- For testing, you can run the screener on a small subset by modifying the `test_tickers` list.

## License

This project is for educational and research