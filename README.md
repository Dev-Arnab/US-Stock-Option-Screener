# US Stock Option Screener

A powerful Python-based screener for US stocks and their options, designed to help you identify the best option opportunities using real-time data, technical analysis, and custom scoring. This project leverages Yahoo Finance and TradingView, computes option Greeks, and ranks results based on configurable filters.

---

## Features

- **Batch Screening:** Analyze hundreds or thousands of US stocks in parallel.
- **Option Chain Analysis:** Fetches option chains and computes Greeks (Delta, Gamma, Theta, Vega).
- **TradingView Integration:** Retrieves technical ratings for each stock.
- **Customizable Filters:** Filter by price, market cap, earnings date, option liquidity, and more.
- **Scoring System:** Combine technical, fundamental, and option metrics into a single score.
- **CSV Export & Visualization:** Saves results to CSV and plots top stocks by score.
- **Manual Stock Check:** Instantly analyze any ticker interactively.

---

## Requirements

- Python 3.7+
- Jupyter Notebook or VS Code

### Python Packages

Install all dependencies with:

```sh
pip install yfinance tradingview_ta pandas numpy tqdm requests beautifulsoup4 joblib scipy matplotlib tenacity
```

---

## Usage

1. **Prepare a CSV file** named `all_stocks.csv` with a column `Symbol` containing the tickers you want to screen.
2. **Open `screener.ipynb`** in Jupyter Notebook or VS Code.
3. **Run all cells** to load dependencies and functions.
4. **Start the screener** by running:

    ```python
    run_screener_from_csv()
    ```

   - This will process all tickers in batches, save results to a timestamped CSV, and plot the top 10 stocks by score.

5. **Check a single stock** interactively:

    ```python
    check_stock("AAPL")
    ```

---

## Configuration

You can adjust screening parameters in the `CONFIG` dictionary at the top of the notebook:

- `MAX_STOCK_PRICE_TO_OWN_100`: Max cost to buy 100 shares if assigned.
- `MAX_PREMIUM_PER_CONTRACT`: Max premium per contract.
- `EARNINGS_WITHIN_DAYS`: Only include stocks with earnings within this many days.
- `REQUIRE_TV_BUY`: Require TradingView "BUY" or "STRONG_BUY" rating.
- `EXPIRY_DAYS_MIN` / `EXPIRY_DAYS_MAX`: Option expiry window.
- `MIN_OPTION_VOLUME` / `MIN_OPTION_OPEN_INTEREST`: Liquidity filters.
- `MIN_MARKET_CAP`: Minimum market capitalization.
- `SCORE_WEIGHTS`: Adjust the importance of each metric in the final score.

---

## Output

- **CSV File:** Results are saved as `us_stock_options_screened_YYYY-MM-DD_HH-MM.csv`.
- **Plot:** Top 10 stocks by score are visualized in a bar chart.
- **Console Output:** Top 5 stocks and filter failure counts are printed.

---

## Notes

- **Data Sources:** Yahoo Finance and TradingView (internet required).
- **Performance:** Screening all US stocks may take several minutes depending on your hardware and network.
- **Error Handling:** The screener tracks and reports filter failures and API errors for transparency.

---

## License

This project is for educational and