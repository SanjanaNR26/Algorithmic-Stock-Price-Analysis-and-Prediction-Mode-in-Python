# Algorithmic-Stock-Price-Analysis-and-Prediction-Mode
Developed a Machine Learning model in Python using Pandas and Scikit-learn for short-term Equity price forecasting, demonstrating skills in data analysis, algorithms, and risk management. This project involved systematic application of core data science methodologies and efficient data structures for processing historical financial data, essential for an Investment Research support role.  

# Algorithmic Stock Price Analysis and Prediction

A compact single-file Streamlit app for short-horizon equity forecasting, walk-forward backtesting, and simple risk diagnostics.

**File:** `Algorithmic Stock Price Analysis and Prediction Mode.py`

**Status:** Ready to run locally. The app uses `yfinance`, `pandas`, `numpy`, `scikit-learn`, and `plotly`.

**Overview**
- **Purpose:** Generate short-term forecasts (returns or next close), run time-series cross-validation, and perform walk-forward backtests to evaluate model stability and forecasting risk.
- **Approach:** Feature engineering (lags, rolling stats, RSI, MACD, ATR, volume features) + scikit-learn pipeline (optional `StandardScaler` + model). Default models: RandomForest, GradientBoosting, Ridge, Lasso, ElasticNet.

**Key Features**
- Technical feature builder: lagged prices/returns, rolling means/std, RSI, MACD, ATR, volume aggregates.
- Target builder: next N-day `return`, `log_return`, or `close` projection.
- TimeSeries cross-validation and walk-forward backtest with expanding training window.
- Efficient hyperparameter search: `HalvingGridSearchCV` (available but disabled by default).
- Progress UI: spinners and progress bars during CV/backtest.
- Caching: `st.cache_data(ttl=3600)` for downloaded price series to speed repeated runs.

**Requirements**
- Python 3.8+ (tested on 3.13)
- Packages: `streamlit`, `yfinance`, `pandas`, `numpy`, `scikit-learn`, `plotly`

Minimal install (use a virtualenv):

```powershell
python -m venv .venv;
.\.venv\Scripts\Activate.ps1;
pip install --upgrade pip;
pip install streamlit yfinance pandas numpy scikit-learn plotly
```

**Run the app**

Start the app from the repository root with a chosen port (8501–8520 are common):

```powershell
streamlit run "Algorithmic Stock Price Analysis and Prediction Mode.py" --server.port 8512
```

If the default port (8509 used historically) is occupied, pick another free port (e.g., `8512`).

**Sidebar configuration (summary)**
- **Ticker:** Example: `RELIANCE.NS`, `AAPL`.
- **Date Range:** Start / end dates for OHLCV history.
- **Forecast Horizon:** 1–10 days ahead.
- **Target Type:** `return`, `log_return`, or `close`.
- **Lag Features:** Number of lag days (1–30).
- **Rolling Windows:** Choose windows (5, 10, 14, 20, 30, 60, 120).
- **Scale features:** Toggle StandardScaler.
- **Model:** RandomForest, GradientBoosting, Ridge, Lasso, ElasticNet.
- **CV splits:** TimeSeriesSplit folds (3–10).
- **Walk-forward:** `Initial train size` and `Step size` control the backtest.
- **Grid Search:** Enable halving grid search (default: OFF).
- **VaR confidence level:** 90%–99%.

**Runtime notes & best practices**
- Grid search is OFF by default to keep the UI responsive. Enable it only for smaller datasets or short date ranges.
- When enabled, the app uses `HalvingGridSearchCV` for efficient progressive search and runs with `n_jobs=1` to avoid streamlit worker freezes.
- For very large datasets (len(X) > 2000), the app prompts for explicit confirmation before running hyperparameter searches.
- Use smaller param grids and fewer CV splits when experimenting interactively.
- If you need extensive hyperparameter sweeps, consider running them offline in a separate script or using randomized/time-budgeted search.

**Typical output**
- Interactive price / predicted charts (Plotly).
- CV metrics table and median summary (MAE, RMSE, R²).
- Walk-forward backtest with true vs predicted series and residual diagnostics.
- Feature importance for tree-based models (or note about permutation importance fallback).
- Downloadable CSVs: `X.csv`, `y.csv`, `backtest.csv`.

**Example quick test**
1. Start the app:

```powershell
streamlit run "Algorithmic Stock Price Analysis and Prediction Mode.py" --server.port 8512
```

2. In the sidebar: set `Ticker` to `AAPL`, `Start date` to `2018-01-01`, `End date` to today, keep `Grid Search` unchecked, choose `Forecast horizon = 1`, then click `Run analysis`.

This should return results quickly because grid search is disabled and the model uses default hyperparameters.

**Troubleshooting**
- If Streamlit reports a port in use, choose another port: add `--server.port 8512` to the run command.
- If no data returned, verify the ticker string and date range. `yfinance` can return empty frames for invalid symbols or missing historical data.
- If grid-search runs very slowly or freezes, cancel and re-run with `Grid Search` OFF, or reduce `CV splits` and param grid sizes.

**Suggested next steps (optional)**
- Add a background worker (e.g., Celery or a simple subprocess) to run long hyperparameter searches off the request thread.
- Add an option for `RandomizedSearchCV` with a user-configurable `n_iter` / time budget.
- Compute permutation importance asynchronously for non-tree models and cache results.

## License

This tool is for educational and research purposes only.

---

If you want, I can:
- Replace the existing `README.md` with this focused README, or
- Keep this as `README_Algorithmic_Stock_Forecasting.md` and also open the app now and tail the logs for live verification.



## OUTPUT

<img width="1135" height="317" alt="Screenshot 2025-12-10 102557" src="https://github.com/user-attachments/assets/4961c12e-44d4-45d1-a80b-a6be7ae66892" />


**The website link is http://localhost:8509**

**The above Webpage in PDF form -** [web.pdf](https://github.com/user-attachments/files/24070551/web.pdf)
