import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from datetime import date, timedelta
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.base import BaseEstimator, TransformerMixin

import plotly.graph_objs as go

# -----------------------------------
# Configuration
# -----------------------------------
st.set_page_config(page_title="Algorithmic Stock Forecasting", layout="wide")
st.title("Algorithmic Stock Price Analysis and Prediction")
st.caption("Single-file Streamlit app using Pandas and scikit-learn for short-term equity forecasting, backtesting, and risk insights.")

# -----------------------------------
# Custom Transformers
# -----------------------------------

class TechnicalFeatures(BaseEstimator, TransformerMixin):
    """
    Compute technical features from OHLCV DataFrame with columns: ['Open','High','Low','Close','Adj Close','Volume'].
    Output is a numeric feature DataFrame aligned with input index.
    """
    def __init__(self, lags=5, roll_windows=(5, 10, 20), include_pct_change=True):
        self.lags = lags
        self.roll_windows = roll_windows
        self.include_pct_change = include_pct_change

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        feats = pd.DataFrame(index=df.index)

        # Returns and price changes
        if self.include_pct_change:
            feats["ret_1d"] = df["Adj Close"].pct_change()
            feats["log_ret_1d"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))

        # Lagged prices and returns
        for l in range(1, self.lags + 1):
            feats[f"lag_close_{l}"] = df["Adj Close"].shift(l)
            feats[f"lag_ret_{l}"] = feats["ret_1d"].shift(l)

        # Rolling statistics
        for w in self.roll_windows:
            feats[f"roll_mean_{w}"] = df["Adj Close"].rolling(w).mean()
            feats[f"roll_std_{w}"] = df["Adj Close"].rolling(w).std()
            feats[f"roll_vol_{w}"] = feats[f"roll_std_{w}"] * np.sqrt(252)
            feats[f"roll_min_{w}"] = df["Adj Close"].rolling(w).min()
            feats[f"roll_max_{w}"] = df["Adj Close"].rolling(w).max()

        # Momentum & Oscillators
        # RSI (Wilder’s)
        window_rsi = 14
        delta = df["Adj Close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(window_rsi).mean()
        roll_down = down.rolling(window_rsi).mean()
        rs = roll_up / (roll_down + 1e-9)
        feats["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD (12-26 EMA)
        ema12 = df["Adj Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Adj Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        feats["macd"] = macd
        feats["macd_signal"] = signal
        feats["macd_hist"] = macd - signal

        # ATR (Average True Range)
        high_low = (df["High"] - df["Low"]).abs()
        high_close = (df["High"] - df["Adj Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Adj Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        feats["atr_14"] = tr.rolling(14).mean()

        # Volume features
        feats["vol_roll_mean_20"] = df["Volume"].rolling(20).mean()
        feats["vol_roll_std_20"] = df["Volume"].rolling(20).std()

        # Price features
        feats["high_low_spread"] = (df["High"] - df["Low"]) / (df["Adj Close"] + 1e-9)

        # Drop initial NaNs due to rolling/lag
        feats = feats.replace([np.inf, -np.inf], np.nan)
        return feats

class TargetBuilder(BaseEstimator, TransformerMixin):
    """
    Build prediction target: next N-day return or next-day close.
    """
    def __init__(self, horizon=1, target_type="return"):
        self.horizon = horizon
        self.target_type = target_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if self.target_type == "return":
            target = df["Adj Close"].shift(-self.horizon).pct_change(self.horizon)
        elif self.target_type == "log_return":
            target = np.log(df["Adj Close"].shift(-self.horizon) / df["Adj Close"])
        else:  # "close"
            target = df["Adj Close"].shift(-self.horizon)
        return target

# -----------------------------------
# Utility Functions
# -----------------------------------

@st.cache_data(ttl=3600)
def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        return data
    
    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        # Extract price columns and flatten
        data.columns = data.columns.get_level_values(0)
    
    # Ensure standard column names
    if 'Adj Close' not in data.columns and 'Close' in data.columns:
        data['Adj Close'] = data['Close']
    
    return data

def align_features_target(raw, lags, roll_windows, horizon, target_type):
    tech = TechnicalFeatures(lags=lags, roll_windows=roll_windows)
    feats = tech.transform(raw)
    tgt_builder = TargetBuilder(horizon=horizon, target_type=target_type)
    target = tgt_builder.transform(raw)
    # Combine and drop rows with NaNs from initial windows or target shift
    df = pd.concat([raw, feats, target.rename("target")], axis=1)
    df = df.dropna()
    X = df[feats.columns]
    y = df["target"]
    return df, X, y

def timeseries_cv_scores(model, X, y, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)
    maes, rmses, r2s = [], [], []
    progress_bar = st.progress(0)
    for i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        maes.append(mean_absolute_error(y_te, pred))
        rmses.append(np.sqrt(mean_squared_error(y_te, pred)))
        r2s.append(r2_score(y_te, pred))
        progress_bar.progress((i + 1) / splits)
    return pd.DataFrame({"MAE": maes, "RMSE": rmses, "R2": r2s})

def walk_forward_backtest(model, X, y, initial_train=252, step=20):
    """
    Walk-forward: train on expanding window, predict next 'step' points repeatedly.
    """
    preds = pd.Series(index=y.index, dtype=float)
    num_steps = max(1, (len(X) - initial_train) // step)
    progress_bar = st.progress(0)
    step_count = 0
    
    for start in range(initial_train, len(X), step):
        end = min(start + step, len(X))
        X_train, y_train = X.iloc[:start], y.iloc[:start]
        X_test_idx = X.index[start:end]
        if len(X_test_idx) == 0:
            break
        model.fit(X_train, y_train)
        preds.loc[X_test_idx] = model.predict(X.iloc[start:end])
        step_count += 1
        progress_bar.progress(min(step_count / num_steps, 1.0))
    
    bt_df = pd.DataFrame({"y_true": y, "y_pred": preds})
    bt_df = bt_df.dropna()
    return bt_df

def error_var(residuals, cl=0.95):
    """
    Value-at-Risk on forecast errors to quantify downside risk from model misses.
    """
    if len(residuals) == 0:
        return np.nan
    return -np.quantile(residuals, 1 - cl)

def feature_importance_df(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    # permutation importance fallback
    from sklearn.inspection import permutation_importance
    return None  # Keep runtime lean; optional to compute permutation importance interactively.

# -----------------------------------
# Sidebar Inputs
# -----------------------------------

with st.sidebar:
    st.header("Inputs")
    st.markdown("- **Label:** Example tickers: RELIANCE.NS, TCS.NS, INFY.NS, AAPL")
    ticker = st.text_input("Ticker", value="RELIANCE.NS")
    start_date = st.date_input("Start date", value=date(2015,1,1))
    end_date = st.date_input("End date", value=date.today())
    horizon = st.number_input("Forecast horizon (days)", value=1, min_value=1, max_value=10, step=1)
    target_type = st.selectbox("Target type", ["return", "log_return", "close"])
    lags = st.number_input("Lag features (days)", value=5, min_value=1, max_value=30, step=1)
    roll_opt = st.multiselect("Rolling windows", options=[5,10,14,20,30,60,120], default=[5,10,20])
    scaler_on = st.checkbox("Scale features (StandardScaler)", value=True)
    model_name = st.selectbox("Model", ["RandomForest", "GradientBoosting", "Ridge", "Lasso", "ElasticNet"])
    cv_splits = st.number_input("TimeSeries CV splits", value=5, min_value=3, max_value=10, step=1)
    initial_train = st.number_input("Initial train size (days)", value=252, min_value=100, step=10)
    step_size = st.number_input("Walk-forward step (days)", value=20, min_value=5, step=5)
    grid_search = st.checkbox("Grid search hyperparameters", value=False)
    cl = st.slider("Confidence level for error VaR", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    run_btn = st.button("Run analysis")

if not run_btn:
    st.stop()

# -----------------------------------
# Data Load
# -----------------------------------

raw = download_data(ticker, start_date, end_date)
if raw.empty:
    st.error("No data returned. Check ticker and date range.")
    st.stop()

st.subheader("Data overview")
st.write(f"{ticker} OHLCV")
st.dataframe(raw.tail(10), use_container_width=True)

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=raw.index, y=raw["Adj Close"], name="Adj Close", mode="lines"))
fig_price.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig_price, use_container_width=True)

# -----------------------------------
# Feature Engineering & Target
# -----------------------------------

df, X, y = align_features_target(raw, lags=int(lags), roll_windows=tuple(roll_opt) if len(roll_opt)>0 else (5,10,20), horizon=int(horizon), target_type=target_type)
if len(X) < 300:
    st.warning("Limited effective samples after feature construction. Consider reducing lags/rolling windows or extending date range.")

st.subheader("Feature sample")
st.dataframe(X.tail(10).round(4), use_container_width=True)

# -----------------------------------
# Pipeline & Models
# -----------------------------------

num_features = X.columns.tolist()
preproc = Pipeline(steps=[("scaler", StandardScaler())]) if scaler_on else "passthrough"

def make_model(name):
    if name == "RandomForest":
        model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=1)
        param_grid = {
            "model__n_estimators": [150, 200],
            "model__max_depth": [10, 15],
            "model__min_samples_leaf": [2, 3]
        }
    elif name == "GradientBoosting":
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "model__n_estimators": [100, 150],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 4]
        }
    elif name == "Ridge":
        model = Ridge(random_state=42)
        param_grid = {"model__alpha": [0.1, 1.0, 10.0]}
    elif name == "Lasso":
        model = Lasso(random_state=42, max_iter=5000)
        param_grid = {"model__alpha": [0.01, 0.1, 1.0]}
    elif name == "ElasticNet":
        model = ElasticNet(random_state=42, l1_ratio=0.5, max_iter=5000)
        param_grid = {
            "model__alpha": [0.1, 1.0],
            "model__l1_ratio": [0.5, 0.8]
        }
    else:
        raise ValueError("Unknown model")
    return model, param_grid

base_model, param_grid = make_model(model_name)

pipe = Pipeline(steps=[
    ("preproc", preproc),
    ("model", base_model)
])

# Grid search with TimeSeriesSplit (use faster halving search for large datasets)
if grid_search:
    tscv = TimeSeriesSplit(n_splits=int(cv_splits))
    # Protect user from accidentally running very long searches on large datasets
    if len(X) > 2000:
        st.warning("Dataset is large — full grid-search may take a long time.")
        confirm = st.checkbox("Confirm and run grid-search on large dataset (may be slow)")
        if not confirm:
            st.info("⚡ Skipping grid-search on large dataset. Using default hyperparameters.")
            best_pipe = pipe.fit(X, y)
        else:
            st.info("🔄 Running an efficient halving grid-search (faster than exhaustive GridSearchCV)")
            gs = HalvingGridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1, factor=2, refit=True)
            with st.spinner("Running HalvingGridSearchCV..."):
                gs.fit(X, y)
            best_pipe = gs.best_estimator_
            st.subheader("Best hyperparameters")
            st.write(gs.best_params_)
    else:
        st.info("🔄 Grid search in progress... (this may take a minute)")
        gs = HalvingGridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1, factor=2, refit=True)
        with st.spinner("Running HalvingGridSearchCV..."):
            gs.fit(X, y)
        best_pipe = gs.best_estimator_
        st.subheader("Best hyperparameters")
        st.write(gs.best_params_)
else:
    st.info("⚡ Using default hyperparameters (grid search disabled)")
    best_pipe = pipe.fit(X, y)

# -----------------------------------
# Cross-validation scores
# -----------------------------------

st.subheader("TimeSeries cross-validation scores")
with st.spinner("Computing cross-validation scores..."):
    cv_df = timeseries_cv_scores(best_pipe, X, y, splits=int(cv_splits))
col1, col2, col3 = st.columns(3)
col1.metric("CV MAE (median)", f"{cv_df['MAE'].median():.6f}")
col2.metric("CV RMSE (median)", f"{cv_df['RMSE'].median():.6f}")
col3.metric("CV R2 (median)", f"{cv_df['R2'].median():.3f}")
st.dataframe(cv_df.round(6), use_container_width=True)

# -----------------------------------
# Walk-forward backtest
# -----------------------------------

st.subheader("Walk-forward backtest")
with st.spinner("Running walk-forward backtest..."):
    bt = walk_forward_backtest(best_pipe, X, y, initial_train=int(initial_train), step=int(step_size))
if bt.empty:
    st.error("Backtest produced no predictions; increase sample size or adjust initial train/step.")
    st.stop()

bt["residual"] = bt["y_true"] - bt["y_pred"]
bt["abs_error"] = bt["residual"].abs()

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(x=bt.index, y=bt["y_true"], name="True", mode="lines"))
fig_bt.add_trace(go.Scatter(x=bt.index, y=bt["y_pred"], name="Pred", mode="lines"))
fig_bt.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig_bt, use_container_width=True)

mae_bt = mean_absolute_error(bt["y_true"], bt["y_pred"])
rmse_bt = np.sqrt(mean_squared_error(bt["y_true"], bt["y_pred"]))
r2_bt = r2_score(bt["y_true"], bt["y_pred"])
err_var_95 = error_var(bt["residual"], cl=float(cl))

colA, colB, colC, colD = st.columns(4)
colA.metric("Backtest MAE", f"{mae_bt:.6f}")
colB.metric("Backtest RMSE", f"{rmse_bt:.6f}")
colC.metric("Backtest R2", f"{r2_bt:.3f}")
colD.metric(f"Error VaR @ {int(cl*100)}%", f"{err_var_95:.6f}")

st.write("Residual diagnostics (errors = true - pred)")
fig_res = go.Figure()
fig_res.add_trace(go.Scatter(x=bt.index, y=bt["residual"], name="Residual", mode="lines"))
fig_res.update_layout(height=280)
st.plotly_chart(fig_res, use_container_width=True)

# Distribution of residuals
hist = go.Figure(data=[go.Histogram(x=bt["residual"], nbinsx=40, name="Residuals")])
hist.update_layout(height=280)
st.plotly_chart(hist, use_container_width=True)

# -----------------------------------
# Feature Importance (if available)
# -----------------------------------

st.subheader("Feature importance")
try:
    model_inner = best_pipe.named_steps["model"]
    fi_df = feature_importance_df(model_inner, num_features)
    if fi_df is not None and not fi_df.empty:
        st.dataframe(fi_df.round(6), use_container_width=True)
        fig_fi = go.Figure([go.Bar(x=fi_df["feature"], y=fi_df["importance"])])
        fig_fi.update_layout(height=320, xaxis_tickangle=-35, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Permutation importance can be enabled for non-tree models, but may be slow. For production, compute offline or on subset.")
except Exception as e:
    st.warning(f"Could not display feature importance: {e}")

# -----------------------------------
# Forecast next horizon
# -----------------------------------

st.subheader("Next-horizon forecast")
# Train on full sample to forecast next horizon-step target
best_pipe.fit(X, y)
last_row = X.iloc[[-1]]
next_pred = best_pipe.predict(last_row)[0]
st.write(f"Model: {model_name} | Target type: {target_type} | Horizon: {int(horizon)} day(s)")
st.metric("Next forecast", f"{next_pred:.6f}")

# Scenario: price projection if target is return/log_return
if target_type in ["return", "log_return"]:
    last_price = df["Adj Close"].iloc[-1]
    if target_type == "return":
        proj_price = last_price * (1 + next_pred)
    else:  # log_return
        proj_price = last_price * np.exp(next_pred)
    st.write(f"Last price: {last_price:.2f}")
    st.metric("Projected price", f"{proj_price:.2f}")

# -----------------------------------
# Calibration and stability checks
# -----------------------------------

st.subheader("Calibration and stability")
# Rolling MAE of backtest
bt["rolling_mae_60"] = bt["abs_error"].rolling(60).mean()
fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(x=bt.index, y=bt["rolling_mae_60"], name="Rolling MAE (60)", mode="lines"))
fig_cal.update_layout(height=280)
st.plotly_chart(fig_cal, use_container_width=True)

# Error autocorrelation (quick look)
err_acf = [bt["residual"].autocorr(lag=k) for k in range(1, 21)]
acf_fig = go.Figure([go.Bar(x=list(range(1,21)), y=err_acf, name="Residual ACF")])
acf_fig.update_layout(height=280)
st.plotly_chart(acf_fig, use_container_width=True)

# -----------------------------------
# Export
# -----------------------------------

st.subheader("Export data and results")
colX, colY, colZ = st.columns(3)
with colX:
    st.write("Feature matrix (X)")
    st.download_button("Download X.csv", data=X.to_csv().encode("utf-8"), file_name=f"{ticker}_X.csv", mime="text/csv")
with colY:
    st.write("Target series (y)")
    st.download_button("Download y.csv", data=y.to_csv().encode("utf-8"), file_name=f"{ticker}_y.csv", mime="text/csv")
with colZ:
    st.write("Backtest predictions")
    st.download_button("Download backtest.csv", data=bt.to_csv().encode("utf-8"), file_name=f"{ticker}_backtest.csv", mime="text/csv")

st.markdown("---")
st.subheader("Method notes")
st.markdown("- **Label:** Target types: return = simple return over horizon; log_return = log difference; close = future adjusted close.")
st.markdown("- **Label:** Walk-forward: expanding training window with fixed-step predictions to emulate live deployment.")
st.markdown("- **Label:** Error VaR quantifies risk of forecast miss at the chosen confidence level, aiding risk management.")
st.markdown("- **Label:** Features include lags, rolling statistics, RSI, MACD, ATR, and volume metrics designed for short-term signals.")
