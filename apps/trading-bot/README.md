# Exness MT5 Algorithmic Trading Bot & Historical Backtester

High-probability top-down algorithmic trading bot targeting **Forex, XAUUSD (Gold), and BTCUSD** on Exness MetaTrader 5.

## Architecture
- **Step 1:** Macro & Quarterly Liquidity Bias (`src/macro_liquidity.py`)
- **Step 2:** Intermediate Structure & Fixed Range Volume Profile (`src/volume_profile.py`, `src/htf_structure.py`)
- **Step 3:** Lower-Timeframe Precision Entry (`src/ltf_entries.py` - QMR & AMD)
- **Step 4:** Strict Risk Management & Minimum 1:5 RR Filter (`src/risk_manager.py`)
- **Execution & Storage:** MT5 Connector (`src/mt5_connector.py`), Backtester (`src/backtester.py`), SQLite DB (`src/database.py`), Live Daemon (`src/bot.py`)
- **Analytics:** Streamlit + Plotly Dashboard (`dashboard/app.py`)
