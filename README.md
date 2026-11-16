# o1mbot | Exness MT5 Algorithmic Trading Bot & Historical Backtester

An end-to-end quantitative algorithmic trading system, historical backtester, and interactive performance dashboard for **Exness (MetaTrader 5)** targeting **Forex, XAUUSD (Gold), and BTCUSD**.

---

## 🏗️ Architecture & Strategy Pipeline

Identical signal evaluation and risk logic is shared 100% deterministically between historical backtesting and live/paper execution.

```
┌────────────────────────────────────────────────────────┐
│ Step 1: Macro & Quarterly Liquidity Bias (D1, W1, MN1) │
│ - Dec 1 Anchored Quarters + 2-Month Checkpoints        │
│ - 20d / 40d / 60d IPDA Highs/Lows                      │
│ - Inducement Sweeps & Primary 60d DOL Targeting        │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ Step 2: Intermediate Structure & FRVP (H4 / H1)        │
│ - Change of Character (CHOCH) Detection                │
│ - 60–100 Bin Fixed Range Volume Profile across Impulse │
│ - POC Alignment with Unmitigated FVG or Breaker Block  │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ Step 3: Lower-Timeframe Precision Entry (M15 / M5)     │
│ - Quasimodo Reversal (QMR: Head, Shoulder, Neckline)   │
│ - Accumulation, Manipulation & Distribution (AMD)      │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ Step 4: Strict Risk & Minimum 1:5 RR Filter            │
│ - SL beyond structural extreme + Dynamic Spread Buffer │
│ - TP at opposing 60-day Draw on Liquidity (DOL)        │
│ - HARD FILTER: Strictly reject if Planned RR < 5.0     │
│ - Exness Lot Sizing (XAUUSDm, BTCUSDm, EURUSDm)        │
└────────────────────────────────────────────────────────┘
```

---

## 📁 Monorepo Structure

```
o1mbot/
├── apps/
│   └── trading-bot/
│       ├── config/
│       │   └── settings.yaml          # Strategy, symbol, and MT5 configuration
│       ├── dashboard/
│       │   └── app.py                 # Streamlit + Plotly interactive analytics dashboard
│       ├── src/
│       │   ├── backtester.py          # Realistic event-driven simulator
│       │   ├── bot.py                 # Live/paper trading execution daemon
│       │   ├── database.py            # SQLite metadata database manager
│       │   ├── htf_structure.py       # CHOCH, FVG, and Breaker block detectors
│       │   ├── ltf_entries.py         # Precision QMR and AMD pattern detectors
│       │   ├── macro_liquidity.py     # IPDA 20/40/60-day quarterly liquidity analyzer
│       │   ├── models.py              # Strongly-typed data models
│       │   ├── mt5_connector.py       # MT5 API wrapper + Mock offline adapter
│       │   ├── risk_manager.py        # Position sizing & >= 5.0 RR enforcement
│       │   ├── strategy.py            # Unified top-down orchestrator
│       │   └── volume_profile.py      # Fixed Range Volume Profile calculator
│       ├── tests/                     # Comprehensive pytest test suite
│       ├── package.json               # Thin Turborepo task wrapper
│       └── pyproject.toml             # Python 3.12+ project & tooling configuration
├── .github/workflows/
│   ├── ci.yml                         # Automated CI pipeline
│   └── release.yml                    # Semantic Release automation
├── .husky/                            # Git hooks (pre-commit, commit-msg)
├── commitlint.config.js               # Conventional commit enforcement
├── package.json                       # Monorepo root configuration
├── pnpm-workspace.yaml                # pnpm workspace definition
├── release.config.js                  # Semantic Release configuration
└── turbo.json                         # Turborepo task pipeline configuration
```

---

## ⚡ Quickstart

### Prerequisites
- Node.js (>= 20) & `pnpm` (>= 9)
- Python (>= 3.11, recommended 3.12) & `uv`
- MetaTrader 5 (Windows only, optional — Mock adapter runs automatically in headless/Linux environments)

### 1. Install Dependencies
```bash
# Install monorepo JS/Turborepo dependencies
pnpm install

# Sync Python virtual environment & dependencies
cd apps/trading-bot
uv sync --extra dev
cd ../..
```

### 2. Run Quality Checks via Turborepo
```bash
# Run linting across the monorepo
pnpm turbo run lint

# Run static type checking with pyright
pnpm turbo run typecheck

# Run pytest unit tests with Turborepo caching
pnpm turbo run test
```

### 3. Launch the Interactive Dashboard
```bash
# Start Streamlit analytics web application
pnpm --filter @o1mbot/trading-bot dev
# Or directly:
cd apps/trading-bot && uv run streamlit run dashboard/app.py
```

### 4. Run the Live/Paper Bot Daemon
```bash
cd apps/trading-bot
uv run python src/bot.py --config config/settings.yaml
```

---

## 🧪 Testing Coverage

The test suite validates:
- **IPDA Lookbacks & Anchors:** 20/40/60-day levels and December anchored quarterly boundaries.
- **Inducement Sweeps:** Detecting sweeps of liquidity into HTF Fair Value Gaps.
- **Volume Profile Math:** Binning algorithm, Point of Control (POC), and 70% Value Area.
- **PD Array Confluence:** Strict alignment of POC with unmitigated FVGs and Breakers.
- **Structure & Patterns:** CHOCH detection, QMR (Head, Shoulder, Neckline), and AMD triggers.
- **Hard Minimum 1:5 RR Filter:** Rejection of trades with Planned RR < 5.0, acceptance when >= 5.0.
- **Exness Contract Specs:** Dynamic lot sizing across Gold (`XAUUSDm`), Bitcoin (`BTCUSDm`), and Forex.
