# 📈 Enhanced DCA Strategy Simulator for S&P 500

This repository provides a robust Python-based backtesting framework for simulating an **Enhanced Dollar-Cost Averaging (DCA)** strategy on the **S&P 500 (SPY)** ETF. The strategy intelligently adds capital during dip-buying opportunities, using technical indicators like **RSI**, **MACD**, **pullbacks**, and **black swan detection**.

---

## 🔍 Features

- 💰 **Fixed Monthly DCA** — invests a constant amount on a specified day each month.
- 📉 **Dynamic Dip Buying**:
  - Oversold RSI triggers
  - Pullbacks from recent highs
  - Extreme drawdowns ("Black Swan" events)
  - MACD-based confirmation
- 📊 **Buy & Hold Comparison** (optional)
- 📈 **Visualizations**:
  - Investment signals plotted on SPY price chart
  - RSI evolution
- 🧮 **Performance Metrics**:
  - ROI, CAGR, Sharpe/Sortino, Max Drawdown, Calmar Ratio, Alpha/Beta

---

## 🧠 Strategy Overview

This enhanced strategy extends basic DCA by adding extra investment under specific market conditions:

| Signal Type         | Trigger Condition                                    | Investment Amount |
|---------------------|------------------------------------------------------|--------------------|
| Fixed DCA           | 1st of each month                                     | $475               |
| Extra Normal        | Uptrend + RSI < 30                                    | $100               |
| Extra Purple Alert  | Extreme negative returns + RSI < 30 (Black Swan)      | $200               |
| Extra Pullback      | Price drops > 5% from 10-day high + RSI < 30          | $150               |
| Extra Extreme Dip   | RSI < 25 (extreme oversold)                           | $250               |

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/ESJavadex/black-swam-detector.git
cd black-swam-detector
```
---

## 📂 Output

- 📈 **Interactive Plots** (via Plotly)
- 📁 `enhanced_dca_transactions.csv` — detailed list of investments
- 📁 `performance_summary.csv` — side-by-side comparison of strategy metrics

---

## 📊 Example Visualization

![Sample Visualization](docs/sample_chart.png)

> Shows SPY price with overlaid investment markers (color-coded by strategy type).

---

## 📄 Configuration

Configuration flags are set at the top of the script:

```python
CONFIG = {
    'fixed_dca': True,
    'extra_normal': True,
    'extra_purple_alert': True,
    'extra_pullback': True,
    'extra_extreme_dip': True,
    'buy_and_hold': False,
}
```

You can toggle components on/off to compare strategies.

---

## 📌 Notes & Best Practices

- **Data Source**: Yahoo Finance via `yfinance`
- **Indicators**: RSI, MACD, SMA, volatility, returns
- **Backtest Range**: 2015–present (default)
- **Output Format**: Daily portfolio evolution with cumulative investment, share count, and value.

---

## ⚠️ Disclaimer

This script is for **educational and informational purposes only**.  
It does **not constitute financial advice**.  
Trading and investing carry significant risk.  
Use a **demo account** and consult a **financial advisor** before making decisions.

---

## 📜 License

MIT License. Feel free to use, modify, and share with attribution.