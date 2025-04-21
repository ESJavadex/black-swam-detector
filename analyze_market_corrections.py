#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Corrections Analysis Script
This script analyzes multiple significant market correction periods to identify
patterns that could indicate market bottoms and trend changes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime, timedelta
import argparse

# Define significant market correction periods to analyze
CORRECTION_PERIODS = {
    "2008_Financial_Crisis": {
        "start": "2008-09-01",
        "bottom": "2009-03-09",
        "end": "2009-06-01",
        "description": "Global Financial Crisis"
    },
    "2011_Debt_Ceiling": {
        "start": "2011-07-01",
        "bottom": "2011-10-03",
        "end": "2011-12-31",
        "description": "US Debt Ceiling Crisis"
    },
    "2015_2016_Oil_Crash": {
        "start": "2015-11-01",
        "bottom": "2016-02-11",
        "end": "2016-04-30",
        "description": "Oil Price Crash and China Slowdown"
    },
    "2018_Q4_Selloff": {
        "start": "2018-10-01",
        "bottom": "2018-12-24",
        "end": "2019-02-28",
        "description": "Fed Rate Hikes and Trade War Fears"
    },
    "2020_Covid_Crash": {
        "start": "2020-02-19",
        "bottom": "2020-03-23",
        "end": "2020-04-13",
        "description": "COVID-19 Pandemic"
    },
    "2022_Rate_Hikes": {
        "start": "2022-01-03",
        "bottom": "2022-10-12",
        "end": "2022-12-31",
        "description": "Fed Rate Hikes and Inflation"
    },
        "2023_Rate_Concerns": {
        "start": "2023-07-31", # Peak before the decline
        "bottom": "2023-10-27", # Lowest point of the correction (-10.3%)
        "end": "2023-11-30",   # Approximate recovery period end (market rebounded quickly after the bottom)
        "description": "Concerns over Fed signals for higher-for-longer interest rates"
    },
    "2025_Q1_Tariff_Fears": {
        "start": "2025-02-19", # Peak before the decline
        "bottom": "2025-04-04", # Approximate bottom, down ~17.5% from peak by early April
        "end": "2025-04-21",   # Still ongoing or very recent as of April 2025, setting end to current date for now.
        "description": "Trade war fears, new tariff policies, economic slowdown concerns"
    }
}

def fetch_data(start_date, end_date, tickers=None):
    """Fetch market data for analysis"""
    if tickers is None:
        tickers = ["SPY", "^VIX", "^TNX"]
    
    data = {}
    for ticker in tickers:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        ticker_data = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = ticker_data
        print(f"Fetched {len(ticker_data)} records for {ticker}.")
    
    return data

def calculate_indicators(df):
    """Calculate technical indicators for analysis"""
    # Calculate RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate volatility (20-day)
    df['Volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
    
    # Calculate percentage drops from recent highs
    for window in [5, 10, 20, 50, 252]:
        rolling_max = df['Close'].rolling(window=window).max()
        df[f'Drop_{window}d'] = ((df['Close'] - rolling_max) / rolling_max) * 100
    
    return df

def analyze_period(period_name, period_data, data):
    """Analyze a specific market correction period"""
    start_date = period_data["start"]
    bottom_date = period_data["bottom"]
    end_date = period_data["end"]
    description = period_data["description"]
    
    # Extract relevant data
    spy_data = data["SPY"].loc[start_date:end_date].copy()
    vix_data = data["^VIX"].loc[start_date:end_date].copy()
    tnx_data = data["^TNX"].loc[start_date:end_date].copy()
    
    # Calculate indicators
    spy_data = calculate_indicators(spy_data)
    
    # Get key prices - ensure they're scalar values, not Series
    start_price = spy_data['Close'].iloc[0]
    if isinstance(start_price, pd.Series):
        start_price = start_price.iloc[0]
    
    end_price = spy_data['Close'].iloc[-1]
    if isinstance(end_price, pd.Series):
        end_price = end_price.iloc[0]
    
    bottom_idx = spy_data.index.get_indexer([bottom_date], method='nearest')[0]
    bottom_price = spy_data['Close'].iloc[bottom_idx]
    if isinstance(bottom_price, pd.Series):
        bottom_price = bottom_price.iloc[0]
    
    # Calculate key metrics
    total_decline = ((end_price - start_price) / start_price) * 100
    max_decline = ((bottom_price - start_price) / start_price) * 100
    recovery = ((end_price - bottom_price) / bottom_price) * 100
    
    # Get VIX values
    start_vix = vix_data['Close'].iloc[0]
    if isinstance(start_vix, pd.Series):
        start_vix = start_vix.iloc[0]
    
    bottom_vix = vix_data['Close'].iloc[bottom_idx]
    if isinstance(bottom_vix, pd.Series):
        bottom_vix = bottom_vix.iloc[0]
    
    max_vix = vix_data['Close'].max()
    if isinstance(max_vix, pd.Series):
        max_vix = max_vix.iloc[0]
    
    max_vix_date = vix_data['Close'].idxmax()
    max_vix_date_str = max_vix_date.strftime('%Y-%m-%d') if hasattr(max_vix_date, 'strftime') else str(max_vix_date)
    
    # Get bond yield values
    start_yield = tnx_data['Close'].iloc[0]
    if isinstance(start_yield, pd.Series):
        start_yield = start_yield.iloc[0]
    
    bottom_yield = tnx_data['Close'].iloc[bottom_idx]
    if isinstance(bottom_yield, pd.Series):
        bottom_yield = bottom_yield.iloc[0]
    
    min_yield = tnx_data['Close'].min()
    if isinstance(min_yield, pd.Series):
        min_yield = min_yield.iloc[0]
    
    min_yield_date = tnx_data['Close'].idxmin()
    min_yield_date_str = min_yield_date.strftime('%Y-%m-%d') if hasattr(min_yield_date, 'strftime') else str(min_yield_date)
    
    # Get bottom day metrics
    bottom_day = spy_data.loc[bottom_date].copy() if bottom_date in spy_data.index else spy_data.iloc[bottom_idx]
    
    print(f"\n=== {period_name}: {description} ===")
    print(f"Analysis Period: {start_date} to {end_date}")
    print(f"Market Bottom: {bottom_date}")
    print(f"SPY Start Price: ${start_price:.2f}")
    print(f"SPY Bottom Price: ${bottom_price:.2f}")
    print(f"SPY End Price: ${end_price:.2f}")
    print(f"Total Decline: {max_decline:.2f}%")
    print(f"Recovery from Bottom: {recovery:.2f}%")
    
    print("\n--- Market Conditions at Bottom ---")
    print(f"VIX Level: {bottom_vix:.2f} (Peak: {max_vix:.2f} on {max_vix_date_str})")
    print(f"10Y Treasury Yield: {bottom_yield:.2f}% (Low: {min_yield:.2f}% on {min_yield_date_str})")
    
    # Print detailed metrics for the bottom day
    print("\n--- Technical Indicators at Bottom ---")
    try:
        # Extract scalar values for each metric
        rsi = bottom_day['RSI']
        if isinstance(rsi, pd.Series):
            rsi = rsi.iloc[0]
        
        volatility = bottom_day['Volatility_20d']
        if isinstance(volatility, pd.Series):
            volatility = volatility.iloc[0]
        
        drop_5d = bottom_day['Drop_5d']
        if isinstance(drop_5d, pd.Series):
            drop_5d = drop_5d.iloc[0]
        
        drop_10d = bottom_day['Drop_10d']
        if isinstance(drop_10d, pd.Series):
            drop_10d = drop_10d.iloc[0]
        
        drop_20d = bottom_day['Drop_20d']
        if isinstance(drop_20d, pd.Series):
            drop_20d = drop_20d.iloc[0]
        
        drop_50d = bottom_day['Drop_50d']
        if isinstance(drop_50d, pd.Series):
            drop_50d = drop_50d.iloc[0]
        
        drop_252d = bottom_day['Drop_252d']
        if isinstance(drop_252d, pd.Series):
            drop_252d = drop_252d.iloc[0]
        
        # Print the metrics
        print(f"RSI (14-day): {rsi:.2f}")
        print(f"Volatility (20-day): {volatility:.2f}%")
        print(f"5-day Drop: {drop_5d:.2f}%")
        print(f"10-day Drop: {drop_10d:.2f}%")
        print(f"20-day Drop: {drop_20d:.2f}%")
        print(f"50-day Drop: {drop_50d:.2f}%")
        print(f"252-day Drop: {drop_252d:.2f}%")
        
        # Check if price is below moving averages
        close_price = bottom_day['Close']
        if isinstance(close_price, pd.Series):
            close_price = close_price.iloc[0]
        
        ma50 = bottom_day['MA50'] if 'MA50' in bottom_day else None
        if isinstance(ma50, pd.Series):
            ma50 = ma50.iloc[0]
        
        ma200 = bottom_day['MA200'] if 'MA200' in bottom_day else None
        if isinstance(ma200, pd.Series):
            ma200 = ma200.iloc[0]
        
        below_50ma = "Yes" if ma50 is not None and close_price < ma50 else "No/N/A"
        below_200ma = "Yes" if ma200 is not None and close_price < ma200 else "No/N/A"
        
        print(f"Below 50-day MA: {below_50ma}")
        print(f"Below 200-day MA: {below_200ma}")
    except Exception as e:
        print(f"Error calculating some metrics: {e}")
    
    print("========================================")
    
    # Extract scalar values for metrics to return
    try:
        rsi = bottom_day['RSI']
        if isinstance(rsi, pd.Series):
            rsi = rsi.iloc[0]
            
        volatility = bottom_day['Volatility_20d']
        if isinstance(volatility, pd.Series):
            volatility = volatility.iloc[0]
            
        drop_5d = bottom_day['Drop_5d']
        if isinstance(drop_5d, pd.Series):
            drop_5d = drop_5d.iloc[0]
            
        drop_10d = bottom_day['Drop_10d']
        if isinstance(drop_10d, pd.Series):
            drop_10d = drop_10d.iloc[0]
            
        drop_20d = bottom_day['Drop_20d']
        if isinstance(drop_20d, pd.Series):
            drop_20d = drop_20d.iloc[0]
            
        drop_50d = bottom_day['Drop_50d']
        if isinstance(drop_50d, pd.Series):
            drop_50d = drop_50d.iloc[0]
            
        drop_252d = bottom_day['Drop_252d']
        if isinstance(drop_252d, pd.Series):
            drop_252d = drop_252d.iloc[0]
    except Exception as e:
        print(f"Warning: Could not extract some metrics: {e}")
        rsi = np.nan
        volatility = np.nan
        drop_5d = np.nan
        drop_10d = np.nan
        drop_20d = np.nan
        drop_50d = np.nan
        drop_252d = np.nan
    
    return {
        "period": period_name,
        "description": description,
        "start_date": start_date,
        "bottom_date": bottom_date,
        "end_date": end_date,
        "start_price": start_price,
        "bottom_price": bottom_price,
        "end_price": end_price,
        "max_decline": max_decline,
        "recovery": recovery,
        "bottom_vix": bottom_vix,
        "max_vix": max_vix,
        "bottom_yield": bottom_yield,
        "min_yield": min_yield,
        "bottom_metrics": {
            "rsi": rsi,
            "volatility": volatility,
            "drop_5d": drop_5d,
            "drop_10d": drop_10d,
            "drop_20d": drop_20d,
            "drop_50d": drop_50d,
            "drop_252d": drop_252d
        }
    }

def plot_comparison(results):
    """Plot comparative analysis of market bottoms"""
    periods = [r["period"] for r in results]
    max_declines = [r["max_decline"] for r in results]
    bottom_vix = [r["bottom_vix"] for r in results]
    bottom_rsi = [r["bottom_metrics"]["rsi"] for r in results]
    drop_20d = [r["bottom_metrics"]["drop_20d"] for r in results]
    
    # Convert to numeric values and handle NaN
    max_declines = [float(x) if not np.isnan(x) else 0 for x in max_declines]
    bottom_vix = [float(x) if not np.isnan(x) else 0 for x in bottom_vix]
    bottom_rsi = [float(x) if not np.isnan(x) else 0 for x in bottom_rsi]
    drop_20d = [float(x) if not np.isnan(x) else 0 for x in drop_20d]
    
    # Create figure with indices instead of strings for x-axis
    x = np.arange(len(periods))
    width = 0.7
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Max decline
    axs[0, 0].bar(x, max_declines, width, color='r')
    axs[0, 0].set_title('Maximum Decline (%)')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(periods, rotation=45)
    
    # VIX at bottom
    axs[0, 1].bar(x, bottom_vix, width, color='orange')
    axs[0, 1].set_title('VIX at Market Bottom')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(periods, rotation=45)
    
    # RSI at bottom
    axs[1, 0].bar(x, bottom_rsi, width, color='g')
    axs[1, 0].set_title('RSI at Market Bottom')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(periods, rotation=45)
    
    # 20-day drop at bottom
    axs[1, 1].bar(x, drop_20d, width, color='b')
    axs[1, 1].set_title('20-day Drop at Market Bottom (%)')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(periods, rotation=45)
    
    plt.tight_layout()
    plt.savefig('market_corrections_comparison.png')
    plt.close()
    
    print("\nComparison chart saved as 'market_corrections_comparison.png'")
    
    # Create a table with the data for easier comparison
    print("\n=== MARKET CORRECTIONS COMPARISON TABLE ===")
    headers = ["Period", "Max Decline", "VIX Level", "RSI", "20-day Drop"]
    print(f"{headers[0]:<20} {headers[1]:<12} {headers[2]:<12} {headers[3]:<8} {headers[4]:<12}")
    print("-" * 65)
    
    for i, period in enumerate(periods):
        print(f"{period:<20} {max_declines[i]:>8.2f}% {bottom_vix[i]:>10.2f} {bottom_rsi[i]:>7.2f} {drop_20d[i]:>10.2f}%")
    
    print("========================================")

def identify_patterns(results):
    """Identify common patterns across market bottoms"""
    # Extract key metrics
    rsi_values = [r["bottom_metrics"]["rsi"] for r in results]
    vix_values = [r["bottom_vix"] for r in results]
    drop_20d_values = [r["bottom_metrics"]["drop_20d"] for r in results]
    drop_10d_values = [r["bottom_metrics"]["drop_10d"] for r in results]
    volatility_values = [r["bottom_metrics"]["volatility"] for r in results]
    
    # Calculate averages
    avg_rsi = np.nanmean(rsi_values)
    avg_vix = np.nanmean(vix_values)
    avg_drop_20d = np.nanmean(drop_20d_values)
    avg_drop_10d = np.nanmean(drop_10d_values)
    avg_volatility = np.nanmean(volatility_values)
    
    print("\n=== COMMON PATTERNS AT MARKET BOTTOMS ===")
    print(f"Average RSI: {avg_rsi:.2f}")
    print(f"Average VIX: {avg_vix:.2f}")
    print(f"Average 10-day Drop: {avg_drop_10d:.2f}%")
    print(f"Average 20-day Drop: {avg_drop_20d:.2f}%")
    print(f"Average Volatility: {avg_volatility:.2f}%")
    
    # Identify optimal thresholds
    optimal_rsi = np.nanpercentile(rsi_values, 75)  # Higher is better for catching bottoms
    optimal_vix = np.nanpercentile(vix_values, 25)  # Lower is better to avoid false signals
    optimal_drop = np.nanpercentile(drop_20d_values, 75)  # Less negative is better
    
    print("\n=== SUGGESTED DETECTION PARAMETERS ===")
    print(f"RSI Threshold: {optimal_rsi:.2f}")
    print(f"VIX Threshold: {optimal_vix:.2f}")
    print(f"20-day Drop Threshold: {optimal_drop:.2f}%")
    print("========================================")
    
    return {
        "avg_rsi": avg_rsi,
        "avg_vix": avg_vix,
        "avg_drop_20d": avg_drop_20d,
        "avg_drop_10d": avg_drop_10d,
        "avg_volatility": avg_volatility,
        "optimal_rsi": optimal_rsi,
        "optimal_vix": optimal_vix,
        "optimal_drop": optimal_drop
    }

def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Analyze market correction periods')
    parser.add_argument('--period', type=str, help='Specific period to analyze')
    args = parser.parse_args()
    
    # Determine which periods to analyze
    periods_to_analyze = {}
    if args.period and args.period in CORRECTION_PERIODS:
        periods_to_analyze = {args.period: CORRECTION_PERIODS[args.period]}
    else:
        periods_to_analyze = CORRECTION_PERIODS
    
    # Find earliest start date and latest end date
    earliest_start = min([pd.to_datetime(period["start"]) for period in periods_to_analyze.values()])
    latest_end = max([pd.to_datetime(period["end"]) for period in periods_to_analyze.values()])
    
    # Add buffer for calculations
    start_date = (earliest_start - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = (latest_end + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Fetch data for all periods
    data = fetch_data(start_date, end_date)
    
    # Analyze each period
    results = []
    for period_name, period_data in periods_to_analyze.items():
        result = analyze_period(period_name, period_data, data)
        results.append(result)
    
    # Plot comparison
    if len(results) > 1:
        plot_comparison(results)
        
        # Identify patterns
        patterns = identify_patterns(results)


def backtest_dip_signals(transactions_csv_path, final_price=None):
    """
    Analyze the profitability of each buy-the-dip signal and their combinations using backtest data.
    Prints summary stats and generates plots for each signal, composite strategies, and for composite (multi-signal) days.
    Normalizes capital allocation for fair comparison (fixed $1000 per trade).
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    print("[DEBUG] Entered backtest_dip_signals. Current working dir:", os.getcwd())

    df = pd.read_csv(transactions_csv_path)
    df = df[df['Type'].notnull()]

    if final_price is None:
        final_price = df['SP500_Close'].iloc[-1]

    def split_signals(sig):
        return sig.split('+') if isinstance(sig, str) else []

    # List of all unique signals
    all_signals = sorted(set(sum(df['Type'].dropna().apply(split_signals), [])))

    # Composite strategies definitions (as boolean masks)
    composite_strategies = {
        'Any 2+ Signals': df['Type'].apply(lambda x: isinstance(x, str) and x.count('+') >= 1),
        'Extreme Dip + Market Capitulation': df['Type'].apply(lambda x: 'Extra_Extreme_Dip' in split_signals(x) and 'Market_Capitulation' in split_signals(x)),
        'Extreme Dip OR Market Capitulation': df['Type'].apply(lambda x: 'Extra_Extreme_Dip' in split_signals(x) or 'Market_Capitulation' in split_signals(x)),
        'Extreme Dip + Vol Spike': df['Type'].apply(lambda x: 'Extra_Extreme_Dip' in split_signals(x) and 'Vol_Spike_RSI_Dip' in split_signals(x)),
        'Extreme Dip + Pullback': df['Type'].apply(lambda x: 'Extra_Extreme_Dip' in split_signals(x) and 'Extra_Pullback' in split_signals(x)),
    }

    # 1. Individual Signal Stats
    signal_stats = {}
    for sig in all_signals:
        mask = df['Type'].apply(lambda x: sig in split_signals(x))
        invested = df.loc[mask, 'Contribution'].sum()
        shares = df.loc[mask, 'Shares_Bought'].sum()
        value = shares * final_price
        profit = value - invested
        n_trades = mask.sum()
        trade_returns = (df.loc[mask, 'SP500_Close'] / df.loc[mask, 'SP500_Close'].iloc[0]).values if n_trades > 0 else []
        avg_return = np.mean(trade_returns) - 1 if n_trades > 0 else np.nan
        win_rate = np.mean(df.loc[mask, 'SP500_Close'] < final_price) if n_trades > 0 else np.nan
        if n_trades > 0:
            dates = pd.to_datetime(df.loc[mask].index if df.loc[mask].index.dtype == 'datetime64[ns]' else df.loc[mask].reset_index()['index'])
            if len(dates) > 0:
                years = (dates.max() - dates.min()).days / 365.25 if (dates.max() - dates.min()).days > 0 else 1
            else:
                years = 1
            cagr = (value / invested) ** (1/years) - 1 if invested > 0 else np.nan
        else:
            cagr = np.nan
        signal_stats[sig] = {
            'invested': invested,
            'shares': shares,
            'value': value,
            'profit': profit,
            'profit_per_dollar': profit / invested if invested > 0 else np.nan,
            'n_trades': n_trades,
            'avg_return': avg_return,
            'win_rate': win_rate,
            'cagr': cagr,
        }

    # 2. Composite/Composite Signal Stats (including multi-signal days)
    for name, mask in composite_strategies.items():
        invested = df.loc[mask, 'Contribution'].sum()
        shares = df.loc[mask, 'Shares_Bought'].sum()
        value = shares * final_price
        profit = value - invested
        n_trades = mask.sum()
        trade_returns = (df.loc[mask, 'SP500_Close'] / df.loc[mask, 'SP500_Close'].iloc[0]).values if n_trades > 0 else []
        avg_return = np.mean(trade_returns) - 1 if n_trades > 0 else np.nan
        win_rate = np.mean(df.loc[mask, 'SP500_Close'] < final_price) if n_trades > 0 else np.nan
        if n_trades > 0:
            dates = pd.to_datetime(df.loc[mask].index if df.loc[mask].index.dtype == 'datetime64[ns]' else df.loc[mask].reset_index()['index'])
            if len(dates) > 0:
                years = (dates.max() - dates.min()).days / 365.25 if (dates.max() - dates.min()).days > 0 else 1
            else:
                years = 1
            cagr = (value / invested) ** (1/years) - 1 if invested > 0 else np.nan
        else:
            cagr = np.nan
        signal_stats[name] = {
            'invested': invested,
            'shares': shares,
            'value': value,
            'profit': profit,
            'profit_per_dollar': profit / invested if invested > 0 else np.nan,
            'n_trades': n_trades,
            'avg_return': avg_return,
            'win_rate': win_rate,
            'cagr': cagr,
        }

    # 3. Normalized (fixed $1000 per trade) cumulative value for each signal/strategy
    norm_invest = 1000
    norm_cum_values = {}
    for sig in list(all_signals) + list(composite_strategies.keys()):
        if sig in composite_strategies:
            mask = composite_strategies[sig]
        else:
            mask = df['Type'].apply(lambda x: sig in split_signals(x))
        idx = df.index[mask]
        norm_shares = norm_invest / df.loc[mask, 'SP500_Close']
        cum_shares = norm_shares.cumsum()
        cum_value = cum_shares * final_price
        norm_cum_values[sig] = (idx, cum_value)

    # 4. Print Summary Table
    print("\n=== Buy-the-Dip Signal Profitability ===")
    print("NOTE: Total profit is not a fair comparison due to different invested capital. Use normalized metrics (profit/$, CAGR, win rate) for signal ranking.\n")
    print(f"{'Signal':<35}{'Trades':>8}{'Invested':>14}{'Profit':>16}{'Profit/$':>12}{'CAGR':>10}{'WinRate':>10}{'AvgRet':>10}")
    print("-"*120)
    for sig, stats in sorted(signal_stats.items(), key=lambda x: -x[1]['profit_per_dollar']):
        print(f"{sig:<35}{stats['n_trades']:>8}{stats['invested']:>14.2f}{stats['profit']:>16.2f}{stats['profit_per_dollar']:>12.2f}{stats['cagr']:>10.2%}{stats['win_rate']:>10.2%}{stats['avg_return']:>10.2%}")
    print("-"*120)

    # 5. Visualization
    try:
        # Bar plot: profit per $ invested
        labels = list(signal_stats.keys())
        profits_per_dollar = [signal_stats[s]['profit_per_dollar'] for s in labels]
        plt.figure(figsize=(12,6))
        plt.bar(labels, profits_per_dollar, color='purple')
        plt.ylabel('Profit per $ Invested')
        plt.title('Buy-the-Dip Signal Power Ranking')
        plt.xticks(rotation=45, ha='right')
        # Draw horizontal line at Fixed_DCA profit per dollar
        if 'Fixed_DCA' in signal_stats:
            fixed_dca_ppd = signal_stats['Fixed_DCA']['profit_per_dollar']
            plt.axhline(fixed_dca_ppd, color='red', linestyle='--', linewidth=2, label='Fixed DCA Baseline')
            plt.legend()
        plt.tight_layout()
        plt.savefig('signal_profit_per_dollar.png')
        plt.close()
        print("\nBar chart saved as 'signal_profit_per_dollar.png'")
    except Exception as e:
        print(f"Plotting failed: {e}")

    # 6. Normalized Cumulative Value Plot
    try:
        plt.figure(figsize=(14,8))
        for sig in sorted(norm_cum_values, key=lambda x: -signal_stats[x]['profit_per_dollar'])[:6]:
            idx, cum_value = norm_cum_values[sig]
            plt.plot(idx, cum_value, label=f'{sig}')
        plt.legend()
        plt.title('Cumulative Value (Normalized $1000/trade): Top Buy-the-Dip Strategies')
        plt.xlabel('Trade #')
        plt.ylabel('Portfolio Value ($)')
        plt.tight_layout()
        plt.savefig('signal_cumulative_value_normalized.png')
        plt.close()
        print("Cumulative value chart (normalized) saved as 'signal_cumulative_value_normalized.png'")
    except Exception as e:
        print(f"Cumulative plot failed: {e}")

    print("[DEBUG] Attempting to save signal_stats_summary.csv...")
    stats_df = pd.DataFrame(signal_stats).T
    stats_df.index.name = 'Signal'
    stats_df.reset_index(inplace=True)
    csv_path = os.path.abspath('signal_stats_summary.csv')
    stats_df.to_csv(csv_path, index=False)
    print(f"[DEBUG] Signal stats saved to {csv_path} in directory: {os.getcwd()}")
    # 8. Return stats for further programmatic use
    return {
        'signal_stats': signal_stats,
        'norm_cum_values': norm_cum_values,
    }

if __name__ == "__main__":
    main()
    # Run the buy-the-dip signal backtest and analysis
    print("\n\n=== Running Buy-the-Dip Signal Backtest ===")
    backtest_dip_signals("enhanced_dca_transactions.csv", final_price=298915.7558)
