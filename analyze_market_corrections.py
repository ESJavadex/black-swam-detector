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

if __name__ == "__main__":
    main()
