#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COVID-19 Market Crash Analysis
This script analyzes the market data during the COVID-19 crash (Feb-Apr 2020)
to identify patterns for better black swan detection.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Date range for analysis
START_DATE = '2020-02-01'
END_DATE = '2020-04-13'

# Key dates to highlight
BOTTOM_DATE = '2020-03-23'  # Market bottom

def fetch_data(ticker, start_date, end_date):
    """Fetch historical data for a given ticker."""
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data fetched for {ticker}.")
    print(f"Fetched {len(df)} records for {ticker}.")
    return df

def calculate_indicators(df_spy, df_vix, df_tnx):
    """Calculate all relevant indicators for analysis."""
    # Combine data
    df = pd.DataFrame()
    df['SPY_Close'] = df_spy['Close']
    df['VIX_Close'] = df_vix['Close']
    df['TNX_Close'] = df_tnx['Close']
    
    # Calculate rolling max values
    df['SPY_5d_Max'] = df['SPY_Close'].rolling(window=5).max()
    df['SPY_10d_Max'] = df['SPY_Close'].rolling(window=10).max()
    df['SPY_20d_Max'] = df['SPY_Close'].rolling(window=20).max()
    df['SPY_50d_Max'] = df['SPY_Close'].rolling(window=50).max()
    df['SPY_252d_Max'] = df['SPY_Close'].rolling(window=252).max()
    
    # Calculate percentage drops from recent highs
    for window in [5, 10, 20, 50, 252]:
        df[f'SPY_{window}d_Drop'] = ((df['SPY_Close'] - df[f'SPY_{window}d_Max']) / df[f'SPY_{window}d_Max']) * 100
    
    # Calculate RSI (14-day)
    delta = df['SPY_Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate VIX changes
    df['VIX_1d_Change'] = df['VIX_Close'].pct_change() * 100
    df['VIX_5d_Change'] = df['VIX_Close'].pct_change(5) * 100
    
    # Calculate Bond Yield changes
    df['TNX_1d_Change'] = df['TNX_Close'].pct_change() * 100
    df['TNX_5d_Change'] = df['TNX_Close'].pct_change(5) * 100
    
    # Calculate volume indicators if available
    if 'Volume' in df_spy.columns:
        df['SPY_Volume'] = df_spy['Volume']
        df['SPY_Volume_20d_Avg'] = df['SPY_Volume'].rolling(window=20).mean()
        df['SPY_Volume_Ratio'] = df['SPY_Volume'] / df['SPY_Volume_20d_Avg']
    
    # Calculate moving averages
    df['SPY_SMA_50'] = df['SPY_Close'].rolling(window=50).mean()
    df['SPY_SMA_200'] = df['SPY_Close'].rolling(window=200).mean()
    
    # Calculate volatility
    df['SPY_Volatility_20d'] = df['SPY_Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
    
    return df

def analyze_specific_dates(df, dates):
    """Analyze specific dates in detail."""
    if isinstance(dates, str):
        dates = [dates]
    
    for date in dates:
        if date in df.index:
            print(f"\n=== DETAILED ANALYSIS FOR {date} ===")
            row = df.loc[date]
            
            print(f"SPY Price: ${row['SPY_Close']:.2f}")
            print(f"VIX Level: {row['VIX_Close']:.2f}")
            print(f"10Y Treasury Yield: {row['TNX_Close']:.2f}%")
            print(f"RSI (14-day): {row['RSI']:.2f}")
            
            print("\nPrice Drops from Recent Highs:")
            for window in [5, 10, 20, 50, 252]:
                print(f"  {window}-day Drop: {row[f'SPY_{window}d_Drop']:.2f}%")
            
            print("\nVIX Changes:")
            print(f"  1-day Change: {row['VIX_1d_Change']:.2f}%")
            print(f"  5-day Change: {row['VIX_5d_Change']:.2f}%")
            
            print("\nBond Yield Changes:")
            print(f"  1-day Change: {row['TNX_1d_Change']:.2f}%")
            print(f"  5-day Change: {row['TNX_5d_Change']:.2f}%")
            
            if 'SPY_Volume_Ratio' in row:
                print(f"\nVolume Ratio (vs 20-day Avg): {row['SPY_Volume_Ratio']:.2f}x")
            
            print(f"Volatility (20-day): {row['SPY_Volatility_20d']:.2f}%")
            
            # Check if it's below moving averages
            below_50d = row['SPY_Close'] < row['SPY_SMA_50']
            below_200d = row['SPY_Close'] < row['SPY_SMA_200']
            print(f"\nBelow 50-day MA: {'Yes' if below_50d else 'No'}")
            print(f"Below 200-day MA: {'Yes' if below_200d else 'No'}")
            
            print("=" * 40)
        else:
            print(f"Date {date} not found in the data.")

def plot_crash_period(df):
    """Create plots for the crash period."""
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Plot 1: SPY Price with highlighted bottom
    axs[0].plot(df.index, df['SPY_Close'], label='SPY Price')
    axs[0].scatter(BOTTOM_DATE, df.loc[BOTTOM_DATE, 'SPY_Close'], 
                  color='red', s=100, zorder=5, label='Market Bottom')
    axs[0].set_title('SPY Price during COVID-19 Crash')
    axs[0].set_ylabel('Price ($)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: VIX
    axs[1].plot(df.index, df['VIX_Close'], color='orange', label='VIX')
    axs[1].axhline(y=40, color='r', linestyle='--', label='VIX Threshold (40)')
    axs[1].set_title('VIX (Volatility Index)')
    axs[1].set_ylabel('VIX')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot 3: RSI
    axs[2].plot(df.index, df['RSI'], color='purple', label='RSI')
    axs[2].axhline(y=30, color='r', linestyle='--', label='Oversold Threshold (30)')
    axs[2].axhline(y=70, color='g', linestyle='--', label='Overbought Threshold (70)')
    axs[2].set_title('RSI (Relative Strength Index)')
    axs[2].set_ylabel('RSI')
    axs[2].set_ylim(0, 100)
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot 4: Price Drops from Recent Highs
    for window, color in zip([20, 50, 252], ['blue', 'green', 'red']):
        axs[3].plot(df.index, df[f'SPY_{window}d_Drop'], 
                   label=f'{window}-day Drop', color=color)
    axs[3].axhline(y=-20, color='r', linestyle='--', label='20% Drop Threshold')
    axs[3].axhline(y=-30, color='darkred', linestyle='--', label='30% Drop Threshold')
    axs[3].set_title('Price Drops from Recent Highs')
    axs[3].set_ylabel('Drop (%)')
    axs[3].set_xlabel('Date')
    axs[3].legend()
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('covid_crash_analysis.png')
    print("Plot saved as 'covid_crash_analysis.png'")

def find_optimal_parameters(df):
    """Find optimal parameters for detecting the market bottom."""
    # Define date range around the bottom
    bottom_date = pd.to_datetime(BOTTOM_DATE)
    date_range = pd.date_range(start='2020-03-20', end='2020-03-26')
    
    # Parameters to test
    drop_thresholds = [-15, -20, -25, -30]
    vix_thresholds = [30, 35, 40, 45, 50]
    rsi_thresholds = [20, 25, 30, 35]
    
    results = []
    
    # Test all combinations
    for drop_threshold in drop_thresholds:
        for vix_threshold in vix_thresholds:
            for rsi_threshold in rsi_thresholds:
                # Create conditions
                condition_drop = df['SPY_20d_Drop'] < drop_threshold
                condition_vix = df['VIX_Close'] > vix_threshold
                condition_rsi = df['RSI'] < rsi_threshold
                
                # Combine conditions (at least 2 of 3)
                conditions_met = condition_drop.astype(int) + condition_vix.astype(int) + condition_rsi.astype(int)
                combined_condition = conditions_met >= 2
                
                # Check if bottom date is detected
                bottom_detected = combined_condition.loc[BOTTOM_DATE]
                
                # Count total signals in the period
                total_signals = combined_condition.sum()
                
                # Add to results
                results.append({
                    'Drop_Threshold': drop_threshold,
                    'VIX_Threshold': vix_threshold,
                    'RSI_Threshold': rsi_threshold,
                    'Bottom_Detected': bottom_detected,
                    'Total_Signals': total_signals
                })
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Filter for combinations that detect the bottom
    valid_combinations = results_df[results_df['Bottom_Detected']]
    
    # Sort by total signals (fewer is better for specificity)
    if not valid_combinations.empty:
        best_params = valid_combinations.sort_values('Total_Signals').iloc[0]
        print("\n=== OPTIMAL PARAMETERS ===")
        print(f"Drop Threshold: {best_params['Drop_Threshold']}%")
        print(f"VIX Threshold: {best_params['VIX_Threshold']}")
        print(f"RSI Threshold: {best_params['RSI_Threshold']}")
        print(f"Total Signals in Period: {best_params['Total_Signals']}")
        print("=" * 40)
        return best_params
    else:
        print("No parameter combination detected the market bottom.")
        return None

def main():
    # Fetch data
    df_spy = fetch_data('SPY', START_DATE, END_DATE)
    df_vix = fetch_data('^VIX', START_DATE, END_DATE)
    df_tnx = fetch_data('^TNX', START_DATE, END_DATE)
    
    # Calculate indicators
    df = calculate_indicators(df_spy, df_vix, df_tnx)
    
    # Analyze the entire period
    print("\n=== OVERVIEW OF COVID-19 CRASH PERIOD ===")
    print(f"Analysis Period: {START_DATE} to {END_DATE}")
    print(f"SPY Start Price: ${df['SPY_Close'].iloc[0]:.2f}")
    print(f"SPY End Price: ${df['SPY_Close'].iloc[-1]:.2f}")
    print(f"SPY Bottom Price: ${df.loc[BOTTOM_DATE, 'SPY_Close']:.2f}")
    print(f"Total Decline: {((df['SPY_Close'].iloc[-1] / df['SPY_Close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"Max Decline: {((df.loc[BOTTOM_DATE, 'SPY_Close'] / df['SPY_Close'].iloc[0]) - 1) * 100:.2f}%")
    
    # Analyze specific dates
    key_dates = ['2020-02-19', '2020-03-16', '2020-03-23', '2020-03-24', '2020-04-13']
    analyze_specific_dates(df, key_dates)
    
    # Find optimal parameters
    best_params = find_optimal_parameters(df)
    
    # Plot the crash period
    plot_crash_period(df)
    
    # Export data to CSV for further analysis
    df.to_csv('covid_crash_data.csv')
    print("Data exported to 'covid_crash_data.csv'")

if __name__ == "__main__":
    main()
