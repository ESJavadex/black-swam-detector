# -*- coding: utf-8 -*-
"""
Enhanced Dollar-Cost Averaging (DCA) Strategy for S&P 500 (SPY)
with Dip Buying Opportunities based on Technical Indicators.

Version: v12 (Fix performance metrics ambiguity, fix plot Y-axis range)

Disclaimer:
This script is for educational and informational purposes ONLY. It is NOT financial advice.
Trading and investing in financial markets involve substantial risk of loss.
Past performance is not indicative of future results.
The strategies implemented here are based on historical data and technical indicators,
which may not accurately predict future market movements.
Any decisions based on this script are at your own risk.
You should consult with a qualified financial advisor before making any investment decisions.
The use of a demo account is strongly recommended for testing any trading strategy.
The authors and providers of this script are not liable for any losses incurred.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import argparse
from datetime import datetime
import sys
import traceback
from typing import List, Dict, Optional, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ----------------------------
# Configuration
# ----------------------------
CONFIG = {
    'fixed_dca': True,          # Enable regular monthly DCA
    'extra_normal': True,       # Enable extra buys on dips during uptrends
    'extra_purple_alert': True, # Enable extra buys on 'Black Swan' signals
    'extra_pullback': True,     # Enable extra buys on pullback signals
    'extra_extreme_dip': True,  # Enable extra buys on extreme RSI dips
    'buy_and_hold': False,      # Enable Buy and Hold strategy for comparison
}
FIXED_INVESTMENT = 475          # Amount for regular monthly DCA
EXTRA_NORMAL_INVESTMENT = 100   # Amount for 'Extra Normal' buys
EXTRA_PURPLE_INVESTMENT = 200   # Amount for 'Extra Purple Alert' buys
EXTRA_PULLBACK_INVESTMENT = 150 # Amount for 'Extra Pullback' buys
EXTRA_EXTREME_DIP_INVESTMENT = 250 # Amount for 'Extra Extreme Dip' buys
BUY_AND_HOLD_INVESTMENT = 5000 # Initial lump sum for Buy and Hold strategy

PREDICTION_WINDOW = 20          # Window for rolling volatility/return calculations
BLACK_SWAN_STD_MULTIPLIER = 3   # Std deviation multiplier for Black Swan detection
PULLBACK_THRESHOLD = -5         # Percentage drop from recent high for Pullback detection (%)
PULLBACK_WINDOW = 10            # Lookback window (days) for Pullback high

RSI_WINDOW = 14                 # Window for RSI calculation
RSI_OVERBOUGHT = 70             # RSI level considered overbought
RSI_OVERSOLD = 30               # RSI level considered oversold
EXTREME_RSI_OVERSOLD = 25       # RSI level for 'Extreme Dip' detection

MACD_FAST = 12                  # Fast EMA window for MACD
MACD_SLOW = 26                  # Slow EMA window for MACD
MACD_SIGNAL = 9                 # Signal line EMA window for MACD

MAX_PURPLE_ALERTS_PER_YEAR = 2  # Max times 'Extra Purple' can trigger per calendar year
MAX_PULLBACKS_PER_YEAR = 3      # Max times 'Extra Pullback' can trigger per calendar year
MAX_EXTREME_DIPS_PER_YEAR = 2   # Max times 'Extra Extreme Dip' can trigger per calendar year

TICKER_SP500 = 'SPY'            # Ticker for S&P 500 ETF
TICKER_VIX = '^VIX'             # Ticker for VIX Index
TICKER_BOND_YIELD = '^TNX'      # Ticker for 10-Year Treasury Yield

DATA_START_DATE = '2015-01-01'  # Start date for historical data fetching
FIXED_INVESTMENT_DAY = 1        # Day of the month for regular fixed DCA
EXTRA_INVESTMENT_DAY = 15       # Day of the month to check for 'Extra Normal' DCA opportunity

# ----------------------------
# Argument Parsing
# ----------------------------
def parse_arguments() -> pd.Timestamp:
    """ Parse command-line arguments for the simulation end date. """
    parser = argparse.ArgumentParser(description='Enhanced DCA Strategy Simulation')
    parser.add_argument('--date', type=str, help='Simulation end date in DD-MM-YYYY or YYYY-MM-DD format. Defaults to today.')
    args = parser.parse_args()
    simulation_date = None
    if args.date:
        try:
            simulation_date = pd.to_datetime(args.date, format='%d-%m-%Y')
            print(f"Parsed simulation date as DD-MM-YYYY: {simulation_date.date()}")
        except ValueError:
            try:
                simulation_date = pd.to_datetime(args.date, format='%Y-%m-%d')
                print(f"Parsed simulation date as YYYY-MM-DD: {simulation_date.date()}")
            except ValueError:
                print("Error: Incorrect date format. Use DD-MM-YYYY or YYYY-MM-DD.")
                sys.exit(1)
    else:
        # Use current real-world date if no date is provided
        simulation_date = pd.to_datetime(datetime.now()) # Use current date and time
        print(f"No date provided. Using current date: {simulation_date.date()}")

    simulation_date = simulation_date.normalize() # Set time to 00:00:00

    # If the simulation date falls on a weekend, adjust to the previous Friday
    if simulation_date.weekday() >= 5: # 5 = Saturday, 6 = Sunday
        print(f"\n{simulation_date.date()} is weekend. Adjusting sim date to previous Friday.")
        simulation_date = simulation_date - pd.Timedelta(days=(simulation_date.weekday() - 4))
        print(f"Adjusted simulation date to {simulation_date.date()}.")

    return simulation_date

# ----------------------------
# 1. Fetch Historical Data
# ----------------------------
def fetch_data(ticker: str, start_date: str, end_date: pd.Timestamp) -> pd.DataFrame:
    """ Fetch historical data for a given ticker. """
    print(f"Fetching data for {ticker} from {start_date} to {end_date.date()}...")
    try:
        start_dt = pd.to_datetime(start_date)
        # Fetch data up to the day *after* the end_date to ensure the end_date itself is included
        df = yf.download(ticker, start=start_dt, end=end_date + pd.Timedelta(days=1), progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data fetched for {ticker}.")
        print(f"Fetched {len(df)} records for {ticker}.")
        # Remove timezone information if present
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        raise

# ----------------------------
# 2. Data Preprocessing
# ----------------------------
def prepare_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """ Prepare fetched data: resample daily, forward fill, select relevant column. """
    print(f"Preparing DataFrame for {name}...")
    try:
        df_processed = df.copy()
        # Resample to daily frequency, take the last value of the day, and forward fill missing days (weekends/holidays)
        df_processed = df_processed.resample('D').last().ffill()
        if 'Close' in df_processed.columns:
            df_processed = df_processed[['Close']].rename(columns={'Close': f'{name}_Close'})
        else:
            raise KeyError(f"Could not find 'Close' column for {name}.")
        print(f"DataFrame for {name} prepared. Index range: {df_processed.index.min().date()} to {df_processed.index.max().date()}")
        return df_processed
    except Exception as e:
        print(f"Error preparing DataFrame for {name}: {e}")
        raise

# ----------------------------
# 3. Feature Engineering
# ----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """ Calculate technical indicators and features. """
    print("Starting feature engineering...")
    if df.empty:
        print("Warning: Input DataFrame for feature engineering is empty.")
        return df
    try:
        df_eng = df.copy()
        df_eng.index.name = None # Remove index name if exists

        # Basic Returns and Changes
        df_eng['SP500_Return'] = df_eng['SP500_Close'].pct_change()
        df_eng['VIX_Change'] = df_eng['VIX_Close'].pct_change()
        df_eng['Bond_Yield_Change'] = df_eng['10Y_Bond_Yield_Close'].pct_change()

        # Moving Averages (SP500)
        df_eng['SP500_SMA_50'] = df_eng['SP500_Close'].rolling(window=50).mean()
        df_eng['SP500_SMA_200'] = df_eng['SP500_Close'].rolling(window=200).mean()

        # Volatility (SP500)
        df_eng['SP500_Volatility'] = df_eng['SP500_Return'].rolling(window=PREDICTION_WINDOW).std() * np.sqrt(252) # Annualized

        # RSI (Relative Strength Index - SP500)
        delta = df_eng['SP500_Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
        avg_loss = loss.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
        rs = avg_gain / avg_loss
        df_eng['RSI'] = 100.0 - (100.0 / (1.0 + rs))
        df_eng['RSI'] = df_eng['RSI'].fillna(50) # Fill initial NaNs with neutral 50

        # MACD (Moving Average Convergence Divergence - SP500)
        exp1 = df_eng['SP500_Close'].ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = df_eng['SP500_Close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df_eng['MACD'] = exp1 - exp2
        df_eng['MACD_Signal'] = df_eng['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df_eng['MACD_Hist'] = df_eng['MACD'] - df_eng['MACD_Signal'] # MACD Histogram

        # Drop rows with NaN values created by rolling calculations
        initial_rows = len(df_eng)
        df_eng.dropna(inplace=True)
        final_rows = len(df_eng)
        print(f"Feature engineering completed. Dropped {initial_rows - final_rows} rows due to NaN values.")
        print(f"Data range after feature engineering: {df_eng.index.min().date()} to {df_eng.index.max().date()}")
        return df_eng
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        traceback.print_exc()
        raise

# ----------------------------
# 4. Define Investment Dates & Signals
# ----------------------------
def get_investment_dates(df_index: pd.DatetimeIndex, day_of_month: int, simulation_date: pd.Timestamp) -> List[pd.Timestamp]:
    """
    Get investment dates for a specific day of the month.
    If the target day is not a trading day, it finds the *next available* trading day
    within the DataFrame's index that falls on or before the simulation_date.
    """
    print(f"Generating investment dates for day {day_of_month} of each month up to {simulation_date.date()}...")
    investment_dates = []
    df_index = df_index.sort_values() # Ensure index is sorted

    # Get unique months present in the index up to the simulation date
    unique_months = df_index[df_index <= simulation_date].to_period('M').unique()

    for period in unique_months:
        month_start = period.to_timestamp()
        # Construct the target date for the given day in the current month/year
        try:
            # Use the exact day if possible
            target_date_naive = f"{period.year}-{period.month:02d}-{day_of_month:02d}"
            target_date = pd.to_datetime(target_date_naive)
        except ValueError:
            # Handle cases like Feb 30th - use the end of the month instead
            target_date = period.end_time

        target_date = target_date.normalize() # Set time to 00:00:00

        # Skip if the calculated target date is beyond the simulation end date
        if target_date > simulation_date:
            continue

        # Find the first index date that is >= target_date
        # Use searchsorted for efficiency on the sorted index
        insertion_point = df_index.searchsorted(target_date, side='left')

        # Check if a valid insertion point was found within the index bounds
        if insertion_point < len(df_index):
            actual_investment_date = df_index[insertion_point]

            # Ensure this date is not after the simulation end date
            if actual_investment_date <= simulation_date:
                # Check if the found date is in the correct target month, OR
                # if it's the first trading day of the *next* month because the target day
                # and all subsequent days in the target month were non-trading days.
                is_correct_month = actual_investment_date.month == target_date.month
                # Check if the previous date in the index belonged to the target month,
                # and the current date is in the next month (handling year rollover)
                is_next_month_start = (
                    insertion_point > 0 and
                    df_index[insertion_point - 1].month == target_date.month and
                    (actual_investment_date.month == target_date.month + 1 or
                     (actual_investment_date.month == 1 and target_date.month == 12))
                )

                if is_correct_month or is_next_month_start:
                     # Add the date only if it's not a duplicate of the last added date
                     if not investment_dates or actual_investment_date != investment_dates[-1]:
                         investment_dates.append(actual_investment_date)

    investment_dates = sorted(list(set(investment_dates))) # Ensure uniqueness and sort
    print(f"Generated {len(investment_dates)} investment dates.")
    return investment_dates


def apply_frequency_limit(dates: List[pd.Timestamp], max_per_year: int) -> List[pd.Timestamp]:
    """Limits the number of trigger dates per calendar year."""
    if not dates:
        return []
    dates_df = pd.DataFrame({'Date': pd.to_datetime(dates)})
    dates_df['Year'] = dates_df['Date'].dt.year
    # Sort by Year, then Date to ensure we take the earliest dates within each year
    dates_df = dates_df.sort_values(by=['Year', 'Date'])
    # Group by year and take the first 'max_per_year' dates
    limited_dates = dates_df.groupby('Year').head(max_per_year)
    return limited_dates['Date'].tolist()

# ----------------------------
# 5. Detect Dip Opportunities
# ----------------------------
def detect_black_swans(df: pd.DataFrame) -> List[pd.Timestamp]:
    """ Detect Black Swan events: extreme negative returns + oversold RSI. """
    print("Detecting Black Swan opportunities...")
    try:
        rolling_mean = df['SP500_Return'].rolling(window=PREDICTION_WINDOW).mean()
        rolling_std = df['SP500_Return'].rolling(window=PREDICTION_WINDOW).std()
        # Define threshold as mean return minus X standard deviations
        threshold = rolling_mean - (BLACK_SWAN_STD_MULTIPLIER * rolling_std)

        # Condition 1: Daily return is below the dynamic threshold
        condition_return = df['SP500_Return'] < threshold
        # Condition 2: RSI is below the oversold level
        condition_rsi = df['RSI'] < RSI_OVERSOLD

        # Combine conditions and get the dates
        black_swan_signals = df.loc[condition_return & condition_rsi].index.tolist()
        print(f"Detected {len(black_swan_signals)} potential Black Swan dates (before limit).")
        return black_swan_signals
    except Exception as e:
        print(f"Error detecting Black Swans: {e}")
        traceback.print_exc()
        return []

def detect_pullbacks(df: pd.DataFrame) -> List[pd.Timestamp]:
    """ Detect Pullbacks: significant decline from recent high + oversold RSI. """
    print("Detecting Pullback opportunities...")
    try:
        df_temp = df.copy()
        df_temp.index.name = None # Ensure index has no name

        required_cols = ['SP500_Close', 'RSI']
        if not all(col in df_temp.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df_temp.columns]
            print(f"Error: Missing required columns in detect_pullbacks: {missing}")
            return []

        # Calculate the rolling maximum closing price over the pullback window
        rolling_max = df_temp['SP500_Close'].rolling(window=PULLBACK_WINDOW, min_periods=1).max()

        # Calculate the drawdown percentage from the rolling max
        drawdown = ((df_temp['SP500_Close'] - rolling_max) / rolling_max) * 100

        # Debug info: Check types and shapes if issues arise
        # print(f"drawdown type: {type(drawdown)}, shape: {drawdown.shape if hasattr(drawdown, 'shape') else 'N/A'}")

        # If drawdown ended up as a DataFrame (shouldn't normally happen), extract the Series
        if isinstance(drawdown, pd.DataFrame):
            print("Warning: Converting drawdown from DataFrame to Series")
            drawdown = drawdown.iloc[:, 0] # Assume the first column is the correct one

        # Condition 1: Drawdown is below the defined threshold (e.g., -5%)
        condition_drawdown = drawdown <= PULLBACK_THRESHOLD
        # Condition 2: RSI is below the oversold level
        condition_rsi = df_temp['RSI'] < RSI_OVERSOLD

        # Ensure conditions are boolean Series with the same index as df_temp
        condition_drawdown = pd.Series(condition_drawdown, index=df_temp.index)
        condition_rsi = pd.Series(condition_rsi, index=df_temp.index)

        # Combine conditions using logical AND, fill any NaNs with False
        combined_condition = (condition_drawdown & condition_rsi).fillna(False)

        # Debug info: Check combined condition
        # print(f"combined_condition type: {type(combined_condition)}, shape: {combined_condition.shape}")
        # if isinstance(combined_condition, pd.Series):
            # print(f"combined_condition True count: {combined_condition.sum()}")

        # Get the index dates where both conditions are met
        pullback_dates = df_temp.index[combined_condition].tolist()
        print(f"Detected {len(pullback_dates)} potential Pullback dates (before limit).")
        return pullback_dates
    except Exception as e:
        print(f"Error detecting Pullbacks: {e}")
        traceback.print_exc()
        return []


def detect_extreme_dips(df: pd.DataFrame) -> List[pd.Timestamp]:
    """ Detect Extreme Dips: RSI below the extreme oversold threshold. """
    print("Detecting Extreme Dip opportunities...")
    try:
        # Condition: RSI is below the extreme oversold level
        condition_rsi_extreme = df['RSI'] < EXTREME_RSI_OVERSOLD
        # Get the dates where the condition is met
        extreme_dip_signals = df.loc[condition_rsi_extreme].index.tolist()
        print(f"Detected {len(extreme_dip_signals)} potential Extreme Dip dates (before limit).")
        return extreme_dip_signals
    except Exception as e:
        print(f"Error detecting Extreme Dips: {e}")
        traceback.print_exc()
        return []

def detect_extra_normal_signals(df: pd.DataFrame, extra_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
    """ Detect Extra Normal opportunities: uptrend + oversold RSI on specific check dates. """
    print("Detecting Extra Normal DCA opportunities...")
    try:
        # Filter the DataFrame to include only the potential extra investment dates
        valid_extra_dates = [d for d in extra_dates if d in df.index]
        if not valid_extra_dates:
            print("No valid dates provided for Extra Normal check within the data range.")
            return []

        relevant_df = df.loc[valid_extra_dates].copy()

        # Condition 1: Market is potentially in an uptrend (SMA50 > SMA200)
        condition_uptrend = relevant_df['SP500_SMA_50'] > relevant_df['SP500_SMA_200']
        # Condition 2: RSI is below the oversold level on these specific dates
        condition_rsi = relevant_df['RSI'] < RSI_OVERSOLD

        # Combine conditions and get the dates
        extra_normal_signals = relevant_df.loc[condition_uptrend & condition_rsi].index.tolist()
        print(f"Detected {len(extra_normal_signals)} Extra Normal DCA dates.")
        return extra_normal_signals
    except KeyError as e:
        print(f"KeyError detecting Extra Normal signals: Missing column - {e}. Ensure SMA/RSI calculated.")
        return []
    except Exception as e:
        print(f"Error detecting Extra Normal signals: {e}")
        traceback.print_exc()
        return []

# ----------------------------
# 6. Define and Combine Investments
# ----------------------------
def create_investment_df(dates: List[pd.Timestamp], investment_type: str, amount: float, df_data: pd.DataFrame) -> pd.DataFrame:
    """ Helper function to create a DataFrame for a specific investment type. """
    if not dates:
        # print(f"No dates provided for {investment_type}.")
        return pd.DataFrame()

    # Filter dates to ensure they exist in the main data index
    valid_dates = [d for d in dates if d in df_data.index]
    if not valid_dates:
        # print(f"No valid dates found in data index for {investment_type}.")
        return pd.DataFrame()

    # Create DataFrame with investment dates as index
    investments = pd.DataFrame(index=pd.DatetimeIndex(valid_dates))
    investments['Type'] = investment_type
    investments['Contribution'] = amount

    # Look up the closing price on the investment date
    try:
        investments['SP500_Close'] = df_data.loc[investments.index, 'SP500_Close']
    except KeyError:
        print(f"Error: Price lookup failed for {investment_type} on some dates. Ensure 'SP500_Close' exists.")
        # Return empty if price lookup fails critically, or handle missing prices
        # For simplicity, we'll drop rows where price lookup might have failed if structure changed
        investments['SP500_Close'] = np.nan # Set to NaN initially
        investments['SP500_Close'] = investments['SP500_Close'].fillna(df_data.loc[investments.index, 'SP500_Close'])
        if investments['SP500_Close'].isnull().any():
             print(f"Warning: Could not find prices for all {investment_type} dates. Dropping investments without prices.")
             investments.dropna(subset=['SP500_Close'], inplace=True)
             if investments.empty: return pd.DataFrame()
        # return pd.DataFrame() # Or safer to return empty

    # Calculate shares bought, handle potential zero or NaN prices
    # Avoid division by zero or NaN
    valid_price_mask = investments['SP500_Close'].notnull() & (investments['SP500_Close'] > 0)
    investments['Shares_Bought'] = np.nan # Initialize column
    investments.loc[valid_price_mask, 'Shares_Bought'] = investments.loc[valid_price_mask, 'Contribution'] / investments.loc[valid_price_mask, 'SP500_Close']

    # Handle cases where shares couldn't be calculated
    if not valid_price_mask.all():
        print(f"Warning: Invalid prices found for {investment_type} on dates: {investments.index[~valid_price_mask].tolist()}. These investments are excluded.")
        investments = investments[valid_price_mask] # Keep only valid investments

    if investments['Shares_Bought'].isnull().any():
        print(f"Warning: NaN Shares_Bought calculated for {investment_type}. Dropping these entries.")
        investments.dropna(subset=['Shares_Bought'], inplace=True)

    investments.index.name = None # Remove index name
    return investments

def define_all_investments(df_data: pd.DataFrame, simulation_date: pd.Timestamp) -> pd.DataFrame:
    """ Defines and combines all planned investments based on configuration and signals. Handles aggregation for same-day investments. """
    print("\n--- Defining All Investments ---")
    all_investments_list = []
    # Filter data up to the simulation date
    df_sim = df_data[df_data.index <= simulation_date].copy()
    if df_sim.empty:
        print("Error: No data available for the simulation period.")
        return pd.DataFrame()

    # --- Define Each Investment Type ---
    if CONFIG['fixed_dca']:
        fixed_dates = get_investment_dates(df_sim.index, FIXED_INVESTMENT_DAY, simulation_date)
        fixed_inv = create_investment_df(fixed_dates, 'Fixed_DCA', FIXED_INVESTMENT, df_sim)
        all_investments_list.append(fixed_inv)
        print(f"Defined {len(fixed_inv)} Fixed DCA investments.")

    if CONFIG['extra_normal']:
        # Get potential check dates first
        dates_check = get_investment_dates(df_sim.index, EXTRA_INVESTMENT_DAY, simulation_date)
        # Detect signals only on those check dates
        dates_trig = detect_extra_normal_signals(df_sim, dates_check)
        inv = create_investment_df(dates_trig, 'Extra_Normal', EXTRA_NORMAL_INVESTMENT, df_sim)
        all_investments_list.append(inv)
        print(f"Defined {len(inv)} Extra Normal investments.")

    if CONFIG['extra_purple_alert']:
        dates_raw = detect_black_swans(df_sim)
        dates_lim = apply_frequency_limit(dates_raw, MAX_PURPLE_ALERTS_PER_YEAR)
        inv = create_investment_df(dates_lim, 'Extra_Purple', EXTRA_PURPLE_INVESTMENT, df_sim)
        all_investments_list.append(inv)
        print(f"Defined {len(inv)} Extra Purple investments (after limit).")

    if CONFIG['extra_pullback']:
        dates_raw = detect_pullbacks(df_sim)
        dates_lim = apply_frequency_limit(dates_raw, MAX_PULLBACKS_PER_YEAR)
        inv = create_investment_df(dates_lim, 'Extra_Pullback', EXTRA_PULLBACK_INVESTMENT, df_sim)
        all_investments_list.append(inv)
        print(f"Defined {len(inv)} Extra Pullback investments (after limit).")

    if CONFIG['extra_extreme_dip']:
        dates_raw = detect_extreme_dips(df_sim)
        dates_lim = apply_frequency_limit(dates_raw, MAX_EXTREME_DIPS_PER_YEAR)
        inv = create_investment_df(dates_lim, 'Extra_Extreme_Dip', EXTRA_EXTREME_DIP_INVESTMENT, df_sim)
        all_investments_list.append(inv)
        print(f"Defined {len(inv)} Extra Extreme Dip investments (after limit).")

    # --- Combine and Process All Investments ---
    if not all_investments_list:
        print("Warning: No investment types were enabled or triggered any signals.")
        return pd.DataFrame()

    # Concatenate all non-empty investment DataFrames
    all_investments = pd.concat([inv for inv in all_investments_list if not inv.empty])

    if all_investments.empty:
        print("Warning: Combined investment DataFrame is empty after filtering.")
        return pd.DataFrame()

    print(f"Initial combined investments before aggregation: {len(all_investments)}")

    # Aggregate investments made on the same day
    duplicates = all_investments.index.duplicated(keep=False)
    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicate index entries (multiple investments on the same day). Aggregating...")
        # Define aggregation functions
        agg_funcs = {
            'Contribution': 'sum',          # Sum contributions
            'Shares_Bought': 'sum',         # Sum shares bought
            'SP500_Close': 'first',         # Keep the price from the first entry (should be same)
            'Type': lambda x: '+'.join(sorted(x.unique())) # Combine type names
        }
        all_investments = all_investments.groupby(all_investments.index).agg(agg_funcs)
        print(f"Investments after aggregation: {len(all_investments)}")
    else:
        print("No duplicate investment dates found.")

    # Ensure index is DatetimeIndex and sort
    all_investments.index = pd.to_datetime(all_investments.index)
    all_investments.sort_index(inplace=True)

    # --- Integrate with Daily Market Data ---
    # Ensure the main simulation data index is unique before reindexing
    if not df_sim.index.is_unique:
        print("Error: df_sim index contains duplicates before reindexing. Fixing...")
        df_sim = df_sim[~df_sim.index.duplicated(keep='first')]

    df_sim.sort_index(inplace=True)

    # Reindex the investment data to the full simulation data index, forward filling values
    try:
        # Use 'ffill' to carry forward the last known portfolio state on non-investment days
        all_investments = all_investments.reindex(df_sim.index, method=None) # Reindex first, then fill NaNs
    except ValueError as e:
         # This can happen with duplicate indices if not handled above
        print(f"ERROR during reindex: {e}. Check index uniqueness.")
        raise

    # Fill NaNs created by reindexing
    # Copy SP500 prices from the main df for all days
    all_investments['SP500_Close'] = df_sim['SP500_Close']
    # Fill 0 for contributions/shares on days with no transactions
    all_investments['Contribution'].fillna(0, inplace=True)
    all_investments['Shares_Bought'].fillna(0, inplace=True)
    # Mark days with no transactions
    all_investments['Type'].fillna('No Transaction', inplace=True)

    # Calculate cumulative values
    all_investments['Cumulative_Shares'] = all_investments['Shares_Bought'].cumsum()
    all_investments['Cumulative_Invested'] = all_investments['Contribution'].cumsum()

    # Drop any initial rows where cumulative shares might still be NaN (if first day had no investment)
    all_investments.dropna(subset=['Cumulative_Shares'], inplace=True)
    if all_investments.empty:
         print("Warning: Investment DataFrame became empty after dropping initial NaN cumulative shares.")
         return pd.DataFrame()


    # Calculate cumulative portfolio value
    all_investments['Cumulative_Value'] = all_investments['Cumulative_Shares'] * all_investments['SP500_Close']
    # Forward fill value for days where price might be missing momentarily (though ffill on df_sim should prevent this)
    all_investments['Cumulative_Value'].fillna(method='ffill', inplace=True)


    print(f"\nTotal investment transactions made: {len(all_investments[all_investments['Contribution'] > 0])}")
    if not all_investments.empty:
        print(f"Final portfolio value on {simulation_date.date()}: {all_investments['Cumulative_Value'].iloc[-1]:,.2f}")
        print(f"Total amount invested: {all_investments['Cumulative_Invested'].iloc[-1]:,.2f}")
    else:
        print("No investments were made during the simulation period.")

    return all_investments

# ----------------------------
# 7. Buy and Hold Strategy (Baseline)
# ----------------------------
def buy_and_hold_strategy(df_data: pd.DataFrame, simulation_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """ Simulates a simple Buy and Hold strategy for comparison. """
    print("\n--- Simulating Buy and Hold Strategy ---")
    if not CONFIG['buy_and_hold']:
        print("Buy and Hold strategy is disabled in the configuration.")
        return None

    # Filter data up to the simulation date
    df_sim = df_data[df_data.index <= simulation_date].copy()
    if df_sim.empty:
        print("Error: No data available for the Buy and Hold simulation period.")
        return None

    # Find the first valid trading day in the simulation period
    start_date = df_sim.index[0]
    initial_price = df_sim.loc[start_date, 'SP500_Close']

    if pd.isna(initial_price) or initial_price <= 0:
        print(f"Error: Invalid initial price ({initial_price}) for Buy and Hold on {start_date.date()}. Cannot simulate.")
        return None

    # Calculate shares bought with the initial investment
    shares_bought = BUY_AND_HOLD_INVESTMENT / initial_price

    # Create a portfolio DataFrame for B&H
    bnh_portfolio = pd.DataFrame(index=df_sim.index)
    bnh_portfolio['SP500_Close'] = df_sim['SP500_Close'] # Include market price for reference
    bnh_portfolio['Cumulative_Shares'] = shares_bought
    bnh_portfolio['Cumulative_Invested'] = BUY_AND_HOLD_INVESTMENT # Investment happens once

    # Calculate the value of the holding over time
    bnh_portfolio['Cumulative_Value'] = bnh_portfolio['Cumulative_Shares'] * bnh_portfolio['SP500_Close']

    # Add columns to match the structure of the DCA portfolio for metric calculation
    bnh_portfolio['Contribution'] = 0
    bnh_portfolio.loc[start_date, 'Contribution'] = BUY_AND_HOLD_INVESTMENT # Mark the initial investment day
    bnh_portfolio['Type'] = 'Buy_and_Hold'

    print(f"Buy and Hold: Invested {BUY_AND_HOLD_INVESTMENT:,.2f} on {start_date.date()}.")
    print(f"Final Buy and Hold portfolio value on {simulation_date.date()}: {bnh_portfolio['Cumulative_Value'].iloc[-1]:,.2f}")
    return bnh_portfolio


# ----------------------------
# 8. Calculate Performance Metrics
# ----------------------------
def calculate_performance_metrics(portfolio: pd.DataFrame, strategy_name: str, df_market: pd.DataFrame) -> Optional[Dict[str, any]]:
    """ Calculate performance metrics for a given portfolio DataFrame. """
    print(f"\nCalculating performance metrics for: {strategy_name}")
    if portfolio is None or portfolio.empty:
        print(f"Warning: Cannot calculate metrics for {strategy_name}. Portfolio data is missing or empty.")
        return None
    if 'Cumulative_Value' not in portfolio.columns or 'Cumulative_Invested' not in portfolio.columns:
        print(f"Warning: Cannot calculate metrics for {strategy_name}. 'Cumulative_Value' or 'Cumulative_Invested' column missing.")
        return None
    if portfolio['Cumulative_Value'].isnull().all():
        print(f"Warning: 'Cumulative_Value' contains only NaN values for {strategy_name}.")
        return None

    # Clean portfolio data - focus on necessary columns and drop NaNs that might interfere
    # Need to keep the full series for date calculations before dropping NaNs for returns etc.
    portfolio_calc = portfolio[['Cumulative_Value', 'Cumulative_Invested']].copy()

    # Drop rows where value is NaN, necessary for return calculations
    portfolio_calc.dropna(subset=['Cumulative_Value'], inplace=True)

    if len(portfolio_calc) < 2:
        print(f"Warning: Not enough data points ({len(portfolio_calc)}) after NaN drop for {strategy_name} metrics calculation.")
        return None

    try:
        # --- Basic Metrics ---
        start_date = portfolio_calc.index[0]
        end_date = portfolio_calc.index[-1]
        num_years = (end_date - start_date).days / 365.25
        num_years = max(num_years, 1e-6) # Avoid division by zero if duration is very short

        start_val = portfolio_calc['Cumulative_Value'].iloc[0]
        final_val = portfolio_calc['Cumulative_Value'].iloc[-1]
        total_inv = portfolio_calc['Cumulative_Invested'].iloc[-1] # Get the final cumulative investment

        profit_loss = final_val - total_inv
        roi = profit_loss / total_inv if total_inv > 0 else 0.0

        # CAGR - Handle cases where start_val might be zero or negative (though unlikely here)
        if pd.isna(start_val) or start_val <= 0:
             print(f"Warning: Invalid start value ({start_val}) for CAGR calculation in {strategy_name}.")
             cagr = np.nan
        else:
            # Ensure final_val is also valid before calculating CAGR
             if pd.isna(final_val):
                  print(f"Warning: Final value is NaN for CAGR calculation in {strategy_name}.")
                  cagr = np.nan
             else:
                cagr = ((final_val / start_val) ** (1 / num_years)) - 1


        # --- Risk & Risk-Adjusted Metrics ---
        port_ret = portfolio_calc['Cumulative_Value'].pct_change().dropna()
        ann_vol, sharpe, sortino, calmar, max_dd, beta, alpha = [np.nan] * 7 # Initialize with NaN

        if not port_ret.empty and len(port_ret) >= 2:
            # Annualized Volatility (Standard Deviation of daily returns)
            ann_vol = port_ret.std() * np.sqrt(252)

            # Sharpe Ratio (requires CAGR)
            if not pd.isna(cagr) and ann_vol > 0:
                # Assuming Risk-Free Rate = 0 for simplicity
                sharpe = cagr / ann_vol
            else:
                sharpe = np.nan # Cannot calculate if CAGR is NaN or vol is zero

            # Max Drawdown
            cummax = portfolio_calc['Cumulative_Value'].cummax()
            drawdown = (portfolio_calc['Cumulative_Value'] - cummax) / cummax
            max_dd = drawdown.min()
            # Ensure max_dd is negative or zero; if positive (no drawdown), set to 0
            max_dd = 0.0 if pd.isna(max_dd) or max_dd > 0 else max_dd


            # Sortino Ratio (uses downside deviation)
            neg_ret = port_ret[port_ret < 0]
            if not neg_ret.empty:
                down_std = neg_ret.std() * np.sqrt(252)
                if not pd.isna(cagr) and down_std > 0:
                     # Assuming Risk-Free Rate = 0
                    sortino = cagr / down_std
                else:
                    sortino = np.nan # Cannot calculate if CAGR NaN or downside std is zero
            else:
                # If no negative returns, Sortino is arguably infinite or undefined. Set to NaN.
                sortino = np.nan


            # Calmar Ratio (CAGR / Max Drawdown)
            if not pd.isna(cagr) and not pd.isna(max_dd) and max_dd < 0:
                calmar = cagr / abs(max_dd)
            else:
                 calmar = np.nan # Cannot calculate if CAGR or max_dd is NaN, or if max_dd is zero/positive


            # --- Beta & Alpha (requires market data) ---
            # Align market data (SP500 returns) with portfolio return dates
            if df_market is not None and 'SP500_Return' in df_market.columns:
                # Ensure market index is unique and get data relevant to the portfolio period
                market_data_aligned = df_market[df_market.index >= start_date].copy()
                market_data_aligned = market_data_aligned[~market_data_aligned.index.duplicated(keep='first')]

                # Find common dates between portfolio returns and market returns
                common_idx_ret = port_ret.index.intersection(market_data_aligned.index)

                if len(common_idx_ret) > 1:
                    port_ret_aligned = port_ret.loc[common_idx_ret]
                    mkt_ret = market_data_aligned['SP500_Return'].loc[common_idx_ret].dropna()

                    # Re-align after potential dropna in market returns
                    final_common_idx = port_ret_aligned.index.intersection(mkt_ret.index)
                    if len(final_common_idx) > 1:
                        port_ret_aligned = port_ret_aligned.loc[final_common_idx]
                        mkt_ret = mkt_ret.loc[final_common_idx]

                        # Calculate Beta
                        # Check variance to avoid division by zero
                        if mkt_ret.var() > 0:
                             # Use numpy cov for potentially better handling of small numbers/precision
                             cov_matrix = np.cov(port_ret_aligned, mkt_ret)
                             # Ensure cov_matrix is 2x2 before accessing elements
                             if cov_matrix.shape == (2, 2):
                                 covariance = cov_matrix[0][1]
                                 market_variance = mkt_ret.var() # Or use cov_matrix[1][1]
                                 beta = covariance / market_variance

                                 # Calculate Alpha (requires Beta and Market Return)
                                 # Assuming Risk-Free Rate = 0
                                 if not pd.isna(beta) and not pd.isna(cagr):
                                     # Calculate annualized market return over the same period
                                     ann_mkt_ret = mkt_ret.mean() * 252
                                     alpha = cagr - (beta * ann_mkt_ret) # Simplified alpha
                                 else:
                                     alpha = np.nan # Need beta and cagr
                             else:
                                 print(f"Warning: Covariance matrix shape unexpected ({cov_matrix.shape}) for {strategy_name}. Cannot calculate Beta/Alpha.")
                                 beta, alpha = np.nan, np.nan
                        else:
                            print(f"Warning: Market return variance is zero for {strategy_name}. Cannot calculate Beta/Alpha.")
                            beta, alpha = np.nan, np.nan # Market variance is zero
                    else:
                        print(f"Warning: Not enough common return data points ({len(final_common_idx)}) after final alignment for {strategy_name} Beta/Alpha.")
                        beta, alpha = np.nan, np.nan
                else:
                     print(f"Warning: Not enough common return data points ({len(common_idx_ret)}) for {strategy_name} Beta/Alpha.")
                     beta, alpha = np.nan, np.nan
            else:
                 print(f"Warning: Market data unavailable for {strategy_name}. Cannot calculate Beta/Alpha.")
                 beta, alpha = np.nan, np.nan # Market data missing

        else: # Handle case where not enough return points for risk calcs
            print(f"Warning: Not enough return points ({len(port_ret)}) to calculate detailed risk metrics for {strategy_name}.")
            # Keep initialized NaN values for ann_vol, sharpe, etc.


        # --- Market Comparison Metrics ---
        mkt_cum_ret = np.nan
        mkt_cagr = np.nan
        if df_market is not None and 'SP500_Close' in df_market.columns:
             # Align market prices to the portfolio start/end dates
             market_data_aligned = df_market.loc[start_date:end_date, 'SP500_Close'].dropna()
             if len(market_data_aligned) >= 2:
                 mkt_start_p = market_data_aligned.iloc[0]
                 mkt_end_p = market_data_aligned.iloc[-1]

                 # Ensure valid market prices before calculation
                 if not pd.isna(mkt_start_p) and not pd.isna(mkt_end_p) and mkt_start_p > 0:
                     mkt_cum_ret = (mkt_end_p / mkt_start_p) - 1
                     mkt_cagr = ((mkt_end_p / mkt_start_p) ** (1 / num_years)) - 1
                 else:
                     print(f"Warning: Invalid market start/end prices ({mkt_start_p}, {mkt_end_p}) for {strategy_name} market comparison.")
             else:
                  print(f"Warning: Not enough aligned market price data points ({len(market_data_aligned)}) for {strategy_name} market comparison.")


        # --- Compile Metrics Dictionary ---
        metrics = {
            'Strategy': strategy_name,
            'Start Date': start_date.date(),
            'End Date': end_date.date(),
            'Duration (Years)': round(num_years, 2),
            'Final Value': round(final_val, 2),
            'Total Invested': round(total_inv, 2),
            'Profit/Loss': round(profit_loss, 2),
            'ROI (%)': round(roi * 100, 2),
            'CAGR (%)': round(cagr * 100, 2) if not pd.isna(cagr) else 'N/A',
            'Annual Volatility (%)': round(ann_vol * 100, 2) if not pd.isna(ann_vol) else 'N/A',
            'Sharpe Ratio': round(sharpe, 2) if not pd.isna(sharpe) else 'N/A',
            'Sortino Ratio': round(sortino, 2) if not pd.isna(sortino) else 'N/A',
            'Max Drawdown (%)': round(max_dd * 100, 2) if not pd.isna(max_dd) else 'N/A',
            'Calmar Ratio': round(calmar, 2) if not pd.isna(calmar) else 'N/A',
            'Beta': round(beta, 2) if not pd.isna(beta) else 'N/A',
            'Alpha (%)': round(alpha * 100, 2) if not pd.isna(alpha) else 'N/A', # Alpha usually presented annualized %
            'Market CAGR (%)': round(mkt_cagr * 100, 2) if not pd.isna(mkt_cagr) else 'N/A',
            'Market Cumulative Return (%)': round(mkt_cum_ret * 100, 2) if not pd.isna(mkt_cum_ret) else 'N/A'
        }
        print(f"Metrics calculation successful for {strategy_name}.")
        return metrics

    except Exception as e:
        print(f"Error calculating performance metrics for {strategy_name}: {e}")
        traceback.print_exc()
        return None

# ----------------------------
# 9. Visualizations
# ----------------------------
def generate_visualizations(df_data: pd.DataFrame, enhanced_dca_portfolio: pd.DataFrame):
    """
    Generate interactive plots using Plotly.
    Creates separate plots for SP500+Signals and RSI.
    Adds explicit Y-axis range setting for the SP500 plot.
    """
    print("\n--- Generating Visualizations ---")
    if enhanced_dca_portfolio is None or enhanced_dca_portfolio.empty:
        print("Warning: Portfolio data is empty. Skipping visualizations.")
        return

    # Determine plot range from portfolio index (which should be aligned with df_data)
    start_plot_date = enhanced_dca_portfolio.index.min()
    end_plot_date = enhanced_dca_portfolio.index.max()

    # Select the relevant slice of the main data DataFrame for plotting indicators
    plot_data_base = df_data.loc[start_plot_date:end_plot_date].copy()
    if plot_data_base.empty:
        print("Warning: Base data for the plotting period is empty.")
        return

    # Prepare investment points data for plotting markers
    # Filter portfolio data for actual investment days (Contribution > 0)
    investment_points = enhanced_dca_portfolio[enhanced_dca_portfolio['Contribution'] > 0].copy()

    # Ensure investment points exist within the base plot data range
    valid_investment_dates = investment_points.index.intersection(plot_data_base.index)
    investment_points = investment_points.loc[valid_investment_dates]

    if not investment_points.empty:
        # Get the SP500 price at the time of investment for plotting markers correctly
        # Use plot_data_base prices which are aligned to the plot range
        investment_points['Price_at_Invest'] = plot_data_base.loc[valid_investment_dates, 'SP500_Close']
    else:
        print("Warning: No valid investment points found within the plot data range.")
        # Add the column even if empty to prevent errors later
        investment_points['Price_at_Invest'] = np.nan


    # --- Calculate Y-axis range for SP500 plot ---
    # Consider both the SP500 price series and the prices where investments occurred
    valid_prices = plot_data_base['SP500_Close'].dropna()
    valid_marker_prices = investment_points['Price_at_Invest'].dropna()

    # Combine all valid Y values to find the overall min/max
    all_valid_y_values = pd.concat([valid_prices, valid_marker_prices])

    yaxis_range = None # Default to None (Plotly auto-range)
    if all_valid_y_values.empty:
        print("Warning: No valid Y values found for SP500 plot range calculation. Using Plotly auto-range.")
    else:
        try:
            # Find min and max, ensuring they are scalar values
            min_val = all_valid_y_values.min()
            max_val = all_valid_y_values.max()

            # Convert potential Series results (if only one value) to scalar
            min_y = min_val.item() if hasattr(min_val, 'item') else float(min_val)
            max_y = max_val.item() if hasattr(max_val, 'item') else float(max_val)

            # Add padding to the range
            padding = (max_y - min_y) * 0.05 # 5% padding top and bottom
            # Handle case where min_y == max_y (flat line)
            if padding == 0:
                padding = max_y * 0.1 # Use 10% of the value as padding

            yaxis_range = [min_y - padding, max_y + padding]
            print(f"Calculated Y-axis range for SP500 plot: {yaxis_range}")
        except Exception as e:
            print(f"Error calculating Y-axis range: {e}. Falling back to Plotly auto-range.")
            yaxis_range = None # Fallback


    # --- Generate Plots ---
    try:
        # --- Plot 1: SP500 Price and Investment Signals ---
        print("Generating SP500 Price + Signals Plot...")
        fig_sp500_signals = go.Figure()

        # Add SP500 Price Trace
        # Ensure we are plotting a Series, not a DataFrame column that might cause issues
        sp500_series = plot_data_base['SP500_Close']
        if isinstance(sp500_series, pd.DataFrame):
            print("Warning: SP500_Close is a DataFrame, flattening to Series.")
            sp500_series = sp500_series.iloc[:, 0] # Take the first column

        fig_sp500_signals.add_trace(go.Scatter(
            x=plot_data_base.index,
            y=sp500_series,
            mode='lines',
            name='SP500 Price',
            line=dict(color='blue', width=1.5)
        ))

        # Add Investment Markers (if any)
        if not investment_points.empty:
            # Define colors and symbols for different investment types
            colors = {'Fixed_DCA':'green', 'Extra_Normal':'blue', 'Extra_Purple':'purple', 'Extra_Pullback':'red', 'Extra_Extreme_Dip':'black', 'Buy_and_Hold': 'orange'}
            symbols = {'Fixed_DCA':'circle', 'Extra_Normal':'triangle-up', 'Extra_Purple':'diamond', 'Extra_Pullback':'star', 'Extra_Extreme_Dip':'cross', 'Buy_and_Hold': 'square'}
            default_color, default_symbol = 'grey', 'circle-open' # Fallback

            # Group by investment type to plot them with distinct markers/colors
            grouped_investments = investment_points.groupby('Type')
            for inv_type, type_data in grouped_investments:
                 if not type_data.empty:
                     # Calculate average contribution for this type to add to legend
                     # Handle potential combined types like 'Fixed_DCA+Extra_Pullback'
                     type_names = inv_type.split('+')
                     avg_contrib_str = ""
                     first_type = type_names[0] # Use first type for color/symbol lookup
                     if len(type_names) == 1: # Single investment type
                         contrib_mean = type_data['Contribution'].mean()
                         avg_contrib_str = f" (~${contrib_mean:,.0f})" # Add avg amount
                     else: # Combined investment type
                         contrib_sum = type_data['Contribution'].sum() # Show total for combined
                         avg_contrib_str = f" (${contrib_sum:,.0f} total)"


                     fig_sp500_signals.add_trace(go.Scatter(
                         x=type_data.index,
                         y=type_data['Price_at_Invest'],
                         mode='markers',
                         name=f'{inv_type}{avg_contrib_str}',
                         marker=dict(
                             color=colors.get(first_type, default_color), # Color based on first type in combined
                             symbol=symbols.get(first_type, default_symbol), # Symbol based on first type
                             size=8,
                             line=dict(width=1, color='DarkSlateGrey') # Marker border
                         ),
                         hoverinfo='x+y+name' # Show date, price, and type name on hover
                     ))

        # Layout for SP500 + Signals Plot
        fig_sp500_signals.update_layout(
            title=f'S&P 500 Price with Investment Signals ({start_plot_date.date()} to {end_plot_date.date()})',
            xaxis_title='Date',
            yaxis_title='SP500 Price (USD)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified', # Show hover data for all traces at a given x-coordinate
            yaxis_range=yaxis_range # Apply calculated Y-axis range (or None for auto)
        )
        fig_sp500_signals.show()
        print("SP500 Price + Signals plot generated.")

        # --- Plot 2: RSI Indicator ---
        print("Generating RSI Indicator Plot...")
        fig_rsi = go.Figure()

        # Add RSI Trace
        fig_rsi.add_trace(go.Scatter(
            x=plot_data_base.index,
            y=plot_data_base['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='magenta', width=1.5)
        ))

        # Add Overbought/Oversold Lines
        fig_rsi.add_hline(y=RSI_OVERBOUGHT, line=dict(color='red', width=1, dash='dash'), name='Overbought', annotation_text="Overbought", annotation_position="bottom right")
        fig_rsi.add_hline(y=RSI_OVERSOLD, line=dict(color='green', width=1, dash='dash'), name='Oversold', annotation_text="Oversold", annotation_position="top right")
        # Add Extreme Oversold line if that strategy is enabled
        if CONFIG['extra_extreme_dip']:
            fig_rsi.add_hline(y=EXTREME_RSI_OVERSOLD, line=dict(color='darkgreen', width=1.5, dash='dot'), name='Extreme Oversold', annotation_text="Extreme", annotation_position="top right")

        # Layout for RSI Plot
        fig_rsi.update_layout(
            title=f'Relative Strength Index (RSI) ({start_plot_date.date()} to {end_plot_date.date()})',
            xaxis_title='Date',
            yaxis_title='RSI',
            yaxis_range=[0, 100], # RSI is bounded between 0 and 100
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig_rsi.show()
        print("RSI plot generated.")

    except Exception as e:
        print(f"Error generating visualizations: {e}")
        traceback.print_exc()


# ----------------------------
# 10. Main Execution Function
# ----------------------------
def main():
    """ Main function to run the DCA strategy simulation. """
    start_time = datetime.now()
    print("--- Starting Enhanced DCA Simulation ---")
    print(f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}")
    print("\n******************** IMPORTANT DISCLAIMER ********************")
    print("This script is for EDUCATIONAL and INFORMATIONAL purposes ONLY.")
    print("It is NOT financial advice. Trading involves SUBSTANTIAL RISK.")
    print("Past performance is NOT indicative of future results.")
    print("Use a DEMO account for testing. Consult a qualified financial advisor.")
    print("************************************************************\n")

    try:
        # 1. Setup: Parse arguments and determine date range
        simulation_date = parse_arguments()
        data_end_date = simulation_date # Use the simulation date as the end date for fetching data

        # 2. Fetch Data for all required tickers
        tickers = {'SP500': TICKER_SP500, 'VIX': TICKER_VIX, 'Bond_Yield': TICKER_BOND_YIELD}
        data_frames = {}
        valid_data = True
        max_available_date = pd.Timestamp.min # Track the latest date available across all fetched data

        for key, ticker in tickers.items():
            try:
                df = fetch_data(ticker, DATA_START_DATE, data_end_date)
                data_frames[key] = df
                # Update max_available_date based on the fetched data's end date
                if not df.empty:
                     current_max = df.index.max()
                     max_available_date = max(max_available_date, current_max) if max_available_date != pd.Timestamp.min else current_max

            except Exception as e:
                # Decide if fatal: If SP500 data fails, it's likely fatal. Others might be recoverable depending on strategy.
                if key == 'SP500':
                    print(f"FATAL ERROR: Failed to fetch essential SP500 data: {e}")
                    valid_data = False
                    break
                else:
                    print(f"Warning: Failed to fetch data for {ticker}: {e}. Proceeding without it if possible.")
                    # Mark DataFrame as potentially missing if needed later
                    data_frames[key] = pd.DataFrame() # Assign empty DF

        if not valid_data:
            sys.exit(1) # Exit if essential data fetching failed

        # Adjust simulation_date if it's later than the latest available data point across all sources
        # Use the *earliest* end date among the successfully fetched non-empty dataframes as the true ceiling
        earliest_end_date = pd.Timestamp.max
        fetched_dfs = [df for df in data_frames.values() if not df.empty]
        if not fetched_dfs:
            print("FATAL ERROR: No data could be fetched successfully.")
            sys.exit(1)
        for df in fetched_dfs:
             earliest_end_date = min(earliest_end_date, df.index.max())


        if simulation_date > earliest_end_date:
             print(f"\nWarning: Requested simulation date {simulation_date.date()} is later than the latest available data point ({earliest_end_date.date()}).")
             print(f"Adjusting simulation date to {earliest_end_date.date()}.")
             simulation_date = earliest_end_date
             # Re-normalize just in case adjustment resulted in time component
             simulation_date = simulation_date.normalize()


        print(f"Final simulation date being used: {simulation_date.date()}")

        # 3. Preprocess and Merge Data
        # Prepare each dataframe (resample, fill, select column)
        sp500 = prepare_dataframe(data_frames.get('SP500', pd.DataFrame()), 'SP500')
        vix = prepare_dataframe(data_frames.get('VIX', pd.DataFrame()), 'VIX')
        bond_yield = prepare_dataframe(data_frames.get('Bond_Yield', pd.DataFrame()), '10Y_Bond_Yield')

        # Check if essential SP500 data is present
        if sp500.empty:
             print("FATAL ERROR: SP500 data preparation failed or resulted in empty DataFrame.")
             sys.exit(1)

        # Combine into a single DataFrame, joining on the index (Date)
        # Use outer join initially to see full date range, then handle NaNs
        data = pd.concat([sp500, vix, bond_yield], axis=1, join='outer')
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]


        # Forward fill data to handle potential gaps if sources had slightly different holidays/trading days
        # data.ffill(inplace=True) # Apply ffill *after* concat

        # Diagnostic: Print columns before dropping NaNs for SP500_Close
        print(f"Columns in merged data before dropna: {list(data.columns)}")
        if 'SP500_Close' not in data.columns:
            print("FATAL ERROR: 'SP500_Close' column is missing from merged data. Available columns:", list(data.columns))
            sys.exit(1)
        # Crucially, drop any rows where the primary instrument (SP500) is still NaN after ffill
        data.dropna(subset=['SP500_Close'], inplace=True)


        if data.empty:
            print("FATAL ERROR: Merged data is empty after processing and NaN handling.")
            sys.exit(1)

        print(f"\nMerged data range available: {data.index.min().date()} to {data.index.max().date()}")

        # 4. Feature Engineering
        data = feature_engineering(data)
        if data.empty:
            print("FATAL ERROR: Data became empty after feature engineering (likely due to initial NaNs from rolling calculations).")
            sys.exit(1)

        # Ensure the final simulation date exists in the processed data after potential drops from feature eng.
        if simulation_date not in data.index:
            # Find the latest date in the data that is <= the target simulation date
            available_dates = data.index[data.index <= simulation_date]
            if not available_dates.empty:
                simulation_date = available_dates[-1] # Use the last available date
                print(f"Warning: Original simulation date not found after feature engineering.")
                print(f"Adjusted simulation date to the last processed day: {simulation_date.date()}")
            else:
                # This shouldn't happen if checks above worked, but as a safeguard:
                print(f"FATAL ERROR: Simulation date {simulation_date.date()} is outside the processed data range: {data.index.min().date()} to {data.index.max().date()}.")
                sys.exit(1)


        # 5. Define Investments & Simulate Enhanced DCA
        enhanced_dca_portfolio = define_all_investments(data, simulation_date)

        # 6. Simulate Buy and Hold (if enabled)
        bnh_portfolio = buy_and_hold_strategy(data, simulation_date) # Will return None if disabled

        # 7. Calculate Performance Metrics
        metrics_list = []
        # Prepare market data subset needed for Beta/Alpha/Market comparison
        market_data = data[['SP500_Close', 'SP500_Return']].copy()

        if enhanced_dca_portfolio is not None and not enhanced_dca_portfolio.empty:
            metrics = calculate_performance_metrics(enhanced_dca_portfolio, "Enhanced DCA", market_data)
            if metrics:
                metrics_list.append(metrics)
            # Save transactions for Enhanced DCA
            try:
                 # Select only actual investment rows
                 trans = enhanced_dca_portfolio[enhanced_dca_portfolio['Contribution'] > 0].copy()
                 # Select and format columns for output
                 trans_cols = ['Type','Contribution','SP500_Close','Shares_Bought','Cumulative_Shares','Cumulative_Invested','Cumulative_Value']
                 trans[trans_cols].to_csv('enhanced_dca_transactions.csv', float_format='%.2f', date_format='%Y-%m-%d')
                 print("\nEnhanced DCA transactions saved to enhanced_dca_transactions.csv")
            except Exception as e:
                 print(f"\nError saving Enhanced DCA transactions: {e}")
        else:
            print("\nNo Enhanced DCA transactions to analyze or save.")

        if bnh_portfolio is not None and not bnh_portfolio.empty:
            metrics = calculate_performance_metrics(bnh_portfolio, "Buy and Hold", market_data)
            if metrics:
                metrics_list.append(metrics)
        elif CONFIG['buy_and_hold']: # Only print warning if it was enabled but failed
            print("\nBuy and Hold strategy enabled but no results generated for metrics.")


        # 8. Display and Save Summary
        if metrics_list:
            summary_df = pd.DataFrame(metrics_list)
            if 'Strategy' in summary_df.columns:
                 # Set Strategy as index for better display
                 summary_df = summary_df.set_index('Strategy')
                 print("\n--- Performance Summary ---")
                 # Transpose for better readability in console
                 print(summary_df.T.to_string()) # Use to_string for full display
                 try:
                     summary_df.T.to_csv('performance_summary.csv', float_format='%.2f')
                     print("\nPerformance summary saved to performance_summary.csv")
                 except Exception as e:
                     print(f"\nError saving performance summary: {e}")
            else:
                 print("\nWarning: 'Strategy' column missing in metrics data. Cannot format summary correctly.")
                 print(summary_df) # Print raw data instead
        else:
            print("\nNo performance metrics were calculated for any strategy.")

        # 9. Generate Visualizations (only for Enhanced DCA for now)
        if enhanced_dca_portfolio is not None and not enhanced_dca_portfolio.empty:
            generate_visualizations(data, enhanced_dca_portfolio)
        else:
            print("\nSkipping visualization generation as Enhanced DCA portfolio is empty.")


        # 10. Check for Investment Alerts for the *Simulation Date*
        print(f"\n--- Investment Alerts for Simulation Date: {simulation_date.date()} ---")
        today_investments = pd.DataFrame() # Initialize empty
        alert_found = False

        if enhanced_dca_portfolio is not None and not enhanced_dca_portfolio.empty and simulation_date in enhanced_dca_portfolio.index:
            # Check if any *extra* investment was triggered on the simulation date
            today_investments = enhanced_dca_portfolio.loc[[simulation_date]].copy() # Use double brackets to keep DataFrame structure
            # Filter for actual contributions > 0
            today_investments = today_investments[today_investments['Contribution'] > 0]

            if not today_investments.empty:
                # Iterate through potentially combined investments for the day
                for idx, row_data in today_investments.iterrows():
                     # Check if the type indicates an 'Extra' investment (could be combined)
                     # This logic assumes 'Fixed_DCA' is not considered an "alert" in the same way
                     is_extra = any(t in row_data['Type'] for t in ['Extra', 'Purple', 'Pullback', 'Extreme'])
                     if is_extra:
                         print(f"ALERT ({row_data['Type']}): Potential extra investment of ${row_data['Contribution']:,.2f} indicated today ({idx.date()}) based on signals.")
                         print(f"  -> Corresponding SP500 Close: ${row_data['SP500_Close']:,.2f}")
                         alert_found = True

        # Separately check for the regular Fixed DCA on the simulation date
        if CONFIG['fixed_dca']:
             # Regenerate fixed dates up to the simulation date
             fixed_dates = get_investment_dates(data.index, FIXED_INVESTMENT_DAY, simulation_date)
             if simulation_date in fixed_dates:
                 # Check if this Fixed DCA was already part of a combined alert printed above
                 is_alerted_already = False
                 if not today_investments.empty and 'Fixed_DCA' in today_investments['Type'].iloc[0]:
                     is_alerted_already = True

                 if not is_alerted_already:
                     # Get the price for the fixed investment day
                     today_price = data.loc[simulation_date, 'SP500_Close']
                     print(f"ALERT (Fixed_DCA): Regular monthly investment of ${FIXED_INVESTMENT:,.2f} scheduled today ({simulation_date.date()}).")
                     print(f"  -> Corresponding SP500 Close: ${today_price:,.2f}")
                     alert_found = True


        if not alert_found:
            print("No specific investment alerts triggered for today based on the configured strategies.")


    except FileNotFoundError as e:
         print(f"\n--- File Access Error ---")
         print(f"Error: {e}. Check if required files exist or if you have permissions.")
    except KeyError as e:
         print(f"\n--- Data Error ---")
         print(f"KeyError: {e}. A required column or index was not found. Check data fetching and processing steps.")
         traceback.print_exc()
    except Exception as e:
        # Catch any other unexpected errors
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")
    finally:
        # This block will always execute, even if errors occur
        end_time = datetime.now()
        print(f"\n--- Simulation Finished ---")
        print(f"End Time: {end_time:%Y-%m-%d %H:%M:%S}")
        print(f"Total Execution Duration: {end_time - start_time}")

# ----------------------------
# 11. Entry Point
# ----------------------------

if __name__ == "__main__":
    # Standard practice to ensure the main function runs only when the script is executed directly
    print("\nExecuting Enhanced DCA Strategy Script")
    print("-" * 40)
    print("Disclaimer: EDUCATIONAL ONLY. NOT financial advice. RISK involved.")
    print("Use DEMO account. Consult professional.")
    print("-" * 40 + "\n")
    main()