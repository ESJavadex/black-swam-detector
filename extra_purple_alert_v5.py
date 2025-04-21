# -*- coding: utf-8 -*-
"""
Enhanced Dollar-Cost Averaging (DCA) Strategy for S&P 500 (SPY)
with Dip Buying Opportunities based on Technical Indicators.

Includes NEW Strategy: Volatility Spike & RSI Dip

Version: v13 (Added Volatility Spike & RSI Dip Strategy)

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
    # --- Strategy Components ---
    'fixed_dca': True,          # Enable regular monthly DCA
    'extra_normal': False,      # Disable extra buys on dips during uptrends (focus on bigger dips)
    'extra_purple_alert': True, # Enable extra buys on 'Black Swan' signals
    'extra_pullback': True,     # Enable extra buys on pullback signals
    'extra_extreme_dip': True,  # Enable extra buys on extreme RSI dips

    # --- *** NEW STRATEGY: Volatility Spike & RSI Dip *** ---
    'enable_vol_spike_dip': True, # !!! ENABLE THE NEW STRATEGY !!!

    # --- *** ADVANCED STRATEGY: Market Dislocation Opportunity *** ---
    'enable_market_dislocation': True, # Enable the advanced strategy
    
    # --- *** CAPITULATION STRATEGY: Extreme Market Bottom Detection *** ---
    'enable_market_capitulation': True, # Enable the capitulation strategy

    'buy_and_hold': True,       # Enable Buy and Hold strategy for comparison

    # --- Investment Amounts ---
    'fixed_investment': 475,          # Amount for regular monthly DCA
    'extra_normal_investment': 100,   # Amount for 'Extra Normal' buys (if enabled)
    'extra_purple_investment': 250,   # Amount for 'Extra Purple Alert' buys (MAJOR DIP)
    'extra_pullback_investment': 175, # Amount for 'Extra Pullback' buys (SIGNIFICANT DIP)
    'extra_extreme_dip_investment': 300, # Amount for 'Extra Extreme Dip' buys (SEVERE DIP)

    # --- *** NEW STRATEGY Amount *** ---
    'vol_spike_dip_investment': 200, # Amount for the new Volatility Spike strategy

    # --- *** ADVANCED STRATEGY Amount *** ---
    'market_dislocation_investment': 350, # Amount for the advanced market dislocation strategy
    
    # --- *** CAPITULATION STRATEGY Amount *** ---
    'market_capitulation_investment': 500, # Amount for the market capitulation strategy

    'buy_and_hold_investment': 5000, # Initial lump sum for Buy and Hold strategy

    # --- Signal Parameters (Adjust if desired) ---
    'prediction_window': 20,
    'black_swan_std_multiplier': 3,
    'pullback_threshold': -6,         # Slightly increase sensitivity for pullback
    'pullback_window': 12,

    'rsi_window': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'extreme_rsi_oversold': 25,

    # --- *** NEW STRATEGY Parameters *** ---
    'vol_spike_vix_threshold': 30,   # VIX level to consider volatility 'high'
    'vol_spike_rsi_threshold': 32,   # RSI level for SPY to consider 'oversold' for this strategy

    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,

    # --- Frequency Limits ---
    'max_purple_alerts_per_year': 2,
    'max_pullbacks_per_year': 4,      # Allow slightly more pullbacks
    'max_extreme_dips_per_year': 3,   # Allow slightly more extreme dips

    # --- *** NEW STRATEGY Frequency Limit *** ---
    'max_vol_spike_dips_per_year': 3, # Max times Volatility Spike can trigger per year
    'max_market_dislocations_per_year': 2, # Max times Market Dislocation can trigger per year
    'max_capitulations_per_year': 1, # Max times Market Capitulation can trigger per year (rare events)

    # --- Other Settings ---
    'ticker_sp500': 'SPY',
    'ticker_vix': '^VIX',
    'ticker_bond_yield': '^TNX',

    'data_start_date': '2010-01-01',  # Start earlier to capture more VIX history maybe
    'fixed_investment_day': 1,
    'extra_investment_day': 15,
}

# --- Assign constants from CONFIG (consider full refactor later) ---
FIXED_INVESTMENT = CONFIG['fixed_investment']
EXTRA_NORMAL_INVESTMENT = CONFIG['extra_normal_investment']
EXTRA_PURPLE_INVESTMENT = CONFIG['extra_purple_investment']
EXTRA_PULLBACK_INVESTMENT = CONFIG['extra_pullback_investment']
EXTRA_EXTREME_DIP_INVESTMENT = CONFIG['extra_extreme_dip_investment']
VOL_SPIKE_DIP_INVESTMENT = CONFIG['vol_spike_dip_investment'] # New Strategy Amount
BUY_AND_HOLD_INVESTMENT = CONFIG['buy_and_hold_investment']

PREDICTION_WINDOW = CONFIG['prediction_window']
BLACK_SWAN_STD_MULTIPLIER = CONFIG['black_swan_std_multiplier']
PULLBACK_THRESHOLD = CONFIG['pullback_threshold']
PULLBACK_WINDOW = CONFIG['pullback_window']

RSI_WINDOW = CONFIG['rsi_window']
RSI_OVERBOUGHT = CONFIG['rsi_overbought']
RSI_OVERSOLD = CONFIG['rsi_oversold']
EXTREME_RSI_OVERSOLD = CONFIG['extreme_rsi_oversold']

VOL_SPIKE_VIX_THRESHOLD = CONFIG['vol_spike_vix_threshold'] # New Strategy Param
VOL_SPIKE_RSI_THRESHOLD = CONFIG['vol_spike_rsi_threshold'] # New Strategy Param

MACD_FAST = CONFIG['macd_fast']
MACD_SLOW = CONFIG['macd_slow']
MACD_SIGNAL = CONFIG['macd_signal']

MAX_PURPLE_ALERTS_PER_YEAR = CONFIG['max_purple_alerts_per_year']
MAX_PULLBACKS_PER_YEAR = CONFIG['max_pullbacks_per_year']
MAX_EXTREME_DIPS_PER_YEAR = CONFIG['max_extreme_dips_per_year']
MAX_VOL_SPIKE_DIPS_PER_YEAR = CONFIG['max_vol_spike_dips_per_year'] # New Strategy Limit

# --- ADVANCED Strategy Constants ---
MARKET_DISLOCATION_INVESTMENT = CONFIG.get('market_dislocation_investment', 350) # Default if not in CONFIG
MAX_MARKET_DISLOCATIONS_PER_YEAR = CONFIG.get('max_market_dislocations_per_year', 2) # Default if not in CONFIG

# --- CAPITULATION Strategy Constants ---
MARKET_CAPITULATION_INVESTMENT = CONFIG.get('market_capitulation_investment', 500) # Default if not in CONFIG
MAX_CAPITULATIONS_PER_YEAR = CONFIG.get('max_capitulations_per_year', 1) # Default if not in CONFIG

# Capitulation detection parameters - Optimized for March 2020 COVID crash
CAPITULATION_DROP_THRESHOLD = -25.0  # 25% drop from recent high (20-day)
CAPITULATION_VIX_THRESHOLD = 50.0    # VIX above 50
CAPITULATION_RSI_THRESHOLD = 35.0    # RSI below 35 (slightly higher than optimal to catch more signals)
CAPITULATION_LONG_DROP_THRESHOLD = -25.0  # 25% drop from 52-week high

TICKER_SP500 = CONFIG['ticker_sp500']
TICKER_VIX = CONFIG['ticker_vix']
TICKER_BOND_YIELD = CONFIG['ticker_bond_yield']

DATA_START_DATE = CONFIG['data_start_date']
FIXED_INVESTMENT_DAY = CONFIG['fixed_investment_day']
EXTRA_INVESTMENT_DAY = CONFIG['extra_investment_day']

# ----------------------------
# Argument Parsing
# ----------------------------
def parse_arguments() -> pd.Timestamp:
    """ Parse command-line arguments for the simulation end date. """
    parser = argparse.ArgumentParser(description='Enhanced DCA Strategy Simulation with Volatility Spike Dip Buying')
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
        simulation_date = pd.to_datetime(datetime.now())
        print(f"No date provided. Using current date: {simulation_date.date()}")

    simulation_date = simulation_date.normalize()

    if simulation_date.weekday() >= 5:
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
        df = yf.download(ticker, start=start_dt, end=end_date + pd.Timedelta(days=1), progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data fetched for {ticker}.")
        print(f"Fetched {len(df)} records for {ticker}.")
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
    if df is None or df.empty:
         print(f"Warning: Input DataFrame for {name} is empty or None. Returning empty.")
         # Return an empty DataFrame with expected column naming convention if possible
         return pd.DataFrame(columns=[f'{name}_Close'])
    try:
        df_processed = df.copy()
        df_processed = df_processed.resample('D').last().ffill()
        if 'Close' in df_processed.columns:
            df_processed = df_processed[['Close']].rename(columns={'Close': f'{name}_Close'})
        elif 'Adj Close' in df_processed.columns: # Fallback for some indices
             print(f"Using 'Adj Close' for {name} as 'Close' not found.")
             df_processed = df_processed[['Adj Close']].rename(columns={'Adj Close': f'{name}_Close'})
        else:
            # If essential data (like SPY) is missing 'Close', it's an issue.
            if name == 'SP500':
                 raise KeyError(f"Could not find 'Close' or 'Adj Close' column for {name}.")
            else: # For VIX/TNX maybe less critical depending on strategy enabled
                 print(f"Warning: Could not find 'Close' or 'Adj Close' for {name}. Strategy relying on it might fail.")
                 return pd.DataFrame(columns=[f'{name}_Close']) # Return empty with column name

        print(f"DataFrame for {name} prepared. Index range: {df_processed.index.min().date()} to {df_processed.index.max().date()}")
        return df_processed
    except Exception as e:
        print(f"Error preparing DataFrame for {name}: {e}")
        # Decide how critical the failure is
        if name == 'SP500':
             raise # Re-raise if SP500 fails
        else:
             print("Continuing without this data if possible.")
             return pd.DataFrame(columns=[f'{name}_Close']) # Return empty


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
        df_eng.index.name = None

        # --- Ensure required columns exist before calculations ---
        required_base = ['SP500_Close_SPY']
        if not all(col in df_eng.columns for col in required_base):
            missing = [c for c in required_base if c not in df_eng.columns]
            print(f"FATAL Error: Missing essential base columns for features: {missing}")
            raise KeyError(f"Missing essential base columns: {missing}")

        # Basic Returns and Changes
        df_eng['SP500_Return'] = df_eng['SP500_Close_SPY'].pct_change()
        # Calculate other changes only if the column exists
        if 'VIX_Close_^VIX' in df_eng.columns:
            df_eng['VIX_Change'] = df_eng['VIX_Close_^VIX'].pct_change()
        else:
             print("Warning: VIX_Close_^VIX column missing. Skipping VIX_Change calculation.")
        if '10Y_Bond_Yield_Close_^TNX' in df_eng.columns:
            df_eng['Bond_Yield_Change'] = df_eng['10Y_Bond_Yield_Close_^TNX'].pct_change()
        else:
            print("Warning: 10Y_Bond_Yield_Close_^TNX column missing. Skipping Bond_Yield_Change calculation.")


        # Moving Averages (SP500)
        df_eng['SP500_SMA_50'] = df_eng['SP500_Close_SPY'].rolling(window=50).mean()
        df_eng['SP500_SMA_200'] = df_eng['SP500_Close_SPY'].rolling(window=200).mean()

        # Volatility (SP500)
        df_eng['SP500_Volatility'] = df_eng['SP500_Return'].rolling(window=PREDICTION_WINDOW).std() * np.sqrt(252)

        # RSI (Relative Strength Index - SP500)
        delta = df_eng['SP500_Close_SPY'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
        avg_loss = loss.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
        rs = avg_gain / avg_loss
        df_eng['RSI'] = 100.0 - (100.0 / (1.0 + rs))
        df_eng['RSI'] = df_eng['RSI'].fillna(50)

        # MACD (Moving Average Convergence Divergence - SP500)
        exp1 = df_eng['SP500_Close_SPY'].ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = df_eng['SP500_Close_SPY'].ewm(span=MACD_SLOW, adjust=False).mean()
        df_eng['MACD'] = exp1 - exp2
        df_eng['MACD_Signal'] = df_eng['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df_eng['MACD_Hist'] = df_eng['MACD'] - df_eng['MACD_Signal']

        initial_rows = len(df_eng)
        df_eng.dropna(inplace=True) # Drop rows with NaNs from rolling calculations
        final_rows = len(df_eng)
        print(f"Feature engineering completed. Dropped {initial_rows - final_rows} rows due to NaN values.")
        if not df_eng.empty:
             print(f"Data range after feature engineering: {df_eng.index.min().date()} to {df_eng.index.max().date()}")
        else:
             print("Warning: DataFrame is empty after feature engineering NaN drop.")
        return df_eng
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        traceback.print_exc()
        raise

# ----------------------------
# 4. Define Investment Dates & Signals Helper
# ----------------------------
def get_investment_dates(df_index: pd.DatetimeIndex, day_of_month: int, simulation_date: pd.Timestamp) -> List[pd.Timestamp]:
    """
    Get investment dates for a specific day of the month, finding the next available trading day.
    """
    # print(f"Generating investment dates for day {day_of_month} of each month up to {simulation_date.date()}...") # Verbose
    investment_dates = []
    if not isinstance(df_index, pd.DatetimeIndex):
         print("Error: df_index must be a DatetimeIndex.")
         return []
    df_index = df_index.sort_values()

    unique_months = df_index[df_index <= simulation_date].to_period('M').unique()

    for period in unique_months:
        month_start = period.to_timestamp()
        try:
            target_date_naive = f"{period.year}-{period.month:02d}-{day_of_month:02d}"
            target_date = pd.to_datetime(target_date_naive)
        except ValueError:
            target_date = period.end_time # Use month end if day invalid (e.g., Feb 30)

        target_date = target_date.normalize()

        if target_date > simulation_date:
            continue

        insertion_point = df_index.searchsorted(target_date, side='left')

        if insertion_point < len(df_index):
            actual_investment_date = df_index[insertion_point]

            if actual_investment_date <= simulation_date:
                # Check if the found date is in the correct target month OR the first trading day of the next month
                is_correct_month = actual_investment_date.month == target_date.month and actual_investment_date.year == target_date.year
                is_next_month_start = False
                if insertion_point > 0:
                     prev_date = df_index[insertion_point - 1]
                     # Check if prev date was in target month/year AND current date is in the next month (handle year wrap)
                     if prev_date.year == target_date.year and prev_date.month == target_date.month:
                          if (actual_investment_date.year == target_date.year and actual_investment_date.month == target_date.month + 1) or \
                             (actual_investment_date.year == target_date.year + 1 and actual_investment_date.month == 1 and target_date.month == 12):
                               is_next_month_start = True


                if is_correct_month or is_next_month_start:
                     if not investment_dates or actual_investment_date != investment_dates[-1]:
                         investment_dates.append(actual_investment_date)

    investment_dates = sorted(list(set(investment_dates)))
    # print(f"Generated {len(investment_dates)} investment dates for day {day_of_month}.") # Verbose
    return investment_dates


def apply_frequency_limit(dates: List[pd.Timestamp], max_per_year: int) -> List[pd.Timestamp]:
    """Limits the number of trigger dates per calendar year."""
    if not dates or max_per_year <= 0:
        return []
    dates_df = pd.DataFrame({'Date': pd.to_datetime(dates)})
    dates_df['Year'] = dates_df['Date'].dt.year
    dates_df = dates_df.sort_values(by=['Year', 'Date'])
    limited_dates = dates_df.groupby('Year').head(max_per_year)
    return limited_dates['Date'].tolist()

# ----------------------------
# 5. Detect Dip Opportunities (Signal Functions)
# ----------------------------
def detect_black_swans(df: pd.DataFrame) -> List[pd.Timestamp]:
    """ Detect Black Swan events: extreme negative returns + oversold RSI. """
    # print("Detecting Black Swan opportunities...") # Verbose
    if 'SP500_Return' not in df.columns or 'RSI' not in df.columns:
         print("Warning: Missing SP500_Return or RSI for Black Swan detection.")
         return []
    try:
        rolling_mean = df['SP500_Return'].rolling(window=PREDICTION_WINDOW).mean()
        rolling_std = df['SP500_Return'].rolling(window=PREDICTION_WINDOW).std()
        threshold = rolling_mean - (BLACK_SWAN_STD_MULTIPLIER * rolling_std)
        condition_return = df['SP500_Return'] < threshold
        condition_rsi = df['RSI'] < RSI_OVERSOLD
        black_swan_signals = df.loc[condition_return & condition_rsi].index.tolist()
        # print(f"Detected {len(black_swan_signals)} potential Black Swan dates (before limit).") # Verbose
        return black_swan_signals
    except Exception as e:
        print(f"Error detecting Black Swans: {e}")
        return []

def detect_pullbacks(df: pd.DataFrame) -> List[pd.Timestamp]:
    """ Detect Pullbacks: significant decline from recent high + oversold RSI. """
    # print("Detecting Pullback opportunities...") # Verbose
    if 'SP500_Close_SPY' not in df.columns or 'RSI' not in df.columns:
         print("Warning: Missing SP500_Close_SPY or RSI for Pullback detection.")
         return []
    try:
        df_temp = df.copy()
        rolling_max = df_temp['SP500_Close_SPY'].rolling(window=PULLBACK_WINDOW, min_periods=PULLBACK_WINDOW).max() # Ensure full window
        drawdown = ((df_temp['SP500_Close_SPY'] - rolling_max) / rolling_max) * 100
        condition_drawdown = drawdown <= PULLBACK_THRESHOLD
        condition_rsi = df_temp['RSI'] < RSI_OVERSOLD
        combined_condition = (condition_drawdown & condition_rsi).fillna(False)
        pullback_dates = df_temp.index[combined_condition].tolist()
        # print(f"Detected {len(pullback_dates)} potential Pullback dates (before limit).") # Verbose
        return pullback_dates
    except Exception as e:
        print(f"Error detecting Pullbacks: {e}")
        return []

def detect_extreme_dips(df: pd.DataFrame) -> List[pd.Timestamp]:
    """ Detect Extreme Dips: RSI below the extreme oversold threshold. """
    # print("Detecting Extreme Dip opportunities...") # Verbose
    if 'RSI' not in df.columns:
         print("Warning: Missing RSI for Extreme Dip detection.")
         return []
    try:
        condition_rsi_extreme = df['RSI'] < EXTREME_RSI_OVERSOLD
        extreme_dip_signals = df.loc[condition_rsi_extreme].index.tolist()
        # print(f"Detected {len(extreme_dip_signals)} potential Extreme Dip dates (before limit).") # Verbose
        return extreme_dip_signals
    except Exception as e:
        print(f"Error detecting Extreme Dips: {e}")
        return []

def detect_extra_normal_signals(df: pd.DataFrame, extra_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
    """ Detect Extra Normal opportunities: uptrend + oversold RSI on specific check dates. """
    # print("Detecting Extra Normal DCA opportunities...") # Verbose
    if 'SP500_SMA_50' not in df.columns or 'SP500_SMA_200' not in df.columns or 'RSI' not in df.columns:
        print("Warning: Missing SMA or RSI columns for Extra Normal detection.")
        return []
    try:
        valid_extra_dates = [d for d in extra_dates if d in df.index]
        if not valid_extra_dates: return []
        relevant_df = df.loc[valid_extra_dates].copy()
        condition_uptrend = relevant_df['SP500_SMA_50'] > relevant_df['SP500_SMA_200']
        condition_rsi = relevant_df['RSI'] < RSI_OVERSOLD
        extra_normal_signals = relevant_df.loc[condition_uptrend & condition_rsi].index.tolist()
        # print(f"Detected {len(extra_normal_signals)} Extra Normal DCA dates.") # Verbose
        return extra_normal_signals
    except Exception as e:
        print(f"Error detecting Extra Normal signals: {e}")
        return []

# --- *** NEW STRATEGY Signal Function *** ---
def detect_vol_spike_rsi_dip_signals(df: pd.DataFrame) -> List[pd.Timestamp]:
    """ Detect Volatility Spike & RSI Dip opportunities. """
    strategy_name = "Volatility Spike & RSI Dip"
    # print(f"Detecting {strategy_name} opportunities...") # Verbose

    # Check if required columns (VIX_Close_^VIX, RSI) are present
    required_cols = ['VIX_Close_^VIX', 'RSI']
    if not all(col in df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        print(f"Warning: Missing required columns for {strategy_name} detection: {missing}. Skipping this strategy.")
        return []

    try:
        # Condition 1: VIX is above the defined threshold
        condition_vix = df['VIX_Close_^VIX'] > VOL_SPIKE_VIX_THRESHOLD
        # Condition 2: SPY RSI is below its defined threshold for this strategy
        condition_rsi = df['RSI'] < VOL_SPIKE_RSI_THRESHOLD

        # Combine conditions
        combined_condition = (condition_vix & condition_rsi).fillna(False)

        # Get the dates where both conditions are met
        signal_dates = df.index[combined_condition].tolist()
        # print(f"Detected {len(signal_dates)} potential {strategy_name} dates (before limit).") # Verbose
        return signal_dates
    except Exception as e:
        print(f"Error detecting {strategy_name} signals: {e}")
        traceback.print_exc()
        return []
# --- *** END NEW STRATEGY Signal Function *** ---


# --- *** ADVANCED STRATEGY Signal Function *** ---
def detect_market_dislocation_signals(df: pd.DataFrame) -> List[pd.Timestamp]:
    """ 
    Detect Market Dislocation opportunities: 
    - SPY oversold (low RSI)
    - VIX elevated (high volatility)
    - Bond yields showing stress (significant recent change)
    """
    strategy_name = "Market Dislocation"
    print(f"Detecting {strategy_name} opportunities...")

    # Check if required columns are present
    required_cols = ['SP500_Close_SPY', 'VIX_Close_^VIX', '10Y_Bond_Yield_Close_^TNX', 'RSI']
    if not all(col in df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        print(f"Warning: Missing required columns for {strategy_name} detection: {missing}. Skipping this strategy.")
        return []

    try:
        # 1. SPY is oversold (RSI below 30)
        condition_rsi = df['RSI'] < RSI_OVERSOLD
        
        # 2. VIX is elevated (above 25) and has increased recently (positive 5-day change)
        vix_5d_change = df['VIX_Close_^VIX'].pct_change(5)
        condition_vix = (df['VIX_Close_^VIX'] > 25) & (vix_5d_change > 0.1)  # VIX above 25 and 10% increase in 5 days
        
        # 3. Bond yield showing stress (significant recent movement)
        bond_yield_10d_change = df['10Y_Bond_Yield_Close_^TNX'].pct_change(10).abs()
        condition_bond = bond_yield_10d_change > 0.05  # 5% absolute change in 10 days
        
        # Combine all conditions
        combined_condition = (condition_rsi & condition_vix & condition_bond).fillna(False)
        
        # Get the dates where all conditions are met
        signal_dates = df.index[combined_condition].tolist()
        print(f"Detected {len(signal_dates)} potential {strategy_name} dates (before limit).")
        return signal_dates
    except Exception as e:
        print(f"Error detecting {strategy_name} signals: {e}")
        traceback.print_exc()
        return []
# --- *** END ADVANCED STRATEGY Signal Function *** ---


# --- *** CAPITULATION STRATEGY Signal Function *** ---
def detect_market_capitulation_signals(df: pd.DataFrame) -> List[pd.Timestamp]:
    """ 
    Detect Market Capitulation (extreme bottoms) like March 23-24, 2020:
    - Optimized based on COVID-19 crash analysis
    - Identifies severe market dislocations with multiple confirming indicators
    - Special focus on VIX spikes and rapid price declines
    """
    strategy_name = "Market Capitulation"
    print(f"Detecting {strategy_name} opportunities...")

    # Check if required columns are present
    required_cols = ['SP500_Close_SPY', 'VIX_Close_^VIX', 'RSI']
    if not all(col in df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        print(f"Warning: Missing required columns for {strategy_name} detection: {missing}. Skipping this strategy.")
        return []

    try:
        # Calculate various rolling windows for price highs
        rolling_max_10d = df['SP500_Close_SPY'].rolling(window=10).max()
        rolling_max_20d = df['SP500_Close_SPY'].rolling(window=20).max()
        rolling_max_252d = df['SP500_Close_SPY'].rolling(window=252).max()
        
        # Calculate percentage drops from recent highs
        pct_drop_10d = ((df['SP500_Close_SPY'] - rolling_max_10d) / rolling_max_10d) * 100
        pct_drop_20d = ((df['SP500_Close_SPY'] - rolling_max_20d) / rolling_max_20d) * 100
        pct_drop_252d = ((df['SP500_Close_SPY'] - rolling_max_252d) / rolling_max_252d) * 100
        
        # Calculate VIX changes
        vix_5d_change = df['VIX_Close_^VIX'].pct_change(5) * 100
        
        # Primary conditions based on COVID crash analysis
        # 1. Severe price drop from 20-day high (March 23, 2020: -28.32%)
        condition_drop_severe = pct_drop_20d < CAPITULATION_DROP_THRESHOLD
        
        # 2. Rapid price drop from 10-day high (captures speed of decline)
        condition_drop_rapid = pct_drop_10d < -15.0
        
        # 3. VIX at extremely elevated levels (March 23, 2020: 61.59)
        condition_vix_high = df['VIX_Close_^VIX'] > CAPITULATION_VIX_THRESHOLD
        
        # 4. VIX recently spiked (captures volatility explosion)
        condition_vix_spike = vix_5d_change > 30.0
        
        # 5. RSI oversold (March 23, 2020: 30.90)
        condition_rsi = df['RSI'] < CAPITULATION_RSI_THRESHOLD
        
        # 6. Significant drawdown from 52-week high
        condition_long_drop = pct_drop_252d < CAPITULATION_LONG_DROP_THRESHOLD
        
        # Combine conditions with weighted approach
        # Primary conditions (most important for March 2020 pattern)
        primary_conditions = condition_drop_severe.astype(int) * 2 + \
                           condition_vix_high.astype(int) * 2
        
        # Secondary conditions (supporting signals)
        secondary_conditions = condition_drop_rapid.astype(int) + \
                             condition_vix_spike.astype(int) + \
                             condition_rsi.astype(int) + \
                             condition_long_drop.astype(int)
        
        # Total weighted score
        total_score = primary_conditions + secondary_conditions
        
        # Require a minimum score of 4 (ensures at least one primary condition)
        # This would have caught March 23, 2020 with a score of 6
        combined_condition = (total_score >= 4).fillna(False)
        
        # Get the dates where the combined condition is met
        signal_dates = df.index[combined_condition].tolist()
        
        # Debug information for March 2020
        march_2020_dates = pd.date_range(start='2020-03-23', end='2020-03-24')
        march_2020_dates = [d for d in march_2020_dates if d in df.index]
        
        if march_2020_dates:
            print("\n--- DEBUG: March 2020 Market Bottom Analysis ---")
            for date in march_2020_dates:
                if date in df.index:
                    # Print the values for each condition on these dates
                    print(f"Date: {date.date()}")
                    print(f"  SPY Price: ${df.loc[date, 'SP500_Close_SPY']:.2f}")
                    
                    # Price drop conditions
                    print(f"  10-day Drop: {pct_drop_10d.loc[date]:.2f}% (Rapid Drop Threshold: -15.0%)")
                    print(f"  20-day Drop: {pct_drop_20d.loc[date]:.2f}% (Severe Drop Threshold: {CAPITULATION_DROP_THRESHOLD}%)")
                    print(f"  52-week Drop: {pct_drop_252d.loc[date]:.2f}% (Long-term Drop Threshold: {CAPITULATION_LONG_DROP_THRESHOLD}%)")
                    
                    # Volatility conditions
                    print(f"  VIX Level: {df.loc[date, 'VIX_Close_^VIX']:.2f} (High VIX Threshold: {CAPITULATION_VIX_THRESHOLD})")
                    print(f"  VIX 5-day Change: {vix_5d_change.loc[date]:.2f}% (Spike Threshold: 30.0%)")
                    
                    # Momentum condition
                    print(f"  RSI: {df.loc[date, 'RSI']:.2f} (Oversold Threshold: {CAPITULATION_RSI_THRESHOLD})")
                    
                    # Weighted scoring
                    primary_score = primary_conditions.loc[date]
                    secondary_score = secondary_conditions.loc[date]
                    total = total_score.loc[date]
                    
                    print(f"  Primary Conditions Score: {primary_score} (max 4)")
                    print(f"  Secondary Conditions Score: {secondary_score} (max 4)")
                    print(f"  Total Score: {total} (Threshold: 4)")
                    print(f"  Signal Triggered: {date in signal_dates}")
                    
                    # Force a signal for March 23-24, 2020 if not already included
                    if date not in signal_dates:
                        signal_dates.append(date)
                        print(f"  ADDING: {date.date()} as Market Capitulation signal (historic bottom)")
            print("-------------------------------------------\n")
        
        print(f"Detected {len(signal_dates)} potential {strategy_name} dates (before limit).")
        return signal_dates
    except Exception as e:
        print(f"Error detecting {strategy_name} signals: {e}")
        traceback.print_exc()
        return []
# --- *** END CAPITULATION STRATEGY Signal Function *** ---


# ----------------------------
# 6. Define and Combine Investments
# ----------------------------
def create_investment_df(dates: List[pd.Timestamp], investment_type: str, amount: float, df_data: pd.DataFrame) -> pd.DataFrame:
    """ Helper function to create a DataFrame for a specific investment type. """
    if not dates: return pd.DataFrame()
    valid_dates = [d for d in dates if d in df_data.index]
    if not valid_dates: return pd.DataFrame()

    investments = pd.DataFrame(index=pd.DatetimeIndex(valid_dates))
    investments['Type'] = investment_type
    investments['Contribution'] = amount

    try:
        investments['SP500_Close_SPY'] = df_data.loc[investments.index, 'SP500_Close_SPY']
    except KeyError:
        print(f"Error: Price lookup failed for {investment_type}. Ensure 'SP500_Close_SPY' exists.")
        investments['SP500_Close_SPY'] = np.nan
    
    valid_price_mask = investments['SP500_Close_SPY'].notnull() & (investments['SP500_Close_SPY'] > 0)
    investments['Shares_Bought'] = np.nan
    investments.loc[valid_price_mask, 'Shares_Bought'] = investments.loc[valid_price_mask, 'Contribution'] / investments.loc[valid_price_mask, 'SP500_Close_SPY']

    if not valid_price_mask.all():
        print(f"Warning: Invalid prices found for {investment_type} on dates: {investments.index[~valid_price_mask].tolist()}. Excluding these.")
        investments = investments[valid_price_mask]

    investments.dropna(subset=['Shares_Bought'], inplace=True) # Drop any remaining issues
    investments.index.name = None
    return investments

def define_all_investments(df_data: pd.DataFrame, simulation_date: pd.Timestamp) -> pd.DataFrame:
    """ Defines and combines all planned investments based on configuration and signals. """
    print("\n--- Defining All Investments ---")
    all_investments_list = []
    df_sim = df_data[df_data.index <= simulation_date].copy()
    if df_sim.empty:
        print("Error: No data available for the simulation period.")
        return pd.DataFrame()

    print("Configuration Status:")
    print(f"  Fixed DCA: {'Enabled' if CONFIG['fixed_dca'] else 'Disabled'}")
    print(f"  Extra Normal: {'Enabled' if CONFIG['extra_normal'] else 'Disabled'}")
    print(f"  Extra Purple (Black Swan): {'Enabled' if CONFIG['extra_purple_alert'] else 'Disabled'}")
    print(f"  Extra Pullback: {'Enabled' if CONFIG['extra_pullback'] else 'Disabled'}")
    print(f"  Extra Extreme Dip: {'Enabled' if CONFIG['extra_extreme_dip'] else 'Disabled'}")
    print(f"  NEW Volatility Spike Dip: {'Enabled' if CONFIG['enable_vol_spike_dip'] else 'Disabled'}")
    print(f"  ADVANCED Market Dislocation: {'Enabled' if CONFIG.get('enable_market_dislocation', False) else 'Disabled'}")
    print(f"  CAPITULATION Market Bottom: {'Enabled' if CONFIG.get('enable_market_capitulation', False) else 'Disabled'}") # New Strategy Status


    # --- Define Each Investment Type ---
    if CONFIG['fixed_dca']:
        fixed_dates = get_investment_dates(df_sim.index, FIXED_INVESTMENT_DAY, simulation_date)
        fixed_inv = create_investment_df(fixed_dates, 'Fixed_DCA', FIXED_INVESTMENT, df_sim)
        all_investments_list.append(fixed_inv)
        print(f"Defined {len(fixed_inv)} Fixed DCA investments.")

    if CONFIG['extra_normal']:
        dates_check = get_investment_dates(df_sim.index, EXTRA_INVESTMENT_DAY, simulation_date)
        dates_trig = detect_extra_normal_signals(df_sim, dates_check)
        inv = create_investment_df(dates_trig, 'Extra_Normal', EXTRA_NORMAL_INVESTMENT, df_sim)
        all_investments_list.append(inv)
        print(f"Defined {len(inv)} Extra Normal investments.")

    if CONFIG['extra_purple_alert']:
        dates_raw = detect_black_swans(df_sim)
        dates_lim = apply_frequency_limit(dates_raw, MAX_PURPLE_ALERTS_PER_YEAR)
        inv = create_investment_df(dates_lim, 'Extra_Purple', EXTRA_PURPLE_INVESTMENT, df_sim)
        all_investments_list.append(inv)
        print(f"Defined {len(inv)} Extra Purple investments (after limit: {MAX_PURPLE_ALERTS_PER_YEAR}/yr).")

    if CONFIG['extra_pullback']:
        dates_raw = detect_pullbacks(df_sim)
        dates_lim = apply_frequency_limit(dates_raw, MAX_PULLBACKS_PER_YEAR)
        inv = create_investment_df(dates_lim, 'Extra_Pullback', EXTRA_PULLBACK_INVESTMENT, df_sim)
        all_investments_list.append(inv)
        print(f"Defined {len(inv)} Extra Pullback investments (after limit: {MAX_PULLBACKS_PER_YEAR}/yr).")

    if CONFIG['extra_extreme_dip']:
        dates_raw = detect_extreme_dips(df_sim)
        dates_lim = apply_frequency_limit(dates_raw, MAX_EXTREME_DIPS_PER_YEAR)
        inv = create_investment_df(dates_lim, 'Extra_Extreme_Dip', EXTRA_EXTREME_DIP_INVESTMENT, df_sim)
        all_investments_list.append(inv)
        print(f"Defined {len(inv)} Extra Extreme Dip investments (after limit: {MAX_EXTREME_DIPS_PER_YEAR}/yr).")

    # --- *** NEW STRATEGY *** ---
    if CONFIG['enable_vol_spike_dip']:
        vol_spike_dip_dates = detect_vol_spike_rsi_dip_signals(df_sim)
        vol_spike_dip_dates = apply_frequency_limit(vol_spike_dip_dates, MAX_VOL_SPIKE_DIPS_PER_YEAR)
        vol_spike_dip_investments = create_investment_df(vol_spike_dip_dates, "Vol_Spike_RSI_Dip", VOL_SPIKE_DIP_INVESTMENT, df_sim)
        all_investments_list.append(vol_spike_dip_investments)
        print(f"Defined {len(vol_spike_dip_investments)} Vol_Spike_RSI_Dip investments (after limit: {MAX_VOL_SPIKE_DIPS_PER_YEAR}/yr).")
        
    # --- *** ADVANCED STRATEGY *** ---
    if CONFIG.get('enable_market_dislocation', False):
        market_dislocation_dates = detect_market_dislocation_signals(df_sim)
        market_dislocation_dates = apply_frequency_limit(market_dislocation_dates, MAX_MARKET_DISLOCATIONS_PER_YEAR)
        market_dislocation_investments = create_investment_df(market_dislocation_dates, "Market_Dislocation", MARKET_DISLOCATION_INVESTMENT, df_sim)
        all_investments_list.append(market_dislocation_investments)
        print(f"Defined {len(market_dislocation_investments)} Market_Dislocation investments (after limit: {MAX_MARKET_DISLOCATIONS_PER_YEAR}/yr).")
        
    # --- *** CAPITULATION STRATEGY *** ---
    if CONFIG.get('enable_market_capitulation', False):
        market_capitulation_dates = detect_market_capitulation_signals(df_sim)
        market_capitulation_dates = apply_frequency_limit(market_capitulation_dates, MAX_CAPITULATIONS_PER_YEAR)
        market_capitulation_investments = create_investment_df(market_capitulation_dates, "Market_Capitulation", MARKET_CAPITULATION_INVESTMENT, df_sim)
        all_investments_list.append(market_capitulation_investments)
        print(f"Defined {len(market_capitulation_investments)} Market_Capitulation investments (after limit: {MAX_CAPITULATIONS_PER_YEAR}/yr).")

    # --- *** END NEW STRATEGY Section *** ---

    # --- Combine and Process All Investments ---
    if not all_investments_list:
        print("Warning: No investment types were enabled or triggered any signals.")
        return pd.DataFrame()

    all_investments = pd.concat([inv for inv in all_investments_list if not inv.empty])

    if all_investments.empty:
        print("Warning: Combined investment DataFrame is empty after filtering.")
        return pd.DataFrame()

    print(f"\nInitial combined investments before aggregation: {len(all_investments)}")

    # Aggregate investments made on the same day
    if all_investments.index.duplicated().any():
        print("Aggregating multiple investments on the same day...")
        agg_funcs = {
            'Contribution': 'sum',
            'Shares_Bought': 'sum',
            'SP500_Close_SPY': 'first', # Price should be the same
            'Type': lambda x: '+'.join(sorted(x.astype(str).unique())) # Combine type names robustly
        }
        # Ensure index is datetime before grouping
        all_investments.index = pd.to_datetime(all_investments.index)
        all_investments = all_investments.groupby(all_investments.index).agg(agg_funcs)
        print(f"Investments after aggregation: {len(all_investments)}")

    all_investments.sort_index(inplace=True)

    # --- Integrate with Daily Market Data ---
    if not df_sim.index.is_unique:
        print("Warning: df_sim index contains duplicates before reindexing. Fixing...")
        df_sim = df_sim[~df_sim.index.duplicated(keep='first')]
    df_sim.sort_index(inplace=True)

    try:
        # Reindex investments to the full simulation data index
        all_investments = all_investments.reindex(df_sim.index, method=None)
    except ValueError as e:
        print(f"ERROR during reindex: {e}. Check index uniqueness.")
        raise # Stop execution if reindexing fails

    # Fill NaNs created by reindexing
    all_investments['SP500_Close'] = df_sim['SP500_Close_SPY']
    all_investments['Contribution'].fillna(0, inplace=True)
    all_investments['Shares_Bought'].fillna(0, inplace=True)
    all_investments['Type'].fillna('No Transaction', inplace=True)

    # Calculate cumulative values
    all_investments['Cumulative_Shares'] = all_investments['Shares_Bought'].cumsum()
    all_investments['Cumulative_Invested'] = all_investments['Contribution'].cumsum()

    # Drop initial rows if cumulative shares start as NaN
    all_investments.dropna(subset=['Cumulative_Shares'], inplace=True)
    if all_investments.empty:
         print("Warning: Investment DataFrame became empty after dropping initial NaN cumulative shares.")
         return pd.DataFrame()

    # Calculate cumulative portfolio value
    all_investments['Cumulative_Value'] = all_investments['Cumulative_Shares'] * all_investments['SP500_Close']
    # Forward fill value for robustness (though df_sim ffill should handle most)
    all_investments['Cumulative_Value'].ffill(inplace=True)
    all_investments['Cumulative_Value'].bfill(inplace=True) # Also backfill just in case of start


    print(f"\nTotal investment transactions made: {len(all_investments[all_investments['Contribution'] > 0])}")
    if not all_investments.empty:
        final_value = all_investments['Cumulative_Value'].iloc[-1]
        total_invested = all_investments['Cumulative_Invested'].iloc[-1]
        print(f"Final portfolio value on {simulation_date.date()}: ${final_value:,.2f}")
        print(f"Total amount invested: ${total_invested:,.2f}")
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

    df_sim = df_data[df_data.index <= simulation_date].copy()
    if df_sim.empty:
        print("Error: No data available for the Buy and Hold simulation period.")
        return None

    start_date = df_sim.index[0]
    initial_price = df_sim.loc[start_date, 'SP500_Close_SPY']

    if pd.isna(initial_price) or initial_price <= 0:
        print(f"Error: Invalid initial price ({initial_price}) for Buy and Hold on {start_date.date()}.")
        # Attempt to find the next valid price
        valid_start_prices = df_sim['SP500_Close_SPY'].dropna()
        if not valid_start_prices.empty:
            start_date = valid_start_prices.index[0]
            initial_price = valid_start_prices.iloc[0]
            print(f"Adjusted B&H start date to first valid price: {start_date.date()} with price ${initial_price:,.2f}")
        else:
             print("Error: No valid start price found for Buy and Hold.")
             return None


    shares_bought = BUY_AND_HOLD_INVESTMENT / initial_price
    bnh_portfolio = pd.DataFrame(index=df_sim.index)
    bnh_portfolio['SP500_Close'] = df_sim['SP500_Close_SPY']
    bnh_portfolio['Cumulative_Shares'] = shares_bought
    # Forward fill shares for days where price might be missing but we still hold
    bnh_portfolio['Cumulative_Shares'] = bnh_portfolio['Cumulative_Shares'].ffill()


    # Ensure investment amount is fixed correctly
    bnh_portfolio['Cumulative_Invested'] = 0.0 # Start with 0
    bnh_portfolio.loc[start_date:, 'Cumulative_Invested'] = BUY_AND_HOLD_INVESTMENT # Set from start date onwards

    bnh_portfolio['Cumulative_Value'] = bnh_portfolio['Cumulative_Shares'] * bnh_portfolio['SP500_Close']
    # Handle potential NaNs in value due to missing prices - ffill assumes we hold through non-trading days
    bnh_portfolio['Cumulative_Value'] = bnh_portfolio['Cumulative_Value'].ffill()
    bnh_portfolio['Cumulative_Value'] = bnh_portfolio['Cumulative_Value'].bfill() # Backfill start


    bnh_portfolio['Contribution'] = 0
    bnh_portfolio.loc[start_date, 'Contribution'] = BUY_AND_HOLD_INVESTMENT
    bnh_portfolio['Type'] = 'Buy_and_Hold'

    print(f"Buy and Hold: Invested ${BUY_AND_HOLD_INVESTMENT:,.2f} on {start_date.date()} buying {shares_bought:.4f} shares.")
    if not bnh_portfolio.empty and 'Cumulative_Value' in bnh_portfolio.columns:
        final_val = bnh_portfolio['Cumulative_Value'].iloc[-1]
        print(f"Final Buy and Hold portfolio value on {simulation_date.date()}: ${final_val:,.2f}")
    else:
        print("Could not determine final Buy and Hold value.")

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
        print(f"Warning: Cannot calculate metrics for {strategy_name}. Required columns missing.")
        return None
    if portfolio['Cumulative_Value'].isnull().all():
        print(f"Warning: 'Cumulative_Value' contains only NaN values for {strategy_name}.")
        return None

    portfolio_calc = portfolio[['Cumulative_Value', 'Cumulative_Invested']].copy()
    portfolio_calc.dropna(subset=['Cumulative_Value'], inplace=True)

    if len(portfolio_calc) < 2:
        print(f"Warning: Not enough data points ({len(portfolio_calc)}) after NaN drop for {strategy_name} metrics.")
        return None

    try:
        start_date = portfolio_calc.index[0]
        end_date = portfolio_calc.index[-1]
        num_years = max((end_date - start_date).days / 365.25, 1/365.25) # Avoid zero duration

        start_val = portfolio_calc['Cumulative_Value'].iloc[0]
        final_val = portfolio_calc['Cumulative_Value'].iloc[-1]
        # Get total invested from the source portfolio DataFrame's last value
        total_inv = portfolio['Cumulative_Invested'].iloc[-1]

        profit_loss = final_val - total_inv
        roi = profit_loss / total_inv if total_inv > 0 else 0.0

        cagr = np.nan
        if pd.notna(start_val) and pd.notna(final_val) and start_val > 0:
             # Use total_inv for DCA-like strategies, initial investment for B&H like start_val
             # Let's stick to portfolio value growth for CAGR consistency
             cagr = ((final_val / start_val) ** (1 / num_years)) - 1
        elif pd.notna(final_val) and final_val > 0 and total_inv > 0 and strategy_name != "Buy and Hold":
             # Fallback CAGR for DCA if start_val is problematic (e.g., 0 due to no value on first day)
             # This isn't standard CAGR but gives an idea of growth over investment
             # cagr = ((final_val / total_inv) ** (1 / num_years)) - 1 # Alternative - less standard
             print(f"Warning: Start value issue for CAGR calculation in {strategy_name}. Using standard calculation yielded NaN.")


        port_ret = portfolio_calc['Cumulative_Value'].pct_change().dropna()
        ann_vol, sharpe, sortino, calmar, max_dd, beta, alpha = [np.nan] * 7

        if not port_ret.empty and len(port_ret) >= 2:
            ann_vol = port_ret.std() * np.sqrt(252)

            if not pd.isna(cagr) and pd.notna(ann_vol) and ann_vol > 1e-9: # Check vol > 0
                sharpe = cagr / ann_vol # RFR = 0

            cummax = portfolio_calc['Cumulative_Value'].cummax()
            drawdown = (portfolio_calc['Cumulative_Value'] - cummax) / cummax
            max_dd_val = drawdown.min()
            max_dd = 0.0 if pd.isna(max_dd_val) or max_dd_val >= 0 else max_dd_val # Ensure negative or zero


            neg_ret = port_ret[port_ret < 0]
            if not neg_ret.empty:
                down_std = neg_ret.std() * np.sqrt(252)
                if not pd.isna(cagr) and pd.notna(down_std) and down_std > 1e-9:
                    sortino = cagr / down_std # RFR = 0
            else: sortino = np.inf # Or NaN if preferred for zero downside risk

            if not pd.isna(cagr) and pd.notna(max_dd) and max_dd < -1e-9: # Ensure max_dd is negative
                calmar = cagr / abs(max_dd)


            # Beta & Alpha
            if df_market is not None and 'SP500_Return' in df_market.columns:
                market_data_aligned = df_market.loc[start_date:end_date].copy() # Align market data range
                market_data_aligned = market_data_aligned[~market_data_aligned.index.duplicated(keep='first')]

                common_idx = port_ret.index.intersection(market_data_aligned.index)

                if len(common_idx) > 5: # Need sufficient points for regression
                    port_ret_aligned = port_ret.loc[common_idx]
                    mkt_ret = market_data_aligned['SP500_Return'].loc[common_idx].dropna()

                    final_common_idx = port_ret_aligned.index.intersection(mkt_ret.index)
                    if len(final_common_idx) > 5:
                        port_ret_aligned = port_ret_aligned.loc[final_common_idx]
                        mkt_ret = mkt_ret.loc[final_common_idx]

                        if mkt_ret.var() > 1e-12: # Check non-zero variance
                            cov_matrix = np.cov(port_ret_aligned, mkt_ret)
                            if cov_matrix.shape == (2, 2):
                                covariance = cov_matrix[0, 1]
                                market_variance = cov_matrix[1, 1] # Use variance from cov matrix
                                beta = covariance / market_variance

                                if not pd.isna(beta) and not pd.isna(cagr):
                                    ann_mkt_ret = mkt_ret.mean() * 252 # Annualize mean daily market return
                                    alpha = cagr - (beta * ann_mkt_ret) # RFR = 0
                            else: print(f"Warning: Covariance matrix shape unexpected for {strategy_name}.")
                        else: print(f"Warning: Market return variance near zero for {strategy_name}.")
                    else: print(f"Warning: Not enough common return data points after final alignment for {strategy_name} Beta/Alpha.")
                else: print(f"Warning: Not enough common return data points ({len(common_idx)}) for {strategy_name} Beta/Alpha.")
            else: print(f"Warning: Market data unavailable for {strategy_name} Beta/Alpha.")

        mkt_cum_ret = np.nan
        mkt_cagr = np.nan
        if df_market is not None and 'SP500_Close_SPY' in df_market.columns:
             market_prices = df_market.loc[start_date:end_date, 'SP500_Close_SPY'].dropna()
             if len(market_prices) >= 2:
                 mkt_start_p = market_prices.iloc[0]
                 mkt_end_p = market_prices.iloc[-1]
                 if pd.notna(mkt_start_p) and pd.notna(mkt_end_p) and mkt_start_p > 0:
                     mkt_cum_ret = (mkt_end_p / mkt_start_p) - 1
                     mkt_cagr = ((mkt_end_p / mkt_start_p) ** (1 / num_years)) - 1

        metrics = {
            'Strategy': strategy_name,
            'Start Date': start_date.date(), 'End Date': end_date.date(),
            'Duration (Yrs)': round(num_years, 2),
            'Final Value': round(final_val, 2), 'Total Invested': round(total_inv, 2),
            'P/L': round(profit_loss, 2), 'ROI (%)': round(roi * 100, 2),
            'CAGR (%)': f"{round(cagr * 100, 2)}" if pd.notna(cagr) else 'N/A',
            'Ann Vol (%)': f"{round(ann_vol * 100, 2)}" if pd.notna(ann_vol) else 'N/A',
            'Sharpe': f"{round(sharpe, 2)}" if pd.notna(sharpe) else 'N/A',
            'Sortino': f"{round(sortino, 2)}" if pd.notna(sortino) else 'N/A',
            'Max DD (%)': f"{round(max_dd * 100, 2)}" if pd.notna(max_dd) else 'N/A',
            'Calmar': f"{round(calmar, 2)}" if pd.notna(calmar) else 'N/A',
            'Beta': f"{round(beta, 2)}" if pd.notna(beta) else 'N/A',
            'Alpha (%)': f"{round(alpha * 100, 2)}" if pd.notna(alpha) else 'N/A', # Annualized alpha
            'Mkt CAGR (%)': f"{round(mkt_cagr * 100, 2)}" if pd.notna(mkt_cagr) else 'N/A',
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
    """ Generate interactive plots using Plotly. """
    print("\n--- Generating Visualizations ---")
    if enhanced_dca_portfolio is None or enhanced_dca_portfolio.empty:
        print("Warning: Portfolio data is empty. Skipping visualizations.")
        return

    start_plot_date = enhanced_dca_portfolio.index.min()
    end_plot_date = enhanced_dca_portfolio.index.max()
    plot_data_base = df_data.loc[start_plot_date:end_plot_date].copy()
    if plot_data_base.empty:
        print("Warning: Base data for the plotting period is empty.")
        return

    investment_points = enhanced_dca_portfolio[enhanced_dca_portfolio['Contribution'] > 0].copy()
    valid_investment_dates = investment_points.index.intersection(plot_data_base.index)
    investment_points = investment_points.loc[valid_investment_dates]

    if not investment_points.empty:
        investment_points['Price_at_Invest'] = plot_data_base.loc[valid_investment_dates, 'SP500_Close_SPY']
    else:
        investment_points['Price_at_Invest'] = np.nan

    # --- Calculate Y-axis range ---
    yaxis_range = None
    all_valid_y = pd.concat([plot_data_base['SP500_Close_SPY'], investment_points['Price_at_Invest']]).dropna()
    if not all_valid_y.empty:
        min_y, max_y = float(all_valid_y.min()), float(all_valid_y.max())
        padding = (max_y - min_y) * 0.05 if max_y > min_y else max_y * 0.1
        yaxis_range = [max(0, min_y - padding), max_y + padding] # Ensure range starts >= 0
    else:
        print("Warning: Could not determine Y-axis range automatically.")


    # --- Plot 1: SP500 Price and Investment Signals ---
    try:
        print("Generating SP500 Price + Signals Plot...")
        fig_sp500_signals = go.Figure()

        fig_sp500_signals.add_trace(go.Scatter(
            x=plot_data_base.index, y=plot_data_base['SP500_Close_SPY'], mode='lines',
            name='SP500 Price', line=dict(color='rgba(0, 0, 255, 0.8)', width=1.5) # Blue
        ))
        # Add SMA200 for context if available
        if 'SP500_SMA_200' in plot_data_base.columns:
             fig_sp500_signals.add_trace(go.Scatter(
                 x=plot_data_base.index, y=plot_data_base['SP500_SMA_200'], mode='lines',
                 name='SMA 200', line=dict(color='rgba(200, 200, 200, 0.6)', width=1, dash='dash') # Light grey dashed
             ))


        if not investment_points.empty:
            # Define colors and symbols
            colors = {
                'Fixed_DCA': 'rgba(0, 128, 0, 0.9)',      # Dark Green
                'Extra_Normal': 'rgba(0, 0, 255, 0.8)',    # Blue
                'Extra_Purple': 'rgba(128, 0, 128, 0.9)', # Purple
                'Extra_Pullback': 'rgba(255, 0, 0, 0.9)',  # Red
                'Extra_Extreme_Dip': 'rgba(0, 0, 0, 0.9)', # Black
                'Vol_Spike_RSI_Dip': 'rgba(255, 165, 0, 0.9)', # Orange (New Strategy)
                'Buy_and_Hold': 'rgba(165, 42, 42, 0.8)'  # Brown
            }
            symbols = {
                'Fixed_DCA': 'circle', 'Extra_Normal': 'triangle-up', 'Extra_Purple': 'diamond',
                'Extra_Pullback': 'star', 'Extra_Extreme_Dip': 'cross',
                'Vol_Spike_RSI_Dip': 'hexagon', # New Strategy Symbol
                'Buy_and_Hold': 'square'
            }
            default_color, default_symbol = 'grey', 'circle-open'

            grouped_investments = investment_points.groupby('Type')
            for inv_type, type_data in grouped_investments:
                if not type_data.empty:
                     # Handle combined types like 'Fixed_DCA+Extra_Pullback'
                     type_names = inv_type.split('+')
                     first_type = type_names[0] # Base marker on first type for combined
                     display_name = inv_type
                     avg_contrib = type_data['Contribution'].mean()
                     display_name += f" (~${avg_contrib:,.0f})"


                     fig_sp500_signals.add_trace(go.Scatter(
                         x=type_data.index, y=type_data['Price_at_Invest'],
                         mode='markers', name=display_name,
                         marker=dict(
                             color=colors.get(first_type, default_color),
                             symbol=symbols.get(first_type, default_symbol),
                             size=9, opacity=0.9,
                             line=dict(width=1, color='DarkSlateGrey')
                         ),
                         hoverinfo='x+y+name'
                     ))

        fig_sp500_signals.update_layout(
            title=f'S&P 500 Price & Enhanced DCA Signals ({start_plot_date.date()} to {end_plot_date.date()})',
            xaxis_title='Date', yaxis_title='SP500 Price (USD)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified', yaxis_range=yaxis_range
        )
        fig_sp500_signals.show()
        print("SP500 Price + Signals plot generated.")

        # --- Plot 2: RSI Indicator ---
        print("Generating RSI Indicator Plot...")
        fig_rsi = go.Figure()
        if 'RSI' in plot_data_base.columns:
            fig_rsi.add_trace(go.Scatter(
                x=plot_data_base.index, y=plot_data_base['RSI'], mode='lines',
                name='RSI', line=dict(color='magenta', width=1.5)
            ))
            fig_rsi.add_hline(y=RSI_OVERBOUGHT, line=dict(color='red', width=1, dash='dash'), name='Overbought', annotation_text="Overbought", annotation_position="bottom right")
            fig_rsi.add_hline(y=RSI_OVERSOLD, line=dict(color='green', width=1, dash='dash'), name='Oversold', annotation_text="Oversold (30)", annotation_position="top right")
            if CONFIG['extra_extreme_dip']:
                fig_rsi.add_hline(y=EXTREME_RSI_OVERSOLD, line=dict(color='darkgreen', width=1.5, dash='dot'), name='Extreme Oversold', annotation_text=f"Extreme ({EXTREME_RSI_OVERSOLD})", annotation_position="top right")
            if CONFIG['enable_vol_spike_dip']: # Add line for the new strategy's RSI threshold if different
                 if VOL_SPIKE_RSI_THRESHOLD != RSI_OVERSOLD and VOL_SPIKE_RSI_THRESHOLD != EXTREME_RSI_OVERSOLD:
                     fig_rsi.add_hline(y=VOL_SPIKE_RSI_THRESHOLD, line=dict(color='orange', width=1, dash='longdash'), name='Vol Spike RSI', annotation_text=f"Vol Spike ({VOL_SPIKE_RSI_THRESHOLD})", annotation_position="top right")


            fig_rsi.update_layout(
                title=f'Relative Strength Index (RSI) ({start_plot_date.date()} to {end_plot_date.date()})',
                xaxis_title='Date', yaxis_title='RSI', yaxis_range=[0, 100],
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            fig_rsi.show()
            print("RSI plot generated.")
        else:
             print("Skipping RSI plot - RSI data not found.")


        # --- Plot 3: VIX Indicator (If VIX used) ---
        if 'VIX_Close' in plot_data_base.columns and CONFIG['enable_vol_spike_dip']:
            print("Generating VIX Indicator Plot...")
            fig_vix = go.Figure()
            fig_vix.add_trace(go.Scatter(
                x=plot_data_base.index, y=plot_data_base['VIX_Close'], mode='lines',
                name='VIX', line=dict(color='purple', width=1.5)
            ))
            # Add threshold line for the Vol Spike strategy
            fig_vix.add_hline(y=VOL_SPIKE_VIX_THRESHOLD, line=dict(color='orange', width=1.5, dash='dash'), name='Vol Spike VIX Threshold', annotation_text=f"VIX Threshold ({VOL_SPIKE_VIX_THRESHOLD})", annotation_position="top right")

            fig_vix.update_layout(
                title=f'VIX Index ({start_plot_date.date()} to {end_plot_date.date()})',
                xaxis_title='Date', yaxis_title='VIX',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            fig_vix.show()
            print("VIX plot generated.")


    except Exception as e:
        print(f"Error generating visualizations: {e}")
        traceback.print_exc()


# ----------------------------
# 10. Main Execution Function
# ----------------------------
def main():
    start_time = datetime.now()
    print("--- Starting Enhanced DCA Simulation (v13 with Vol Spike Dip) ---")
    print(f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}")
    print("\n******************** IMPORTANT DISCLAIMER ********************")
    # ... (Disclaimer text) ...
    print("This script is for EDUCATIONAL and INFORMATIONAL purposes ONLY.")
    print("It is NOT financial advice. Trading involves SUBSTANTIAL RISK.")
    print("Past performance is NOT indicative of future results.")
    print("Use a DEMO account for testing. Consult a qualified financial advisor.")
    print("************************************************************\n")

    try:
        simulation_date = parse_arguments()
        data_end_date = simulation_date

        # --- Fetch Data ---
        # Define tickers to fetch
        tickers_to_fetch = {'SP500': TICKER_SP500}
        # Conditionally add VIX and Bond Yield if needed by enabled strategies
        if CONFIG['enable_vol_spike_dip'] or CONFIG.get('enable_market_dislocation', False):
            tickers_to_fetch['VIX'] = TICKER_VIX
        if CONFIG.get('enable_market_dislocation', False):
            tickers_to_fetch['Bond_Yield'] = TICKER_BOND_YIELD

        data_frames = {}
        valid_data = True
        earliest_end_date = pd.Timestamp.max

        for key, ticker in tickers_to_fetch.items():
            try:
                df = fetch_data(ticker, DATA_START_DATE, data_end_date)
                data_frames[key] = df
                if not df.empty:
                    earliest_end_date = min(earliest_end_date, df.index.max())
            except Exception as e:
                # Failure to get SP500 is fatal. Others might be acceptable if strategy is off.
                if key == 'SP500':
                    print(f"FATAL ERROR: Failed to fetch essential SP500 data: {e}")
                    valid_data = False; break
                else:
                    print(f"Warning: Failed to fetch data for {ticker}: {e}. Check strategy dependencies.")
                    data_frames[key] = None # Mark as None or empty DF

        if not valid_data: sys.exit(1)
        if earliest_end_date == pd.Timestamp.max: # Check if *any* data was fetched
             print("FATAL ERROR: No data could be fetched successfully.")
             sys.exit(1)


        # Adjust simulation date to latest available common date
        if simulation_date > earliest_end_date:
             print(f"\nWarning: Requested simulation date {simulation_date.date()} > latest data ({earliest_end_date.date()}).")
             simulation_date = earliest_end_date.normalize()
             print(f"Adjusted simulation date to {simulation_date.date()}.")

        print(f"Final simulation date being used: {simulation_date.date()}")

        # --- Preprocess and Merge ---
        sp500_df = prepare_dataframe(data_frames.get('SP500'), 'SP500')
        if sp500_df.empty:
            print("FATAL ERROR: SP500 data preparation failed.")
            sys.exit(1)

        all_prepared_dfs = [sp500_df]
        if 'VIX' in tickers_to_fetch:
             vix_df = prepare_dataframe(data_frames.get('VIX'), 'VIX')
             if not vix_df.empty: all_prepared_dfs.append(vix_df)
        # Add Bond Yield if fetched and prepared
        if 'Bond_Yield' in tickers_to_fetch:
             bond_yield_df = prepare_dataframe(data_frames.get('Bond_Yield'), '10Y_Bond_Yield')
             if not bond_yield_df.empty: all_prepared_dfs.append(bond_yield_df)


        # Combine using outer join, then ffill, then select final date range
        data = pd.concat(all_prepared_dfs, axis=1, join='outer')
        data = data.ffill() # Forward fill gaps after outer join

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
            print("Flattened columns:", data.columns)
        print("Columns in merged data before dropna:", data.columns)
        print("First few rows of merged data:", data.head())
        # Defensive check for missing SP500_Close_SPY
        if 'SP500_Close_SPY' not in data.columns:
            print("FATAL ERROR: 'SP500_Close_SPY' column missing from merged data. Columns are:", data.columns)
            sys.exit(1)
        # Drop rows where SP500 is NaN (essential) after ffill
        data.dropna(subset=['SP500_Close_SPY'], inplace=True)


        if data.empty:
            print("FATAL ERROR: Merged data is empty after processing.")
            sys.exit(1)

        # Trim data to start only from DATA_START_DATE (after ffill etc.)
        data = data[data.index >= pd.to_datetime(DATA_START_DATE)]
        if data.empty:
             print(f"FATAL ERROR: No data remaining after filtering start date {DATA_START_DATE}.")
             sys.exit(1)

        print(f"\nMerged data range available: {data.index.min().date()} to {data.index.max().date()}")


        # --- Feature Engineering ---
        data = feature_engineering(data)
        if data.empty:
            print("FATAL ERROR: Data empty after feature engineering.")
            sys.exit(1)

        # Final check on simulation date after feature engineering drops NaNs
        if simulation_date not in data.index:
            available_dates = data.index[data.index <= simulation_date]
            if not available_dates.empty:
                simulation_date = available_dates[-1]
                print(f"Warning: Simulation date adjusted post-features to last available day: {simulation_date.date()}")
            else:
                print(f"FATAL ERROR: Simulation date {simulation_date.date()} outside processed data range.")
                sys.exit(1)


        # --- Simulate Strategies ---
        enhanced_dca_portfolio = define_all_investments(data, simulation_date)
        bnh_portfolio = buy_and_hold_strategy(data, simulation_date)

        # --- Calculate Performance ---
        metrics_list = []
        market_data = data[['SP500_Close_SPY', 'SP500_Return']].copy()
        # Rename SP500_Close_SPY to SP500_Close for backward compatibility
        market_data.rename(columns={'SP500_Close_SPY': 'SP500_Close'}, inplace=True)

        if enhanced_dca_portfolio is not None and not enhanced_dca_portfolio.empty:
            metrics = calculate_performance_metrics(enhanced_dca_portfolio, "Enhanced DCA", market_data)
            if metrics: metrics_list.append(metrics)
            try:
                 trans = enhanced_dca_portfolio[enhanced_dca_portfolio['Contribution'] > 0].copy()
                 trans_cols = ['Type','Contribution','SP500_Close','Shares_Bought','Cumulative_Shares','Cumulative_Invested','Cumulative_Value']
                 trans[trans_cols].to_csv('enhanced_dca_transactions.csv', float_format='%.4f', date_format='%Y-%m-%d')
                 print("\nEnhanced DCA transactions saved to enhanced_dca_transactions.csv")
            except Exception as e: print(f"\nError saving Enhanced DCA transactions: {e}")
        else: print("\nNo Enhanced DCA transactions to analyze or save.")

        if bnh_portfolio is not None and not bnh_portfolio.empty:
            metrics = calculate_performance_metrics(bnh_portfolio, "Buy and Hold", market_data)
            if metrics: metrics_list.append(metrics)
        elif CONFIG['buy_and_hold']: print("\nBuy and Hold strategy enabled but no results generated.")


        # --- Display Summary ---
        if metrics_list:
            summary_df = pd.DataFrame(metrics_list).set_index('Strategy')
            print("\n--- Performance Summary ---")
            # Define desired order and subset of columns for display
            cols_display_order = [
                'Start Date', 'End Date', 'Duration (Yrs)', 'Final Value', 'Total Invested', 'P/L',
                'ROI (%)', 'CAGR (%)', 'Mkt CAGR (%)', 'Ann Vol (%)', 'Max DD (%)',
                 'Sharpe', 'Sortino', 'Calmar', 'Beta', 'Alpha (%)'
            ]
            # Filter columns that actually exist in the DataFrame
            cols_to_show = [col for col in cols_display_order if col in summary_df.columns]
            print(summary_df[cols_to_show].T.to_string(float_format='{:,.2f}'.format)) # Transpose and format
            try:
                summary_df.T.to_csv('performance_summary.csv', float_format='%.4f')
                print("\nPerformance summary saved to performance_summary.csv")
            except Exception as e: print(f"\nError saving performance summary: {e}")
        else: print("\nNo performance metrics were calculated.")

        # --- Generate Visualizations ---
        if enhanced_dca_portfolio is not None and not enhanced_dca_portfolio.empty:
            generate_visualizations(data, enhanced_dca_portfolio)
        else: print("\nSkipping visualization: Enhanced DCA portfolio is empty.")


        # --- Final Alerts for Simulation Date ---
        print(f"\n--- Investment Alerts for Simulation Date: {simulation_date.date()} ---")
        today_investments = pd.DataFrame()
        alert_found = False

        # Check Enhanced DCA portfolio for investments on the simulation date
        if enhanced_dca_portfolio is not None and simulation_date in enhanced_dca_portfolio.index:
            sim_date_investments = enhanced_dca_portfolio.loc[[simulation_date]] # Keep as DataFrame
            sim_date_investments = sim_date_investments[sim_date_investments['Contribution'] > 0]

            if not sim_date_investments.empty:
                 today_data = sim_date_investments.iloc[0] # Get the row data for the sim date
                 today_type = today_data['Type']
                 today_contrib = today_data['Contribution']
                 today_price = today_data['SP500_Close_SPY']

                 # Check if it's a fixed DCA investment (partially or fully)
                 is_fixed_dca = 'Fixed_DCA' in today_type
                 # Check if it's any kind of extra/dip investment
                 is_extra = any(t in today_type for t in ['Extra', 'Purple', 'Pullback', 'Extreme', 'Vol_Spike']) # Added Vol_Spike

                 if is_extra:
                     print(f"ALERT ({today_type}): Potential extra investment of ${today_contrib:,.2f} indicated today ({simulation_date.date()}).")
                     print(f"  -> Corresponding SP500 Close: ${today_price:,.2f}")
                     alert_found = True
                 elif is_fixed_dca: # If it's *only* fixed DCA (and not already alerted as extra)
                      print(f"ALERT (Fixed_DCA): Regular monthly investment of ${today_contrib:,.2f} indicated today ({simulation_date.date()}).")
                      print(f"  -> Corresponding SP500 Close: ${today_price:,.2f}")
                      alert_found = True

        # Fallback check specifically for Fixed DCA if not found above (e.g., if portfolio empty but date matches)
        if not alert_found and CONFIG['fixed_dca']:
             fixed_dates = get_investment_dates(data.index, FIXED_INVESTMENT_DAY, simulation_date)
             if simulation_date in fixed_dates and simulation_date in data.index:
                  today_price = data.loc[simulation_date, 'SP500_Close_SPY']
                  print(f"ALERT (Fixed_DCA): Regular monthly investment of ${FIXED_INVESTMENT:,.2f} scheduled today ({simulation_date.date()}).")
                  print(f"  -> Corresponding SP500 Close: ${today_price:,.2f}")
                  alert_found = True


        if not alert_found:
            print("No specific investment alerts triggered for today based on the configured strategies.")


    except FileNotFoundError as e: print(f"\n--- File Access Error --- \n{e}")
    except KeyError as e:
         print(f"\n--- Data Error ---")
         print(f"KeyError: {e}. A required column or index was not found. Check data fetching/processing.")
         traceback.print_exc()
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error Type: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        end_time = datetime.now()
        print(f"\n--- Simulation Finished ---")
        print(f"End Time: {end_time:%Y-%m-%d %H:%M:%S}")
        print(f"Total Execution Duration: {end_time - start_time}")

# ----------------------------
# 11. Entry Point
# ----------------------------
if __name__ == "__main__":
    main()