"""
Markov-switching GARCH model to detect BTC/USD volatility regimes

Applies Markov-Switching to the GJR-GARCH(1, 1) forecast of hourly BTC/USD volatilities.

This project aims to implement the findings of:
Ardia, D., Bluteau, K., & Rüede, M. (2019). Regime changes in Bitcoin GARCH volatility dynamics
https://doi.org/10.1016/j.frl.2018.08.009
"""

import sys
import os
import time
from pathlib import Path
import csv

# Utils
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

# statsmodels for MarkovAutoregression
import statsmodels.api as sm

# Plotting tools
import matplotlib.pyplot as plt
from matplotlib import style, cm
style.use('dark_background')

# ccxt as our currency-fetching library
import ccxt

# arch for the GARCH model
from arch import arch_model
from arch.univariate import GARCH


# ============================================================================
# Configuration Constants
# ============================================================================

# Default data file path
DEFAULT_DATA_FILE = r'./data/BTCUSDT_1h_20190923_20260101.csv'


# ============================================================================
# Data Fetching Functions
# ============================================================================

def fetch_ohlcv(exchange, since, end_time, timeframe='1h', hold=30):
    """
    Fetch BTC/USDT OHLCV from Binance
    
    Parameters:
    -----------
    exchange : ccxt.Exchange
        Binance exchange instance
    since : int
        Starting timestamp in milliseconds
    end_time : int
        Ending timestamp in milliseconds (None for current time)
    timeframe : str
        Timeframe: '1h' for 1-hour, '5m' for 5-minute
    hold : int
        Retry delay in seconds
        
    Returns:
    --------
    list : List of OHLCV candles
    """
    data = []
    now = end_time if end_time else exchange.milliseconds()
    
    # Calculate timeframe in milliseconds
    if timeframe == '1h':
        timeframe_ms = 60 * 60 * 1000  # 1 hour
    elif timeframe == '5m':
        timeframe_ms = 5 * 60 * 1000  # 5 minutes
    else:
        timeframe_ms = 60 * 60 * 1000  # default to 1 hour
    
    while since < now:
        try:
            print(exchange.milliseconds(), 'Fetching candles starting from', exchange.iso8601(since))
            ohlcvs = exchange.fetch_ohlcv('BTC/USDT', timeframe, since, limit=1000)
            print(exchange.milliseconds(), 'Fetched', len(ohlcvs), 'candles')
            if len(ohlcvs) > 0:
                first = ohlcvs[0][0]
                last = ohlcvs[-1][0]
                print('First candle epoch', first, exchange.iso8601(first))
                print('Last candle epoch', last, exchange.iso8601(last))
                
                # Filter out candles beyond end_time
                filtered_ohlcvs = [c for c in ohlcvs if c[0] < now]
                if len(filtered_ohlcvs) > 0:
                    data += filtered_ohlcvs
                
                # Check if we've reached the end time
                if last >= now:
                    # Add the last candle if it's exactly at end_time
                    if last == now:
                        data.append(ohlcvs[-1])
                    break
                    
                # If no new data was added, break to avoid infinite loop
                if len(filtered_ohlcvs) == 0:
                    break
                    
                since = ohlcvs[-1][0] + timeframe_ms
                print('Total length:' + str(len(data)))

        except (ccxt.ExchangeError, ccxt.AuthenticationError, 
                ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
            print('Got an error', type(error).__name__, error.args, ', retrying in', hold, 'seconds...')
            time.sleep(hold)
            
    print("Gathered " + str(len(data)) + " datapoints.")
    return data


def write_to_csv(filename, data, data_dir='./data', use_timestamp=False):
    """
    Write OHLCV data to CSV file
    
    Parameters:
    -----------
    filename : str
        Output CSV filename
    data : list
        List of OHLCV candles
    data_dir : str
        Directory to save CSV file
    use_timestamp : bool
        If True, use 'timestamp' column (for 5m data), else use 'dates' column
    """
    timestamps = []
    dates = []
    open_data = []
    high_data = []
    low_data = []
    close_data = []
    volume_data = []
    
    for candle in data:
        dt = datetime.fromtimestamp(candle[0] / 1000.0)
        timestamps.append(dt.strftime('%Y-%m-%d %H:%M:%S+00:00'))
        dates.append(dt.strftime('%Y-%m-%d %H:%M'))
        open_data.append(candle[1])
        high_data.append(candle[2])
        low_data.append(candle[3])
        close_data.append(candle[4])
        volume_data.append(candle[5] if len(candle) > 5 else 0)
        
    if use_timestamp:
        csv_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_data, 
            "high": high_data, 
            "low": low_data, 
            "close": close_data,
            "volume": volume_data
        })
    else:
        csv_df = pd.DataFrame({
            "open": open_data, 
            "high": high_data, 
            "low": low_data, 
            "close": close_data,
            "volume": volume_data,
            "dates": dates
        })
    
    # Ensure data directory exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(Path(data_dir, filename), index=False)
    print(f"Data saved to {Path(data_dir, filename)}")
    print(f"Total rows: {len(csv_df)}")


def update_data_from_binance(from_datetime='2023-02-03 00:00:00',
                              to_datetime='2026-02-02 23:59:59',
                              timeframe='1h',
                              output_filename='btc_usdt_1h.csv',
                              data_dir='./data'):
    """
    Fetch BTC/USDT data from Binance and save to CSV.
    
    Parameters:
    -----------
    from_datetime : str
        Starting datetime string (format: 'YYYY-MM-DD HH:MM:SS')
    to_datetime : str
        Ending datetime string (format: 'YYYY-MM-DD HH:MM:SS'), None for current time
    timeframe : str
        Timeframe: '1h' for 1-hour, '5m' for 5-minute
    output_filename : str
        Output CSV filename
    data_dir : str
        Directory to save CSV file
    """
    # Initialize exchange
    exchange = ccxt.binance({
        'rateLimit': 10000,
        'enableRateLimit': True,
    })
    
    from_timestamp = exchange.parse8601(from_datetime)
    
    if to_datetime:
        to_timestamp = exchange.parse8601(to_datetime)
    else:
        to_timestamp = None
    
    print(f"Fetching {timeframe} data from {from_datetime} to {to_datetime or 'now'}")
    print("This may take several minutes...")
    
    # Fetch data
    data = fetch_ohlcv(exchange, from_timestamp, to_timestamp, timeframe=timeframe)
    
    # Write to CSV (use dates format for 1h data)
    use_timestamp = False
    write_to_csv(output_filename, data, data_dir, use_timestamp=use_timestamp)


# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def refresh_data(data_file=DEFAULT_DATA_FILE):
    """
    Parse CSV to DataFrame
    
    Parameters:
    -----------
    data_file : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame : DataFrame with OHLCV data and 'dates' column
    """
    df = pd.read_csv(Path(data_file))
    df.reset_index(inplace=True, drop=True)
    
    # Handle different CSV formats
    if 'open_time' in df.columns:
        # New format: open_time column (from download_binance_klines.py)
        # Convert open_time to dates format
        df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
        df['dates'] = df['open_time'].dt.strftime('%Y-%m-%d %H:%M')
    elif 'timestamp' in df.columns:
        # Alternative format: timestamp column
        df['dates'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    elif 'dates' not in df.columns and 'date' in df.columns:
        # Alternative format: date column
        df['dates'] = df['date']
    
    return df


def resample_data(df):
    """
    Resample 1-hour candles to 2-hour candles (only keep even hours)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'dates' column
        
    Returns:
    --------
    pd.DataFrame : Resampled DataFrame
    """
    dta_list = []
    
    for i in range(len(df['dates'])):
        if datetime.strptime(df['dates'].iloc[i], "%Y-%m-%d %H:%M").hour % 2 == 0:
            dta_list.append(df.iloc[i])
    
    dta = pd.DataFrame(dta_list)
    dta.reset_index(inplace=True, drop=True)
    return dta


def normalize_data(data_file=DEFAULT_DATA_FILE):
    """
    Transform DataFrame to log-returns, close price, and date format
    
    Parameters:
    -----------
    data_file : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame : DataFrame with returns, close, returns_abs, date columns
    """
    dta = resample_data(refresh_data(data_file))
    dta = pd.DataFrame({
        'returns': dta.close,
        'close': dta.close,
        'returns_abs': dta.close,
        'date': dta.dates
    })
    
    dta['returns'] = np.log1p(dta['returns'].pct_change())
    dta['returns_abs'] = dta['returns'].abs()
    
    return dta.iloc[1:]


# ============================================================================
# GARCH Model Functions
# ============================================================================

def get_forecasts(show_vis=False, data_file=DEFAULT_DATA_FILE):
    """
    Initialize & fit GJR-GARCH(1, 1) model on full dataset.
    Plot GARCH-Volatility against log-returns if requested.
    
    Parameters:
    -----------
    show_vis : bool
        Whether to show visualization
    data_file : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame : DataFrame with forecast_vol column added
    """
    df = normalize_data(data_file)
    
    # Annual vol (rolling window)
    df['btc_vol'] = df['returns'].rolling(window=12 * 10, center=False).std() * np.sqrt(365 * 12)
    df.dropna(inplace=True)
    
    # Initialize/fit GJR-GARCH(1, 1) model with Skewed Student-T distribution
    model = arch_model(
        df['returns'], 
        p=1,      # GARCH lag
        o=1,      # GJR (asymmetric) term
        q=1,      # ARCH lag
        mean='zero', 
        vol='GARCH', 
        dist='skewt', 
        rescale=True
    )
    res = model.fit(disp='off')
    
    # Calculate forecasted volatility
    df['forecast_vol'] = 0.1 * np.sqrt(
        res.params['omega'] + 
        res.params['alpha[1]'] * res.resid**2 + 
        res.conditional_volatility**2 * res.params['beta[1]']
    )
    
    # Print summary and plot if requested
    if show_vis:
        print(res.summary())
        
        plt.figure(figsize=(17, 9))
        plt.plot(df['returns'], label='Returns')
        plt.plot(df['forecast_vol'], label='Predicted Volatility')
        plt.title('Vol Prediction - Rolling fcast', fontsize=20)
        plt.legend(fontsize=16)
        plt.show()
        
    return df


# ============================================================================
# Markov-Switching Model Classes
# ============================================================================

class MarkovSwitch2:
    """
    Markov-Switching model with 2 regimes (Low-Vol and High-Vol)
    """
    
    def __init__(self):
        self.k = 2
        self.model = None
        self.result = None
    
    def _define_params(self, dta):
        """Define the model parameters"""
        self.model = sm.tsa.MarkovRegression(
            dta, 
            k_regimes=self.k,
            trend='n',  # 'n' = no trend (was 'nc' in older versions) 
            switching_variance=True
        )
        print("Model Parameter's defined.")

    def fit_model(self, dta):
        """
        Fit the Markov-Switching model
        
        Parameters:
        -----------
        dta : pd.DataFrame
            DataFrame with 'forecast_vol' column
        """
        np.random.seed(33)
        
        self._define_params(dta.forecast_vol)

        # Fit model
        self.result = self.model.fit()

        # Print summary
        print(self.result.summary())

        # Get probabilities of the respective regimes
        dta['FilteredProbs1'] = self.result.filtered_marginal_probabilities[0]
        dta['SmoothedProbs1'] = self.result.smoothed_marginal_probabilities[0]
        dta['FilteredProbs2'] = self.result.filtered_marginal_probabilities[1]
        dta['SmoothedProbs2'] = self.result.smoothed_marginal_probabilities[1]
        
    def plot_filtered(self, dta, save_dir=''):
        """
        Plot filtered probabilities
        
        Parameters:
        -----------
        dta : pd.DataFrame
            DataFrame with probabilities and data
        save_dir : str
            Directory to save plot
        """
        dta = dta.copy()
        dta['date'] = range(1, len(dta) + 1)

        # Create the figure
        f1, ax = plt.subplots(self.k + 2, figsize=(19, 15))

        # Plot regime probabilities
        ax[0].plot(dta.date, dta.FilteredProbs1, label='Low-vol')
        ax[0].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[0].set_ylabel('Probabilities', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[0].legend(loc='best')

        ax[1].plot(dta.date, dta.FilteredProbs2, label='High-vol')
        ax[1].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[1].set_ylabel('Probabilities', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[1].legend(loc='best')
        
        # Plot returns
        ax[self.k].plot(dta.date, dta.returns, label='Returns', color='gold', linewidth=2)
        ax[self.k].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k].set_ylabel('Returns', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k].legend(loc='best')

        # Plot close price
        ax[self.k+1].plot(dta.date, dta.close, label='Close Price', color='gold', linewidth=2)
        ax[self.k+1].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k+1].set_ylabel('Close price', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k+1].legend(loc='best')

        plt.grid(linestyle='dotted')
        plt.subplots_adjust(left=0.1, bottom=0.20, right=0.95, top=0.95, wspace=0.2, hspace=0)
        f1.suptitle('Filtered Probabilities of GARCH-Vol for BTC/USD')

        # Save PNG
        if save_dir:
            Path(save_dir + '/visuals').mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir + '/visuals/MSGARCH_2_Filtered_BTCUSD.png')

        plt.show()
            
    def plot_smoothed(self, dta, save_dir=''):
        """
        Plot smoothed probabilities
        
        Parameters:
        -----------
        dta : pd.DataFrame
            DataFrame with probabilities and data
        save_dir : str
            Directory to save plot
        """
        dta = dta.copy()
        dta['date'] = range(1, len(dta) + 1)

        # Create the figure
        f1, ax = plt.subplots(self.k + 2, figsize=(19, 15))

        # Plot regime probabilities
        ax[0].plot(dta.date, dta.SmoothedProbs1, label='Low-vol')
        ax[0].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[0].set_ylabel('Probabilities', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[0].legend(loc='best')

        ax[1].plot(dta.date, dta.SmoothedProbs2, label='High-vol')
        ax[1].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[1].set_ylabel('Probabilities', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[1].legend(loc='best')
        
        # Plot returns
        ax[self.k].plot(dta.date, dta.returns, label='Returns', color='gold', linewidth=2)
        ax[self.k].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k].set_ylabel('Returns', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k].legend(loc='best')

        # Plot close price
        ax[self.k+1].plot(dta.date, dta.close, label='Close Price', color='gold', linewidth=2)
        ax[self.k+1].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k+1].set_ylabel('Close price', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k+1].legend(loc='best')

        plt.grid(linestyle='dotted')
        plt.subplots_adjust(left=0.1, bottom=0.20, right=0.95, top=0.95, wspace=0.2, hspace=0)
        f1.suptitle('Smoothed Probabilities of GARCH-Vol for BTC/USD')

        # Save PNG
        if save_dir:
            Path(save_dir + '/visuals').mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir + '/visuals/MSGARCH_2_Smoothed_BTCUSD.png')

        plt.show()


class MarkovSwitch3:
    """
    Markov-Switching model with 3 regimes (Low-Vol, Medium-Vol, High-Vol)
    """
    
    def __init__(self):
        self.k = 3
        self.model = None
        self.result = None
    
    def _define_params(self, dta):
        """Define the model parameters"""
        self.model = sm.tsa.MarkovRegression(
            dta, 
            k_regimes=self.k,
            trend='n',  # 'n' = no trend (was 'nc' in older versions) 
            switching_variance=True
        )
        print("Model Parameter's defined.")

    def fit_model(self, dta):
        """
        Fit the Markov-Switching model
        
        Parameters:
        -----------
        dta : pd.DataFrame
            DataFrame with 'forecast_vol' column
        """
        np.random.seed(33)
        
        self._define_params(dta.forecast_vol)

        # Fit the model
        self.result = self.model.fit(maxiter=200)

        # Print summary
        print(self.result.summary())

        # Get probabilities of the respective regimes
        dta['FilteredProbs1'] = self.result.filtered_marginal_probabilities[0]
        dta['SmoothedProbs1'] = self.result.smoothed_marginal_probabilities[0]
        
        dta['FilteredProbs2'] = self.result.filtered_marginal_probabilities[1]
        dta['SmoothedProbs2'] = self.result.smoothed_marginal_probabilities[1]
        
        dta['FilteredProbs3'] = self.result.filtered_marginal_probabilities[2]
        dta['SmoothedProbs3'] = self.result.smoothed_marginal_probabilities[2]

    def plot_filtered(self, dta, save_dir=''):
        """
        Plot filtered probabilities
        
        Parameters:
        -----------
        dta : pd.DataFrame
            DataFrame with probabilities and data
        save_dir : str
            Directory to save plot
        """
        dta = dta.copy()
        dta['date'] = range(1, len(dta) + 1)

        # Create the figure
        f1, ax = plt.subplots(self.k + 2, figsize=(19, 15))

        # Plot regime probabilities
        ax[0].plot(dta.date, dta.FilteredProbs1, label='Low-Vol')
        ax[0].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[0].set_ylabel('Probabilities', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[0].legend(loc='best')

        ax[1].plot(dta.date, dta.FilteredProbs2, label='Medium-Vol')
        ax[1].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[1].set_ylabel('Probabilities', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[1].legend(loc='best')

        ax[2].plot(dta.date, dta.FilteredProbs3, label='High-Vol')
        ax[2].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[2].set_ylabel('Probabilities', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[2].legend(loc='best')
        
        # Plot returns
        ax[self.k].plot(dta.date, dta.returns, label='Returns', color='gold', linewidth=2)
        ax[self.k].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k].set_ylabel('Returns', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k].legend(loc='best')

        # Plot close price
        ax[self.k+1].plot(dta.date, dta.close, label='Close Price', color='gold', linewidth=2)
        ax[self.k+1].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k+1].set_ylabel('Close price', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k+1].legend(loc='best')

        plt.grid(linestyle='dotted')
        plt.subplots_adjust(left=0.1, bottom=0.20, right=0.95, top=0.95, wspace=0.2, hspace=0)
        f1.suptitle('Filtered Probabilities of GARCH-Vol for BTC/USD')

        # Save PNG
        if save_dir:
            Path(save_dir + '/visuals').mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir + '/visuals/MSGARCH_3_Filtered_BTCUSD.png')

        plt.show()
            
    def plot_smoothed(self, dta, save_dir=''):
        """
        Plot smoothed probabilities
        
        Parameters:
        -----------
        dta : pd.DataFrame
            DataFrame with probabilities and data
        save_dir : str
            Directory to save plot
        """
        dta = dta.copy()
        dta['date'] = range(1, len(dta) + 1)

        # Create the figure
        f1, ax = plt.subplots(self.k + 2, figsize=(19, 15))

        # Plot regime probabilities
        ax[0].plot(dta.date, dta.SmoothedProbs1, label='Low-Volatility Regime')
        ax[0].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[0].set_ylabel('Probabilities', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[0].legend(loc='best')

        ax[1].plot(dta.date, dta.SmoothedProbs2, label='Medium-Volatility Regime')
        ax[1].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[1].set_ylabel('Probabilities', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[1].legend(loc='best')

        ax[2].plot(dta.date, dta.SmoothedProbs3, label='High-Volatility Regime')
        ax[2].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[2].set_ylabel('Probabilities', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[2].legend(loc='best')
            
        # Plot returns
        ax[self.k].plot(dta.date, dta.returns, label='Returns', color='gold', linewidth=2)
        ax[self.k].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k].set_ylabel('Returns', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k].legend(loc='best')

        # Plot close price
        ax[self.k+1].plot(dta.date, dta.close, label='Close Price', color='gold', linewidth=2)
        ax[self.k+1].set_xlabel('Observations', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k+1].set_ylabel('Close price', horizontalalignment='center', verticalalignment='center', fontsize=12, labelpad=20)
        ax[self.k+1].legend(loc='best')

        plt.grid(linestyle='dotted')
        plt.subplots_adjust(left=0.1, bottom=0.20, right=0.95, top=0.95, wspace=0.2, hspace=0)
        f1.suptitle('Smoothed Probabilities of GARCH-Vol for BTC/USD')

        # Save PNG
        if save_dir:
            Path(save_dir + '/visuals').mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir + '/visuals/MSGARCH_3_Smoothed_BTCUSD.png')

        plt.show()


# ============================================================================
# Main Execution Functions
# ============================================================================

def run_2_regime_model(data_file=DEFAULT_DATA_FILE, save_dir='.'):
    """
    Run 2-Regime Markov-Switching model
    
    Parameters:
    -----------
    data_file : str
        Path to CSV file
    save_dir : str
        Directory to save plots
    """
    print("=" * 60)
    print("Running 2-Regime Markov-Switching Model")
    print("=" * 60)
    
    # Get GARCH forecasts
    df = get_forecasts(show_vis=False, data_file=data_file)
    
    # Clean data
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    # Initialize and fit model
    markov_rs = MarkovSwitch2()
    markov_rs.fit_model(df)
    
    # Plot results
    markov_rs.plot_smoothed(df, save_dir)
    markov_rs.plot_filtered(df, save_dir)
    
    return df, markov_rs


def run_3_regime_model(data_file=DEFAULT_DATA_FILE, save_dir='.'):
    """
    Run 3-Regime Markov-Switching model
    
    Parameters:
    -----------
    data_file : str
        Path to CSV file
    save_dir : str
        Directory to save plots
    """
    print("=" * 60)
    print("Running 3-Regime Markov-Switching Model")
    print("=" * 60)
    
    # Get GARCH forecasts
    df = get_forecasts(show_vis=False, data_file=data_file)
    
    # Clean data
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    # Initialize and fit model
    markov_ar = MarkovSwitch3()
    markov_ar.fit_model(df)
    
    # Plot results
    markov_ar.plot_smoothed(df, save_dir)
    markov_ar.plot_filtered(df, save_dir)
    
    return df, markov_ar


# ============================================================================
# Main Script
# ============================================================================

if __name__ == '__main__':
    # Configuration
    DATA_DIR = './data'
    SAVE_DIR = '.'  # Current directory, or set to your desired path
    
    # Fetch data from Binance (2023-02-03 to 2026-02-02)
    # This will take ~10-15 mins
    print("=" * 60)
    print("Fetching data from Binance")
    print("=" * 60)
    # update_data_from_binance(
    #     from_datetime='2023-02-03 00:00:00',
    #     to_datetime='2026-02-02 23:59:59',
    #     timeframe='1h',  # 1-hour data
    #     output_filename='btc_usdt_1h.csv',
    #     data_dir=DATA_DIR
    # )
    
    print("\n" + "=" * 60)
    print("Running MSGARCH Models")
    print(f"Using data file: {DEFAULT_DATA_FILE}")
    print("=" * 60)
    
    # Run 2-Regime model
    df_2, model_2 = run_2_regime_model(
        data_file=DEFAULT_DATA_FILE, 
        save_dir=SAVE_DIR
    )
    
    # Run 3-Regime model
    df_3, model_3 = run_3_regime_model(
        data_file=DEFAULT_DATA_FILE, 
        save_dir=SAVE_DIR
    )
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
