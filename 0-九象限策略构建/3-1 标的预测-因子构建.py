# test test test
import pandas as pd
import numpy as np
import os
import sys
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)
import datetime
import scipy.stats as stats

def get_resampled_data(path):
    # Load the original data again
    df = pd.read_csv(path,index_col = 0)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Set the datetime as the index
    df.set_index('datetime', inplace=True)

    # Resample the data to daily frequency, calculating different aggregation functions
    daily_data = df.resample('B').agg({'open': 'first', 
                                    'high': 'max', 
                                    'low': 'min', 
                                    'close': 'last', 
                                    'volume': 'sum', 
                                    'amount': 'sum'})

    # Calculate VWAP (Volume Weighted Average Price)
    daily_data['vwap'] = daily_data['amount'] / daily_data['volume']

    # Calculate the daily returns
    daily_data['daily_return_ctc'] = daily_data['close'].pct_change()
    daily_data['daily_return_otc'] = (daily_data['close']-daily_data['open'])/daily_data['open']

    # Resample the original dataframe to 5-minute intervals
    df_resampled_5min = df.resample('5T').last()

    # Calculate the log returns for the resampled data
    df_resampled_5min['log_return'] = np.log(df_resampled_5min['close'] / df_resampled_5min['close'].shift())

    # Group by each trading day and calculate the realized volatility
    realized_vol = pd.DataFrame(df_resampled_5min.groupby(df_resampled_5min.index.date)['log_return'].transform('std'))
    realized_vol.rename(columns={'log_return':'realized_volatility'},inplace=True)

    # Extract the realized volatility at the end of each trading day
    daily_data = daily_data.merge(realized_vol,left_on = daily_data.index, right_on = realized_vol.index,how='left')
    daily_data.rename(columns={'key_0':'date'},inplace=True)
    return daily_data

def BETA(close, i):
    close_series = pd.Series(close) 
    return close_series.cov(close_series.shift(i)) / close_series.shift(i).var()
def RSQR(close, i):
    close = pd.Series(close) 
    return np.power(close.corr(close.shift(i)), 2)

def QTLU(x):
    return x.quantile(0.75)

def QTLD(x):
    return x.quantile(0.25)

def RANK(close):
    return close.rank()

def RSV(close, high, low, i):
    return (close.iloc[-1] - pd.Series.rolling(low, window=i).min()) / (pd.Series.rolling(high, window=i).max() - pd.Series.rolling(low, window=i).min())

def IMAX(high, i):
    return (pd.Series(high).rolling(window=i).apply(lambda x: i - x.argmax(), raw=True) + 1)

def IMIN(low, i):
    return (pd.Series(low).rolling(window=i).apply(lambda x: i - x.argmin(), raw=True) + 1)

def IMXD(high, low, i):
    imax = (pd.Series(high).rolling(window=i).apply(lambda x: x.argmax(), raw=True) + 1)
    imin = (pd.Series(low).rolling(window=i).apply(lambda x: x.argmin(), raw=True) + 1)
    return imax - imin

def CORR(close, volume, i):
    return close.rolling(window=i).corr(volume)

def CORD(close, volume, i):
    return close.pct_change(i).rolling(window=i).corr(volume.pct_change(i))

def CNTP(close, i):
    roc_close = close.pct_change(i)
    return (roc_close > 0).rolling(window=i).sum() / i

def CNTN(close, i):
    roc_close = close.pct_change(i)
    return (roc_close < 0).rolling(window=i).sum() / i

def CNTD(close, i):
    roc_close = close.pct_change(i)
    return (roc_close > 0).rolling(window=i).sum() - (roc_close < 0).rolling(window=i).sum()

def SUMP(close, i):
    diff = close - close.shift(i)
    return diff.where(diff > 0).rolling(window=i).sum() / (diff.abs().rolling(window=i).sum() + 1e-12)

def SUMN(close, i):
    return 1 - SUMP(close, i)

def SUMD(close, i):
    diff = close - close.shift(i)
    return (diff.where(diff > 0).rolling(window=i).sum() - diff.where(diff < 0).rolling(window=i).sum()) / (diff.abs().rolling(window=i).sum() + 1e-12)

def VMA(volume, i):
    return volume.rolling(window=i).mean() / (volume + 1e-12)

def VSTD(volume, i):
    return volume.rolling(window=i).std() / (volume + 1e-12)

def WVMA(close, volume, i):
    abs_diff = (close / close.shift(i) - 1).abs()
    return abs_diff * volume.rolling(window=i).std() / (abs_diff * volume.rolling(window=i).mean() + 1e-12)

def VSUMP(volume, i):
    diff = volume - volume.shift(i)
    return diff.where(diff > 0).rolling(window=i).sum() / (diff.abs().rolling(window=i).sum() + 1e-12)

def VSUMN(volume, i):
    return 1 - VSUMP(volume, i)

def VSUMD(volume, i):
    diff = volume - volume.shift(i)
    return (diff.where(diff > 0).rolling(window=i).sum() - diff.where(diff < 0).rolling(window=i).sum()) / (diff.abs().rolling(window=i).sum() + 1e-12)

def gen_factors(daily_data):
    eps = 1e-12
    daily_data['KMID'] = (daily_data['close'] - daily_data['open']) / daily_data['open']
    daily_data['KLEN'] = (daily_data['high'] - daily_data['low']) / daily_data['open']
    daily_data['KMID2'] = (daily_data['close'] - daily_data['open']) / (daily_data['high'] - daily_data['low'] + eps)
    daily_data['KUP'] = (daily_data['high'] - daily_data[['open', 'close']].max(axis=1)) / daily_data['open']
    daily_data['KUP2'] = (daily_data['high'] - daily_data[['open', 'close']].max(axis=1)) / (daily_data['high'] - daily_data['low'] + eps)
    daily_data['KLOW'] = (daily_data[['open', 'close']].min(axis=1) - daily_data['low']) / daily_data['open']
    daily_data['KLOW2'] = (daily_data[['open', 'close']].min(axis=1) - daily_data['low']) / (daily_data['high'] - daily_data['low'] + eps)
    daily_data['KSFT'] = (2 * daily_data['close'] - daily_data['high'] - daily_data['low']) / daily_data['open']
    daily_data['KSFT2'] = (2 * daily_data['close'] - daily_data['high'] - daily_data['low']) / (daily_data['high'] - daily_data['low'] + eps)
    daily_data['OPEN0'] = daily_data['open']/daily_data['close']
    daily_data['HIGH0'] = daily_data['high']/daily_data['close']
    daily_data['LOW0'] = daily_data['low']/daily_data['close']
    daily_data['VWAP0'] = daily_data['vwap']/daily_data['close']

    i_list = [5,10,20,30,60]
    rolling_n = 40
    for i in i_list:
        daily_data['ROC'+str(i)] = (daily_data['close'] - daily_data['close'].shift(i))/daily_data['close'].shift(i)
        daily_data['MA'+str(i)] = daily_data['close'].rolling(i).mean()
        daily_data[''+str(i)] = daily_data['close'].rolling(i).std()
        daily_data['BETA'+str(i)] = daily_data['close'].rolling(window=rolling_n).apply(BETA, args=(i,), raw=True)
        daily_data['RSQR'+str(i)] = daily_data['close'].rolling(window=rolling_n).apply(RSQR, args=(i,), raw=True)
        #daily_data['RESI'+str(i)] = daily_data['close'].rolling(window=rolling_n).apply(RESI, args=(i,), raw=True)
        daily_data['HIGH'+str(i)] = daily_data['high'].rolling(i).max()
        daily_data['LOW'+str(i)] = daily_data['low'].rolling(i).min()
        daily_data['QTLU'+str(i)] = daily_data['close'].rolling(i).apply(QTLU)
        daily_data['QTLD'+str(i)] = daily_data['close'].rolling(i).apply(QTLD)
        daily_data['RANK'+str(i)] = daily_data['close'].rolling(i).apply(lambda x: RANK(x).iloc[-1])
        daily_data['RSV'+str(i)] = (daily_data['close'] - daily_data['low'].rolling(i).min())/(daily_data['high'].rolling(i).max() - daily_data['low'].rolling(i).min())
        daily_data['IMAX'+str(i)] = IMAX(daily_data['high'],i)
        daily_data['IMIN'+str(i)] = IMIN(daily_data['low'],i)
        daily_data['IMXD'+str(i)] = IMXD(daily_data['high'],daily_data['low'],i)
        daily_data['CORR'+str(i)] = CORR(daily_data['close'],daily_data['volume'],i)
        daily_data['CORD'+str(i)] = CORD(daily_data['close'],daily_data['volume'],i)
        daily_data['CNTP'+str(i)] = CNTP(daily_data['close'], i)
        daily_data['CNTN'+str(i)] = CNTN(daily_data['close'], i)
        daily_data['CNTD'+str(i)] = CNTD(daily_data['close'], i)
        daily_data['SUMP'+str(i)] = SUMP(daily_data['close'], i)
        daily_data['SUMN'+str(i)] = SUMN(daily_data['close'], i)
        daily_data['SUMD'+str(i)] = SUMD(daily_data['close'], i)
        daily_data['VMA'+str(i)] = VMA(daily_data['volume'], i)
        daily_data['VSTD'+str(i)] = VSTD(daily_data['volume'], i)
        daily_data['WVMA'+str(i)] = WVMA(daily_data['close'], daily_data['volume'], i)
        daily_data['VSUMP'+str(i)] = SUMP(daily_data['volume'], i)
        daily_data['VSUMN'+str(i)] = SUMN(daily_data['volume'], i)
        daily_data['VSUMD'+str(i)] = SUMD(daily_data['volume'], i)
    return daily_data

if __name__ == "main":
    path = r"C:\Users\Yue\Desktop\衍生品\策略\index_hh.csv"
    data = get_resampled_data(path)
    data = gen_factors(data)
    data.to_pickle('./data/factors.pkl.gz')
    print('因子生成完毕！')

