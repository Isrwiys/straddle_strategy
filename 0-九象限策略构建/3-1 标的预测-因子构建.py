import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)
import scipy.stats as stats
import pickle
import gzip

def get_resampled_data(path):
    df = pickle.loads(gzip.decompress(open(path, 'rb').read()))
    df.set_index('datetime', inplace=True)
    daily_data = df.resample('B').agg({'open': 'first', 
                                    'high': 'max', 
                                    'low': 'min', 
                                    'close': 'last', 
                                    'volume': 'sum', 
                                    'amount': 'sum'})
    daily_data['daily_return_ctc'] = daily_data['close'].pct_change()
    daily_data['daily_return_otc'] = (daily_data['close']-daily_data['open'])/daily_data['open']
    df_resampled_5min = df.resample('5T').last()
    df_resampled_5min['log_return'] = np.log(df_resampled_5min['close'] / df_resampled_5min['close'].shift())
    realized_vol = pd.DataFrame(df_resampled_5min.groupby(df_resampled_5min.index.date)['log_return'].transform('std'))
    realized_vol.rename(columns={'log_return':'realized_volatility'},inplace=True)
    daily_data = daily_data.merge(realized_vol,left_on = daily_data.index, right_on = realized_vol.index,how='left')
    daily_data.rename(columns={'key_0':'date'},inplace=True)
    return daily_data

def mean_abs_residuals(close):  #计算过去n天回归残差绝对值的平均值
    y = close
    x = range(len(y))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    residuals = y - (slope * x + intercept)
    abs_residuals = np.abs(residuals)
    mean_abs_residuals = np.mean(abs_residuals)
    return mean_abs_residuals

def SUMP(close):
    diff = close - close.shift(1)
    return diff.where(diff > 0).sum() / (diff.abs().sum() + 1e-12)

def WVMA(close, volume, i):
    abs_diff = (close / close.shift(1) - 1).abs()
    return (abs_diff * volume).rolling(window=i).std() / ((abs_diff * volume).rolling(window=i).mean() + 1e-12)

def gen_factors(daily_data):
    eps = 1e-12
    daily_data = daily_data.copy()
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
    daily_data['vwap'] = daily_data['amount']/daily_data['volume']
    daily_data['VWAP0'] = daily_data['vwap']/daily_data['close']
    i_list = [5,10,20,30,60]
    for i in i_list:
        daily_data['ROC'+str(i)] = daily_data['close'].shift(i)/daily_data['close']
        daily_data['MA'+str(i)] = daily_data['close'].rolling(i).mean()/daily_data['close']
        daily_data['STD'+str(i)] = daily_data['close'].rolling(i).std()/daily_data['close']
        daily_data['BETA'+str(i)] = daily_data['close'].rolling(i).apply(lambda x: stats.linregress(x=range(len(x)), y=x).slope)
        daily_data['RSQR'+str(i)] = daily_data['close'].rolling(i).apply(lambda x: stats.linregress(x=range(len(x)), y=x).rvalue)
        daily_data['RESI'+str(i)] = daily_data['close'].rolling(i).apply(mean_abs_residuals)/daily_data['close']
        daily_data['MAX'+str(i)] = daily_data['high'].rolling(i).max()/daily_data['close']
        daily_data['MIN'+str(i)] = daily_data['low'].rolling(i).min()/daily_data['close']
        daily_data['QTLU'+str(i)] = daily_data['close'].rolling(i).quantile(0.8)/daily_data['close']
        daily_data['QTLD'+str(i)] = daily_data['close'].rolling(i).quantile(0.2)/daily_data['close']
        daily_data['RANK'+str(i)] = daily_data['close'].rolling(i).apply(lambda x: x.rank(pct=True).iloc[-1])
        daily_data['RSV'+str(i)] = (daily_data['close'] - daily_data['low'].rolling(i).min())/(daily_data['high'].rolling(i).max() - daily_data['low'].rolling(i).min() + eps)
        daily_data['IMAX'+str(i)] = daily_data['high'].rolling(i).apply(lambda x: (i - 1 - x.argmax())/i)
        daily_data['IMIN'+str(i)] = daily_data['low'].rolling(i).apply(lambda x: (i - 1 - x.argmin())/i)
        daily_data['IMXD'+str(i)] = daily_data['high'].rolling(i).apply(lambda x: (x.argmax())/i) - daily_data['low'].rolling(i).apply(lambda x: (x.argmin())/i)
        daily_data['CORR'+str(i)] = daily_data['close'].rolling(i).corr(daily_data['volume'].apply(lambda x: np.log1p(x)))
        daily_data['CORD'+str(i)] = (daily_data['close']/daily_data['close'].shift(1)).rolling(i).corr((daily_data['volume']/daily_data['volume'].shift(1)).apply(lambda x: np.log1p(x)))
        daily_data['CNTP'+str(i)] = daily_data['close'].rolling(i).apply(lambda x: (x>x.shift(1)).mean())
        ##相关性为-1的
        ##daily_data['CNTN'+str(i)] = daily_data['close'].rolling(i).apply(lambda x: (x<x.shift(1)).mean())
        ##daily_data['CNTD'+str(i)] = daily_data['CNTP'+str(i)] - daily_data['CNTN'+str(i)]
        daily_data['SUMP'+str(i)] = daily_data['close'].rolling(i).apply(lambda x: SUMP(x))
        ##daily_data['SUMN'+str(i)] = 1 - daily_data['SUMP'+str(i)]
        ##daily_data['SUMD'+str(i)] = daily_data['SUMP'+str(i)] - daily_data['SUMN'+str(i)]
        daily_data['VMA'+str(i)] = daily_data['volume'].rolling(i).mean()/(daily_data['volume']+eps)
        daily_data['VSTD'+str(i)] = daily_data['volume'].rolling(i).std()/(daily_data['volume']+eps)
        daily_data['WVMA'+str(i)] = WVMA(daily_data['close'],daily_data['volume'],i)
        daily_data['VSUMP'+str(i)] = daily_data['volume'].rolling(i).apply(lambda x: SUMP(x))
        ##daily_data['VSUMN'+str(i)] = 1 - daily_data['VSUMP'+str(i)]
        ##daily_data['VSUMD'+str(i)] = daily_data['VSUMP'+str(i)] - daily_data['VSUMN'+str(i)]
    daily_data.drop(columns = ['open','high','low','close','volume','amount','vwap'],inplace=True)
    return daily_data

if __name__ == "__main__":
    underlying_code = 'SH510050'
    path = 'data/'+underlying_code+'_index_hh.pkl.gz'
    data = get_resampled_data(path).dropna()
    data = gen_factors(data)
    open('data/'+underlying_code+'_factors.pkl.gz', 'wb').write(gzip.compress(pickle.dumps(data)))
    print('因子生成完毕！')

