import pandas as pd
import numpy as np
import os
import sys
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import scipy.stats as stats
import empyrical
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
warnings.filterwarnings('ignore')
sns.set(color_codes=True)#导入seaborn包设定颜色
from clickhouse_driver import Client
import pickle
import gzip

def generate_label(x):
    sorted_values = sorted(x['T_calendar'].unique())
    label_map = {
        sorted_values[0]: '当月',
        sorted_values[1]: '下月',
        sorted_values[-2]: '下季月',
        sorted_values[-1]: '隔季月'
    }
    x['当月标签'] = x['T_calendar'].map(label_map)
    return x
def trans_date(date_num):
    year = int(np.floor(date_num/10000))
    month = int(np.floor(((date_num - year*10000)/100)))
    day = date_num - year*10000-month*100
    time_datetime = datetime(year,month,day)
    return time_datetime
# 计算ETF期权保证金
def calc_margin(contract_type, contract_settle, underlying_close, strike, contract_unit):
    # 把 contract_settle 和 underlying_close 带入 昨日合约结算价 和 阼日标的收盘价 即可得到开仓保证金
    if contract_type == '认购' :
        margin_reserve = (contract_settle + max( 0.12 * underlying_close - (strike - underlying_close ), 0.07 * underlying_close)) * contract_unit
    elif contract_type == '认沽': 
        margin_reserve = min(contract_settle + max( 0.12 * underlying_close - (underlying_close - strike), 0.07 * strike), strike) * contract_unit
    return round(margin_reserve, 2)
def generate_label_K(x,call = True):
    x = x.copy()
    sorted_values = sorted(x['exercise_price'].unique())
    if call==False:
        sorted_values = sorted(x['exercise_price'].unique(),reverse = True)
    #sorted_values_2 = sorted(x['exercise_price'].unique())
    if len(sorted_values)==1:
        label_map = {
            sorted_values[0]: '平值',
        }  
    elif(len(sorted_values)==2):
        label_map = {
            sorted_values[0]: '平值',
            sorted_values[1]: '虚值一档',
        }  
    elif(len(sorted_values)==3):
        label_map = {
            sorted_values[0]: '平值',
            sorted_values[1]: '虚值一档',
            sorted_values[2]: '虚值二档',
        }  
    else:
        label_map = {
            sorted_values[0]: '平值',
            sorted_values[1]: '虚值一档',
            sorted_values[2]: '虚值二档',
            sorted_values[3]: '虚值三档',
        }    
    x['虚值档'] = x['exercise_price'].map(label_map)
    return x

def select_data(data, call_or_put, time):
    data_selected = data[data['time']==datetime.strptime(time, '%H:%M:%S').time()]
    data_selected = data_selected[data_selected['call_or_put']==call_or_put]     #只保留看涨期权
    data_selected['k_minus_s'] = data_selected['exercise_price']-data_selected['close_mid_index']
    data_selected= data_selected[data_selected['k_minus_s']>0].groupby(['datetime','当月标签']).apply(generate_label_K)
    data_selected = data_selected[data_selected['虚值档'].notna()]
    data_selected['month'] = [i.month for i in data_selected['datetime']]
    data_selected['year'] = [i.year for i in data_selected['datetime']]
    data_selected['year_month'] =  data_selected['year']*100+data_selected['month']
    data_selected['expiry_date'] = data_selected['expiry_date'].astype(int).apply(trans_date)
    return data_selected

#计算日内卖一手call或者put的pnl  
def cal_oneside_pnl(data, call_or_put):
    data_0940 = select_data(data, call_or_put=call_or_put, time='9:40:00') 
    data_1500 = select_data(data, call_or_put=call_or_put, time='15:00:00') 
    df_intraday = pd.DataFrame()
    #df_intraday = pd.DataFrame(pd.concat([data_0940[['datetime','date']],data_1500[['datetime','date']]]).drop_duplicates()).rename_axis(index=["index_datetime", "当月标签",'index']).sort_values(by = 'datetime').reset_index()[['datetime','date']].set_index('datetime')
    df_intraday = pd.DataFrame(pd.concat([data_0940[['datetime','date']],data_1500[['datetime','date']]]).drop_duplicates()).sort_index().reset_index()[['datetime','date']].set_index('datetime')
    #df_intraday['date'] =[datetime.date(datetime.strptime(i, '%Y-%m-%d %H:%M:%S').year,datetime.strptime(i, '%Y-%m-%d %H:%M:%S').month,datetime.strptime(i, '%Y-%m-%d %H:%M:%S').day) for i in df_intraday.index] 
    for dt in df_intraday.index:
        temp = data_0940.loc[(data_0940['datetime']==dt)&(data_0940['当月标签']=='当月')&(data_0940['虚值档']=='平值'),['close_mid_option','StockID_option','settle','close_mid_index','exercise_price','multiplier']]
        if(len(temp)>0):
            df_intraday.loc[dt,['close_mid_option','StockID_option','settle','close_mid_index','exercise_price','multiplier']] = temp.iloc[0]
            #如果剩余到期日小于等于7天
            if((data_0940.loc[(data_0940['datetime']==dt)&(data_0940['当月标签']=='当月')&(data_0940['虚值档']=='平值'),'expiry_date'].iloc[0] - dt).days)<=7:
                temp = data_0940.loc[(data_0940['datetime']==dt)&(data_0940['当月标签']=='下月')&(data_0940['虚值档']=='平值'),['close_mid_option','StockID_option','settle','close_mid_index','exercise_price','multiplier']]
                df_intraday.loc[dt,['close_mid_option','StockID_option','settle','close_mid_index','exercise_price','multiplier']] = temp.iloc[0] if len(temp)>0 else np.nan
    df_intraday['StockID_option'] = df_intraday['StockID_option'].ffill(limit=1)
    data_temp = data[data['time']==datetime.strptime('15:00:00', '%H:%M:%S').time()]
    data_temp = data_temp[data_temp['call_or_put']==call_or_put] 
    for dt in data_1500['datetime'].drop_duplicates().sort_values().to_list():
        StockID_option = df_intraday.loc[dt,'StockID_option']
        temp = data_temp.loc[(data_temp['datetime']==dt)&(data_temp['StockID_option']==StockID_option),['close_mid_option','settle','close_mid_index','exercise_price','multiplier']]
        df_intraday.loc[dt,['close_mid_option','settle','close_mid_index','exercise_price','multiplier']] = temp.iloc[0] if (len(temp) > 0) else np.nan
    df_intraday['time'] = df_intraday.index.to_series().apply(lambda x : x.hour)  #区分开盘和收盘 
    df_intraday[['last_settle','last_underlying_close']] = df_intraday.sort_index().groupby('time').shift(1)[['settle','close_mid_index']]  #昨日合约结算价 和 昨日标的收盘价
    df_intraday['margin'] = df_intraday.apply(lambda x: calc_margin(contract_type='认购', contract_settle=x['last_settle'], underlying_close=x['close_mid_index'], strike=x['exercise_price'], contract_unit=x['multiplier']) ,axis = 1)
    df_intraday.loc[df_intraday['time']==15,'margin'] = np.nan #只保留开仓时候算的保证金
    #df_intraday['sell_lots'] = initial*0.6//df_intraday['margin']   #卖出的期权手数, 取整
    #df_intraday.loc[df_intraday['time']==15,'sell_lots'] = np.nan #只保留开仓时候算的卖出手数 
    return df_intraday

def cal_total_pnl(df_call_intraday, df_put_intraday):
    df_pnl = pd.DataFrame(index = df_call_intraday['date'])
    for date in df_pnl.index:
        condition_open = (df_call_intraday['date']==date)&(df_call_intraday['time']==9)
        condition_close = (df_call_intraday['date']==date)&(df_call_intraday['time']==15)
        try:
            df_pnl.loc[date,'call_pnl'] = -(float(df_call_intraday.loc[condition_close,'close_mid_option'])*float(df_call_intraday.loc[condition_close,'multiplier']) - float(df_call_intraday.loc[condition_open,'close_mid_option'])*float(df_call_intraday.loc[condition_close,'multiplier'])) - 3.6
            df_pnl.loc[date,'call_margin'] = float(df_call_intraday.loc[condition_open, 'margin'])
        except:
            pass
    for date in df_pnl.index:
        condition_open = (df_put_intraday['date']==date)&(df_put_intraday['time']==9)
        condition_close = (df_put_intraday['date']==date)&(df_put_intraday['time']==15)
        try:
            #交易一手的PNL
            df_pnl.loc[date,'put_pnl'] = -(float(df_put_intraday.loc[condition_close,'close_mid_option'])*float(df_put_intraday.loc[condition_close,'multiplier']) - float(df_put_intraday.loc[condition_open,'close_mid_option'])*float(df_put_intraday.loc[condition_close,'multiplier'])) - 3.6
            df_pnl.loc[date,'put_margin'] = float(df_put_intraday.loc[condition_open, 'margin'])
        except:
            pass
    df_pnl['sell_lots'] =  1 #initial*0.6//(df_pnl['call_margin']+df_pnl['put_margin'])
    df_pnl['pnl'] = (df_pnl['call_pnl'] + df_pnl['put_pnl'])*df_pnl['sell_lots']
    return df_pnl

if __name__ == '__main__':
    #data = pd.read_csv('0-九象限策略构建/data/dh_ret50_new1.csv')
    initial = 10**7
    for code in ['SH000016','SH000852','SH510050','SH510300','SH510500','SH588000','SZ159901','SZ159915']:
        #data = pickle.loads(gzip.decompress(open('0-九象限策略构建/data/SH510050_data.pkl.gz', 'rb').read()))
        data = pickle.loads(gzip.decompress(open('data/'+code+'_data.pkl.gz', 'rb').read()))
        data = generate_label_K(data)
        df_call_intraday = cal_oneside_pnl(data, call_or_put='call')
        df_put_intraday = cal_oneside_pnl(data, call_or_put='put')
        df_pnl = cal_total_pnl(df_call_intraday, df_put_intraday)
        df_pnl['ret'] = df_pnl['pnl']/initial
        #open('data/df_pnl.pkl.gz', 'wb').write(gzip.compress(pickle.dumps(df_pnl)))
        open('data/'+code+'_pnl.pkl.gz', 'wb').write(gzip.compress(pickle.dumps(df_pnl)))



