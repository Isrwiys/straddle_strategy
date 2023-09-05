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


if __name__ == '__main__':
    #data = pd.read_csv('data/dh_ret50_new1.csv')
    data = pickle.loads(gzip.decompress(open('data/dh_ret50_new1.pkl.gz', 'rb').read()))
    data.index.name='index'
    data= data.groupby('date').apply(generate_label)
    data['time'] = data['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').time())
    data = generate_label_K(data)

    #data2:call 15:00
    data2 = data[data['time']==datetime.strptime('15:00:00', '%H:%M:%S').time()]
    data2 = data2[data2['call_or_put']=='认购']     #只保留看涨期权
    data2['k_minus_s'] = data2['exercise_price']-data2['index_close_mid']
    data2= data2[data2['k_minus_s']>0].groupby(['datetime','当月标签']).apply(generate_label_K)
    data2 = data2[data2['虚值档'].notna()]
    data2['month'] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S').month for i in data2['datetime']]
    data2['year'] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S').year for i in data2['datetime']]
    data2['year_month'] =  data2['year']*100+data2['month']
    #data3:put 15:00
    data3 = data[data['time']==datetime.strptime('15:00:00', '%H:%M:%S').time()]
    data3 = data3[data3['call_or_put']=='认沽']     
    data3['k_minus_s'] = data3['exercise_price']-data3['index_close_mid']
    data3= data3[data3['k_minus_s']<0].groupby(['datetime','当月标签']).apply(generate_label_K,call=False)
    data3 = data3[data3['虚值档'].notna()]
    data3['month'] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S').month for i in data3['datetime']]
    data3['year'] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S').year for i in data3['datetime']]
    data3['year_month'] =  data3['year']*100+data3['month']
    #data4:call 9:40
    data4 = data[data['time']==datetime.strptime('9:40:00', '%H:%M:%S').time()]
    data4 = data4[data4['call_or_put']=='认购']     #只保留看涨期权
    data4['k_minus_s'] = data4['exercise_price']-data4['index_close_mid']
    data4= data4[data4['k_minus_s']>0].groupby(['datetime','当月标签']).apply(generate_label_K)
    data4 = data4[data4['虚值档'].notna()]
    data4['month'] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S').month for i in data4['datetime']]
    data4['year'] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S').year for i in data4['datetime']]
    data4['year_month'] =  data4['year']*100+data4['month']
    #data5:put 9:40
    data5 = data[data['time']==datetime.strptime('9:40:00', '%H:%M:%S').time()]
    data5 = data5[data5['call_or_put']=='认沽']    
    data5['k_minus_s'] = data5['exercise_price']-data5['index_close_mid']
    data5= data5[data5['k_minus_s']<0].groupby(['datetime','当月标签']).apply(generate_label_K,call=False)
    data5 = data5[data5['虚值档'].notna()]
    data5['month'] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S').month for i in data5['datetime']]
    data5['year'] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S').year for i in data5['datetime']]
    data5['year_month'] =  data5['year']*100+data5['month']

    data4['expiry_date'] = data4['expiry_date'].astype(int).apply(trans_date)
    data3['expiry_date'] = data3['expiry_date'].astype(int).apply(trans_date)
    data2['expiry_date'] = data2['expiry_date'].astype(int).apply(trans_date)
    data5['expiry_date'] = data5['expiry_date'].astype(int).apply(trans_date) 
    #日频 日内 卖straddle  _call
    df_call_intraday = pd.DataFrame()
    #df_call_intraday = pd.DataFrame(pd.concat([data4[['datetime','date']],data2[['datetime','date']]]).drop_duplicates()).rename_axis(index=["index_datetime", "当月标签",'index']).sort_values(by = 'datetime').reset_index()[['datetime','date']].set_index('datetime')
    df_call_intraday = pd.DataFrame(pd.concat([data4[['datetime','date']],data2[['datetime','date']]]).drop_duplicates()).sort_index().reset_index()[['datetime','date']].set_index('datetime')
    df_call_intraday
    #df_call_intraday['date'] =[datetime.date(datetime.strptime(i, '%Y-%m-%d %H:%M:%S').year,datetime.strptime(i, '%Y-%m-%d %H:%M:%S').month,datetime.strptime(i, '%Y-%m-%d %H:%M:%S').day) for i in df_call_intraday.index] 
    df_call_intraday
    for dt in df_call_intraday.index:
        temp = data4.loc[(data4['datetime']==dt)&(data4['当月标签']=='当月')&(data4['虚值档']=='平值'),['price_mid','StockID','settle','index_close_mid','exercise_price','multiplier']]
        if(len(temp)>0):
            df_call_intraday.loc[dt,['price_mid','StockID','settle','index_close_mid','exercise_price','multiplier']] = temp.iloc[0]
            #如果剩余到期日小于等于7天
            if((data4.loc[(data4['datetime']==dt)&(data4['当月标签']=='当月')&(data4['虚值档']=='平值'),'expiry_date'].iloc[0] - datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')).days)<=7:
                temp = data4.loc[(data4['datetime']==dt)&(data4['当月标签']=='下月')&(data4['虚值档']=='平值'),['price_mid','StockID','settle','index_close_mid','exercise_price','multiplier']]
                df_call_intraday.loc[dt,['price_mid','StockID','settle','index_close_mid','exercise_price','multiplier']] = temp.iloc[0] if len(temp)>0 else np.nan
    df_call_intraday['StockID'] = df_call_intraday['StockID'].ffill(limit=1)
    data_temp =data[data['time']==datetime.strptime('15:00:00', '%H:%M:%S').time()]
    data_temp = data_temp[data_temp['call_or_put']=='认购'] 
    for dt in data2['datetime'].drop_duplicates().sort_values().to_list():
        StockID = df_call_intraday.loc[dt,'StockID']
        temp = data_temp.loc[(data_temp['datetime']==dt)&(data_temp['StockID']==StockID),['price_mid','settle','index_close_mid','exercise_price','multiplier']]
        df_call_intraday.loc[dt,['price_mid','settle','index_close_mid','exercise_price','multiplier']] = temp.iloc[0] if (len(temp) > 0) else np.nan
    df_call_intraday['time'] = df_call_intraday.index.to_series().apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)  #区分开盘和收盘
    #昨日合约结算价 和 昨日标的收盘价
    df_call_intraday[['last_settle','last_underlying_close']] = df_call_intraday.sort_index().groupby('time').shift(1)[['settle','index_close_mid']]
    df_call_intraday['margin'] = df_call_intraday.apply(lambda x: calc_margin(contract_type='认购', contract_settle=x['last_settle'], underlying_close=x['index_close_mid'], strike=x['exercise_price'], contract_unit=x['multiplier']) ,axis = 1)
    df_call_intraday.loc[df_call_intraday['time']==15,'margin'] = np.nan #只保留开仓时候算的保证金
    #卖出的期权手数, 取整
    initial = 10**7
    #df_call_intraday['sell_lots'] = initial*0.6//df_call_intraday['margin']
    #df_call_intraday.loc[df_call_intraday['time']==15,'sell_lots'] = np.nan #只保留开仓时候算的卖出手数 
    data.columns
    #日频 日内 卖straddle    _put
    df_put_intraday = pd.DataFrame()
    #df_put_intraday = pd.DataFrame(pd.concat([data5[['datetime','date']],data3[['datetime','date']]]).drop_duplicates().rename_axis(index=["index_datetime", "当月标签",'index']).sort_values(by = 'datetime').reset_index()[['datetime','date']]).set_index('datetime')
    df_put_intraday = pd.DataFrame(pd.concat([data5[['datetime','date']],data3[['datetime','date']]]).drop_duplicates()).sort_index().reset_index()[['datetime','date']].set_index('datetime')
    #df_put_intraday['date'] =[datetime.date(datetime.strptime(i, '%Y-%m-%d %H:%M:%S').year,datetime.strptime(i, '%Y-%m-%d %H:%M:%S').month,datetime.strptime(i, '%Y-%m-%d %H:%M:%S').day) for i in df_put_intraday.index] 
    df_put_intraday
    for dt in df_put_intraday.index:
        temp = data5.loc[(data5['datetime']==dt)&(data5['当月标签']=='当月')&(data5['虚值档']=='平值'),['price_mid','StockID','settle','index_close_mid','exercise_price','multiplier']]
        if(len(temp)>0):
            df_put_intraday.loc[dt,['price_mid','StockID','settle','index_close_mid','exercise_price','multiplier']] = temp.iloc[0]
            #如果剩余到期日小于等于7天
            if((data5.loc[(data5['datetime']==dt)&(data5['当月标签']=='当月')&(data5['虚值档']=='平值'),'expiry_date'].iloc[0] - datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')).days)<=7:
                temp = data5.loc[(data5['datetime']==dt)&(data5['当月标签']=='下月')&(data5['虚值档']=='平值'),['price_mid','StockID','settle','index_close_mid','exercise_price','multiplier']]
                df_put_intraday.loc[dt,['price_mid','StockID','settle','index_close_mid','exercise_price','multiplier']] = temp.iloc[0] if len(temp)>0 else np.nan
    df_put_intraday['StockID'] = df_put_intraday['StockID'].ffill(limit=1)
    data_temp =data[data['time']==datetime.strptime('15:00:00', '%H:%M:%S').time()]
    data_temp = data_temp[data_temp['call_or_put']=='认沽'] 
    for dt in data3['datetime'].drop_duplicates().sort_values().to_list():
        StockID = df_put_intraday.loc[dt,'StockID']
        temp = data_temp.loc[(data_temp['datetime']==dt)&(data_temp['StockID']==StockID),['price_mid','StockID','settle','index_close_mid','exercise_price','multiplier']]
        df_put_intraday.loc[dt,['price_mid','StockID','settle','index_close_mid','exercise_price','multiplier']] = temp.iloc[0] if (len(temp) > 0) else np.nan
    df_put_intraday['time'] = df_put_intraday.index.to_series().apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)  #区分开盘和收盘
    #昨日合约结算价 和 昨日标的收盘价
    df_put_intraday[['last_settle','last_underlying_close']] = df_put_intraday.sort_index().groupby('time').shift(1)[['settle','index_close_mid']]
    df_put_intraday['margin'] = df_put_intraday.apply(lambda x: calc_margin(contract_type='认沽', contract_settle=x['last_settle'], underlying_close=x['index_close_mid'], strike=x['exercise_price'], contract_unit=x['multiplier']) ,axis = 1)
    df_put_intraday.loc[df_put_intraday['time']==15,'margin'] = np.nan #只保留开仓时候算的保证金
    #卖出的期权手数, 取整
    initial = 10000
    df_put_intraday['sell_lots'] =  1 #initial*0.6//df_put_intraday['margin']
    df_put_intraday.loc[df_put_intraday['time']==15,'sell_lots'] = np.nan #只保留开仓时候算的卖出手数 
    """ df_put_pnl = pd.DataFrame(index = df_put_intraday['date'])
    for date in df_put_pnl.index:
        condition_open = (df_put_intraday['date']==date)&(df_put_intraday['time']==9)
        condition_close = (df_put_intraday['date']==date)&(df_put_intraday['time']==15)
        try:
            df_put_pnl.loc[date,'pnl'] = -float(df_put_intraday.loc[condition_open,'sell_lots'])*(float(df_put_intraday.loc[condition_close,'price_mid']) - float(df_put_intraday.loc[condition_open,'price_mid'])) - float(df_put_intraday.loc[condition_open,'sell_lots'])*1.8
        except TypeError:
            df_put_pnl.loc[date,'pnl'] = np.nan  #一天，没有15:00的数据
    df_put_pnl['ret'] = df_put_pnl['pnl']/initial
    df_put_pnl  """
    df_pnl = pd.DataFrame(index = df_call_intraday['date'])
    for date in df_pnl.index:
        condition_open = (df_call_intraday['date']==date)&(df_call_intraday['time']==9)
        condition_close = (df_call_intraday['date']==date)&(df_call_intraday['time']==15)
        try:
            df_pnl.loc[date,'call_pnl'] = -(float(df_call_intraday.loc[condition_close,'price_mid'])*float(df_call_intraday.loc[condition_close,'multiplier']) - float(df_call_intraday.loc[condition_open,'price_mid'])*float(df_call_intraday.loc[condition_close,'multiplier'])) - 3.6
            df_pnl.loc[date,'call_margin'] = float(df_call_intraday.loc[condition_open, 'margin'])
        except:
            pass
    for date in df_pnl.index:
        condition_open = (df_put_intraday['date']==date)&(df_put_intraday['time']==9)
        condition_close = (df_put_intraday['date']==date)&(df_put_intraday['time']==15)
        try:
            #交易一手的PNL
            df_pnl.loc[date,'put_pnl'] = -(float(df_put_intraday.loc[condition_close,'price_mid'])*float(df_put_intraday.loc[condition_close,'multiplier']) - float(df_put_intraday.loc[condition_open,'price_mid'])*float(df_put_intraday.loc[condition_close,'multiplier'])) - 3.6
            df_pnl.loc[date,'put_margin'] = float(df_put_intraday.loc[condition_open, 'margin'])
        except:
            pass
    df_pnl['sell_lots'] =  1 #initial*0.6//(df_pnl['call_margin']+df_pnl['put_margin'])
    df_pnl['pnl'] = (df_pnl['call_pnl'] + df_pnl['put_pnl'])*df_pnl['sell_lots']
    df_pnl['ret'] = df_pnl['pnl']/initial
    #df_pnl.index =  df_pnl.index.to_series().apply(trans_date)
    df_pnl
    #df_pnl.to_pickle('df_pnl_300')
    open('data/df_pnl.pkl.gz', 'wb').write(gzip.compress(pickle.dumps(df_pnl)))



