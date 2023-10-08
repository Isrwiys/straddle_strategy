import pandas as pd
import numpy as np
import os
import sys
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)#导入seaborn包设定颜色
import datetime
import scipy.stats as stats
from clickhouse_driver import Client
import pickle
import gzip

N = stats.norm.cdf
def get_theta(S, K, r, sigma, T, otype='call'):
    """
    根据BSM模型计算期权的Theta值
    :param S: 标的资产当前价格
    :param K: 期权行权价格
    :param r: 无风险收益率
    :param sigma: 标的资产年化波动率
    :param T: 以年计算的期权到期时间
    :param otype: 'call'或'put'
    :return: 返回期权的Theta值
    """
    q = 0
    X = K
    d1 = (np.log(S / X) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N1 = stats.norm.cdf(d1)
    N2 = stats.norm.cdf(d2)
    N1_prime = stats.norm.pdf(d1)
    theta_c = -((S * np.exp(-q * T) * N1_prime * sigma) / (2 * np.sqrt(T))) - r * X * np.exp(-r * T) * N2
    if otype.lower()[0] == 'c':
        return theta_c
    else:
        return theta_c + r * X * np.exp(-r * T)

def get_gamma(S, K, r, sigma, T):
    """
    根据BSM模型计算期权的Gamma值
    :param S: 标的资产当前价格
    :param K: 期权行权价格
    :param r: 无风险收益率
    :param sigma: 标的资产年化波动率
    :param T: 以年计算的期权到期时间
    :return: 返回期权的Gamma值
    """
    q = 0
    X = K
    d1 = (np.log(S / X) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    N1_prime = stats.norm.pdf(d1)
    gamma = np.exp(-q * T) * N1_prime / (S * sigma * np.sqrt(T))
    return gamma

def get_vega(S, K, r, sigma, T):
    """
    定义函数计算期权的Vega
    param S: 标的资产当前价格
    param X: 期权行权价格
    param r: 复合无风险收益率
    param q: 基础资产分红收益率
    param sigma：标的资产年化波动率
    param T: 以年计算的期权到期时间
    return：返回期权的 Vega 值
    """
    q=0
    try:
        d1 = (np.log(np.exp(-q*T)*S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    except ZeroDivisionError:
        return 0.0
    else:
        N_dash = np.exp(1) ** (-d1 ** 2 / 2)/np.sqrt(2 * np.pi)
        return S * N_dash * np.sqrt(T)
    
def calc_price_bs(S, K, T, sigma, otype='call',r=0.03): 
    d1 = (np.log(S/K) + (r+1/2*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    pcall = S*N(d1) - K*np.exp(-r*T)*N(d2)
    pput = K*np.exp(-r*T)*N(-d2) - S*N(-d1)
    # print(otype)
    return pcall if otype=='call' else pput
    pass
#定义函数计算期权的Vega
def vega(S, X, r, q, sigma, T):
    """
    定义函数计算期权的Vega
    param S: 标的资产当前价格
    param X: 期权行权价格
    param r: 复合无风险收益率
    param q: 基础资产分红收益率
    param sigma：标的资产年化波动率
    param T: 以年计算的期权到期时间
    return：返回期权的 Vega 值
    """
    try:
        d1 = (np.log(np.exp(-q*T)*S/X) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    except ZeroDivisionError:
        return 0.0
    else:
        N_dash = np.exp(-d1 ** 2 / 2)/np.sqrt(2 * np.pi)
        return S * N_dash * np.sqrt(T)
def calc_implied_volatility(price, S, K, T, otype='call', r=0.03, sigma_min=0.001, sigma_max=5.0, precision=0.00001, max_error=0.00001):
    q = 0
    X = K
    sigma = 0.5
    exp_minus_qT = np.exp(-q*T)
    sqrt_2pi = np.sqrt(2 * np.pi)
    
    for i in range(100):
        V = calc_price_bs(S, K, T, sigma, otype, r)
        if abs(V - price) < precision:  # 精度达到要求
            return sigma
        try:
            d1 = (np.log(exp_minus_qT * S / X) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
            N_dash = np.exp(-d1 ** 2 / 2) / sqrt_2pi
            vega = S * N_dash * np.sqrt(T)
            sigma = sigma - (V - price) / vega  # 牛顿法迭代计算sigma
        except ZeroDivisionError:
            return np.nan
    return sigma
def get_delta(S, K, r, sigma, T, otype='call'):
    """
    根据BSM模型计算期权价格
    param S: 标的资产当前价格
    param X: 期权行权价格
    param r: 无风险收益率
    param q: 基础资产分红收益率
    param sigma：标的资产年化波动率
    param T: 以年计算的期权到期时间
    param opt_type: 'call'或'put'
    return 返回期权的delta值
    """
    q = 0
    X = K
    d1 = (np.log(S / X) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N1 = stats.norm.cdf(d1)
    delta_c = np.exp(-q * T) * N1
    if otype.lower()[0] == 'c':
        return delta_c
    else:
        return delta_c - 1
    
def trans_date(date_num):
    year = int(np.floor(date_num/10000))
    month = int(np.floor(((date_num - year*10000)/100)))
    day = date_num - year*10000-month*100
    time_datetime = datetime.datetime(year,month,day)
    return time_datetime

def get_columns(client, table_name):
    sql='''
    select table,name,type,comment
    from `system`.columns
    where table in('{}')
    '''.format(table_name)
    df = pd.DataFrame(client.execute(sql))
    return df.loc[:,1]

def import_chain(client, underlying_code, date):
    if date is None:
        sql = f"""
        select * from backbone.optionchain where underlying_code = '{underlying_code}' 
        """
    else:
        sql = f"""
        select * from backbone.optionchain where underlying_code = '{underlying_code}' and date = {date}
        """       
    chain = pd.DataFrame(client.execute(sql))
    chain.columns = get_columns(client, 'optionchain')
    chain.columns.name = ''
    return chain

def import_option_1m(client, stock_ids_str, date):
    if date is None:
        sql = f"""
            SELECT * FROM backbone.option_1m 
            WHERE DATE_FORMAT(datetime, '%H:%i:%s') IN (
                '09:31:00', '09:40:00', '10:00:00',
                '10:30:00', '11:00:00',
                '11:30:00', '13:01:00',
                '13:30:00', '14:00:00',
                '14:30:00', '14:50:00', '15:00:00'
            )
            AND StockID IN {stock_ids_str}
        """
    else:
        sql = f"""
            SELECT * FROM backbone.option_1m 
            WHERE DATE_FORMAT(datetime, '%H:%i:%s') IN (
                '09:31:00', '09:40:00', '10:00:00',
                '10:30:00', '11:00:00',
                '11:30:00', '13:01:00',
                '13:30:00', '14:00:00',
                '14:30:00', '14:50:00', '15:00:00'
            )
            AND StockID IN {stock_ids_str}
            AND DATE_FORMAT(datetime, '%Y%m%d') = '{date}'
        """
    option_hh = pd.DataFrame(client.execute(sql))
    option_hh.columns = get_columns(client, 'option_1m')
    option_hh.columns.name = ''
    return option_hh

def import_index_1m(client, underlying_code, date):
    if date is None:
        sql = f"""
            SELECT * FROM backbone.index_1m 
            WHERE StockID = '{underlying_code}'
            AND DATE_FORMAT(datetime, '%H:%i:%s') IN (
                '09:31:00', '09:40:00', '10:00:00',
                '10:30:00', '11:00:00',
                '11:30:00', '13:01:00',
                '13:30:00', '14:00:00',
                '14:30:00', '14:50:00', '15:00:00'
            )
            ORDER BY datetime DESC
        """
    else:
        sql = f"""
            SELECT * FROM backbone.index_1m 
            WHERE StockID = '{underlying_code}'
            AND DATE_FORMAT(datetime, '%H:%i:%s') IN (
                '09:31:00', '09:40:00', '10:00:00',
                '10:30:00', '11:00:00',
                '11:30:00', '13:01:00',
                '13:30:00', '14:00:00',
                '14:30:00', '14:50:00', '15:00:00'
            )
            AND DATE_FORMAT(datetime, '%Y%m%d') = '{date}'
            ORDER BY datetime DESC
        """
    index_hh = pd.DataFrame(client.execute(sql))
    index_hh.columns = get_columns(client, 'index_1m')
    index_hh.columns.name = ''
    return index_hh

def import_option_1m_bidask(client, stock_ids_str, date):
    if date is None:
        sql = f"""
            select * 
            from backbone.option_1m_bidask 
            where DATE_FORMAT(datetime,'%H:%i:%s') in 
                (   '09:31:00','09:40:00','10:00:00',
                    '10:30:00','11:00:00',
                    '11:30:00','13:01:00', 
                    '13:30:00','14:00:00',
                    '14:30:00','14:50:00','15:00:00') AND 
                StockID in {stock_ids_str} 
            order by datetime desc
        """
    else:
        sql = f"""
            select * 
            from backbone.option_1m_bidask 
            where DATE_FORMAT(datetime,'%H:%i:%s') in 
                (   '09:31:00','09:40:00','10:00:00',
                    '10:30:00','11:00:00',
                    '11:30:00','13:01:00', 
                    '13:30:00','14:00:00',
                    '14:30:00','14:50:00','15:00:00') AND 
                StockID in {stock_ids_str} AND
                toYYYYMMDD(datetime) = {date} 
            order by datetime desc
        """       
    option_bidask = pd.DataFrame(client.execute(sql))
    option_bidask.columns = get_columns(client, 'option_1m_bidask')
    option_bidask.columns.name = ''
    return option_bidask

def import_index_1m_bidask(client, underlying_code, date):
    if date is None:
        sql = f"""
            SELECT *
            FROM backbone.index_1m_bidask
            WHERE DATE_FORMAT(datetime,'%H:%i:%s') in 
                (   '09:31:00','09:40:00','10:00:00',
                    '10:30:00','11:00:00',
                    '11:30:00','13:01:00', 
                    '13:30:00','14:00:00',
                    '14:30:00','14:50:00','15:00:00') AND
                StockID = '{underlying_code}' 
            ORDER BY datetime
        """
    else:
        sql = f"""
            SELECT *
            FROM backbone.index_1m_bidask
            WHERE DATE_FORMAT(datetime,'%H:%i:%s') in 
                (   '09:31:00','09:40:00','10:00:00',
                    '10:30:00','11:00:00',
                    '11:30:00','13:01:00', 
                    '13:30:00','14:00:00',
                    '14:30:00','14:50:00','15:00:00') AND
                StockID = '{underlying_code}' AND
                toYYYYMMDD(datetime) = '{date}'
            ORDER BY datetime
        """
    index_1m_bidask = pd.DataFrame(client.execute(sql))
    index_1m_bidask.columns = get_columns(client, 'index_1m_bidask')
    index_1m_bidask.columns.name = ''
    return index_1m_bidask

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

############################### 主函数 ###################################
def import_data(client, underlying_code='SH510050',date = None):
    # 获取期权链数据
    chain = import_chain(client, underlying_code, date)
    # Chain表中StockName列和交易数据option_hh中的StockName列重复了
    chain.drop(columns = 'StockName',inplace = True)
    # 将日期转换为datetime格式
    chain['date'] = pd.to_datetime(chain['date'].astype('str'))

    # 获取特定标的的期权代码
    optionlist = chain['StockID'].unique()
    stock_ids_str = "("

    for i in optionlist:
        stock_ids_str += "\'"
        stock_ids_str += i
        stock_ids_str += "\'"
        stock_ids_str += ','

    stock_ids_str = stock_ids_str[:-1]  # 去除最后一个逗号
    stock_ids_str += ")"

    # 获取期权数据
    option_hh = import_option_1m(client, stock_ids_str, date)
    option_hh['date'] = option_hh['datetime'].dt.floor('D')
    # 合并期权交易数据和基本信息
    option_data = pd.merge(option_hh,chain,left_on=['date','StockID'],right_on=['date','StockID'],how='inner')

    # 获取标的数据
    index_hh = import_index_1m(client, underlying_code, date)
    # 合并期权和标的数据
    option_and_index = pd.merge(index_hh, option_data, left_on='datetime', right_on='datetime', how='inner', suffixes=('_index', '_option'))
    # 获取期权买卖价格数据
    option_bidask = import_option_1m_bidask(client, stock_ids_str, date)
    # 获取指数买卖价格数据
    index_bidask = import_index_1m_bidask(client,underlying_code, date)
    # 在 option_bidask 中，将所有列名都加上 "_option" 后缀，除了 'datetime'
    option_bidask.drop(columns='close', inplace=True)
    option_bidask.columns = ['{}_option'.format(col) if col not in ['datetime'] else col for col in option_bidask.columns]
    full_data_tmp = pd.merge(option_and_index, option_bidask, on=['datetime', 'StockID_option'])

    # 在 index_bidask 中，将所有列名都加上 "_index" 后缀，除了 'datetime'
    index_bidask.drop(columns='close', inplace=True)
    index_bidask.columns = ['{}_index'.format(col) if col not in ['datetime'] else col for col in index_bidask.columns]
    full_data = pd.merge(full_data_tmp, index_bidask, on=['datetime', 'StockID_index'])
    full_data['cdays_to_expire']  = (full_data['expiry_date'].apply(trans_date) - full_data['date'])
    full_data['T_calendar'] = full_data['cdays_to_expire'].apply(lambda x: x.days)/365
    full_data.loc[full_data['call_or_put']=='认购','call_or_put'] = 'call'
    full_data.loc[full_data['call_or_put']=='认沽','call_or_put'] = 'put'
    #print(full_data.columns)
    ###########################################
    #此处需要添加希腊字母计算的代码
    full_data['iv'] = full_data.apply(lambda x: calc_implied_volatility(price = x.close_option , 
                            S = x.close_index, 
                            K = x.exercise_price, 
                            T = x.T_calendar, 
                            otype=x.call_or_put,
                            r=0.03, 
                            sigma_min=0.001,
                            sigma_max=5.0, 
                            precision=0.00001,
                            max_error=0.00001), axis= 1)
    full_data['delta'] = full_data.apply(lambda x: get_delta( 
                            S = x.close_index, 
                            K = x.exercise_price,
                            sigma = x.iv, 
                            T = x.T_calendar, 
                            otype=x.call_or_put,
                            r=0.03), axis= 1)
    full_data['gamma'] = full_data.apply(lambda x: get_gamma( 
                                S = x.close_index, 
                                K = x.exercise_price,
                                r=0.03,
                                sigma = x.iv, 
                                T = x.T_calendar
                                ), axis= 1)
    full_data['theta'] = full_data.apply(lambda x: get_theta( 
                                S = x.close_index, 
                                K = x.exercise_price,
                                r=0.03,
                                sigma = x.iv, 
                                T = x.T_calendar,
                                otype=x.call_or_put
                                ), axis= 1)
    full_data['vega'] = full_data.apply(lambda x: get_vega( 
                                S = x.close_index, 
                                K = x.exercise_price,
                                r=0.03,
                                sigma = x.iv, 
                                T = x.T_calendar
                                ), axis= 1)
    full_data['close_mid_option'] = (full_data['ask1_option'] + full_data['bid1_option'])/2
    full_data['close_mid_index'] = (full_data['ask1_index'] + full_data['bid1_index'])/2
    full_data['cash_delta'] = full_data['delta']*full_data['close_mid_index']*full_data['multiplier']
    full_data['cash_gamma'] = full_data['gamma']*full_data['close_mid_index']*full_data['close_mid_index']*full_data['multiplier']*0.01
    full_data['cash_vega'] = full_data['vega']*full_data['multiplier']
    full_data['cash_theta'] = full_data['theta']*full_data['multiplier']
    ###########################################
    full_data.sort_values(by=['datetime','StockID_option'],ascending=True,inplace=True)
    full_data.index.name='index'
    full_data= full_data.groupby('date').apply(generate_label)
    full_data['time'] = full_data['datetime'].apply(lambda x: x.time())
    return full_data

if __name__ == '__main__':
    underlying_code='SH000852'
    #underlying_code='SH000852'
    '''[('SZ399001', '深证成指'),
        ('SZ399330', '深证100'),
        ('SH000001', '上证指数'),
        ('SH000016', '上证50'),
        ('SH000300', '沪深300'),
        ('SH510050', '50ETF'),
        ('SZ159901', '深100ETF'),
        ('SZ399006', '创业板指'),
        ('SH000852', '中证1000'),
        ('SH000905', '中证500'),
        ('SH510300', '300ETF'),
        ('SH510500', '500ETF'),
        ('SZ159915', '创业板'),
        ('SZ159919', '300ETF'),
        ('SZ159922', '500ETF'),
        ('SH000688', '科创50'),
        ('SH588000', '科创50'),
        ('SH588080', '科创板50'),
        ('CSI932000', '中证2000')]'''
    #date = '20230718'  #只提取1天
    date = None    #历史所有期
    client = Client(host='62.234.171.209',user='RUC_QUANT_READER',password='ruc_quant_reader_2023')
    data = import_data(client, underlying_code,date)
    if date is None:
        #open('0-九象限策略构建/data/dh_ret50.pkl.gz', 'wb').write(gzip.compress(pickle.dumps(data)))
        open('0-九象限策略构建/data/'+underlying_code+'_data.pkl.gz', 'wb').write(gzip.compress(pickle.dumps(data)))
        #data.to_csv(underlying_code+'_data.csv')
    else:
        open('0-九象限策略构建/data/dh_ret50_'+date+'.pkl.gz', 'wb').write(gzip.compress(pickle.dumps(data)))

