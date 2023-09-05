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
from clickhouse_driver import Client
import gzip
import pickle

# SH510050 SH000852 SH510300 

def trans_date(date_num):
    year = int(np.floor(date_num/10000))
    month = int(np.floor(((date_num - year*10000)/100)))
    day = date_num - year*10000-month*100
    time_datetime = datetime.datetime(year,month,day)
    return time_datetime

def import_data(underlying_code='SH510050'): 
    underlying_code = "\'"  +underlying_code+"\'"
    ##########################Import data##############################
    ##########################option chain##############################
    sql = '''
    select * from backbone.optionchain where underlying_code =  
    '''+underlying_code
    chain = client.execute(sql)
    chain = pd.DataFrame(chain)

    sql='''
    select table,name,type,comment
    from `system`.columns
    where table in('optionchain')
    '''
    chain.columns = pd.DataFrame(client.execute(sql)).loc[:,1]
    chain.columns.name = ''
    ##########################option processing##############################
    optionlist=chain['StockID'].unique()
    optionlist
    str="("
    for i in optionlist:
        str+="\'"
        str+= i
        str+="\'"
        str+=','
    str+=")"
    str
    # Get option data
    sql = '''select * from backbone.option_1m where DATE_FORMAT(datetime,'%H:%i:%s') in 
                (   '09:31:00','09:40:00','10:00:00',
                    '10:30:00','11:00:00',
                    '11:30:00','13:01:00', 
                    '13:30:00','14:00:00',
                    '14:30:00','14:50:00','15:00:00') and StockID in ''' + str + '''order by datetime desc'''


    option_hh = client.execute(sql)
    option_hh = pd.DataFrame(option_hh)

    sql='''
    select table,name,type,comment
    from `system`.columns
    where table in('option_1m')
    '''
    option_hh.columns = pd.DataFrame(client.execute(sql)).loc[:,1]
    option_hh.columns.name = ''
    option_hh['price']= option_hh.apply(lambda x: x['open'] if (x['datetime'].time().strftime('%H:%M') == '09:31')|(x['datetime'].time().strftime('%H:%M') == '13:31')  else x['close'] , axis=1)
    ##########################index data##############################
    sql = '''
    select * from backbone.index_1m where StockID = '''+underlying_code+''' and DATE_FORMAT(datetime,'%H:%i:%s') in 
                (   '09:31:00','09:40:00','10:00:00',
                    '10:30:00','11:00:00',
                    '11:30:00','13:01:00', 
                    '13:30:00','14:00:00',
                    '14:30:00','14:50:00','15:00:00') order by datetime desc
    '''
    index_hh = res = client.execute(sql)
    index_hh = pd.DataFrame(index_hh)
    sql='''
    select table,name,type,comment
    from `system`.columns
    where table in('index_1m')
    '''
    index_hh.columns = pd.DataFrame(client.execute(sql)).loc[:,1]
    index_hh.columns.name = ''
    ##########################list of trading days##############################
    sql = '''
    select * from backbone.calendar 
    '''
    tradingdays = client.execute(sql)
    tradingdays = pd.DataFrame(tradingdays)

    sql='''
    select table,name,type,comment
    from `system`.columns
    where table in('calendar')
    '''
    tradingdays.columns = pd.DataFrame(client.execute(sql)).loc[:,1]
    tradingdays.columns.name = ''
    tradingdays = tradingdays.loc[(tradingdays['istradingday']==True)]
    tradingdays['date1'] = tradingdays['date'].apply(trans_date)
    tradingday_list = list(tradingdays['date1'].drop_duplicates())
    tradingdaynum_list = list(tradingdays['date'].drop_duplicates())
    ##########################option selection##############################
    option_hh['date'] = option_hh['datetime'].apply(lambda x: x.date().year*10000+x.date().month*100+x.date().day)
    tmp = pd.merge(chain,option_hh,on = ['StockID','date'],how ='right')
    tmp1 = index_hh.loc[:,["datetime","close"]]
    tmp1.rename(columns={'close':'index_close'}, inplace = True)
    tmp2 = pd.merge(tmp,tmp1,on = ['datetime'],how ='left')
    # 查找所有50ETF的虚值期权 find OTM and ATM options
    chain_select = tmp2
    chain_select['K/S']=chain_select['exercise_price']/chain_select['index_close'] * 100
    #chain_select = chain_select.loc[((chain_select['call_or_put']=='认购')&(chain_select['K/S']>=100))|((chain_select['call_or_put']=='认沽')&(chain_select['K/S']<=100))]
    chain_select['cdays_to_expire']  = (chain_select['expiry_date'].apply(trans_date) - chain_select['date'].apply(trans_date))
    
    chain_select['T_calendar'] = chain_select['cdays_to_expire'].apply(lambda x: x.days)/365
    chain_select['otype'] = chain_select['call_or_put'].apply(lambda x: 'call' if x == '认购' else 'put' )

    #get option bid ask
    # Get option data
    sql = '''select * from backbone.option_1m_bidask where DATE_FORMAT(datetime,'%H:%i:%s') in 
                (   '09:31:00','09:40:00','10:00:00',
                    '10:30:00','11:00:00',
                    '11:30:00','13:01:00', 
                    '13:30:00','14:00:00',
                    '14:30:00','14:50:00','15:00:00') and StockID in ''' + str + '''order by datetime desc'''


    option_bidask = client.execute(sql)
    option_bidask = pd.DataFrame(option_bidask)

    sql='''
    select table,name,type,comment
    from `system`.columns
    where table in('option_1m_bidask')
    '''
    option_bidask.columns = pd.DataFrame(client.execute(sql)).loc[:,1]
    option_bidask.columns.name = ''
    option_bidask

    ##index_1m_bidask
    sql = '''
    select * from backbone.index_1m_bidask where StockID = '''+underlying_code+''' and DATE_FORMAT(datetime,'%H:%i:%s') in 
                (   '09:31:00','09:40:00','10:00:00',
                    '10:30:00','11:00:00',
                    '11:30:00','13:01:00', 
                    '13:30:00','14:00:00',
                    '14:30:00','14:50:00','15:00:00') order by datetime
    '''
    res = client.execute(sql)
    index_1m_bidask = pd.DataFrame(res)
    sql='''
    select table,name,type,comment
    from `system`.columns
    where table in('index_1m_bidask')
    '''
    index_1m_bidask.columns = pd.DataFrame(client.execute(sql)).loc[:,1]
    index_1m_bidask.columns.name = ''
    #index_1m_bidask.index = index_1m_bidask['datetime']
    index_1m_bidask
    #index_1m_bidask = pd.read_csv(r"index_bidask_SH510050.csv",sep=',')
    index_1m_bidask['index_close_mid'] =( index_1m_bidask['ask1']+index_1m_bidask['bid1'])/2
    index_1m_bidask['index_bidask_spread'] = index_1m_bidask['ask1']-index_1m_bidask['bid1']

    index_1m_bidask

    # half-hour iv and greeks
    from datetime import datetime,date,time 
    import scipy.stats as st
    import numpy as np
    N = st.norm.cdf

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

    chain_select['iv'] = chain_select.apply(lambda x: calc_implied_volatility(price = x.close , 
                            S = x.index_close, 
                            K = x.exercise_price, 
                            T = x.T_calendar, 
                            otype=x.otype,
                            r=0.03, 
                            sigma_min=0.001,
                            sigma_max=5.0, 
                            precision=0.00001,
                            max_error=0.00001), axis= 1)
    chain_select.loc[:,['StockID','datetime','iv']]

    from scipy import stats
    # 计算期权delta
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
        
    chain_select['delta'] = chain_select.apply(lambda x: get_delta( 
                            S = x.index_close, 
                            K = x.exercise_price,
                            sigma = x.iv, 
                            T = x.T_calendar, 
                            otype=x.otype,
                            r=0.03), axis= 1)
    chain_select.loc[:,['StockID','datetime','delta']]

    option50 = chain_select



    #50ETF数据 optionchain
    sql = '''
    select * from backbone.optionchain where underlying_code = '''+underlying_code

    chain = client.execute(sql)
    chain = pd.DataFrame(chain)

    sql='''
    select table,name,type,comment
    from `system`.columns
    where table in('optionchain')
    '''
    chain.columns = pd.DataFrame(client.execute(sql)).loc[:,1]
    chain.columns.name = ''
    chain

    optionlist=chain['StockID'].unique()

    optionlist
    str="("
    for i in optionlist:
        str+="\'"
        str+= i
        str+="\'"
        str+=','
    str+=")"
    str

    #option_bidask = pd.read_csv(r"option_bidask_SH510050.csv",sep=',')
    option_bidask['price_mid'] =( option_bidask['ask1']+option_bidask['bid1'])/2
    option_bidask['option_bidask_spread'] = option_bidask['ask1']-option_bidask['bid1']

    #option50_1 = option50.loc[:,['StockID','StockName_x','datetime','price','call_or_put','K/S','T_calendar','volume','index_close','delta']]
    option50_1 = option50.copy()
    option50_1.sort_values(by=['StockID','datetime'],ascending=True,inplace=True)
    option50_1 = pd.merge(option50_1,option_bidask[['StockID','datetime','price_mid','option_bidask_spread','ask1','bid1']],on = ['StockID','datetime'],how = 'left')
    option50_1.rename(columns={'ask1':'option_ask1','bid1':'option_bid1'}, inplace = True)
    option50_1 = pd.merge(option50_1,index_1m_bidask[['datetime','index_close_mid','index_bidask_spread','ask1','bid1']],on = ['datetime'],how = 'left')
    option50_1.rename(columns={'ask1':'index_ask1','bid1':'index_bid1'}, inplace = True)
    option50_1['P&L'] = option50_1.groupby('StockID')['price_mid'].diff()-(option50_1.groupby('StockID')['delta'].shift(1))*option50_1.groupby('StockID')['index_close_mid'].diff()
    option50_1['ret'] = option50_1['P&L']/option50_1.groupby('StockID')['price_mid'].shift(1) 
    return option50_1


if __name__ == '__main__':
    client = Client(host='62.234.171.209',user='RUC_QUANT_READER',password='ruc_quant_reader_2023')
    dat = import_data('SH510050')
    #dat.to_csv(r"data/dh_ret50_new1_0719.csv",index=False)
    open('data/dh_ret50_new1.pkl.gz', 'wb').write(gzip.compress(pickle.dumps(dat)))



