import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip

underlying_code = 'SH510050'
path = 'data/'+underlying_code+'_pnl.pkl.gz'
pnl = pickle.loads(gzip.decompress(open(path, 'rb').read()))
pnl.reset_index(inplace=True)
pnl['date'] = pd.to_datetime(pnl['date'].astype(str))
pnl.drop_duplicates(subset=['date'],inplace=True)

full_data = pickle.loads(gzip.decompress(open('data/'+underlying_code+'_pnl.pkl.gz', 'rb').read())).sort_index()
result_ret = pickle.loads(gzip.decompress(open('data/'+underlying_code+'_return_prediction.pkl.gz', 'rb').read()))
result_vol = pickle.loads(gzip.decompress(open('data/'+underlying_code+'_rv_prediction.pkl.gz', 'rb').read()))

#full_data = pd.merge(daily_data,pnl,on='date',how='left')
full_data.dropna(inplace=True)

full_data['日期'] = full_data.index
full_data['卖购收益'] = full_data['call_pnl']
full_data['卖沽收益'] = full_data['put_pnl']
full_data['双卖收益'] = full_data['pnl']
full_data['买购收益'] = - (full_data['call_pnl'] - 1.8) - 1.8
full_data['买沽收益'] = - (full_data['put_pnl'] - 1.8) - 1.8
full_data['双买收益'] = - (full_data['pnl'] - 3.6) - 3.6

# 合并预测数据和真实数据
#full_prediction_result = pd.merge(full_data, result_ret, left_index=True, right_on='Date', how='left')
#full_prediction_result = pd.merge(full_prediction_result, result_vol, left_index=True, right_on='Date', how='left')
full_prediction_result = full_data.copy()
full_prediction_result['Prediction_ret'] = result_ret['Prediction']
full_prediction_result['true_ret'] = result_ret['Ground Truth']
full_prediction_result['Prediction_vol'] = result_vol['Prediction']
full_prediction_result['true_vol'] = result_vol['Ground Truth']

# 确定滚动窗口大小
window_size = 20

# 计算每20天的收益率阈值
upper_threshold_return = full_prediction_result['Prediction_ret'].rolling(window_size).apply(lambda x: x.quantile(0.66))
lower_threshold_return = full_prediction_result['Prediction_ret'].rolling(window_size).apply(lambda x: x.quantile(0.33))

# 计算每20天的波动率阈值
upper_threshold_volatility = full_prediction_result['Prediction_vol'].rolling(window_size).apply(lambda x: x.quantile(0.66))
lower_threshold_volatility = full_prediction_result['Prediction_vol'].rolling(window_size).apply(lambda x: x.quantile(0.33))


# Create the '涨跌标签' column
full_prediction_result.loc[full_prediction_result['Prediction_ret'] > upper_threshold_return, '涨跌标签'] = '价格大涨'
full_prediction_result.loc[full_prediction_result['Prediction_ret'] < lower_threshold_return, '涨跌标签'] = '价格下跌'
full_prediction_result.loc[(full_prediction_result['Prediction_ret'] >= lower_threshold_return) & (full_prediction_result['Prediction_ret'] <= upper_threshold_return), '涨跌标签'] = '价格不变'

# Create the '波动率标签' column
full_prediction_result.loc[full_prediction_result['Prediction_vol'] > upper_threshold_volatility, '波动率标签'] = '波动率较大'
full_prediction_result.loc[full_prediction_result['Prediction_vol'] < lower_threshold_volatility, '波动率标签'] = '波动率较小'
full_prediction_result.loc[(full_prediction_result['Prediction_vol'] >= lower_threshold_volatility) & (full_prediction_result['Prediction_vol'] <= upper_threshold_volatility), '波动率标签'] = '波动率适中'

full_prediction_result.tail(10)


def strategy1(row):
    if row['涨跌标签']=='价格大涨':
        if row['波动率标签']=='波动率较大':
            return row['买购收益']
        elif row['波动率标签']=='波动率适中' or row['波动率标签']=='波动率较小':
            return row['卖沽收益']
    elif row['涨跌标签']=='价格不变':
        if row['波动率标签']=='波动率较大':
            return row['双买收益'] 
        elif row['波动率标签']=='波动率适中' or row['波动率标签']=='波动率较小':
            return row['双卖收益'] 
    elif row['涨跌标签']=='价格下跌':
        if row['波动率标签']=='波动率较大':
            return row['买沽收益']  
        elif row['波动率标签']=='波动率适中' or row['波动率标签']=='波动率较小':
            return row['卖购收益']  
    else:
        return row['put_pnl']
    
def strategy2(row):
    if row['波动率标签']=='波动率较大':
        return 0  # '不操作'，返回0
    elif row['波动率标签']=='波动率适中' or row['波动率标签']=='波动率较小':
        return row['双卖收益']  # 假设你有一个名为'双卖收益'的列
    else:
        return 0
    
def strategy3(row):
    if row['波动率标签'] == '波动率较大':
        return row['双买收益']  # '不操作'，返回0
    elif row['波动率标签'] in ['波动率适中', '波动率较小']:
        return row['双卖收益']  
    else:
        return 0
    

full_prediction_result['策略1收益'] = full_prediction_result.apply(strategy1, axis=1)
full_prediction_result['策略2收益'] = full_prediction_result.apply(strategy2, axis=1)
full_prediction_result['策略3收益'] = full_prediction_result.apply(strategy3, axis=1)

full_prediction_result.index = full_prediction_result['日期']
test = full_prediction_result.loc[(full_prediction_result['日期']>=pd.to_datetime('2019-01-01'))]
# 然后画出'双卖收益'的累积和
plt.plot(10000+test['双卖收益'].cumsum(), label='straddle')
plt.plot(10000+test['策略3收益'].cumsum(), label='strategy 1')
plt.plot(10000+test['策略2收益'].cumsum(), label='strategy 2')
plt.plot(10000+test['策略1收益'].cumsum(), label='strategy 3')

# 添加图例
plt.legend()

# 显示图形
plt.show()