import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip

def gen_lstm_prediction(daily_data,target_var='daily_return_ctc'):
    # 保存日期列
    daily_data.set_index('date', inplace=True)
    dates = daily_data.index[60:]


    #为什么加入128个因子预测出来就是空值？
    #factors = daily_data[[ 'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2']].values
    factors = daily_data[['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0',\
                           'LOW0', 'VWAP0', 'ROC5', 'MA5', 'STD5', 'BETA5', 'RSQR5', 'RESI5', 'MAX5', 'MIN5', 'QTLU5', \
                            'QTLD5', 'RANK5', 'RSV5', 'IMAX5', 'IMIN5', 'IMXD5', 'CORR5', 'CORD5', 'CNTP5', 'SUMP5', \
                            'VMA5', 'VSTD5', 'WVMA5', 'VSUMP5', 'ROC10', 'MA10', 'STD10', 'BETA10', 'RSQR10', 'RESI10',\
                            'MAX10', 'MIN10', 'QTLU10', 'QTLD10', 'RANK10', 'RSV10', 'IMAX10', 'IMIN10', 'IMXD10',\
                            'CORR10', 'CORD10', 'CNTP10', 'SUMP10', 'VMA10', 'VSTD10', 'WVMA10', 'VSUMP10', 'ROC20', \
                            'MA20', 'STD20', 'BETA20', 'RSQR20', 'RESI20', 'MAX20', 'MIN20', 'QTLU20', 'QTLD20', 'RANK20',\
                             'RSV20', 'IMAX20', 'IMIN20', 'IMXD20', 'CORR20', 'CORD20', 'CNTP20', 'SUMP20', 'VMA20', \
                            'VSTD20', 'WVMA20', 'VSUMP20', 'ROC30', 'MA30', 'STD30', 'BETA30', 'RSQR30', 'RESI30', 'MAX30',\
                            'MIN30', 'QTLU30', 'QTLD30', 'RANK30', 'RSV30', 'IMAX30', 'IMIN30', 'IMXD30', 'CORR30',\
                             'CORD30', 'CNTP30', 'SUMP30', 'VMA30', 'VSTD30', 'WVMA30', 'VSUMP30', 'ROC60', 'MA60', \
                            'STD60', 'BETA60', 'RSQR60', 'RESI60', 'MAX60', 'MIN60', 'QTLU60', 'QTLD60', 'RANK60', 'RSV60', \
                            'IMAX60', 'IMIN60', 'IMXD60', 'CORR60', 'CORD60', 'CNTP60', 'SUMP60', 'VMA60', 'VSTD60', 'WVMA60',\
                             'VSUMP60']].values
    target = daily_data['daily_return_ctc'].shift(-1).values

    # 将数据重新形状为(samples, timesteps, features)
    data = np.array([factors[i-60:i] for i in range(60, len(factors))])
    target = target[60:]

    # 找到2019年的索引位置
    split_index = np.where(dates.year == 2019)[0][0]

    # 根据2019年的位置划分数据集
    data_train = data[:split_index]
    data_test = data[split_index:]
    target_train = target[:split_index]
    target_test = target[split_index:]
    dates_train = dates[:split_index]
    dates_test = dates[split_index:]

    # 缩放数据
    scaler_data = MinMaxScaler()
    data_train = scaler_data.fit_transform(data_train.reshape(-1, data_train.shape[-1])).reshape(data_train.shape)
    data_test = scaler_data.transform(data_test.reshape(-1, data_test.shape[-1])).reshape(data_test.shape)

    scaler_target = MinMaxScaler()
    target_train = scaler_target.fit_transform(target_train.reshape(-1, 1))
    target_test = scaler_target.transform(target_test.reshape(-1, 1))

    # 定义模型
    model = Sequential()
    model.add(LSTM(32, input_shape=(data_train.shape[1], data_train.shape[2])))
    model.add(Dense(1))

    # 编译并训练模型
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(data_train, target_train, epochs=10, batch_size=32, validation_split=0.2)

    # 预测下一个Return
    next_return = model.predict(data_test)

    # 反归一化预测结果
    next_return = scaler_target.inverse_transform(next_return)

    # 反归一化真实结果
    target_test = scaler_target.inverse_transform(target_test)

    # 创建一个DataFrame用于保存日期，预测的返回值和实际的返回值
    result_ret = pd.DataFrame({
        'Date': dates_test,
        'Prediction': next_return.flatten(),
        'Ground Truth': target_test.flatten()
    })

    return result_ret

if __name__ == '__main__':
    underlying_code = 'SH510050'
    path = 'data/'+underlying_code+'_factors.pkl.gz'
    data = pickle.loads(gzip.decompress(open(path, 'rb').read()))
    # 设置Date为index
    result_ret = gen_lstm_prediction(data,target_var='daily_return_ctc')
    result_ret.set_index('Date', inplace=True)
    result_ret.sort_index(inplace=True)
    # 画出真实的return和预测的return
    plt.figure(figsize=(12, 6))
    plt.plot(result_ret['Ground Truth'], label="Ground Truth")
    plt.plot(result_ret['Prediction'], label="Predictions")

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.title("Return Prediction using LSTM")
    plt.show()
    open('data/'+underlying_code+'_return_prediction.pkl.gz', 'wb').write(gzip.compress(pickle.dumps(result_ret)))
    print("收益预测完毕！")
