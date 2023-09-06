import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

def gen_lstm_prediction(daily_data,target_var='daily_return_ctc'):
    # 保存日期列
    daily_data.set_index('date', inplace=True)
    dates = daily_data.index[20:]

    factors = daily_data[['open', 'high', 'low', 'close', 'volume', 'vwap', 'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2']].values
    target = daily_data['daily_return_ctc'].values

    # 将数据重新形状为(samples, timesteps, features)
    data = np.array([factors[i-20:i] for i in range(20, len(factors))])
    target = target[20:]

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
    data = pd.read_pickle('./data/factors.pkl.gz')
    # 设置Date为index
    result_ret = gen_lstm_prediction(data,target_var='daily_return_ctc')
    result_ret.set_index('Date', inplace=True)
    result_ret.sort_index(inplace=True)
    # 画出真实的return和预测的return
    plt.figure(figsize=(12, 6))
    plt.plot(result_ret['Prediction'], label="Predictions")
    plt.plot(result_ret['Ground Truth'], label="Ground Truth")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.title("Return Prediction using LSTM")
    plt.show()
    result_ret.to_pickle('./data/return_prediction.pkl.gz')
    print("收益预测完毕！")
