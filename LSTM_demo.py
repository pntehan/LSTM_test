# 数据预处理

# import pandas as pd
# import matplotlib.pyplot as plt
#
# def parser(x):
# 	return pd.datetime.strptime('190'+x, '%Y-%m')
#
# # 读取数据
# series = pd.read_csv('sales-of-shampoo-over-a-three-ye.csv',
# 					header=0, parse_dates=[0], index_col=0,
# 					squeeze=True, date_parser=parser)
#
# # 划分数据
# x = series.values
# train, test = x[0:-12], x[-12:]
#
# # # 前向模型验证
# # # 已t-1时刻的数据作为t时刻的数据
# # from sklearn.metrics import mean_squared_error
# # from math import sqrt
# #
# # history = [x for x in train]
# # predictions = list()
# # for i in range(len(test)):
# # 	# 预测
# # 	predictions.append(history[-1])
# # 	# 观测
# # 	history.append(test[i])
# # rmse = sqrt(mean_squared_error(test, predictions))
# # print('RMSE:%.3f'%rmse)
# # plt.plot(test, color='blue')
# # plt.plot(predictions, color='red')
# # plt.show()
#
# # 时间序列转换为监督学习
# def timeseries_to_supervised(data, lag=1):
# 	df = pd.DataFrame(data)
# 	columns = [df.shift(i) for i in range(1, lag+1)]
# 	columns.append(df)
# 	df = pd.concat(columns, axis=1)
# 	df.fillna(0, inplace=True)
# 	return df
#
# # supervised = timeseries_to_supervised(x, 1)
# # print(supervised)
#
# # 做差分使时间序列趋于平稳
# def difference(dataset, interval=1):
# 	diff = list()
# 	for i in range(interval, len(dataset)):
# 		value = dataset[i] - dataset[i - interval]
# 		diff.append(value)
# 	return pd.Series(diff)
#
# # 差分的复原
# def inverse_difference(history, yhat, interval=1):
# 	return yhat + history[-interval]
#
# # print(series.head())
# differenced = difference(series, 1)
# # print(differenced.head())
# inverted = list()
# for i in range(len(differenced)):
# 	value = inverse_difference(series, differenced[i], len(series)-i)
# 	inverted.append(value)
# inverted = pd.Series(inverted)
# # print(inverted.head())
#
# # 差分后的数据再次绘图，趋于平稳
# # plt.plot(series.values, label='Origin')
# # plt.plot(differenced, label='diff')
# # plt.legend()
# # plt.show()
#
# # 数据的归一化处理
# from sklearn.preprocessing import MinMaxScaler
#
# y = x.reshape(len(x), 1)
# # 构建归一化模型
# scaler = MinMaxScaler(feature_range=(-1, 1))
# # 带入模型
# scaler = scaler.fit(y)
# scaler_y = scaler.transform(y)
#
# scaler_series = pd.Series(scaler_y[:, 0])
# print(scaler_series.head())
#
# # 还原归一化数据
# inverted_y = scaler.inverse_transform(scaler_y)
# inverted_series = pd.Series(inverted_y[:, 0])
# print(inverted_series.head())

# LSTM模型构建
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

def parser(x):
	# 将数据集转换为符合要求的原始时间序列
	return pd.datetime.strptime('190'+x, '%Y-%m')

def timeseries_to_supervised(data, lag=1):
	# 时间序列转为监督学习
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

def difference(dataset, interval=1):
	# 将时间序列数据进行差分处理，使其趋于平稳
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

def inverse_difference(history, yhat, interval=1):
	# 还原差分数据
	return yhat + history[-interval]

def scale(train, test):
	# 将测试集数据和训练集数据进行归一化处理
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler =scaler.fit(train)
	# 将数据带入模型
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

def invert_scale(scaler, x, value):
	# 将归一化的数据还原
	new_row = [i for i in x] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

def fit_lstm(train, batch_size, rb_epoch, neurons):
	# 将训练集数据带入LSTM模型
	x, y = train[:, 0:-1], train[:, -1]
	x = x.reshape(x.shape[0], 1, x.shape[1])
	# 使用序贯模型
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, x.shape[1], x.shape[2]), stateful=True))
	model.add(Dense(1))
	# 编译用来配置模型的学习过程，此处配置了优化器和损失函数
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(rb_epoch):
		model.fit(x, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

def forecast_lstm(model, batch_size, x):
	# 预测数据
	x = x.reshape(1, 1, len(x))
	yhat = model.predict(x, batch_size=batch_size)
	return yhat[0, 0]

# 按照要求读取时间序列数据
series = pd.read_csv('sales-of-shampoo-over-a-three-ye.csv',
					header=0, parse_dates=[0],
					index_col=0, squeeze=True,
					date_parser=parser)

# 提取数据
raw_values = series.values
# 对数据进行一阶差分，返回的是长度减一的数组
diff_values = difference(raw_values, 1)
# 对差分数据进行监督学习的转换，以t-1的观测值作为输入指向t时刻的输出
supervised = timeseries_to_supervised(diff_values, 1)
# 提取监督学习的对应关系
supervised_values = supervised.values
# 划分数据为训练集与测试集
train, test = supervised_values[0:-12], supervised_values[-12:]
# 对训练集数据和测试集数据进行归一化处理
scaler, train_scaled, test_scaled = scale(train, test)
# 将训练集数据带入LSTM模型，并且训练数据3000次
lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
# 去除训练集的输入值进行数组变换
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
# 预测数据模型
lstm_model.predict(train_reshaped, batch_size=1)

predictions = list()
for i in range(len(test_scaled)):
	# 取测试集的输入与输出
	x, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	# 将测试集的输入带入进行预测，返回预测值
	yhat = forecast_lstm(lstm_model, 1, x)
	# 将预测出的数据还原成正常数据
	yhat = invert_scale(scaler, x, yhat)
	# 进行反差分
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# 存入预测集
	predictions.append(yhat)
	# 期望值
	expected = raw_values[len(train)+i+1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# 求预测集的均方根误差
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(raw_values[-12:], label='Real')
plt.plot(predictions, label='Pred')
plt.legend()
plt.show()





















