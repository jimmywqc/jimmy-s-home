# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 22:06:06 2020

@author: Administrator
"""


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv1D,MaxPooling1D,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

"""
数据预处理
"""
#把所有数据放在一张表上
features=pd.read_csv("features.csv").drop(columns="IsHoliday")
stores=pd.read_csv("stores.csv")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
dataset=train.merge(stores,how="left").merge(features,how="left")#merge 多个表格融合，保留train文件中对应的所有信息，其余文件找到有相同的对应行

#删掉2011-11-11以后的数据
dataset=dataset[dataset.Date >= '2011-11-11']
dataset=dataset.fillna(0)#将NAN填充为0
dataset =pd.get_dummies(dataset, columns=["Type",'IsHoliday']) #one-hot 编码
dataset['Month']=pd.to_datetime(dataset['Date']).dt.month##pandas 还可以时间相减
dataset['Year']=pd.to_datetime(dataset['Date']).dt.year
dataset['Day']=pd.to_datetime(dataset['Date']).dt.day
dataset=dataset.drop(columns='Date')


def conv1_nn(train_x,train_y,test_x):
    train_x=train_x.reshape(train_x.shape[0],20,1)
    test_x =test_x.reshape(test_x.shape[0],20,1)
    model=Sequential()
    model.add(Conv1D(15, 3, activation='relu', input_shape=(20, 1)))#尝试采用一维卷积
    print(model.output_shape)
    model.add(Conv1D(7, 3, activation='relu'))
    print(model.output_shape)
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='MSE', optimizer='adam')
    model.fit(train_x,train_y, epochs=30, batch_size=128,verbose=2, shuffle=False) #可以添加validation
    test_y=model.predict(test_x)
    return test_y
#
def calculate_error(test_y, predicted, weights):
    a=mean_absolute_error(test_y, predicted, sample_weight=weights)
    return a


#构建训练集
train_y=dataset.Weekly_Sales.values #dataset.Weekly_Sales会输出一个series的类型，需要输出其values，也可以输出索引index
train_y=train_y.reshape(-1,1) #一维的array  需要重新输出其格式（-1，1），不然无法归一化
train_x=dataset.drop(columns='Weekly_Sales')


#0-1标准化
scaler1 = MinMaxScaler(feature_range=(0, 1))
train_x = scaler1.fit_transform(train_x)
scaler2 = MinMaxScaler(feature_range=(0, 1))
train_y = scaler2.fit_transform(train_y)
       

#十折交叉生成数据 并调用参数
error=[]
kf=KFold(n_splits=5)
min_error=np.float64('inf')
for train__x_index, validation_x_index in kf.split(train_x):
    train_x_new=train_x[train__x_index]
    train_y_new=train_y[train__x_index]
    validation_x=train_x[validation_x_index]
    validation_y=train_y[validation_x_index]
    validation_y=scaler2.inverse_transform(validation_y)
#构建权重
    weights=np.zeros(shape=(len(validation_x),1))
    for i in range (int(len(validation_x))):
        if validation_x[i,15]==1:
            weights[i]=1
        else:
            weights[i]=5
#一维卷积   
    validation_y_conv1_nn=conv1_nn(train_x_new,train_y_new,validation_x).reshape(-1,1)
    validation_y_conv1_nn=scaler2.inverse_transform(validation_y_conv1_nn)
    error_conv1_nn=calculate_error(validation_y,validation_y_conv1_nn,weights)
    error.append(error_conv1_nn)
    print(error_conv1_nn)
    print('='*50)
    
    
test=test.merge(stores,how="left").merge(features,how="left")
test=test.fillna(0)#将NAN填充为0       
test =pd.get_dummies(test, columns=["Type",'IsHoliday']) 
test['Month']=pd.to_datetime(test['Date']).dt.month 
test['Year']=pd.to_datetime(test['Date']).dt.year
test['Day']=pd.to_datetime(test['Date']).dt.day

#误差的权重  
IsHoliday_False=test.IsHoliday_False.values.reshape(-1,1)
weights_test=np.zeros(shape=(len(IsHoliday_False),1))
for i in range (int(len(IsHoliday_False))):
    if IsHoliday_False[i]==1:
        weights_test[i]=1
    else:
        weights_test[i]=5

#归一化
test_x=test
test_x=test_x.drop(columns='Date')
scaler3=MinMaxScaler(feature_range=(0,1))
test_x=scaler3.fit_transform(test_x)

#调用模型
predict_y=conv1_nn(train_x,train_y,test_x).reshape(-1,1)
predict_y=scaler2.inverse_transform(predict_y)

#输出csv
out_put_data=pd.DataFrame(np.zeros(shape=(len(predict_y),2)),columns=['id','Weekly_Sales'])
out_put_data['Weekly_Sales']=predict_y
out_put_data['id']=test['Store'].astype(str)+'_'+test['Dept'].astype(str)+'_'+test['Date'].astype(str)#astype numpy中将对应的数据类型转换为另一种 #dtype是展示对应的数据了类型
out_put_data.to_csv('data_predict.csv',index=False)
