# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:27:35 2020

@author: Administrator
"""

import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM, Flatten,Conv2D,MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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



##疯狂调用回归类型的包 sklearn 中有 分类 回归 聚类 降维 文本挖掘 模型优化 数据预处理
def knn(train_x,train_y,test_x,k):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(train_x,train_y)
    test_y=knn.predict(test_x)
    return test_y

def extraTreesRegressor(train_x,train_y,test_x):
    clf = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1,n_jobs=1)
    clf.fit(train_x,train_y)
    test_y=clf.predict(test_x)
    return test_y

def randomForestRegressor(train_x,train_y,test_x):
    clf = RandomForestRegressor(n_estimators=100,max_features='log2', verbose=1)
    clf.fit(train_x,train_y)
    test_y=clf.predict(test_x)
    return test_y

def svm(train_x,train_y,test_x):
    clf = SVR(kernel='rbf', gamma='auto')
    clf.fit(train_x,train_y)
    test_y=clf.predict(test_x)
    return test_y

def nn(train_x,train_y,test_x):
    clf = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', verbose=1)
    clf.fit(train_x,train_y)
    test_y=clf.predict(test_x)
    return test_y


def build_LSTM(train_x,train_y,test_x):
    train_x=train_x.reshape(train_x.shape[0],1,train_x.shape[1])
    test_x =test_x.reshape(test_x.shape[0],1,test_x.shape[1])
    model=Sequential()
    model.add(LSTM(64, input_shape=(train_x.shape[1],train_x.shape[2]),return_sequences=True))
    model.add(LSTM(16,return_sequences=False))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='MSE', optimizer='adam')
    model.fit(train_x,train_y, epochs=50, batch_size=72,verbose=2, shuffle=False) #可以添加validation
    test_y=model.predict(test_x)
    return test_y

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
kf=KFold(n_splits=10)
min_error=np.float32('inf')
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
#knn    
    validation_y_knn=knn(train_x_new,train_y_new,validation_x,5).reshape(-1,1)
    validation_y_knn=scaler2.inverse_transform(validation_y_knn)
    error_knn=calculate_error(validation_y,validation_y_knn,weights)
# extraTreesRegressor  
    validation_y_extraTreesRegressor=extraTreesRegressor(train_x_new,train_y_new,validation_x).reshape(-1,1)
    validation_y_extraTreesRegressor=scaler2.inverse_transform(validation_y_extraTreesRegressor)
    error_extraTreesRegressor=calculate_error(validation_y,validation_y_extraTreesRegressor,weights)
#randomForestRegressor
    validation_y_randomForestRegressor=randomForestRegressor(train_x_new,train_y_new,validation_x).reshape(-1,1)
    validation_y_randomForestRegressor=scaler2.inverse_transform(validation_y_randomForestRegressor)
    error_randomForestRegressor=calculate_error(validation_y,validation_y_randomForestRegressor,weights)
#svr    
    validation_y_svm=svm(train_x_new,train_y_new,validation_x).reshape(-1,1)
    validation_y_svm=scaler2.inverse_transform(validation_y_svm)
    error_svm=calculate_error(validation_y,validation_y_svm,weights)
#nn    
    validation_y_nn=nn(train_x_new,train_y_new,validation_x).reshape(-1,1)
    validation_y_nn=scaler2.inverse_transform(validation_y_nn)
    error_nn=calculate_error(validation_y,validation_y_nn,weights)
#LSTM    
    validation_y_build_LSTM=build_LSTM(train_x_new,train_y_new,validation_x).reshape(-1,1)
    validation_y_build_LSTM=scaler2.inverse_transform(validation_y_build_LSTM)
    error_build_LSTM=calculate_error(validation_y,validation_y_build_LSTM,weights)
#输出最好模型
    list=[error_knn,error_extraTreesRegressor,error_randomForestRegressor,error_svm,error_nn,error_build_LSTM]
    error_index=list.index(min(list)) #返回index
    if list[error_index] < min_error:
        min_error=min(list)
        best_model_index=error_index
        print('='*50)
        
        
if best_model_index==0:
    print('最优模型为knn')
if best_model_index==1:
    print('最优模型为extraTreesRegressor')
if best_model_index==2:
    print('最优模型为randomForestRegressor')
if best_model_index==3:
    print('最优模型为svr')
if best_model_index==4:
    print('最优模型为nn')
if best_model_index==5:
    print('最优模型为lstm')

    
    



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

#调用最优模型
predict_y=extraTreesRegressor(train_x,train_y,test_x).reshape(-1,1)
predict_y=scaler2.inverse_transform(predict_y)

#输出csv
out_put_data=pd.DataFrame(np.zeros(shape=(len(predict_y),2)),columns=['id','Weekly_Sales'])
out_put_data['Weekly_Sales']=predict_y
out_put_data['id']=test['Store'].astype(str)+'_'+test['Dept'].astype(str)+'_'+test['Date'].astype(str)#astype numpy中将对应的数据类型转换为另一种 #dtype是展示对应的数据了类型
out_put_data.to_csv('data_predict.csv',index=False)

