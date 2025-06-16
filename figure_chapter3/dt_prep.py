import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import pandas as pd
# from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error  # mse
from sklearn.metrics import mean_absolute_error  # mae
from sklearn.metrics import mean_absolute_percentage_error # mape
from sklearn.metrics import r2_score  # R square
# import os, math

def count_na_interpolate(df):
    """
    interpolation of dataset
    :param df: dataframe of air pollutants
    :return: dataset with interpolation
    """
    rows = df.index
    cols = df.columns
    r, c = len(list(rows)),len(list(cols))    # print(rows,cols,r,c)
    for col in cols:
        num_na = df[col].isna().sum()
        # print('num of na:',num_na)
        if num_na >0:
            df[col].interpolate(method='polynomial',order=2, inplace=True)
    return df
#################################################################################
def data_scaler(df):
    """
    scale the dataset using standardization
    :param df: df with interpolation
    :return: dt_out: the whole dataset after standardization, scaler: all scalers used, for later inverse transform
    """
    dt = df.values
    dt = dt.astype("float32")
    r,c = dt.shape # row=5114
    dt_scaler= dt[:,0].reshape((r,1)) #(5114,)-> (5114,1)
    scaler = []
    dt_out = np.empty((r,0))
    for i in np.arange(c-1):
        dt_scaler =np.concatenate((dt_scaler,dt[:,i+1].reshape((r,1))),axis=1)  # reshape to (5114,1) for every col
    ##################################### normalization
    # for j in np.arange(c):
    #     scaler.append(MinMaxScaler())
    #     # scaler.append(StandardScaler())
    #     tempt = scaler[j].fit_transform(dt_scaler[:, j].reshape((r, 1)))
    #     dt_out = np.concatenate((dt_out,tempt),axis=1)
    # ##################################### standardization
    for j in np.arange(c):
        scaler.append(StandardScaler())
        tempt = scaler[j].fit_transform(dt_scaler[:, j].reshape((r, 1)))
        dt_out = np.concatenate((dt_out,tempt),axis=1)
    # print(dt_out[:5,:])
    return dt_out,scaler

def create_dataset(dt,seq):
    """
    reshape the dataset into (sequence,prediction) form, the first col dt[:,0] is the "health consequence", like mortality, cvd ...
    :param dt: dataset after interpolation and rescaling
    :param seq: sequence
    :return: (sequenced data,prediction)
    """
    r,c = dt.shape # (5114,cols+1), air pollutants + mortality
    dataX, dataY = [], [] # class list
    ################### impact of the day when consequence occurs is included
    for i in range(r - seq + 1):
        x = dt[i:(i + seq), 1:c] # (seq,c-1)
        y = dt[i + seq - 1,0] # (look_back,1)
        dataX.append(x)
        dataY.append([y])
    ################### not included, only previous days' exposure are included
    # for i in range(r - seq):
    #     x = dt[i:(i + seq), 1:c]  # (seq,c-1)
    #     y = dt[i + seq, 0]  # (look_back,1)
    #     dataX.append(x)
    #     dataY.append([y])
    return np.array(dataX), np.array(dataY)

def dataset_partition(dataX,dataY):
    """
    partition the dataset into training set and test set
    :param dataX: input sequenced features-air pollutants sequence
    :param dataY: mortality/morbidity
    :return:
    """
    r,c = dataY.shape # r = 5114-seq+1
    train_size = int(r * 0.7) # proportion for partition
    test_size = r - train_size

    torch_dataX= torch.from_numpy(dataX).type(torch.float32) # (r,seq,cols)
    torch_dataY= torch.from_numpy(dataY).type(torch.float32) # (r,1)

    train_x = torch_dataX[:train_size,:] #(train_size,seq,cols)
    train_y = torch_dataY[:train_size]   #(test_size,1)
    test_x = torch_dataX[train_size:,:]
    test_y = torch_dataY[train_size:]
    return torch_dataX,torch_dataY,train_x,train_y,train_size,test_x,test_y,test_size

# df = count_na_interpolate(df[['death','temp','pm10','o3']])
def read_data(features):
    filepath = './chic_1987_2000.xlsx' # 3117,106 >> (62+61)/2=61
    df = pd.read_excel(filepath, engine='openpyxl')  # outlier: 3117 106>>(62+60)/2=61
    df = df[features] #[['death','tmpd','tmean','pm10mean','o3mean','so2mean','no2mean','comean']]
    # print(df.head(5))
    dt, scaler = data_scaler(df) # (5114,3)
    return dt

def split_dataset(dt,seq):
    # seq = 7 # 7
    dataX, dataY = create_dataset(dt, seq) # (r-(seq-1),seq,cols) (r,1)
    torch_dataX,torch_dataY,train_x,train_y,train_size,test_x,test_y,test_size = dataset_partition(dataX,dataY)
    batch_train, seq_train, feature_size = train_x.shape
    # print(feature_size)
    # print("train_size:",train_size) # seq: 1-8 train_size: [3579,3578,3577,3577,3576,3575, 3574 ,3574]
    return dataX, dataY, torch_dataX, torch_dataY, feature_size, train_x, train_y, train_size, test_x, test_y, test_size

############################################################################## performance metrics: MSE, RMSE, MAE, RS
def performance_metrics1(y_predict, y_true):
    loss_error = y_predict - y_true
    loss_mse = np.sum(loss_error**2)/len(loss_error)
    loss_rmse = loss_mse**0.5
    loss_mae = np.sum(np.absolute(loss_error)) / len(loss_error)
    r_squared = 1-(np.sum((y_predict-y_true)**2)/np.var(y_true)/len(loss_error))
    MAPE = np.sum(np.absolute(loss_error/y_true)) / len(loss_error)
    return loss_error,loss_mse,loss_rmse,loss_mae,MAPE,r_squared
def performance_metrics2(data_predict,data_true, train_size):
    error = data_predict-data_true
    mse_train = mean_squared_error(data_true[0:train_size],data_predict[0:train_size])
    rmse_train = mse_train**0.5
    mae_train = mean_absolute_error(data_true[0:train_size],data_predict[0:train_size])
    mape_train = mean_absolute_percentage_error(data_true[0:train_size],data_predict[0:train_size])
    r2_train = r2_score(data_true[0:train_size],data_predict[0:train_size])

    mse_test = mean_squared_error(data_true[train_size:],data_predict[train_size:])
    rmse_test = mse_test**0.5
    mae_test = mean_absolute_error(data_true[train_size:],data_predict[train_size:])
    mape_test = mean_absolute_percentage_error(data_true[train_size:],data_predict[train_size:])
    r2_test = r2_score(data_true[train_size:],data_predict[train_size:])
    print("Train: using sklearn-metrics")
    print("MSE:{:.4f},RMSE:{:.4f},MAE:{:.4f},MAPE:{:.4f},R2:{:.4f}".format(mse_train, rmse_train, mae_train, mape_train, r2_train))
    print("Test: using sklearn-metrics")
    print("MSE:{:.4f},RMSE:{:.4f},MAE:{:.4f},MAPE:{:.4f},R2:{:.4f}".format(mse_test, rmse_test, mae_test, mape_test, r2_test))
    # print(np.around(mse_test,decimals=4),np.around(mae_test,decimals=4),np.around(mape_test,decimals=4),np.around(r2_test,decimals=4))  # 2.2719309220510002 1.182904648034428 -1.7949149334803214
    return mse_train, rmse_train, mae_train, mape_train, r2_train, mse_test, rmse_test, mae_test, mape_test, r2_test