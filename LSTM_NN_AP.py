
# This module gives the methods to build a LSTM neural network model for rectangular regression / Air pollution related health assessment.


#  packages/libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error  # mse
from sklearn.metrics import mean_absolute_error  # mae
from sklearn.metrics import mean_absolute_percentage_error # mape
from sklearn.metrics import r2_score  # R square
import os, math
import glob
import re
import lime
import lime.lime_tabular
import random


# data_preprocess
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
        
        if num_na >0:
            df[col] = df[col].interpolate(method='polynomial',order=2)
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
    dt_scaler= dt[:,0].reshape((r,1)) 
    scaler = []
    dt_out = np.empty((r,0))
    for i in np.arange(c-1):
        dt_scaler =np.concatenate((dt_scaler,dt[:,i+1].reshape((r,1))),axis=1)  
    ##################################### normalization
    # for j in np.arange(c):
    #     # scaler.append(MinMaxScaler())
    #     scaler.append(StandardScaler())
    #     tempt = scaler[j].fit_transform(dt_scaler[:, j].reshape((r, 1)))
    #     dt_out = np.concatenate((dt_out,tempt),axis=1)
    ##################################### standardization
    for j in np.arange(c):
        scaler.append(StandardScaler())
        tempt = scaler[j].fit_transform(dt_scaler[:, j].reshape((r, 1)))
        dt_out = np.concatenate((dt_out,tempt),axis=1)
    
    return dt_out,scaler

def create_dataset(dt,seq,cont_seq=True):
    """
    reshape the dataset into (sequence,prediction) form, the first col dt[:,0] is the "health consequence", like mortality, cvd ...
    :param dt: dataset after interpolation and rescaling
    :param seq: sequence when cont_seq==True or lag vector when cont_seq == False
    :param cont_seq: Logic value representing whether the seq is continous sequence. If it is False, then the seq parameter is an array. Like [0,3,4] means lag0, lag3 and lag4 data.
    :return: (sequenced data,prediction)
    """
    r,c = dt.shape 
    dataX, dataY = [], [] # class list
    if cont_seq : ############ When the seq is a number and lags are continuous
      ################### impact of the day when consequence occurs is included
      for i in range(r - seq + 1):
        x = dt[i:(i + seq), 1:c] # (seq,c-1) the order is from lag_(seq-1) to lag0
        y = dt[i + seq - 1,0] # (look_back,1)
        dataX.append(x)
        dataY.append([y])
      ################### not included, only previous days' exposure are included
      # for i in range(r - seq):
      #     x = dt[i:(i + seq), 1:c]  # (seq,c-1)
      #     y = dt[i + seq, 0]  # (look_back,1)
      #     dataX.append(x)
      #     dataY.append[y])
      return np.array(dataX), np.array(dataY)
    else:  ################# When the seq is an array with non-continuous lags
      lag_max = seq.max() # the largest lag day
      for i in range(r - lag_max):
        x = dt[i+lag_max-seq, 1:c] # the order of the day inputs are opposite to the continuous one. from lag0 to lag_max
        y = dt[i + lag_max,0] 
        dataX.append(x)
        dataY.append([y])
      return np.array(dataX), np.array(dataY)
    
      
      
      
      

def dataset_partition(dataX,dataY,train_proportion=0.7):
    """
    partition the dataset into training set and test set
    :param dataX: input sequenced features-air pollutants sequence
    :param dataY: mortality/morbidity
    :return:
    """
    r,c = dataY.shape # r = 5114-seq+1
    train_size = int(r * train_proportion) # proportion for partition
    test_size = r - train_size

    torch_dataX= torch.from_numpy(dataX).type(torch.float32) # (r,seq,cols)
    torch_dataY= torch.from_numpy(dataY).type(torch.float32) # (r,1)

    train_x = torch_dataX[:train_size,:] #(train_size,seq,cols)
    train_y = torch_dataY[:train_size]   #(test_size,1)
    test_x = torch_dataX[train_size:,:]
    test_y = torch_dataY[train_size:]
    
    ############################ run on GPU
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    torch_dataX = torch_dataX.to(device)
    torch_dataY = torch_dataY.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    
    return torch_dataX,torch_dataY,train_x,train_y,train_size,test_x,test_y,test_size





# BuiltModel
class att(nn.Module):
    def __init__(self,hidden):
        super(att,self).__init__()
        self.input = nn.Sequential(
            
            nn.Linear(hidden,1) # attention on all output of hidden layers
            
        )
    def forward(self, x):
        ############################# attention on the all output
        w = self.input(x) # (batch_size, seq, hidden) >> (batch_size,seq,1)
        
        ws = F.softmax(w.squeeze(-1),dim=1) # torch.Size(batch_size,seq)
        out_att0 = (x * ws.unsqueeze(-1)).sum(dim=1) #output: (batch_size,hidden)  (b,seq,hidden_size)*(b,seq,1)=(b,seq,hidden_size) after sum: (b,13)
        # print("out_att:", out_att.shape)
        ############################# attention on the last output x[:,-1,:] (batch_size,hidden_size)
        # w = self.input(encoder) # (batch_size, hidden) >> (batch_size,hidden)
        # ws = F.softmax(w,dim=1) # (batch_size,-1, hidden)
        # out_att = encoder*ws #(batch_size,hidden) (b,13)*(b,13)=(b,13)
        #############################
        return out_att0

############################################################ RNN class
class LSTM_model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size,num_layers) -> None:
        super(LSTM_model, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = att(hidden_size)
        self.linear_out = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(True),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        
        x, _ = self.rnn(x) # (batch_size, seq, input_size)>>(batch_size, seq, hidden_size)
        
        out_att = self.attention(x) 
        
        out_in = self.linear_out(out_att) # 

        return out_in
#############################################################

# Read&SplitData
def read_data(features,filepath = './data_nmmaps/nmmaps_chic_1987_2000.xlsx',sheet_name=0):
    
    
    df = pd.read_excel(filepath, engine='openpyxl',sheet_name=sheet_name)  
   
    df = df[features] 
   
    dt, scaler = data_scaler(df) 
    return dt, scaler

def split_dataset(dt,seq,train_proportion = 0.7,cont_seq=True):
    """
    prepare the normalized data into training dataset and testing dataset with regard to the looking back steps/seq
    
    """
    dataX, dataY = create_dataset(dt, seq, cont_seq ) 
    torch_dataX,torch_dataY,train_x,train_y,train_size,test_x,test_y,test_size = dataset_partition(dataX,dataY,train_proportion)
    batch_train, seq_train, feature_size = train_x.shape
    
    return dataX, dataY, torch_dataX, torch_dataY, feature_size, train_x, train_y, train_size, test_x, test_y, test_size



# training
def train_model(feature_size, train_x, train_y, train_size, epochs_set):
    h_size = 13
    o_size = 1
    n_layers = 5
    l_rate = 5e-2 
    epochs = epochs_set  
    step = 100 
    
    # seed = 65
    # torch.manual_seed(seed)# set random seed
    ############################################################################
    nn_net = LSTM_model(input_size = feature_size, hidden_size = h_size, out_size = o_size, num_layers = n_layers)
    
    ###### run the model on GPU################################################
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    nn_net = nn_net.to(device)
    
    loss_fun = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(nn_net.parameters(), lr=l_rate)
    step_schedule = torch.optim.lr_scheduler.StepLR(step_size=step, gamma=0.95, optimizer=optimizer)

    running_loss = 0.0
   
   
    loss_show = []
    for epoch in range(epochs):
        var_x = train_x
        var_y = train_y.reshape(train_size, -1)
        out = nn_net(var_x)
        loss = loss_fun(out, var_y)
        loss_show.append(loss.item())
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_schedule.step()

        
    return nn_net, loss_show


# testing
def test_model(nn_net, dataY, torch_dataX, scaler):
    test_net = nn_net.eval()  
    train_predict = test_net(torch_dataX)
    
    if torch.cuda.is_available():
      train_predict = train_predict.to('cpu') # if the model is running on GUP, move it to CPU
    data_predict = train_predict.data.numpy()   # torch.Size([5110,1])
    data_true = dataY                           # torch.Size([5110,1])
    ####################################################### inversed scaling of mortality
    data_predict = scaler[0].inverse_transform(data_predict)
    data_true = scaler[0].inverse_transform(data_true)
    
    
    
    return data_predict, data_true


