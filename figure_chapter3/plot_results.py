import numpy as np
import matplotlib.pyplot as plt

def plot_configure():
    """
    :return: configure figure size, font, etc.
    """
    # xlen = 3.27 * 2.2
    # ylen = 3.27 * 0.75 * 2.5*ratio
    xlen = 3.27
    ylen = 3.27 * 0.75
    config = {
        "font.family": 'serif',  # 'Times New Roman'
        # "font.serif": ['simsun'],
        "font.serif": ['Times New Roman'],
        "font.size": 10,
        "mathtext.fontset": 'stix',
        "axes.unicode_minus": False,  # 用来正常显示负号
        "figure.figsize": (xlen, ylen),
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
    plt.rcParams.update(config)
    return xlen,ylen
def plot_r(seq, loss_show, data_predict,data_true, train_size, mse_train, rmse_train, mae_train, mape_train, r2_train, mse_test, rmse_test, mae_test, mape_test, r2_test):
    # prediction and true mortality
    fig,ax = plt.subplots()
    ax.axvline(x=train_size, c='g', linestyle='--')
    ax.plot(data_true,color='red',label='Ground Truth')
    ax.plot(np.arange(0,train_size), data_predict[:train_size],color='cyan',label='Predict_train',alpha=0.5) # print(train_size,dataY_len)
    ax.plot(np.arange(train_size,data_true.shape[0]),data_predict[train_size:],color='blue',label='Predict_test',alpha=0.5)
    ax.set_title('Mortality Prediction with LSTM')
    ax.legend()

    # training loss
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("epoch",fontsize=10)
    ax1.set_ylabel("loss{}".format(seq),fontsize=10)
    ax1.plot(np.arange(len(loss_show)),loss_show,color='tab:orange',label='Loss')
    ax1.set_title('Loss{} VS epoch'.format(seq))
    ax1.text(1,0.9,"PM10+O3+T, Seq:{}, loss:{:.4f}".format(seq,loss_show[len(loss_show)-1]),fontsize=10,color="red")
    ax1.text(1,0.8,"Train-MSE:{:.4f},RMSE:{:.4f},MAE:{:.4f},MAPE:{:.4f},RS:{:.4f}".format(mse_train, rmse_train, mae_train, mape_train, r2_train),fontsize=10,color="blue")
    ax1.text(1,0.7,"Test-MSE:{:.4f},RMSE:{:.4f},MAE:{:.4f},MAPE:{:.4f},RS:{:.4f}".format(mse_test, rmse_test, mae_test, mape_test, r2_test),fontsize=10,color="blue")
    ax1.legend()
    # plt.savefig('./test_result/loss{}.png'.format(seq), dpi=300,  bbox_inches='tight', transparent=True) #指定分辨率保存

    ###########################
    #plot 加注释？pandas读取列不等长xlsx？
    ###########################
    plt.show()
def plot_single(seq,train_size,data_true,data_predict,loss_show,mse_train, rmse_train, mae_train, mape_train, r2_train,mse_test, rmse_test, mae_test, mape_test, r2_test):
    font_size=12
    lw = 1.3  # linewidth
    x_len = len(data_true)
    # seg = train_size
    # dataY_len = dataY.shape[0]

    fig, ax = plt.subplots()
    ax.set_xlabel(r'Time', size=font_size, labelpad=1)
    ax.set_xticks([0, 730, 1460, 2191, 2921, 3652, 4382, 5112], [r'1987', r'1989', r'1991', r'1993', r'1995', r'1997', r'1999', r'2001'], size=font_size)
    ax.axvline(x=train_size, c='g', linestyle='--')
    ax.plot(data_true, color='red', label='Ground Truth')
    ax.plot(np.arange(0, train_size), data_predict[:train_size], color='cyan', label='Predict_train', alpha=0.5)  # print(train_size,dataY_len)
    ax.plot(np.arange(train_size, x_len), data_predict[train_size:], color='blue', label='Predict_test', alpha=0.5)
    ax.set_title('Mortality Prediction with LSTM')
    ax.legend()

    # training loss
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("epoch", fontsize=10)
    ax1.set_ylabel("loss{}".format(seq), fontsize=10)
    ax1.plot(np.arange(len(loss_show)), loss_show, color='tab:orange', label='Loss')
    ax1.set_title('Loss{} VS epoch'.format(seq))
    ax1.text(1, 0.9, "PM10+O3+T, Seq:{}, loss:{:.4f}".format(seq, loss_show[len(loss_show) - 1]), fontsize=10,
             color="red")
    ax1.text(1, 0.8,
             "Train-MSE:{:.4f},RMSE:{:.4f},MAE:{:.4f},MAPE:{:.4f},RS:{:.4f}".format(mse_train, rmse_train, mae_train, mape_train, r2_train),
             fontsize=10, color="blue")
    ax1.text(1, 0.7,
             "Test-MSE:{:.4f},RMSE:{:.4f},MAE:{:.4f},MAPE:{:.4f},RS:{:.4f}".format(mse_test, rmse_test, mae_test, mape_test, r2_test),
             fontsize=10, color="blue")
    ax1.legend()
    plt.show()
