import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, math
import dt_prep
import plot_results
from sklearn.metrics import mean_squared_error  # mse
from sklearn.metrics import mean_absolute_error  # mae

def plot_prediction1(seqs,train_sizes,xlen,ylen):
    file_prediction = "./epoch5000_PM10_O3_CO_NO2/predictions.xlsx"
    if os.path.isfile(file_prediction):
        print("Plotting prediction and actual mortality...")
    df_prediction = pd.read_excel(file_prediction, sheet_name=None)
    font_size, mk_size = 10, 0.8
    l, r, t, b = 0.08, 0.96, 0.96, 0.07  # bord width
    h, w = 0.13, 0.07  # width among sub-figures
    lw = 1.3  # linewidth
    lpx, lpy = 0.05, 1  # label pad
    # seqs = [1, 2, 3]
    # train_sizes = [3579, 3578, 3577]
    len_seqs = len(seqs)
    nr, nc = math.ceil(len_seqs/2), 2  # row and column numbers
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(xlen*1.85,ylen*3.4),nrows=nr, ncols=nc, sharex=True, sharey=True)
    for i in range(nr):
        for j in range(nc):
            if i!=nr-1:
                ax[i,j].tick_params(axis="x", labelbottom=False)
            else:
                ax[i, j].set_xticks([0,730,1460,2191,2921,3652,4382,5112],[r'1987', r'1989', r'1991', r'1993', r'1995', r'1997', r'1999', r'2001'], size=font_size)
            if j==1:
                ax[i, j].tick_params(axis="y", labelbottom=False)
            # else:
            #     ax[i, j].set_ylabel('Daily NonAM', size=font_size, labelpad=1)  # 输入乘号？
            if (len_seqs%2!=0)&(i==nr-1)&(j==1):
                ax[i,j].plot()
            else:
                true_value = df_prediction['seq={}'.format(2 * i + j + 1)]['true']
                predict_value = df_prediction['seq={}'.format(2 * i + j + 1)]['predict']
                x = len(true_value)
                seg = train_sizes[2 * i + j]
                ymin = np.floor(min(min(true_value), min(predict_value)) - 1)
                ymax = np.ceil(max(max(true_value),max(predict_value))+1)
                # ax[i, j].set_ylim(-5, 7)
                # ax[i, j].set_yticks(np.arange(-5, 7, 2))
                # ax[i, j].set_ylim(-10, 10)
                ax[i, j].set_ylim(ymin, ymax)
                ax[i, j].set_yticks(np.arange(ymin, ymax, 2))
                print(ymin,ymax)
                if (i==0) and (j==0):

                    ax[i, j].plot(np.arange(x), true_value, color="#E69F00", linestyle='--', linewidth=lw,
                                  label='Actual Mortality')
                    ax[i, j].plot(np.arange(0, seg), predict_value[:seg], color='#009E73', linestyle='-.', alpha=1,
                                  linewidth=lw, label='Prediction_train')
                    ax[i, j].plot(np.arange(seg, x), predict_value[seg:], color='#56B4E9', linestyle=':', alpha=0.8,
                                  linewidth=lw, label='Prediction_test')
                    ax[i, j].axvline(x=seg, color='#F0E442', linestyle='-', linewidth=lw + 0.6, alpha=0.7)

                    ax[i, j].text(2500, -4.5, "$m$={}".format(2 * i + j + 1), fontsize=12, color="red")
                else:
                    # ax[i, j].set_ylim(ymin, ymax)
                    # ax[i, j].set_yticks(np.arange(ymin, ymax, 2))
                    ax[i, j].plot(np.arange(x), true_value, color="#E69F00", linestyle='--', linewidth=lw,
                                  label='Actual Mortality')
                    ax[i, j].plot(np.arange(0, seg), predict_value[:seg], color='#009E73', linestyle='-.', alpha=1,
                                  linewidth=lw, label='Prediction_train')
                    ax[i, j].plot(np.arange(seg, x), predict_value[seg:], color='#56B4E9', linestyle=':', alpha=0.8,
                                  linewidth=lw, label='Prediction_test')

                    ax[i, j].axvline(x=seg, color='#F0E442', linestyle='-', linewidth=lw + 0.6, alpha=0.7)

                    ax[i, j].text(2500, -4.5, "$m$={}".format(2*i+j+1), fontsize=12,color="red")
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3,frameon=True,loc='upper center', labelspacing=0.2, columnspacing=0.5, handlelength=1.0, handletextpad=0.2, borderaxespad=0.1,fontsize=9)

    fig.text(0.015, 0.5, 'Daily Mortality', va='center', rotation='vertical', fontsize=10)
    fig.text(0.5, 0.035, 'Time', va='center', rotation='horizontal', fontsize=10)
    plt.subplots_adjust(left=l, right=r, top=t, bottom=b, hspace=h, wspace = w)
    plt.savefig('./c3_figs/3_52.pdf', dpi=300,  bbox_inches='tight', transparent=False)
def plot_loss_var(seqs,epochs_set,xlen,ylen):
    file = "./epoch5000_PM10_O3/losses.xlsx"
    if os.path.isfile(file):
        print("Plotting loss...")
    df_loss = pd.read_excel(file, sheet_name=None)
    font_size, mk_size = 10, 0.8
    l, r, t, b = 0.08, 0.96, 0.96, 0.07
    h, w = 0.13, 0.07  # width among sub-figures
    lw = 1.3  # linewidth
    lpx, lpy = 0.05, 1  # label pad
    len_seqs = len(seqs)
    nr, nc = math.ceil(len_seqs/2), 2  # row and column numbers

    x = epochs_set
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(xlen*1.85,ylen*3.4),nrows=nr, ncols=nc, sharex=True, sharey=True)
    for i in range(nr):
        for j in range(nc):
            if i % 2 == 0:
                ax[i, j].set_yticks(np.arange(0, 1.2, 0.2))
            if i!=nr-1:
                ax[i,j].tick_params(axis="x", labelbottom=False)
            else:
                ax[i, j].set_xticks(np.arange(0,5001,1000))
            if j==1:
                ax[i, j].tick_params(axis="y", labelbottom=False)
            if (len_seqs%2!=0)&(i==nr-1)&(j==1):
                ax[i,j].plot()
            else:
                loss = df_loss['seq={}'.format(2 * i + j + 1)]['loss']
                # ymin = np.floor(min(min(true_value), min(predict_value)) - 1)
                # ymax = np.ceil(max(max(true_value),max(predict_value))+1)
                # ax[i, j].set_ylim(ymin, ymax)
                # ax[i, j].set_yticks(np.arange(ymin, ymax, 2))
                ax[i, j].plot(np.arange(x),loss,color="tab:orange",linewidth=lw)
                ax[i, j].text(round(0.75*x), 0.8, "$m$={}".format(2*i+j+1), fontsize=12,color="red")
    fig.text(0.01, 0.5, 'Loss', va='center', rotation='vertical', fontsize=10)
    fig.text(0.5, 0.035, 'epoch', va='center', rotation='horizontal', fontsize=10)
    plt.subplots_adjust(left=l, right=r, top=t, bottom=b, hspace=h, wspace = w)
    plt.savefig('./c3_figs/3_4.pdf', dpi=300,  bbox_inches='tight', transparent=False)
def plot_loss_prediction(seqs,epochs_set,train_sizes,xlen,ylen): # c2
    # file_prediction = "./epoch5000_PM10_O3_CO/predictions.xlsx"
    # file_loss = "./epoch5000_PM10_O3_CO/losses.xlsx"
    file_prediction = "./epoch5000_PM10_O3_CO_NO2/predictions.xlsx"
    file_loss = "./epoch5000_PM10_O3_CO_NO2/losses.xlsx"
    if os.path.isfile(file_prediction) & os.path.isfile(file_loss):
        print("Plotting loss and prediction...")
    df_prediction = pd.read_excel(file_prediction, sheet_name=None)
    df_loss = pd.read_excel(file_loss, sheet_name=None)
    font_size, mk_size = 10, 0.8
    l, r, t, b = 0.08, 0.96, 0.96, 0.07  # bord width
    h, w = 0.13, 0.07  # width among sub-figures
    lw = 1.3  # linewidth
    lpx, lpy = 0.05, 1  # label pad
    # seqs = [1, 2, 3]
    # train_sizes = [3579, 3578, 3577]
    len_seqs = len(seqs)
    nr, nc = math.ceil(len_seqs/2), 2  # row and column numbers
    plt.style.use('ggplot')
    ##############################################
    x = epochs_set
    fig1, ax1 = plt.subplots(figsize=(xlen*1.85,ylen*3.4),nrows=nr, ncols=nc, sharex=True, sharey=True)
    for i in range(nr):
        for j in range(nc):
            if i % 2 == 0:
                ax1[i, j].set_yticks(np.arange(0, 1.2, 0.2))
            if i!=nr-1:
                ax1[i,j].tick_params(axis="x", labelbottom=False)
            else:
                ax1[i, j].set_xticks(np.arange(0,5001,1000))
            if j==1:
                ax1[i, j].tick_params(axis="y", labelbottom=False)
            if (len_seqs%2!=0)&(i==nr-1)&(j==1):
                ax1[i,j].plot()
            else:
                loss = df_loss['seq={}'.format(2 * i + j + 1)]['loss']
                # ymin = np.floor(min(min(true_value), min(predict_value)) - 1)
                # ymax = np.ceil(max(max(true_value),max(predict_value))+1)
                # ax[i, j].set_ylim(ymin, ymax)
                # ax[i, j].set_yticks(np.arange(ymin, ymax, 2))
                ax1[i, j].plot(np.arange(x),loss,color="tab:orange",linewidth=lw)
                ax1[i, j].text(round(0.75*x), 0.8, "$m$={}".format(2*i+j+1), fontsize=12,color="red")
    fig1.text(0.01, 0.5, 'Loss', va='center', rotation='vertical', fontsize=10)
    fig1.text(0.5, 0.035, 'epoch', va='center', rotation='horizontal', fontsize=10)
    plt.subplots_adjust(left=l, right=r, top=t, bottom=b, hspace=h, wspace = w)
    plt.savefig('./c3_figs/3_42.pdf', dpi=300,  bbox_inches='tight', transparent=False)
##############################################
    fig2, ax2 = plt.subplots(figsize=(xlen*1.85,ylen*3.4),nrows=nr, ncols=nc, sharex=True, sharey=True)
    for i in range(nr):
        for j in range(nc):
            if i!=nr-1:
                ax2[i,j].tick_params(axis="x", labelbottom=False)
            # elif (i==nr-1) & (len_seqs%2!=0)&(j==1):
            #     ax[i, 1].tick_params(axis="x", labelbottom=False)
            #     # ax[i, 1].tick_params(axis="y", labelbottom=False)
            else:
                ax2[i, j].set_xticks([0,730,1460,2191,2921,3652,4382,5112],[r'1987', r'1989', r'1991', r'1993', r'1995', r'1997', r'1999', r'2001'], size=font_size)
            if j==1:
                ax2[i, j].tick_params(axis="y", labelbottom=False)
            if (len_seqs%2!=0)&(i==nr-1)&(j==1):
                ax2[i,j].plot()
            else:
                true_value = df_prediction['seq={}'.format(2 * i + j + 1)]['true']
                predict_value = df_prediction['seq={}'.format(2 * i + j + 1)]['predict']
                x = len(true_value)
                seg = train_sizes[2 * i + j]
                ymin = np.floor(min(min(true_value), min(predict_value)) - 1)
                ymax = np.ceil(max(max(true_value),max(predict_value))+1)
                # ax[i, j].set_ylim(-5, 7)
                # ax[i, j].set_yticks(np.arange(-5, 7, 2))
                # ax[i, j].set_ylim(-10, 10)
                ax2[i, j].set_ylim(ymin, ymax)
                ax2[i, j].set_yticks(np.arange(ymin, ymax, 2))
                # print(ymin,ymax)
                if (i==0) and (j==0):
                    ax2[i, j].plot(np.arange(x), true_value, color="tab:red", linewidth=lw, label='Actual Mortality')
                    ax2[i, j].plot(np.arange(0, seg), predict_value[:seg], color='forestgreen', alpha=1,linewidth=lw, label='Prediction_train')
                    ax2[i, j].plot(np.arange(seg, x), predict_value[seg:], color='tab:blue', alpha=0.8,linewidth=lw, label='Prediction_test')
                    ax2[i, j].axvline(x=seg, color='blue', linestyle='-', linewidth=lw + 0.6, alpha=0.7)
                    ax2[i, j].text(2500, -4.5, "$m$={}".format(2 * i + j + 1), fontsize=12, color="red")
                else:
                    # ax[i, j].set_ylim(ymin, ymax)
                    # ax[i, j].set_yticks(np.arange(ymin, ymax, 2))
                    ax2[i, j].plot(np.arange(x),true_value,color="tab:red",linewidth=lw)
                    ax2[i, j].plot(np.arange(0, seg), predict_value[:seg], alpha=1,color='forestgreen', linewidth=lw)
                    ax2[i, j].plot(np.arange(seg, x), predict_value[seg:], alpha=0.8,color='tab:blue', linewidth=lw)
                    ax2[i, j].axvline(x=seg, color='blue', linestyle='-',linewidth=lw+0.6,alpha=0.7)
                    ax2[i, j].text(2500, -4.5, "$m$={}".format(2*i+j+1), fontsize=12,color="red")
    handles, labels = ax2[0,0].get_legend_handles_labels()
    fig2.legend(handles, labels, ncol=3,frameon=True,loc='upper center', labelspacing=0.2, columnspacing=0.5, handlelength=1.0, handletextpad=0.2, borderaxespad=0.1,fontsize=9)

    fig2.text(0.015, 0.5, 'Daily Mortality', va='center', rotation='vertical', fontsize=10)
    fig2.text(0.5, 0.035, 'Time', va='center', rotation='horizontal', fontsize=10)
    plt.subplots_adjust(left=l, right=r, top=t, bottom=b, hspace=h, wspace = w)
    plt.savefig('./c3_figs/3_52.pdf', dpi=300,  bbox_inches='tight', transparent=False)
def plot_multipollutant(xlen,ylen):
    file = "./multipollutant.xlsx"
    if os.path.isfile(file):
        print("Plotting loss and prediction for multipollutant...")
    df = pd.read_excel(file)
    # print(df.head(3))
    n = 5000
    loss1 = df['loss1'].iloc[:n].to_numpy()
    true1 = df['true1'].to_numpy()
    predict1 = df['predict1'].to_numpy()
    loss2 = df['loss2'].iloc[:n].to_numpy()
    true2 = df['true2'].to_numpy()
    predict2 = df['predict2'].to_numpy()
    loss3 = df['loss3'].iloc[:n].to_numpy()
    true3 = df['true3'].to_numpy()
    predict3 = df['predict3'].to_numpy()
    loss4 = df['loss4'].iloc[:n].to_numpy()
    true4 = df['true4'].to_numpy()
    predict4 = df['predict4'].to_numpy()
    loss5 = df['loss5'].iloc[:n].to_numpy()
    true5 = df['true5'].to_numpy()
    predict5 = df['predict5'].to_numpy()

    font_size, mk_size = 10, 0.8
    l, r, t, b = 0.075, 0.96, 0.96, 0.07  # bord width
    h, w = 0.15, 0.15  # width among sub-figures
    lw = 1.3  # linewidth
    lpx, lpy = 0.05, 1  # label pad
    nr, nc = 5, 2  # row and column numbers

    x1 = len(loss1)
    print('length of loss data:',x1)
    x2 = len(true1)
    seg =3575 # train_size for m = 7
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(xlen*1.85,ylen*2.8),nrows=nr, ncols=nc)
    for j in range(nc):
        if j==0:
            for i in range(nr):
                ax[i, j].plot(np.arange(x1), locals()['loss{}'.format(i+1)], color="#E69F00", linewidth=lw)
                ax[i, j].tick_params(axis="x", labelbottom=False)
                ax[i, j].set_yticks(np.arange(0, 1.2, 0.2))
                # ax[i, j].set_xlim(-200,5100)
                # ax[i, j].arrow(5000,0.55, 100,0, width=0.06,color='k',head_width=0.1,head_length=0.15)
                if i==4:
                #     # ax[i, j].annotate('$PM_{10}+O_{3}+CO+NO_{2}+SO_{2}$', xy=(3000, 0.5), xytext=(2000, 0.5), arrowprops=dict(facecolor='lightblue', shrink=0.05))
                    ax[i, j].tick_params(axis="x", labelbottom=True)
                    ax[i, j].set_xlabel(r'epoch', labelpad=1)
        else:
            for i in range(nr):
                ax[i, j].plot(np.arange(x2), locals()['true{}'.format(i + 1)], color="#D55E00", linewidth=lw, label='Actual Mortality')
                ax[i, j].tick_params(axis="x", labelbottom=False)
                ax[i, j].set_ylim(-4, 5.5)
                ax[i, j].set_yticks(np.arange(-4, 4.1, 2))
                ax[i, j].plot(np.arange(0, seg), locals()['predict{}'.format(i + 1)][:seg], color='#009E73', alpha=0.8,linewidth=lw, label='Prediction_train')
                # ax[i, j].plot(np.arange(0, seg), locals()['predict{}'.format(i + 1)][:seg], color='tab:green', alpha=0.6, linewidth=lw, label='Prediction_train')
                ax[i, j].plot(np.arange(seg, x2), locals()['predict{}'.format(i + 1)][seg:], color='#0072B2', alpha=0.8,linewidth=lw, label='Prediction_test')
                ax[i, j].axvline(x=seg, color='#0072B2', linestyle='-', linewidth=lw + 0.6, alpha=0.7)
                if i==4:
                    ax[i, j].tick_params(axis="x", labelbottom=True)
                    ax[i, j].set_xticks([0, 730, 1460, 2191, 2921, 3652, 4382, 5112],[r'1987', r'1989', r'1991', r'1993', r'1995', r'1997', r'1999', r'2001'])
                    ax[i, j].set_xlabel(r'Time', labelpad=1)
    handles, labels = ax[0,1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3,frameon=True,loc='upper center', labelspacing=0.2, columnspacing=0.5, handlelength=1.0, handletextpad=0.2, borderaxespad=0.1,fontsize=9)

    fig.text(0.2,0.125,'${\mathrm{PM}}_{10}+{\mathrm{O}}_{3}+{\mathrm{CO}}+{\mathrm{NO}}_{2}+{\mathrm{SO}}_{2}$', color="#56B4E9",  rotation='horizontal')
    fig.text(0.2, 0.125+0.19, '${\mathrm{PM}}_{10}+{\mathrm{O}}_{3}+{\mathrm{CO}}+{\mathrm{NO}}_{2}$',color="#56B4E9", rotation='horizontal')
    fig.text(0.2, 0.125+0.19*2, '${\mathrm{PM}}_{10}+{\mathrm{O}}_{3}+{\mathrm{CO}}$',color="#56B4E9", rotation='horizontal')
    fig.text(0.2, 0.125+0.19*3, '${\mathrm{PM}}_{10}+{\mathrm{O}}_{3}$',color="#56B4E9", rotation='horizontal')
    fig.text(0.2, 0.125+0.19*4, '${\mathrm{PM}}_{10}$',color="#56B4E9", rotation='horizontal')
    fig.text(0.01, 0.5, 'Loss', va='center', rotation='vertical')
    fig.text(0.5, 0.5, 'Daily Mortality', va='center', rotation='vertical')

    plt.arrow(0.5,0.55, 100,0, width=0.06,color='k',head_width=0.1,head_length=0.15)

    plt.subplots_adjust(left=l, right=r, top=t, bottom=b, hspace=h, wspace = w)
    plt.savefig('./c3_figs/3_6.pdf', dpi=300,  bbox_inches='tight', transparent=False)
    plt.show()

def performance_cal(seqs,train_sizes):
    # file_prediction = "./epoch5000_PM10_O3/predictions.xlsx"
    # file_prediction = "./epoch5000_PM10_O3_CO/predictions.xlsx"
    file_prediction = "./epoch5000_PM10_O3_CO_NO2/predictions.xlsx"
    df_prediction = pd.read_excel(file_prediction, sheet_name=None)
    # file = "./epoch5000_PM10_O3/losses.xlsx"
    # df_loss = pd.read_excel(file, sheet_name=None)
    multipollutant = "./multipollutant.xlsx"
    df_multi = pd.read_excel(multipollutant)

    TrS_rmse1 = []
    TrS_mae1 = []
    TeS_rmse1 = []
    TeS_mae1 = []
    TrS_rmse2 = []
    TrS_mae2 = []
    TeS_rmse2 = []
    TeS_mae2 = []
    for seq in seqs:
        data_true= df_prediction['seq={}'.format(seq)]['true']
        data_prediction = df_prediction['seq={}'.format(seq)]['predict']
        train_size = train_sizes[seq-1]
        mse_train, rmse_train, mae_train, mape_train, r2_train, mse_test, rmse_test, mae_test, mape_test, r2_test\
            =dt_prep.performance_metrics2(data_prediction, data_true, train_size)
        TrS_rmse1.append(rmse_train)
        TrS_mae1.append(mae_train)
        TeS_rmse1.append(rmse_test)
        TeS_mae1.append(mae_test)
    print("=============Performance comparison for different m==============")
    print(np.round(TrS_rmse1,3),np.round(TrS_mae1,3),np.round(TeS_rmse1,3),np.round(TeS_mae1,3), sep="\n")

    fig1,ax1 = plt.subplots()
    ax1.plot(np.arange(len(TrS_rmse1)),TrS_rmse1,label='TrS_rmse1')
    ax1.plot(np.arange(len(TrS_mae1)), TrS_mae1,label='TrS_mae1')
    ax1.plot(np.arange(len(TeS_rmse1)), TeS_rmse1,label='TeS_rmse1')
    ax1.plot(np.arange(len(TeS_mae1)), TeS_mae1,label='TeS_mae1')
    plt.legend() # 1. fig.legned vs. ax.legend??? 2.局部批量变量修改？
    plt.title("Performance comparison for different m")
    for i in range(5):
        true= df_multi['true{}'.format(i++1)]
        predict = df_multi['predict{}'.format(i++1)]
        train_size = train_sizes[5-1]
        mse_train, rmse_train, mae_train, mape_train, r2_train, mse_test, rmse_test, mae_test, mape_test, r2_test\
            =dt_prep.performance_metrics2(predict, true, train_size)
        TrS_rmse2.append(rmse_train)
        TrS_mae2.append(mae_train)
        TeS_rmse2.append(rmse_test)
        TeS_mae2.append(mae_test)
    print("=============Performance comparison for multipollutant==============")
    print(np.round(TrS_rmse2,3), np.round(TrS_mae2,3), np.round(TeS_rmse2,3), np.round(TeS_mae2,3), sep="\n")
    fig2,ax2 = plt.subplots()
    ax2.plot(np.arange(len(TrS_rmse2)),TrS_rmse2,label='TrS_rmse1')
    ax2.plot(np.arange(len(TrS_mae2)), TrS_mae2,label='TrS_mae1')
    ax2.plot(np.arange(len(TeS_rmse2)), TeS_rmse2,label='TeS_rmse1')
    ax2.plot(np.arange(len(TeS_mae2)), TeS_mae2,label='TeS_mae1')
    plt.legend() # fig.legned vs. ax.legend???
    plt.title("Performance comparison for multipollutant")
    plt.show()

def compare_with_gam(train_sizes,xlen, ylen):
    def z_score_normalize(data):
        mean_val = np.mean(data)
        std_dev = np.std(data)
        normalized_data = (data - mean_val) / std_dev
        return normalized_data
    file_gam = "./results_gam.xlsx"
    df_gam = pd.read_excel(file_gam, sheet_name=None)
    file_lstm = "./results_lstm_PM10.xlsx"
    df_lstm = pd.read_excel(file_lstm, sheet_name=None)

    font_size, mk_size = 10, 0.8
    l, r, t, b = 0.07, 0.96, 0.96, 0.09  # bord width
    h, w = 0.13, 0.06  # width among sub-figures
    lw = 1.3  # linewidth
    lpx, lpy = 0.05, 1  # label pad
    nr, nc = 4, 2  # row and column numbers

    seqs = [2,4,6,8]
    # x1 = len(loss1)
    # print('length of loss data:',x1)
    # x2 = len(true1)
    seg =3575 # train_size for m = 7
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(xlen*1.85,ylen*2.4),nrows=nr, ncols=nc)
    for j in range(nc):
        if j==0:
            for i in range(nr):
                y_true = df_lstm['seq={}'.format(2*(i+1))]['true']
                y_pred = df_lstm['seq={}'.format(2 * (i + 1))]['predict']
                x = len(y_true)
                x_train = train_sizes[i]

                ax[i, j].plot(np.arange(x), y_true, color="#D55E00", linewidth=lw, label='Actual Mortality')
                ax[i, j].plot(np.arange(0, x_train), y_pred[:x_train], color='#009E73', alpha=1, linewidth=lw,
                              label='Prediction_train')
                ax[i, j].plot(np.arange(x_train, x), y_pred[x_train:], color='#0072B2', alpha=0.8, linewidth=lw,
                              label='Prediction_test')
                ax[i, j].axvline(x=x_train, color='#56B4E9', linestyle='-', linewidth=lw + 0.6, alpha=0.7)
                ax[i, j].text(round(0.5 * x), -4.5, r"$m$={}".format(2 * (i + 1)), fontsize=12, color="#CC79A7")

                ax[i, j].tick_params(axis="x", labelbottom=False)
                ax[i, j].set_ylim(-5, 5)
                ax[i, j].set_yticks(np.arange(-5, 6, 2))
                # ymin = np.floor(min(min(y_true), min(y_pred)) - 1)
                # ymax = np.ceil(max(max(y_true), max(y_pred)) + 1)
                # ax[i, j].set_ylim(ymin, ymax)
                # ax[i, j].set_yticks(np.arange(ymin, ymax, 2))
                if i==nr-1:
                #     # ax[i, j].annotate('$PM_{10}+O_{3}+CO+NO_{2}+SO_{2}$', xy=(3000, 0.5), xytext=(2000, 0.5), arrowprops=dict(facecolor='lightblue', shrink=0.05))
                    ax[i, j].tick_params(axis="x", labelbottom=True)
                    ax[i, j].set_xticks([0, 730, 1460, 2191, 2921, 3652, 4382, 5112],[r'1987', r'1989', r'1991', r'1993', r'1995', r'1997', r'1999', r'2001'])
                mse_train, rmse_train, mae_train, mape_train, r2_train, mse_test, rmse_test, mae_test, mape_test, r2_test \
                    = dt_prep.performance_metrics2(y_pred, y_true, x_train)
                print("==========seq={}=========:".format(2 * (i + 1)), rmse_train, mae_train, rmse_test, mae_test)
        else:
            for i in range(nr):
                y_true = z_score_normalize(df_gam['Sheet1']['death'])
                y_pred = z_score_normalize(df_gam['Sheet1']['lag{}'.format(i + 1)])
                x = len(y_true)
                ax[i, j].plot(np.arange(x), y_true, color="#D55E00", linewidth=lw, label='Actual Mortality')
                ax[i, j].plot(np.arange(x), y_pred, color="#009E73", linewidth=lw, label='Prediction_train')
                ax[i, j].text(round(0.5 * x), -4.5, r"$lag$={}".format(2 * (i + 1)-1), fontsize=12, color="#CC79A7")
                ax[i, j].set_ylim(-5, 5)
                ax[i, j].set_yticks(np.arange(-5, 6, 2))
                ax[i, j].tick_params(axis="x", labelbottom=False)
                ax[i, j].tick_params(axis="y", labelleft=False)
                if i==nr-1:
                    ax[i, j].tick_params(axis="x", labelbottom=True)
                    ax[i, j].set_xticks([0, 730, 1460, 2191, 2921, 3652, 4382, 5112],[r'1987', r'1989', r'1991', r'1993', r'1995', r'1997', r'1999', r'2001'])
                mse_train = mean_squared_error(y_true[2*(i+1)-1:,], y_pred[2*(i+1)-1:,]) # 空值参与运算会error
                rmse_train = mse_train ** 0.5
                mae_train = mean_absolute_error(y_true[2*(i+1)-1:,], y_pred[2*(i+1)-1:,])
                print("==========lag={}=========:".format(2 * (i + 1)-1), rmse_train, mae_train)
    fig.text(0.015, 0.5, 'Daily Mortality', va='center', rotation='vertical')
    fig.text(0.5, 0.025, 'Time', va='center', rotation='horizontal')
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=True, loc='upper center', labelspacing=0.2, columnspacing=0.5,
               handlelength=1.0, handletextpad=0.2, borderaxespad=0.1, fontsize=9)
    # handles, labels = ax[0,1].get_legend_handles_labels()
    # fig.legend(handles, labels, ncol=3,frameon=True,loc='upper center', labelspacing=0.2, columnspacing=0.5, handlelength=1.0, handletextpad=0.2, borderaxespad=0.1,fontsize=9)
    plt.subplots_adjust(left=l, right=r, top=t, bottom=b, hspace=h, wspace = w)
    plt.savefig('./c3_figs/3_7.pdf', dpi=300,  bbox_inches='tight', transparent=False)

    # % seq = 2: 0.2360182458382641
    # 0.13539741644919434
    # 1.9298611543868713
    # 1.4055134788145842
    # % seq = 4: 0.09607432342947987
    # 0.051872593712875024
    # 1.6528251628519681
    # 1.2745340132243164
    # % seq = 6: 0.14948425895923678
    # 0.05872809174272672
    # 1.496081562523617
    # 1.1821744874707834
    # % seq = 8: 0.0744297112818418
    # 0.03629012015519458
    # 1.4096487976124052
    # 1.1192772314730066
    # % lag = 1: 0.8818472927674151
    # 0.6992424313489776
    # % lag = 3: 0.8822706699820559
    # 0.6994716757774044
    # % lag = 5: 0.8817800082135634
    # 0.6992463647644995
    # % lag = 7: 0.8824013666576705
    # 0.6994023717556132
    plt.show()
    ##################################################################################
def plot_c4():
    xlength = 3.27
    ylength = 3.27 * 0.75
    config = {
        "font.family": 'serif',  # 'Times New Roman'
        # "font.serif": ['simsun'],
        "font.serif": ['Times New Roman'],
        "font.size": 10,
        "mathtext.fontset": 'stix',
        "axes.unicode_minus": False,  # 用来正常显示负号
        "figure.figsize": (xlength, ylength),
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
    plt.rcParams.update(config)
    plt.style.use('ggplot')
    file2 = "./c4_figs/LSTM_cities_withseed.xlsx"
    # file3 = "./c4_figs/LSTM_cities_trial30.xlsx"
    file3 = "./c4_figs/LSTM_cities_trial30_training.xlsx"
    file4 = "./c4_figs/Comp_GAM_LSTM2_m5.xlsx"
    # 0.986328,     0.035194
    # 1.99805,      0.0116418
    # 2.99609,     -0.00116418
    # 3.99414,      0.0103881
    # 5.00586,     -0.00564179
    # 6.00391,      0.00555224
    # 7.00195,      0.00779104
    # 8,           -0.00295522
    df1 =[0.035194, 0.0116418, -0.00116418, 0.0103881,
          -0.00564179, 0.00555224, 0.00779104, -0.00295522]
    df2 = pd.read_excel(file2, engine='openpyxl')
    df3 = pd.read_excel(file3, engine='openpyxl') # df3 = pd.read_excel(file3, sheet_name=None) 无法通过列读取df['col'] error!
    df4 = pd.read_excel(file4, engine='openpyxl')

    font_size, mk_size = 10, 0.8
    l, r, t, b = 0.1, 0.96, 0.96, 0.1  # bord width
    h, w = 0.13, 0.06  # width among sub-figures
    lw = 1.3  # linewidth

    fig1,ax1 = plt.subplots(figsize=(xlength*1.8,ylength*2.0))
    # ax1.set_xlim(0, 168)
    ax1.set_ylim(-0.008, 0.040)
    ax1.set_xticks(np.arange(0, 8, 1), np.arange(1,9,1))
    # ax1.set_yticks(np.arange(50 * 0.25, 251 * 0.25, 50 * 0.25), fontproperties='Times New Roman', size=8)
    ax1.set_xlabel(r'$m$', labelpad=1, size=10)
    ax1.set_ylabel(r'$\beta$',rotation=0, labelpad=3, size=10)  # config里的font不起作用？
    ax1.plot(df1, 'k-o',linewidth=0.81, markersize='5', markerfacecolor='white', markeredgecolor='tab:blue',markeredgewidth=1.2)
    # ax1.grid()
    plt.subplots_adjust(left=0.10, right=r, top=t, bottom=b, hspace=h, wspace = w)
    plt.savefig('./c4_figs/4_1.pdf', dpi=300,  bbox_inches='tight', transparent=False)
    ###################################
    fig2, ax2 = plt.subplots(figsize=(xlength*1.8,ylength*2.2))
    ax2.set_ylim(-0.04, 0.06)
    ax2.set_xticks(np.arange(0, 8, 1), np.arange(1, 9, 1))
    ax2.set_yticks(np.arange(-0.04, 0.061, 0.02))
    ax2.set_xlabel(r'$m$', labelpad=1, size=10)
    ax2.set_ylabel(r'$\beta$', rotation=0,labelpad=3, size=10)  # 输入乘号？
    la = []
    ny = []
    dlft = []
    hous = []
    sand = []
    miam = []
    sanb = []
    sanj = []
    rive = []
    phil = []
    for i in range(8):
        la.append(df2['slope'].values.tolist()[0+i*10])
        ny.append(df2['slope'].values.tolist()[1 + i*10])
        dlft.append(df2['slope'].values.tolist()[2 + i*10])
        hous.append(df2['slope'].values.tolist()[3 + i * 10])
        sand.append(df2['slope'].values.tolist()[4 + i * 10])
        miam.append(df2['slope'].values.tolist()[5 + i * 10])
        sanb.append(df2['slope'].values.tolist()[6 + i * 10])
        sanj.append(df2['slope'].values.tolist()[7 + i * 10])
        rive.append(df2['slope'].values.tolist()[8 + i * 10])
        phil.append(df2['slope'].values.tolist()[9 + i * 10])
    # print(la)
    # ax2.plot(la, '-o', color='tab:blue',linewidth=0.81, markersize='4', markerfacecolor='white', markeredgecolor='tab:blue', markeredgewidth=1.2,label='la')
    # ax2.plot(ny, '-o', color='tab:orange',linewidth=0.81, markersize='5', markerfacecolor='white', markeredgecolor='tab:orange', markeredgewidth=1.2, label='ny')
    # ax2.plot(dlft, '-o', color='tab:green',linewidth=0.81, markersize='5', markerfacecolor='white', markeredgecolor='tab:green', markeredgewidth=1.2, label='dlft')
    # ax2.plot(hous, '-o',color='tab:red',linewidth=0.81, markersize='5', markerfacecolor='white', markeredgecolor='tab:red', markeredgewidth=1.2,label='hous')
    # ax2.plot(sand, '-o', color='tab:purple',linewidth=0.81, markersize='5', markerfacecolor='white', markeredgecolor='tab:purple', markeredgewidth=1.2, label='sand')
    # ax2.plot(miam, '-o', color='tab:brown',linewidth=0.81, markersize='5', markerfacecolor='white', markeredgecolor='tab:brown', markeredgewidth=1.2, label='miam')
    # ax2.plot(sanb, '-o',color='tab:pink',linewidth=0.81, markersize='5', markerfacecolor='white', markeredgecolor='tab:pink', markeredgewidth=1.2,label='sanb')
    # ax2.plot(sanj, '-o', color='tab:gray',linewidth=0.81, markersize='5', markerfacecolor='white', markeredgecolor='tab:gray', markeredgewidth=1.2, label='sanj')
    # ax2.plot(rive, '-o', color='tab:olive',linewidth=0.81, markersize='5', markerfacecolor='white', markeredgecolor='tab:olive', markeredgewidth=1.2, label='rive')
    # ax2.plot(phil, '-o', color='tab:cyan',linewidth=0.81, markersize='5', markerfacecolor='white', markeredgecolor='tab:cyan', markeredgewidth=1.2, label='phil')
    ax2.plot(la, '-o', color='tab:blue', linewidth=0.81, markersize='3',
             markeredgecolor='tab:blue', markeredgewidth=1.2, label='la')
    ax2.plot(ny, '-o', color='tab:orange', linewidth=0.81, markersize='3',
             markeredgecolor='tab:orange', markeredgewidth=1.2, label='ny')
    ax2.plot(dlft, '-o', color='tab:green', linewidth=0.81, markersize='3',
             markeredgecolor='tab:green', markeredgewidth=1.2, label='dlft')
    ax2.plot(hous, '-o', color='tab:red', linewidth=0.81, markersize='3',
             markeredgecolor='tab:red', markeredgewidth=1.2, label='hous')
    ax2.plot(sand, '-o', color='tab:purple', linewidth=0.81, markersize='3',
             markeredgecolor='tab:purple', markeredgewidth=1.2, label='sand')
    ax2.plot(miam, '-o', color='tab:brown', linewidth=0.81, markersize='3',
             markeredgecolor='tab:brown', markeredgewidth=1.2, label='miam')
    ax2.plot(sanb, '-o', color='tab:pink', linewidth=0.81, markersize='3',
             markeredgecolor='tab:pink', markeredgewidth=1.2, label='sanb')
    ax2.plot(sanj, '-o', color='tab:gray', linewidth=0.81, markersize='3',
             markeredgecolor='tab:gray', markeredgewidth=1.2, label='sanj')
    ax2.plot(rive, '-o', color='tab:olive', linewidth=0.81, markersize='3',
             markeredgecolor='tab:olive', markeredgewidth=1.2, label='rive')
    ax2.plot(phil, '-o', color='tab:cyan', linewidth=0.81, markersize='3',
             markeredgecolor='tab:cyan', markeredgewidth=1.2, label='phil')
    plt.legend(ncol=2, frameon=True, loc='upper center', labelspacing=0.2, columnspacing=0.5,
               handlelength=1.0, handletextpad=0.2, borderaxespad=0.1, fontsize=10)
    # # ax1.grid()
    plt.subplots_adjust(left=0.10, right=r, top=t, bottom=b, hspace=h, wspace = w)
    plt.savefig('./c4_figs/4_2.pdf', dpi=300,  bbox_inches='tight', transparent=False)
    ###################################
    fig3, ax3 = plt.subplots(figsize=(xlength * 1.9, ylength * 3.4),nrows=5, ncols=2)

    la_l = []
    ny_l = []
    dlft_l = []
    hous_l = []
    sand_l = []
    miam_l = []
    sanb_l = []
    sanj_l = []
    rive_l = []
    phil_l = []
    # n = 240
    la = df3['slope'].values.tolist()[0 : 240]
    ny = df3['slope'].values.tolist()[240: 240*2]
    dlft = df3['slope'].values.tolist()[240*2: 240*3]
    hous = df3['slope'].values.tolist()[240*3: 240* 4]
    sand = df3['slope'].values.tolist()[240*4: 240* 5]
    miam = df3['slope'].values.tolist()[240*5: 240* 6]
    sanb = df3['slope'].values.tolist()[240*6: 240* 7]
    sanj = df3['slope'].values.tolist()[240*7: 240* 8]
    rive = df3['slope'].values.tolist()[240*8: 240* 9]
    phil= df3['slope'].values.tolist()[240*9 : 240* 10]

    for i in range(8):
        la_l.append(np.array(la[i*30:(i+1)*30]))
        ny_l.append(np.array(ny[i * 30:(i + 1) * 30]))
        dlft_l.append(np.array(dlft[i * 30:(i + 1) * 30]))
        hous_l.append(np.array(hous[i * 30:(i + 1) * 30]))
        sand_l.append(np.array(sand[i*30:(i+1)*30]))
        miam_l.append(np.array(miam[i * 30:(i + 1) * 30]))
        sanb_l.append(np.array(sanb[i * 30:(i + 1) * 30]))
        sanj_l.append(np.array(sanj[i * 30:(i + 1) * 30]))
        rive_l.append(np.array(rive[i * 30:(i + 1) * 30]))
        phil_l.append(np.array(phil[i * 30:(i + 1) * 30]))

    # print(la[0], la[-1])
    # print(sand[0],sand[-1])

    # ax3.set_ylim(-0.06, 0.06)
    # ax3.set_xticks(np.arange(0, 8, 1), np.arange(1, 9, 1))
    # ax3.set_yticks(np.arange(-0.06, 0.061, 0.02))
    # ax3.set_xlabel(r'$m$', labelpad=1, size=10)
    # ax3.set_ylabel(r'$\beta$', rotation=0, labelpad=3)  # 输入乘号？

    for j in range(2):
        if j==0:
            # ax3[0, j].plot(la, color='tab:blue', alpha=0.8, linewidth=lw,  label='la')
            ax3[0, j].boxplot(la_l,
                              medianprops={'color': 'red', 'linewidth': '1.5'},
                              meanline=True,showmeans=True,
                              meanprops={'color': 'blue', 'ls': '--', 'linewidth': 1},
                              flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 2},
                              labels=np.arange(8))
            # ax3[0, j].set_title('la')
            # ax3[0, j].boxplot(la_l,
            #             medianprops={'color': 'red', 'linewidth': '1.5'},
            #             meanline=True,
            #             showmeans=True,
            #             meanprops={'color': 'blue', 'ls': '--', 'linewidth': 3},
            #             flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
            #             labels='la')
            ax3[1, j].boxplot(ny_l, medianprops={'color':'tab:orange', 'linewidth':lw})
            ax3[2, j].boxplot(dlft_l, medianprops={'color':'tab:green', 'linewidth':lw})
            ax3[3, j].boxplot(hous_l, medianprops={'color':'tab:red', 'linewidth':lw})
            ax3[4, j].boxplot(sand_l, medianprops={'color':'tab:purple', 'linewidth':lw},labels=np.arange(1,9,1))

            for i in range(5):
                if i<4:
                    ax3[i, j].tick_params(axis="x", labelbottom=False)
                    ax3[i, j].set_ylim(-0.06, 0.06)
                    ax3[i, j].set_yticks(np.arange(-0.06, 0.061, 0.02))
                else:
                    ax3[i, j].set_ylim(-0.06, 0.06)
                    ax3[i, j].set_yticks(np.arange(-0.06, 0.061, 0.02))
                    ax3[i, j].tick_params(axis="x", labelbottom=True)
                    # ax3[i, j].set_xticks(np.arange(0, 8, 1), np.arange(1, 9, 1))
        else:
            ax3[0, j].boxplot(miam_l, medianprops={'color':'tab:brown', 'linewidth':lw})
            # ax3[0, j].boxplot(miam,
            #             medianprops={'color': 'red', 'linewidth': '1.5'},
            #             meanline=True,
            #             showmeans=True,
            #             meanprops={'color': 'blue', 'ls': '--', 'linewidth': 3},
            #             flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
            #             labels='miam')
            ax3[1, j].boxplot(sanb_l, medianprops={'color':'tab:pink', 'linewidth':lw})
            ax3[2, j].boxplot(sanj_l, medianprops={'color':'tab:gray', 'linewidth':lw})
            ax3[3, j].boxplot(rive_l, medianprops={'color':'tab:olive', 'linewidth':lw})
            ax3[4, j].boxplot(phil_l, medianprops={'color':'tab:cyan', 'linewidth':lw},labels=np.arange(1,9,1))
            for i in range(5):
                if i<4:
                    ax3[i, j].tick_params(axis="both", labelbottom=False, labelleft=False)
                    ax3[i, j].set_ylim(-0.06, 0.06)
                    # ax3[i, j].set_yticks(np.arange(-0.06, 0.061, 0.02))
                else:
                    ax3[i, j].set_ylim(-0.06, 0.06)
                    ax3[i, j].tick_params(axis="y", labelleft=False)
                    # ax3[i, j].set_xticks(np.arange(0, 8, 1), np.arange(1, 9, 1))
    #
    fig3.text(0.45, 0.2, r'sand', va='center', rotation=0)
    fig3.text(0.45, 0.385, r'hous', va='center', rotation=0)
    fig3.text(0.45, 0.570, r'dlft', va='center', rotation=0)
    fig3.text(0.45, 0.755, r'ny', va='center', rotation=0)
    fig3.text(0.45, 0.940, r'la', va='center', rotation=0)

    fig3.text(0.90, 0.2, r'phil', va='center', rotation=0)
    fig3.text(0.90, 0.385, r'rive', va='center', rotation=0)
    fig3.text(0.90, 0.570, r'sanj', va='center', rotation=0)
    fig3.text(0.90, 0.755, r'sanb', va='center', rotation=0)
    fig3.text(0.90, 0.940, r'miam', va='center', rotation=0)

    fig3.text(0.025, 0.5, r'$\beta$', va='center', rotation=0)
    fig3.text(0.5, 0.025, r'$m$', va='center', rotation='horizontal')
    plt.subplots_adjust(left=0.11, right=r, top=t, bottom=0.06, hspace=h, wspace = w)
    plt.savefig('./c4_figs/4_31.pdf', dpi=300,  bbox_inches='tight', transparent=False)

    ###################################
    # fig4, ax4 = plt.subplots()
    plt.show()
def plot_c4_box():
    labels = 'A'  # 标签设置为A
    data = np.random.rand(4)  # 生成含4个随机元素的列表
    plt.grid(True)  # 显示网格
    plt.boxplot(data,
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': 3},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                labels=labels)
    plt.yticks(np.arange(0.1, 0.9, 0.1))
    plt.savefig('4.5箱线图1.png', dpi=400)
    plt.show()
    ###########################################
    # # 创建一些随机数据
    # data1 = np.random.normal(0, 1, 100)
    # data2 = np.random.normal(1, 1.5, 100)
    # data3 = np.random.normal(2, 2, 100)
    # data = [data1, data2, data3]
    # # print((data))
    # # 绘制箱线图
    # plt.boxplot(data)
    # # 添加标题和标签
    # plt.title('Boxplot Example')
    # plt.xticks([1, 2, 3], ['Data 1', 'Data 2', 'Data 3'])
    # plt.ylabel('Value')
    # # 显示图形
    # plt.show()
    ###########################################
    # # 假设data是一个包含多个组数据的列表
    # data = [
    #     [1, 3, 5, 7, 9],
    #     [2, 4, 6, 8, 10],
    #     [15, 16, 17, 18, 19],
    # ]
    # # 设置数据的标签
    # labels = ['组1', '组2', '组3']
    # # 创建箱型图
    # plt.boxplot(data, labels=labels)
    # # 添加图例
    # plt.legend()
    # # 显示图形
    # plt.show()
if __name__=="__main__":
    print(dt_prep)
    features = ['death', 'tempt','PM10', 'O3', 'CO']#[['death','tempt','PM10','O3','SO2', 'CO','NO2']]
    dt = dt_prep.read_data(features)
    epochs_set = 5000
    ###################### set the distributed lags and training epochs
    seqs = [1,2,3,4,5,6,7,8,9,10,11,12]
    ###################### compare lstm with gam
    # seqs = [2, 4, 6, 8]
    ######################
    train_sizes = []
    for seq in seqs:
        dataX, dataY, torch_dataX, torch_dataY, feature_size, train_x, train_y, train_size, test_x, test_y, test_size = dt_prep.split_dataset(dt, seq)
        train_sizes.append(train_size)
    # print(train_sizes)
    # plot_results.plot_configure(0.8)
    ###############################################################################################
    #  ###################### plot prediction and actual mortality for different m
    xlen,ylen = plot_results.plot_configure()
    # plot_prediction1(seqs,train_sizes,xlen,ylen)
    # plot_loss_var(seqs, epochs_set,xlen,ylen)
    # plot_loss_prediction(seqs, epochs_set, train_sizes, xlen, ylen)
    # plt.show()
    #  ###################### plot prediction and actual mortality for multipollutant
    # xlen,ylen = plot_results.plot_configure()
    # plot_multipollutant(xlen,ylen)
    #  ####################### calculate rmse and mae PM10+O3
    # performance_cal(seqs,train_sizes)
    #  ####################### calculate rmse and mae PM10+O3+CO
    # performance_cal(seqs,train_sizes)
    #  ####################### calculate rmse and mae PM10+O3+CO+NO2
    # performance_cal(seqs,train_sizes)
    #  ####################### compare lstm with gam seqs = [2, 4, 6, 8]
    # xlen, ylen = plot_results.plot_configure()
    compare_with_gam(train_sizes,xlen, ylen)

    # ######################### plots in Chapter 4
    # plot_c4()
    # plot_c4_box()





