#!/usr/bin/env Python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from NSNPs import transferFunc
import pandas as pd

import warnings

warnings.filterwarnings("ignore")
import time


def splitData(dataset, ratio=0.85):
    len_train_data = int(dataset.shape[1] * ratio)
    return dataset[:, :len_train_data], dataset[:, len_train_data:]


# form feature matrix from sequence
def create_dataset(seq, belta, Order, current_node):
    Nc, K = seq.shape
    samples = np.zeros(shape=(K, Order * Nc + 2))
    for m in range(Order, K):
        for n_idx in range(Nc):
            for order in range(Order):
                samples[m - Order, n_idx * Order + order + 1] = transferFunc(seq[n_idx, m - 1 - order])
        samples[m - Order, 0] = 1
        samples[m - Order, -1] = seq[current_node, m]
    return samples


def predict(samples, weight, steepness, belta):
    K, _ = samples.shape
    predicted_data = np.zeros(shape=(1, K))
    for t in range(K):
        features = samples[t, :-1]
        # -----------------------------------------
        length = len(features)
        for i in range(length):
            features[i] = transferFunc(features[i], belta)
        # -----------------------------------------
        predicted_data[0, t] = steepness * np.dot(weight, features)
    return predicted_data


# normalize data set
def normalize(ori_data, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:
        N, K = data.shape
        minV = np.zeros(shape=K)
        maxV = np.zeros(shape=K)
        for i in range(N):
            minV[i] = np.min(data[i, :])
            maxV[i] = np.max(data[i, :])
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':
                    data[i, :] = (data[i, :] - minV[i]) / (maxV[i] - minV[i])
                else:
                    data[i, :] = 2 * (data[i, :] - minV[i]) / (maxV[i] - minV[i]) - 1
        return data, maxV, minV
    else:
        minV = np.min(data)
        maxV = np.max(data)
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':
                data = (data - minV) / (maxV - minV)
            else:
                data = 2 * (data - minV) / (maxV - minV) - 1
        return data, maxV, minV


#  re-normalize data set

def re_normalize(ori_data, maxV, minV, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:
        Nc, K = data.shape
        for i in range(Nc):
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':
                    data[i, :] = data[i, :] * (maxV[i] - minV[i]) + minV[i]
                else:
                    data[i, :] = (data[i, :] + 1) * (maxV[i] - minV[i]) / 2 + minV[i]
    else:
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':
                data = data * (maxV - minV) + minV
            else:
                data = (data + 1) * (maxV - minV) / 2 + minV
    return data



def NSNP(dataset1, ratio=0.7, plot_flag=False):
    dataset_name = 'gas'

    normalize_style = '-01'
    dataset, maxV, minV = normalize(dataset1, normalize_style)

    #In order to adapt to NSST transform,a stack of source data is developed according to the characteristics
    character_num = 2
    # number of characters PM2.5=8 Air=13 gas=2 nasdaq100=82 sunspot=4
    clo_mun = 78  # 84
    # PM2.5=84  Air=78 nasdaq100=574 gas = 78 sunspot=8
    dataset_recur = np.zeros(shape=(clo_mun, dataset.shape[1]))
    for i in range(0, clo_mun, character_num):
        dataset_recur[i:i + character_num, :] = dataset

    np.savetxt('./Normalize_dataset/'+dataset_name+'.csv', dataset_recur, delimiter=',')
    belta = 1

    # partition dataset into train set and test set
    if dataset.shape[1] > 30:
        train_data, test_data = splitData(dataset, ratio)
    else:
        train_data, test_data = splitData(dataset, 1)
        test_data = train_data

    len_train_data = train_data.shape[1]
    len_test_data = test_data.shape[1]

    alpha_list = [1e-12]  # [1e-12, 1e-14, 1e-20]

    Nc = 12  # Number of neurons
    Order = 2
    L = 12  # decomposition level

    best_predict = np.zeros(shape=(2, len_train_data))

    row = dataset.shape[0]
    clo = dataset.shape[1]
    coffis01 = np.zeros((L, row, clo))
    # Read the transformed coefficient
    coffis01[0] = pd.read_csv('./dataset_coeffs/' + dataset_name + '/one/01_' + dataset_name + '.csv', delimiter=',',
                              header=None).values[0:row, :]
    coffis01[1] = pd.read_csv('./dataset_coeffs/' + dataset_name + '/one/02_' + dataset_name + '.csv', delimiter=',',
                              header=None).values[0:row, :]
    coffis01[2] = pd.read_csv('./dataset_coeffs/' + dataset_name + '/two/1_' + dataset_name + '.csv', delimiter=',',
                              header=None).values[0:row, :]
    coffis01[3] = pd.read_csv('./dataset_coeffs/' + dataset_name + '/two/2_' + dataset_name + '.csv', delimiter=',',
                              header=None).values[0:row, :]
    num = 8
    for i in range(num):
        coffis01[i + 4] = pd.read_csv('./dataset_coeffs/' + dataset_name + '/three/' + str(i + 1) + '_' + dataset_name + '.csv', delimiter=',',header=None).values[0:row, :]

    #  Extract the same features from L layers transformation
    coffis0 = np.zeros((row, L, clo))
    for n in range(row):
        for i in range(L):
            coffis0[n, i, :] = coffis01[i, n, :]

    data_predicted = np.zeros((row, L, clo))

    new_data_predicted_recur = np.zeros(shape=(L, clo_mun, dataset.shape[1]))
    for coffis_num in range(row):
        for alpha in alpha_list:
            coffis = coffis0[coffis_num]
            U_train = coffis[:, :len_train_data]
            #
            # the regression
            tol = 1e-24
            from sklearn import linear_model
            clf = linear_model.Ridge(alpha=alpha, max_iter=100, fit_intercept=False, tol=tol)
            # clf = linear_model.Lasso(alpha=alpha, fit_intercept=False, tol=tol)

            # solving x = Aw to obtain x(x is the weight vector corresponding to certain node)
            # learned weight matrix
            W_learned = np.zeros(shape=(Nc, Nc * Order + 1))
            samples_train = {}
            for node_solved in range(Nc):
                samples = create_dataset(U_train, belta, Order, node_solved)
                samples_train[node_solved] = samples[:-Order, :]
                clf.fit(samples[:, :-1], samples[:, -1])
                W_learned[node_solved, :] = clf.coef_

            steepness = np.max(np.abs(W_learned), axis=1)
            for i in range(Nc):
                if steepness[i] > 1:
                    W_learned[i, :] /= steepness[i]

            # ------------------------------------------------
            # wi, wj = W_learned.shape
            # for ii in range(wi):
            #     for jj in range(wj):
            #         if ii == jj:
            #             W_learned[ii, jj] = 0
            # ----------------------------------------------------------

            # predict on training data set
            trainPredict = np.zeros(shape=(Nc, len_train_data))
            for i in range(Nc):
                trainPredict[i, :Order] = U_train[i, :Order]
                trainPredict[i, Order:] = predict(samples_train[i], W_learned[i, :], steepness[i], belta)

            new_trainPredict = trainPredict
            rmse = -1
            nmse = -1
            best_predict = new_trainPredict

        if dataset.shape[1] <= 30:
            data_predicted = best_predict
            data_predicted = re_normalize(data_predicted, maxV, minV, normalize_style)
            return data_predicted, rmse, nmse, Order, Nc, alpha
        else:

            coffis = coffis0[coffis_num]
            U_test = coffis[:, len_train_data - Order:]

            testPredict = np.zeros(shape=(Nc, len_test_data))
            samples_test = {}
            for i in range(Nc):
                samples = create_dataset(U_test, belta, Order, i)
                samples_test[i] = samples[:-Order, :]
                testPredict[i, :] = predict(samples_test[i], W_learned[i, :], steepness[i], belta)

            new_testPredict = testPredict
            data_predicted[coffis_num] = np.hstack((best_predict, new_testPredict))

    # Scatter the same feature into L layers
    data_predicted_tempt = np.zeros((L, row, clo))
    for n in range(L):
        for i in range(row):
            data_predicted_tempt[n, i, :] = data_predicted[i, n, :]

    for coffis_num in range(L):
        for i in range(0, clo_mun, character_num):
            new_data_predicted_recur[coffis_num][i:i + character_num, :] = data_predicted_tempt[coffis_num]

    # Save trained coefficients
    np.savetxt('./re_coffis/' + dataset_name + '/one/01_' + dataset_name + '.csv', new_data_predicted_recur[0],
               delimiter=',')
    np.savetxt('./re_coffis/' + dataset_name + '/one/02_' + dataset_name + '.csv', new_data_predicted_recur[1],
               delimiter=',')
    np.savetxt('./re_coffis/' + dataset_name + '/two/01_' + dataset_name + '.csv', new_data_predicted_recur[2],
               delimiter=',')
    np.savetxt('./re_coffis/' + dataset_name + '/two/02_' + dataset_name + '.csv', new_data_predicted_recur[3],
               delimiter=',')
    num = 8
    for i in range(num):
        np.savetxt('./re_coffis/' + dataset_name + '/three/' + str(i + 1) + '_' + dataset_name + '.csv',
                   new_data_predicted_recur[i + 4], delimiter=',')
    new_data_predicted = pd.read_csv('./reconctruct/' + dataset_name + '/' + dataset_name + '.csv', delimiter=',',
                                     header=None).values[0:row, :]

    # np.savetxt("./predicted/"+dataset_name + '.txt', new_data_predicted)
    new_data_predicted = re_normalize(new_data_predicted, maxV, minV, normalize_style)

    return new_data_predicted, Order, Nc, alpha




def main():
    # load time series data

    dataset_name = 'gas'

    # # data set 01: Nasdaq100
    # oil_production_src = "./datasets/nasdaq100_padding.csv"
    # # dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d')
    # oil_production = pd.read_csv(oil_production_src, delimiter=',').values
    # dataset0 = np.array(oil_production, dtype=np.float)
    # dataset = np.zeros(shape=(dataset0.shape[1], dataset0.shape[0]))
    # for i in range(82):
    #     dataset[i, :] = dataset0[:, i]
    # ratio = 13/15


    # # data set 02: PM2.5
    # oil_production_src = "./datasets/pollution.csv"
    # # dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d')
    # oil_production = pd.read_csv(oil_production_src, delimiter=',').values
    # encoder = LabelEncoder()
    # oil_production[:, 5] = encoder.fit_transform(oil_production[:, 5])
    # # 保证为float ensure all data is float
    # oil_production = oil_production[:,1:].astype('float32')
    # dataset0 = np.array(oil_production[:,:], dtype=np.float)
    # dataset = np.zeros(shape=(dataset0.shape[1], dataset0.shape[0]))
    # for i in range(8):
    #     dataset[i, :] = dataset0[:, i]
    # ratio = 0.8

    # data set 03: gas
    oil_production_src = "./datasets/gas.csv"
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d')
    oil_production = pd.read_csv(oil_production_src, delimiter=',').values
    dataset0 = np.array(oil_production[6:, 1:], dtype=np.float)
    dataset = np.zeros(shape=(dataset0.shape[1], dataset0.shape[0]))
    for i in range(2):
        dataset[i, :] = dataset0[:, i]
    ratio = 0.69

    # data set 04: sunspot
    # oil_production_src = "./datasets/sunspot2.csv"
    # # dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d')
    # oil_production = pd.read_csv(oil_production_src, delimiter=';',header=None).values
    # dataset0 = np.array(oil_production[:, 3:], dtype=np.float)
    # dataset = np.zeros(shape=(dataset0.shape[1], dataset0.shape[0]))
    # for i in range(4):
    #     dataset[i, :] = dataset0[:, i]

    # data set 05:AirQualityUCI
    # oil_production_src = "./datasets/AirQualityUCI_last.csv"
    # oil_production = pd.read_csv(oil_production_src, delimiter=',').values
    # dataset0 = np.array(oil_production[:, 1:14], dtype=np.float)
    # dataset = np.zeros(shape=(dataset0.shape[1], dataset0.shape[0]))
    # for i in range(13):
    #     dataset[i, :] = dataset0[:, i]
    # time = oil_production[:, 0]



    # ratio = 0.6063 #gas=0.69pollution=0.8992 sunspot=0.6933 AirQualityUCI_last = 0.8
    # dataset_two = dataset




    # partition dataset into train set and test set
    length = dataset.shape[1]
    len_train_data = int(length * ratio)
    len_test_data = length - len_train_data

    # perform prediction
    data_predicted, Order, Nc, alpha = NSNP(dataset, ratio)
    '''***************************************'''
    # dataset_Nor_org = pd.read_csv('./Normalize_dataset/'+ dataset_name + '.csv', delimiter=',',header=None).values
    # np.savetxt('./predicted/chaotic/' + dataset_name + '.txt', data_predicted)
    # dataset_Nor_org, maxV, minV = normalize(dataset, '-01')
    # data_predicted_Nor = np.loadtxt('./predicted/' + dataset_name + '.txt')
    # data_predicted = re_normalize(data_predicted_Nor,maxV,minV)
    # np.savetxt('./predicted/' + dataset_name + '.txt', data_predicted)


    # Outcomes
    print('*' * 80)
    print('The ratio is %f' % ratio)
    print('Order is %d, Nc is %d, alpha is %g' % (Order, Nc, alpha))
    mse, rmse, mae, nmse, rae, rrse, smape, nrmse = statistics(dataset[:,len_train_data:], data_predicted[:,len_train_data:])
    # mse, rmse, mae, nmse, rae, rrse, smape, nrmse = statistics(dataset_Nor_org[80:81, len_train_data:],data_predicted_Nor[80:81, len_train_data:])
    print('Forecasting on test dataset: RMSE|MAE|SMAPE|NRMSE is : |%f |%f|%f|%f  ' % (rmse, mae, smape,nrmse))

    # print length of each subdatasets

    print('The whole length is %d' % length)
    print('Train dataset length is %d' % (len_train_data))
    print('Test dataset length is %d' % len_test_data)


    fig4 = plt.figure() # figsize=(15, 5)
    fig5 = plt.figure()
    ax41 = fig4.add_subplot(111)
    ax51 = fig5.add_subplot(111)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax41.set_ylabel(..., fontsize=15)
    ax51.set_ylabel(..., fontsize=15)

    time = [i for i in range(len_test_data)]
    ax41.plot(time, dataset[-1, -1:-len_test_data - 1:-1].ravel(), 'r-', label='the original data')
    ax41.plot(time, data_predicted[-1, -1:-len_test_data - 1:-1].ravel(), 'g--', label='the predicted data')
    ax51.plot(time, np.abs(dataset[-1, -1:-len_test_data - 1:-1]-data_predicted[-1, -1:-len_test_data - 1:-1]).ravel(), 'b--', label='Absolute error')

    ax41.set_ylabel("Magnitude")
    ax41.set_xlabel('Time')
    ax41.set_title('GAS')
    ax51.set_ylabel("Magnitude")
    ax51.set_xlabel('Time')
    ax51.set_title('GAS')

    ax41.legend()
    ax51.legend()
    plt.tight_layout()
    plt.show()




def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    return mape


def Mdape(y_true, y_pred):
    return np.median(np.abs((y_pred - y_true) / y_true)) * 100


def statistics(origin, predicted):
    # # compute RMSE
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import median_absolute_error
    from sklearn.metrics import r2_score
    ssr = ((origin - origin.mean()) ** 2).sum()
    sst = ((predicted - origin) ** 2).sum()
    rsq = 1 - (ssr / sst)
    # r2 = r2_score(origin, predicted)
    t1 = np.abs(predicted - origin).sum()
    t2 = np.abs(origin - origin.mean()).sum()
    rae = t1 / t2
    mae = mean_absolute_error(origin, predicted)
    mse = mean_squared_error(origin, predicted)

    # mape = MAPE(origin,predicted)
    # mdape = Mdape(origin,predicted)
    smape = 2.0 * np.mean(np.abs(predicted - origin) / (np.abs(predicted) + np.abs(origin)))
    mse0 = (mse*origin.shape[1])/(origin.shape[1]-1)
    rmse = np.sqrt(mse)
    meanV = np.mean(origin)
    dominator = np.linalg.norm(predicted - meanV, 2)
    Len = origin.shape[1]
    avr = np.sqrt(Len * (Len) * np.power(dominator, 2))
    return mse, rmse, mae, mse / np.power(dominator, 2), rae, np.sqrt(1 / (1 - rsq)), smape,Len*mse/avr


if __name__ == '__main__':
    main()
