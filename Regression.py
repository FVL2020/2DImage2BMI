import json
from sklearn.metrics import mean_absolute_error
from sklearn.kernel_ridge import KernelRidge  # KRR
from sklearn.ensemble import BaggingRegressor
import sklearn.gaussian_process  # GPR
from sklearn import tree  # DTR
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ExpSineSquared, RBF, ConstantKernel
from sklearn.gaussian_process.kernels import Exponentiation, RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.svm import SVR  # SVR
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import csv


def RMSE(MSE):
    return np.sqrt(MSE)


Sex_loc = {'author': 5, 'VGG_pretrained': 1, 'Ours': 1}
BMI_loc = {'author': 9, 'VGG_pretrained': 0, 'Ours': 0}


def BMI_Category(raw_data, name):
    loc = BMI_loc[name]
    sex_loc = Sex_loc[name]

    under_data = [[], []]
    normal_data = [[], []]
    over_data = [[], []]
    obese_data = [[], []]

    for data in raw_data:
        # print(loc)
        data[sex_loc] = 0
        if data[loc] <= 18.5:
            under_data[int(data[sex_loc])].append(data)
        elif data[loc] > 18.5 and data[loc] <= 25:
            normal_data[int(data[sex_loc])].append(data)
        elif data[loc] > 25 and data[loc] <= 30:
            over_data[int(data[sex_loc])].append(data)
        elif data[loc] > 30:
            obese_data[int(data[sex_loc])].append(data)

    # print(under_data)

    sex = 0
    under_data = np.asarray(under_data[sex])
    normal_data = np.asarray(normal_data[sex])
    over_data = np.asarray(over_data[sex])
    obese_data = np.asarray(obese_data[sex])

    return Feature(under_data, name), Feature(normal_data, name), Feature(over_data, name), Feature(
        obese_data, name)


def Male_Female_data(raw_data, name):
    loc = Sex_loc[name]
    male_data = []
    female_data = []

    for data in raw_data:
        if data[loc] == 0:
            female_data.append(data)
        else:
            male_data.append(data)
    male_data = np.asarray(male_data)
    female_data = np.asarray(female_data)

    return Feature(male_data, name), Feature(female_data, name)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def Feature(data, name, df=15):
    if (name == 'author'):
        x_5f = data[:, 0:5]
        y = data[:, 9]
        return x_5f, y
    elif (name == 'VGG_pretrained'):
        x_df = data[:, 2:]
        y = data[:, 0]
        return x_df, y
    elif (name == 'Ours'):
        x_5f = data[:, 3:8]
        # print(np.delete(data[:, 2:9], 0, axis=1).shape)
        x_7f = data[:, 2:9]
        # x_7f = np.delete(x_7f, [1], axis=1)
        x_df = data[:, 9:9 + df]
        y = data[:, 0]
        return x_5f, x_7f, x_df, y


def Stdm(x):
    Mean = np.mean(x, axis=0)
    Std = np.std(x, axis=0)
    return Mean, Std


def Pre(raw_data, name):
    if (name != 'VGG_pretrained'):
        raw_data = raw_data.iloc[:, 1:]
    raw_data = raw_data.replace([np.inf, -np.inf], np.nan)
    #     raw_data = raw_data.fillna(raw_data.mean())
    raw_data = raw_data.replace(np.nan, 0)
    raw_data = raw_data.values.astype(np.float64)
    return raw_data


def DataProcess(method, df=15, mode='TEST', Net='Resnet101'):
    if method == 'Ours':
        raw_data_train = pd.read_csv(
            'ALL_feature/Image_train.csv'.format(Net))
        raw_data_test = pd.read_csv(
            'ALL_feature/Image_test.csv'.format(Net), header=None)
    raw_data_name = raw_data_test.values
    raw_data_train = Pre(raw_data_train, method)
    raw_data_test = Pre(raw_data_test, method)
    if method == 'Ours':
        x_5f_train, x_7f_train, x_df_train, y_train = Feature(raw_data_train, method, df=df)
        x_5f_test, x_7f_test, x_df_test, y_test = Feature(raw_data_test, method, df=df)
        (x_5f_train_m, x_7f_train_m, x_df_train_m, y_train_m), (
            x_5f_train_f, x_7f_train_f, x_df_train_f, y_train_f) = Male_Female_data(raw_data_train, 'Ours')
        (x_5f_test_m, x_7f_test_m, x_df_test_m, y_test_m), (
            x_5f_test_f, x_7f_test_f, x_df_test_f, y_test_f) = Male_Female_data(raw_data_test, 'Ours')
        (_, x_7f_test_u, x_df_test_u, y_test_u), (_, x_7f_test_n, x_df_test_n, y_test_n), (
            _, x_7f_test_ov, x_df_test_ov, y_test_ov), (_, x_7f_test_ob, x_df_test_ob, y_test_ob) = BMI_Category(
            raw_data_test, 'Ours')
    else:
        x_train, y_train = Feature(raw_data_train, method)
        x_test, y_test = Feature(raw_data_test, method)

    if method == 'author':
        Mean, Std = Stdm(x_train)
        x_train = (x_train - Mean) / Std
        x_test = (x_test - Mean) / Std
    elif method == 'Ours':
        x_train = x_7f_train
        x_test = x_7f_test
        Mean, Std = Stdm(x_train)
        x_test = x_7f_test
        # print('Mean: ', Mean)
        # print('Std: ', Std)
        x_train = (x_train - Mean) / Std
        x_test = (x_test - Mean) / Std
        x_train = np.append(x_train, x_df_train, axis=1)
        x_test = np.append(x_test, x_df_test, axis=1)

        y_train = y_train
        y_test = y_test

        # only df
        # x_train = x_df_train
        # x_test = x_df_test

    if mode == 'DEMO':
        return raw_data_name, x_test, y_test

    return x_train, y_train, x_test, y_test


def Regression(method, Net='Resnet101'):
    # 初始化
    svr = SVR(kernel='rbf')
    dtr = tree.DecisionTreeRegressor()
    krr = KernelRidge()
    KN = Exponentiation(RationalQuadratic(), exponent=2)
    gpr = sklearn.gaussian_process.GaussianProcessRegressor(kernel=KN, alpha=1e-3)

    x_train, y_train, x_test, y_test = DataProcess(method, Net=Net)
    print("Train size:", len(x_train))
    print("Test size:", len(x_test))

    regressors = [svr, krr, dtr, gpr]
    y_pred = [[], [], [], []]
    reg_name = ['SVR', 'KRR', 'DTR', 'GPR']

    for i, reg in enumerate(regressors):
        reg.fit(x_train, y_train)
        y_pred[i] = reg.predict(x_test)
        print(reg_name[i], ': MAE: ', mean_absolute_error(y_test, y_pred[i]),
              'MAPE: ', mean_absolute_percentage_error(y_test, y_pred[i]),
              'R2: ', r2_score(y_test, y_pred[i]),
              'RMSE: ', RMSE(mean_squared_error(y_test, y_pred[i])),
              )


if __name__ == '__main__':
    Regression('Ours')
