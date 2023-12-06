import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from script.utility import cnv_sparse_mat_to_coo_tensor
from sklearn import preprocessing


def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = pd.read_excel(os.path.join(dataset_path, 'adj.xlsx'))
    adj = np.array(adj.values[:, 1:], dtype=float)
    adj_sp = sp.csc_matrix(adj)
    # 建立对称归一化矩阵
    adj = adj_sp + adj_sp.T.multiply(adj_sp.T > adj_sp) - adj_sp.multiply(adj_sp.T > adj_sp)
    # adj = adj.todense()  # 变成密集矩阵
    adj = adj.tocsc()
    n_vertex = 0

    if dataset_name == 'zhejiang-province(N0_01)':
        n_vertex = 101
    elif dataset_name == 'zhejiang-province(Include_02)':
        n_vertex = 101

    return adj, n_vertex


# # 加载负荷数据
# def load_data_sql(dataset_name, len_train, len_val):
#     dataset_path = './data'
#     dataset_path = os.path.join(dataset_path, dataset_name)
#     head_rows = pd.read_excel(os.path.join(dataset_path, 'adj.xlsx'), nrows=0)
#     head_rows = list(head_rows)[1:]
#     dataset_path = os.path.join(dataset_path, 'history-data')
#     history_sql_data = []   # 历史负荷数据
#     for head_row in head_rows:
#         file_name = dataset_path + '/' + head_row + '.xlsx'
#         data = pd.read_excel(file_name, usecols=[1])
#         data = data.values
#         if head_row == '长兴燃气':
#             history_sql_data = data
#         else:
#             history_sql_data = np.concatenate((history_sql_data, data), axis=1)
#     # 去除nan值加归一化
#     history_sql_data = np.nan_to_num(history_sql_data)
#     zscore = preprocessing.StandardScaler()
#     history_sql_data = zscore.fit_transform(history_sql_data)
#
#     train = history_sql_data[: len_train]
#     val = history_sql_data[len_train: len_train + len_val]
#     test = history_sql_data[len_train + len_val:]
#     # data_df = pd.DataFrame(history_sql_data)
#     # writer = pd.ExcelWriter('new.xlsx')
#     # data_df.to_excel(writer, 'page_1', float_format='%.5f')
#     # writer.save()
#     return train, val, test, zscore


# 加载所有数据
def load_data_sql(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    head_rows = pd.read_excel(os.path.join(dataset_path, 'adj.xlsx'), nrows=0)
    head_rows = list(head_rows)[1:]
    n_vertex = len(head_rows)
    n_channel = 5
    dataset_path = os.path.join(dataset_path, 'history-data')
    history_sql_data = []   # 历史负荷数据
    history_all_same_data = []   # 历史所有相同的数据[year, day of year, month, week, day, holiday_weekday]， year需要归一化
    history_max_tem_data = []  # 历史最高温度数据
    history_avg_tem_data = []  # 历史平均温度数据
    history_min_tem_data = []  # 历史最小温度数据
    history_rain_data = []  # 历史降水数据
    for head_row in head_rows:
        file_name = dataset_path + '/' + head_row + '.xlsx'
        data = pd.read_excel(file_name, usecols=[1, 24, 25, 26, 27])
        data = data.values
        # print(f'data:{data.shape}, head_row:{head_row}')
        if head_row == '长兴燃气':
            history_sql_data = data[:, 0].reshape(-1, 1)
            # 在长兴燃气上加高斯噪声
            # print(f'shape:{history_sql_data.shape}, len:{len(history_sql_data)}')
            # noise = np.random.normal(loc=0, scale=0.1 * history_sql_data, size=len(history_sql_data))
            # history_sql_data = history_sql_data + noise
            # history_sql_data = history_sql_data.reshape(-1, 1)
            history_max_tem_data = data[:, 1].reshape(-1, 1)
            history_avg_tem_data = data[:, 2].reshape(-1, 1)
            history_min_tem_data = data[:, 3].reshape(-1, 1)
            history_rain_data = data[:, 4].reshape(-1, 1)
            history_all_same_data = pd.read_excel(file_name, usecols=("D:X"))
        else:
            history_sql_data = np.concatenate((history_sql_data, data[:, 0].reshape(-1, 1)), axis=1)
            history_max_tem_data = np.concatenate((history_max_tem_data, data[:, 1].reshape(-1, 1)), axis=1)
            history_avg_tem_data = np.concatenate((history_avg_tem_data, data[:, 2].reshape(-1, 1)), axis=1)
            history_min_tem_data = np.concatenate((history_min_tem_data, data[:, 3].reshape(-1, 1)), axis=1)
            history_rain_data = np.concatenate((history_rain_data, data[:, 4].reshape(-1, 1)), axis=1)
    # 去除nan值，有nan值的变为0
    history_sql_data = np.nan_to_num(history_sql_data)
    history_max_tem_data = np.nan_to_num(history_max_tem_data)
    history_avg_tem_data = np.nan_to_num(history_avg_tem_data)
    history_min_tem_data = np.nan_to_num(history_min_tem_data)
    history_rain_data = np.nan_to_num(history_rain_data)
    history_all_same_data = np.nan_to_num(history_all_same_data.values)
    # 归一化
    zscore = preprocessing.StandardScaler()
    history_sql_data = zscore.fit_transform(history_sql_data)
    zscore_max = preprocessing.StandardScaler()
    history_max_tem_data = zscore_max.fit_transform(history_max_tem_data)
    zscore_avg = preprocessing.StandardScaler()
    history_avg_tem_data = zscore_avg.fit_transform(history_avg_tem_data)
    zscore_min = preprocessing.StandardScaler()
    history_min_tem_data = zscore_min.fit_transform(history_min_tem_data)
    zscore_rain = preprocessing.StandardScaler()
    history_rain_data = zscore_rain.fit_transform(history_rain_data)
    # 数据导入到数据集中
    n_t = len(history_sql_data)  # 时间尺度
    all_data = np.zeros([n_channel, n_t, n_vertex])
    all_data[0, :, :] = history_sql_data
    all_data[1, :, :] = history_max_tem_data
    all_data[2, :, :] = history_avg_tem_data
    all_data[3, :, :] = history_min_tem_data
    all_data[4, :, :] = history_rain_data
    # for i in range (int(history_all_same_data.shape[1])):
    #     temp = history_all_same_data[:, i]
    #     expand_data = np.expand_dims(temp, 1).repeat(n_vertex, axis=1)
    #     all_data[5+i, :, :] = expand_data
    # 训练集、验证集、测试集
    # print(f'history_sql:{history_sql_data}, all_data:{all_data[0]}')
    train = all_data[:, : len_train, :]
    val = all_data[:, len_train: len_train + len_val, :]
    test = all_data[:, len_train + len_val:, :]
    # data_df = pd.DataFrame(history_sql_data)
    # writer = pd.ExcelWriter('new.xlsx')
    # data_df.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # train_same_data = history_all_same_data[: len_train, :]
    # val_same_data = history_all_same_data[len_train: len_train + len_val, :]
    # test_same_data = history_all_same_data[len_train + len_val:, :]
    return train, val, test, zscore


# def data_transform(data, n_his, n_pred, device):
#     # produce data slices for x_data and y_data
#     n_vertex = data.shape[1]
#     len_record = len(data)
#     num = len_record - n_his - n_pred
#
#     x = np.zeros([num, 1, n_his, n_vertex])
#     y = np.zeros([num, n_pred, n_vertex])
#
#     for i in range(num):
#         head = i
#         tail = i + n_his
#         x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
#         y[i, :, :] = data[tail: tail + n_pred].reshape(n_pred, n_vertex)
#
#     return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

# 加载全部数据的transform
def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data
    n_channels = data.shape[0]
    n_vertex = data.shape[2]
    len_record = data.shape[1]
    num = len_record - n_his - n_pred
    # date_feature_num = same_data.shape[1]

    x = np.zeros([num, n_channels, n_his, n_vertex])
    y = np.zeros([num, n_pred, n_vertex])
    # x_mask = np.zeros([num, n_his, date_feature_num])

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[:, head: tail, :].reshape(n_channels, n_his, n_vertex)
        y[i, :, :] = data[0][tail: tail + n_pred].reshape(n_pred, n_vertex)
        # x_mask[i, :, :] = same_data[head: tail, :]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

