import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch.nn.init as init


# 计算拉普拉斯算子
def calc_gso(dir_adj):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # 计算adj波浪
    adj = dir_adj + id
    # 计算拉普拉斯算子
    row_sum = adj.sum(axis=1).A1
    row_sum_inv_sqrt = np.power(row_sum, -0.5)
    row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
    deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
    # A_{sym} = D^{-0.5} * A * D^{-0.5}
    sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)
    gso = sym_norm_adj
    # gso = cnv_sparse_mat_to_coo_tensor(gso)

    return gso


def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    # 将稀疏矩阵转变为张量形式
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device,
                                       requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape  # shape:[B, C, T, N]
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)],
                          dim=1)  # 残差连接,保存原来的x值
        else:
            x = x

        return x


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # bs, c_in, ts, n_vertex = x.shape
        x = torch.permute(x, (0, 2, 3, 1))  # 等式左边 shape[bs, ts, n_vertex, c_in]

        first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul

        return graph_conv


class GraphConvLayer(nn.Module):
    def __init__(self, c_in, c_out, gso, bias):
        super(GraphConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.gso = gso
        self.graph_conv = GraphConv(c_out, c_out, gso, bias)

    def forward(self, x):
        x_gc_in = self.align(x)  # x_gc_in：shape:[B, C, T, N]
        x_gc = self.graph_conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)  # 等式左边 shape[bs, c_in, ts, n_vertex]-[B, C, T, N]

        return x_gc_out


# 原文章引用的图卷积形式
# class GraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, x, adj):
#         support = torch.mm(x, self.weight)
#         output = torch.spmm(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'
#
#
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)


def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1, 101)
            y = y.reshape(len(x), -1, 101)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n

        return mse


def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            # y = scaler.inverse_transform(y.cpu().numpy().reshape(-1, 101)).reshape(-1)
            # y_pred = scaler.inverse_transform(model(x).view(len(x), -1, 101).cpu().numpy().reshape(-1, 101)).reshape(-1)
            # 测试不用反变换 直接在归一化的数据中进行
            y = y.cpu().numpy().reshape(-1, 101).reshape(-1)
            y_pred = model(x).view(len(x), -1, 101).cpu().numpy().reshape(-1, 101).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            # mape += (d / y).tolist()
            # print(f'd/y:{d/y},d:{d},y:{y}')
            mape += np.where(y < 0, d, d/y).tolist()
            # print(f'mape:{mape}')
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        # return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE, MAPE


def output_value(model, data_iter, scaler):
    model.eval()
    y_save_pred = []
    y_save_real = []
    with torch.no_grad():
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy().reshape(-1, 101))
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1, 101).cpu().numpy().reshape(-1, 101))
            y_save_pred = np.append(y_save_pred, y_pred).reshape(-1, 101)
            y_save_real = np.append(y_save_real, y).reshape(-1, 101)
    # 将每个用户的值去掉
    y_save_real[y_save_real < 0] = 0
    y_save_pred[y_save_pred < 0] = 0
    # 将ndarray格式转换成Dataframe格式
    y_save_real_df = pd.DataFrame(y_save_real)
    y_save_pred_df = pd.DataFrame(y_save_pred)
    writer = pd.ExcelWriter('new_output_and_real_pred_1.xlsx')
    y_save_real_df.to_excel(writer, 'page_1', float_format='%.5f')
    y_save_pred_df.to_excel(writer, 'page_2', float_format='%.5f')
    writer.save()
