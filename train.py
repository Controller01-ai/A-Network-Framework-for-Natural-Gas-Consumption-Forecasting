import logging
import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.utils.data
import time
from thop import profile

from script import dataloader, utility, earlystopping
from model import New_Transformer_2_1
from loss.custom_loss import Asymmetric_Gaussian_Mse


def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_parameters():
    parser = argparse.ArgumentParser(description='STTN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=30, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='zhejiang-province(No_02)',
                        choices=['zhejiang-province(No_01)', 'zhejiang-province(Include_02)'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=1,
                        help='the number of time interval for prediction, default as 3')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--in_channels', type=int, default=5)
    parser.add_argument('--time_num', type=int, default=366)

    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dilation_t', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--drop_rate', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=300, help='epochs, default as 100')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=2)  # 之前是2
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--loss_parameter', type=int, default=0.05, help='loss function super parameter')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))
    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 设计embed_size的大小
    embed_size = []
    embed_size.append([64])
    for l in range(args.num_layers):
        embed_size.append([64, 32, 64])  # 原来都是64
    embed_size.append([64])

    return args, device, embed_size


def data_preparate(args, device):
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(dir_adj=adj)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)
    adj_torch = adj.toarray()
    adj_torch = adj_torch.astype(dtype=np.float32)
    args.adj = torch.from_numpy(adj_torch).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    dataset_path = os.path.join(dataset_path, 'history-data')
    data_col = pd.read_excel(os.path.join(dataset_path, '长兴燃气.xlsx')).values.shape[0]
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test, zscore = dataloader.load_data_sql(args.dataset,
                                                        len_train,
                                                        len_val)
    # zscore = preprocessing.StandardScaler()
    # train = zscore.fit_transform(train)
    # val = zscore.transform(val)
    # test = zscore.transform(test)
    # 保存val
    # val_df = pd.DataFrame(val)
    # writer = pd.ExcelWriter('new_1.xlsx')
    # val_df.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()

    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, device)
    # print(f'len_train；{len_train}, len_val；{len_val}')
    # print(f'val_x:{x_val[0, 0, :, :]}, val_y:{y_val[0, :, :]}')

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter


def prepare_model(args, device, embed_size):
    loss = nn.MSELoss()
    my_loss = Asymmetric_Gaussian_Mse(args.loss_parameter)
    es = earlystopping.EarlyStopping(patience=args.patience, verbose=True)

    # model = ST_Transformer.STTransformer(adj=args.adj, gso=args.gso, in_channels=args.in_channels, embed_size=args.
    #                                      embed_size, time_num=args.time_num, num_layers=args.num_layers, T_dim=args.n_his,
    #                                      output_T_dim=args.n_pred, heads=args.heads, enable_bias=args.enable_bias).to(device)
    model = New_Transformer_2_1.STTransformer(adj=args.adj, gso=args.gso, in_channels=args.in_channels, embed_size=
                                            embed_size, time_num=args.time_num, num_layers=args.num_layers,
                                            T_dim=args.n_his, output_T_dim=args.n_pred, heads=args.heads,
                                            enable_bias=args.enable_bias, kernel_size=args.kernel_size, dilation_t=
                                            args.dilation_t, batch_size=args.batch_size).to(device)
    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler, my_loss


def train(loss, my_loss, args, optimizer, scheduler, es, model, train_iter, val_iter):
    min_val_loss = np.inf
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            y_pred = model(x).view(len(x),-1, 101)  # [batch_size, n_pred, num_nodes]
            y = y.reshape(len(x), -1, 101)
            # print(f'y_pred:{y_pred.shape}, y:{y.shape}')
            # l = loss(y_pred, y)
            l = my_loss(y_pred, y)
            # 计算FLOPS
            # flops, params = profile(model, inputs=(x, ))
            # print(f'FLOPS:{flops}, Params:{params}')
            # 看损失函数的大小
            # print(l.item())
            # print(f'my_loss:{l_my.item()}')
            optimizer.zero_grad()
            l.backward()
            # l_my.backward()
            optimizer.step()
            # scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            # print(f'x:{x[0:4, 0, 0, :]}')
            # print(y[0:4, 25:30])
            # print(f'y:{y}, y_pred:{y_pred}')
            # print(f'l:{l.item()}, l_sum:{l_sum}')
        scheduler.step()
        val_loss = val(model, val_iter)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        es(val_loss, model)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'. \
              format(epoch + 1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        if es.early_stop:
            print("Early stopping.")
            break
    print('\nTraining finished.\n')


@torch.no_grad()
def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1, 101).reshape(-1)
        y = y.reshape(len(x), -1, 101).reshape(-1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)


@torch.no_grad()
def test(zscore, loss, model, test_iter, args):
    model.eval()
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE, test_MAPE = utility.evaluate_metric(model, test_iter, zscore)
    # utility.output_value(model, test_iter, zscore)
    print(
        f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f} | MAPE {test_MAPE:.6f}')


if __name__ == "__main__":
    # Logging
    # logger = logging.getLogger('stgcn')
    # logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    start_time = time.time()
    args, device, embed_size = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler, my_loss = prepare_model(args, device, embed_size)
    train(loss, my_loss, args, optimizer, scheduler, es, model, train_iter, val_iter)
    test(zscore, loss, model, test_iter, args)
    end_time = time.time()
    print(f'模型运行时间： {end_time-start_time}秒')
