import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=366):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, T):
        return self.pe[:T, :]


# 代替升维的conv1
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=1, padding=0, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, T, d_model, num_elements, batch_size, n_vertex):
        super(FixedEmbedding, self).__init__()

        # self.w = torch.zeros(batch_size, T, d_model).float()
        self.num_elements = num_elements
        self.d_model = d_model
        self.T = T
        self.n_vertex = n_vertex
        self.device = torch.device('cuda')

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.w = torch.zeros(self.batch_size, self.T, self.d_model).float()
        self.w.require_grad = False
        for l in range(self.num_elements):
            temp = x[:, :, l].unsqueeze(2)
            expand_data = temp.repeat(1, 1, int(self.w[:, :, l::self.num_elements].shape[2]))
            # print(f'x:{x.shape},d_model:{self.d_model}, temp:{temp.shape}, expand:{expand_data.shape}, self.w:{self.w[:, :, l::self.num_elements].shape}')
            self.w[:, :, l::self.num_elements] = expand_data
        pe = self.w.expand(self.n_vertex, self.batch_size, self.T, self.d_model).permute(1, 3, 2, 0)
        return pe


class TemporalEmbedding(nn.Module):
    def __init__(self, T, d_model, batch_size, n_vertex):
        super(TemporalEmbedding, self).__init__()
        self.T = T
        self.d_model = d_model
        self.device = torch.device('cuda')
        Embed = FixedEmbedding
        day_of_year_size = 12
        month_size = 2
        week_size = 2
        day_size = 2
        holiday_size = 3
        self.day_of_year_embed = Embed(T, d_model, day_of_year_size, batch_size, n_vertex).to(self.device)
        self.month_embed = Embed(T, d_model, month_size, batch_size, n_vertex).to(self.device)
        self.week_embed = Embed(T, d_model, week_size, batch_size, n_vertex).to(self.device)
        self.day_embed = Embed(T, d_model, day_size, batch_size, n_vertex).to(self.device)
        self.holiday_embed = Embed(T, d_model, holiday_size, batch_size, n_vertex).to(self.device)

    def forward(self, x_mask):  # x_mask:[B, T, c_in]
        day_of_year = self.day_of_year_embed(x_mask[:, :, 0:12])
        month = self.month_embed(x_mask[:, :, 12:14])
        week = self.week_embed(x_mask[:, :, 14:16])
        day = self.day_embed(x_mask[:, :, 16:18])
        holiday = self.holiday_embed(x_mask[:, :, 18:21])

        return day_of_year + month + week + day + holiday
