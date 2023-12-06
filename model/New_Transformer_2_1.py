import torch
import torch.nn as nn
from script.utility import GraphConvLayer, Align
from model.layer import TemporalConvLayer
from script import embed


class SSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        B, N, T, C = query.shape

        # Split the embedding into self.heads different pieces
        values = values.reshape(B, N, T, self.heads, self.head_dim)  # embed_size维拆成 heads×head_dim
        keys = keys.reshape(B, N, T, self.heads, self.head_dim)
        query = query.reshape(B, N, T, self.heads, self.head_dim)

        values = self.values(values)  # (B, N, T, heads, head_dim)
        keys = self.keys(keys)  # (B, N, T, heads, head_dim)
        queries = self.queries(query)  # (B, N, T, heads, heads_dim) B:batch_size; N: n_vertex; T:ts;

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("bqthd,bkthd->bqkth", [queries, keys])  # 空间self-attention
        # queries shape: (B, N, T, heads, heads_dim),
        # keys shape: (B, N, T, heads, heads_dim)
        # energy: (B, N, N, T, heads)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)  # 在K维做softmax，和为1
        # attention shape: (B, N, N, T, heads)

        out = torch.einsum("bqkth,bkthd->bqthd", [attention, values]).reshape(
            B, N, T, self.heads * self.head_dim
        )
        # attention shape: (B, N, N, T, heads)
        # values shape: (B, N, T, heads, heads_dim)
        # out after matrix multiply: (N, T, heads, head_dim), then
        # we reshape and flatten the last two dimensions. (B, N, T, heads*head_dim)

        out = self.fc_out(out)
        # print(f'embed_size:{self.embed_size}')
        # Linear layer doesn't modify the shape, final shape will be
        # (B, N, T, embed_size)

        return out


class TSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, kernel_size=3):
        super(TSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.padding = (kernel_size - 1)
        self.padding_operator = nn.ConstantPad2d((0, 0, self.padding, 0), 0)

        self.queries = nn.Conv2d(embed_size, embed_size, (kernel_size, 1), padding=0, bias=True)
        self.keys = nn.Conv2d(embed_size, embed_size, (kernel_size, 1), padding=0, bias=True)
        self.values = nn.Conv2d(embed_size, embed_size, kernel_size=1, padding=0, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        B, N, T, C = query.shape

        # 通过conv聚合时间方向的信息
        query = self.queries(self.padding_operator(query.permute(0, 3, 2, 1))).permute(0, 3, 2, 1)
        keys = self.keys(self.padding_operator(keys.permute(0, 3, 2, 1))).permute(0, 3, 2, 1)
        values = self.values(values.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        # Split the embedding into self.heads different pieces
        values = values.reshape(B, N, T, self.heads, self.head_dim)  # embed_size维拆成 heads×head_dim
        keys = keys.reshape(B, N, T, self.heads, self.head_dim)
        query = query.reshape(B, N, T, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        energy = torch.einsum("bnqhd,bnkhd->bnqkh", [query, keys])  # 时间self-attention
        # queries shape: (B, N, T, heads, heads_dim),
        # keys shape: (B, N, T, heads, heads_dim)
        # energy: (B, N, T, T, heads)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # 在K维做softmax，和为1
        # attention shape: (B, N, query_len, key_len, heads)

        out = torch.einsum("bnqkh,bnkhd->bnqhd", [attention, values]).reshape(
            B, N, T, self.heads * self.head_dim
        )
        # attention shape: (B, N, T, T, heads)
        # values shape: (B, N, T, heads, heads_dim)
        # out after matrix multiply: (B, N, T, heads, head_dim), then
        # we reshape and flatten the last two dimensions. shape:[B, N, T, heads*head_dim]

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (B, N, T, embed_size)

        return out


class STransformer(nn.Module):
    def __init__(self, c_in, c_out, heads, adj, gso, dropout, forward_expansion, enable_bias):
        super(STransformer, self).__init__()
        # Spatial Embedding
        self.adj = adj
        self.D_S = nn.Parameter(adj)  # [N, N]
        self.embed_liner = nn.Linear(adj.shape[0], c_in)  # shape:[N, C]

        self.attention = SSelfAttention(c_in, heads)
        self.norm1 = nn.LayerNorm(c_in)
        self.norm2 = nn.LayerNorm(c_in)

        self.feed_forward = nn.Sequential(
            nn.Linear(c_in, forward_expansion * c_in),
            nn.ReLU(),
            nn.Linear(forward_expansion * c_in, c_in),
        )

        # 调用GCN
        # self.gcn = GCN(embed_size, embed_size * 2, embed_size, dropout)
        # self.norm_adj = nn.InstanceNorm2d(1)  # 对邻接矩阵归一化
        self.gcn_mine = GraphConvLayer(c_in=c_in, c_out=c_out, gso=gso, bias=enable_bias)

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(c_in, c_out)
        self.fg = nn.Linear(c_out, c_out)

    def forward(self, value, key, query):
        # Spatial Embedding 部分
        B, N, T, C = query.shape
        D_S = self.embed_liner(self.D_S)
        D_S = D_S.expand(T, N, C)
        D_S = D_S.permute(1, 0, 2)  # D_S:[N, T, C]
        D_S = D_S.expand(B, N, T, C)  # shape:[B, N, T, C] 嵌入时空位置关系

        # GCN 修改版
        X_G = torch.Tensor(self.gcn_mine(torch.permute(query, (0, 3, 2, 1))))  # 左边shape:[B, C, T, N]
        X_G = torch.permute(X_G, (0, 3, 2, 1))  # 左边shape:[B, N, T, C]

        # Spatial Transformer 部分
        query = query + D_S
        attention = self.attention(value, key, query)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))  # 左边shape:[B, N, T, C]

        # 融合 STransformer and GCN
        g = torch.sigmoid(self.fs(U_S) + self.fg(X_G))  # (7)
        out = g * self.fs(U_S) + (1 - g) * self.fg(X_G)  # (8)  左边shape:[B, N, T, C]
        # out = self.fs(U_S)

        return out


class TTransformer(nn.Module):
    def __init__(self, c_in, c_out, heads, time_num, dropout, forward_expansion, kernel_size, dilation_t):
        super(TTransformer, self).__init__()

        # Temporal embedding One hot
        self.time_num = time_num
        # 将D_T也放入Tensor里面去，cuda
        self.device = torch.device('cuda')
        # self.one_hot = One_hot_encoder(embed_size, time_num)  # temporal embedding选用one-hot方式 或者
        self.temporal_embedding = nn.Embedding(time_num, c_in,
                                               device=self.device)  # temporal embedding选用nn.Embedding
        # self.position_embedding = embed.PositionalEmbedding(d_model=c_in, max_len=time_num)
        self.attention = TSelfAttention(c_in, heads)
        self.temp_conv = TemporalConvLayer(kernel_size, c_in, c_out, dilation_t=dilation_t)
        self.norm1 = nn.LayerNorm(c_in)
        self.norm2 = nn.LayerNorm(c_in)

        self.feed_forward = nn.Sequential(
            nn.Linear(c_in, forward_expansion * c_in),
            nn.ReLU(),
            nn.Linear(forward_expansion * c_in, c_in),
        )
        self.dropout = nn.Dropout(dropout)

        self.fs = nn.Linear(c_in, c_out)
        self.fg = nn.Linear(c_out, c_out)

    def forward(self, value, key, query):
        B, N, T, C = query.shape

        # D_T = self.one_hot(t, N, T)  # temporal embedding选用one-hot方式 或者
        D_T = self.temporal_embedding(
            torch.Tensor(torch.arange(0, T)).to(self.device))  # temporal embedding选用nn.Embedding
        # D_T = self.position_embedding(T).to(self.device)
        D_T = D_T.expand(N, T, C)
        # 增加batch_size这个维度
        D_T = D_T.expand(B, N, T, C)  # 左边shape:[B, N, T, C]

        # Temporal ConvLayer Layer部分, 需要将输入改成[B, C, T, N]
        X_T = torch.Tensor(self.temp_conv(torch.permute(query, (0, 3, 2, 1))))
        X_T = torch.permute(X_T, (0, 3, 2, 1))  # 左边shape:[B, N, T, C]
        # Temporal Transformer部分
        # temporal embedding加到query。 原论文采用concatenated
        query = query + D_T
        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_T = self.dropout(self.norm2(forward + x))

        # 融合temporal transformer 和 temporal convolution layer
        g = torch.sigmoid(self.fs(U_T) + self.fg(X_T))
        out = g * self.fs(U_T) + (1 - g) * self.fg(X_T)  # (8)  左边shape:[B, N, T, C]
        # out = self.fg(X_T)

        return out  # 左边shape:[B, N, T, C]


class STTransformerBlock(nn.Module):
    def __init__(self, last_embed_size, embed_size, heads, adj, gso, time_num, dropout, forward_expansion, enable_bias, kernel_size,
                 dilation_t):
        super(STTransformerBlock, self).__init__()
        self.TTransformer_1 = TTransformer(last_embed_size, embed_size[0], heads, time_num, dropout, forward_expansion, kernel_size,
                                           dilation_t)
        self.STransformer = STransformer(embed_size[0], embed_size[1], heads, adj, gso, dropout, forward_expansion, enable_bias)
        self.TTransformer_2 = TTransformer(embed_size[1], embed_size[2], heads, time_num, dropout, forward_expansion, kernel_size,
                                           dilation_t)
        self.norm1 = nn.LayerNorm(embed_size[0])
        self.norm2 = nn.LayerNorm(embed_size[1])
        self.norm3 = nn.LayerNorm(embed_size[2])
        self.dropout = nn.Dropout(dropout)
        self.align1 = Align(last_embed_size, embed_size[0])
        self.align2 = Align(embed_size[0], embed_size[1])
        self.align3 = Align(embed_size[1], embed_size[2])

    def forward(self, value, key, query):
        # Add skip connection,run through normalization and finally dropout
        # x1 = self.norm1(self.STransformer(value, key, query) + query)
        # x2 = self.dropout(self.norm2(self.TTransformer(x1, x1, x1) + x1))
        x1 = self.norm1(self.TTransformer_1(value, key, query) + self.align1(query.permute(0, 3, 2, 1)).permute(0, 3, 2, 1))
        x2 = self.norm2(self.STransformer(x1, x1, x1) + self.align2(x1.permute(0, 3, 2, 1)).permute(0, 3, 2, 1))
        x3 = self.dropout(self.norm3(self.TTransformer_2(x2, x2, x2) + self.align3(x2.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)))
        return x3


class Encoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            adj,
            gso,
            time_num,
            device,
            forward_expansion,
            dropout,
            enable_bias,
            kernel_size,
            dilation_t
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                STTransformerBlock(
                    embed_size[l][-1],
                    embed_size[l+1],
                    heads,
                    adj,
                    gso,
                    time_num,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    enable_bias=enable_bias,
                    kernel_size=kernel_size,
                    dilation_t=dilation_t
                )
                for l in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(x)
        # In the Encoder the query, key, value are all the same.
        i = 0
        for layer in self.layers:
            i += 1
            out = layer(out, out, out)
            # print(f'第{i}个layer out:{out[0:4, :, 0, 0]}')
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            adj,
            gso,
            embed_size,
            num_layers=3,
            heads=2,
            time_num=366,
            forward_expansion=4,
            dropout=0,
            device="cuda",
            enable_bias=False,
            kernel_size=3,
            dilation_t=1

    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            adj,
            gso,
            time_num,
            device,
            forward_expansion,
            dropout,
            enable_bias,
            kernel_size,
            dilation_t
        )
        self.device = device

    def forward(self, src):
        enc_src = self.encoder(src)
        return enc_src


class STTransformer(nn.Module):
    def __init__(
            self,
            adj,
            gso,
            embed_size,
            in_channels=1,
            time_num=366,
            num_layers=3,
            T_dim=12,
            output_T_dim=1,
            heads=2,
            enable_bias=False,
            kernel_size=3,
            dilation_t=1,
            batch_size=16
    ):
        super(STTransformer, self).__init__()
        # 第一次卷积扩充通道数
        # self.conv1 = nn.Conv2d(in_channels, embed_size[0][0], 1)
        self.token_conv = embed.TokenEmbedding(in_channels, embed_size[0][0])
        self.temporal_embeding = embed.TemporalEmbedding(T_dim, embed_size[0][0], batch_size, n_vertex=101)
        self.Transformer = Transformer(
            adj,
            gso,
            embed_size,
            num_layers,
            heads,
            time_num,
            enable_bias=enable_bias,
            kernel_size=kernel_size,
            dilation_t=dilation_t
        )

        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size[-1][0], 1, 1)
        self.relu = nn.ReLU()
        # cuda
        self.device = torch.device('cuda')

    def forward(self, x):
        # input x shape[ C, N, T] 改进后的input x shape [B, C, T, N]
        # C:通道数量。  N:传感器数量。  T:时间数量

        # 时间关系嵌入：
        # temporal = self.temporal_embeding(x_mask).to(self.device)
        # print(f'temporal:{temporal.shape}')
        # x = x.unsqueeze(0)
        input_Transformer = self.token_conv(x)
        # 换成[B, N, T, C]
        input_Transformer = torch.permute(input_Transformer, (0, 3, 2, 1))
        # input_Transformer = input_Transformer.squeeze(0)
        # input_Transformer = input_Transformer.permute(1, 2, 0)

        # input_Transformer shape[N, T, C] 改进后：[B, N, T, C]
        output_Transformer = self.Transformer(input_Transformer)
        # output_Transformer变成[B, T, N, C] 对应下面这个
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)

        out = self.relu(self.conv2(output_Transformer))  # 等号左边 out shape: [B, output_T_dim, N, C]
        out = out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, C, N, output_T_dim]
        out = self.conv3(out)  # 等号左边 out shape: [B, 1, N, output_T_dim]
        # out = out.squeeze(0).squeeze(0)
        out = out.squeeze(1)  # 等式左边 out shape: [B, N, T]
        out = out.permute(0, 2, 1)  # 等式左边out shape: [B, T, N]

        return out
        # return out shape: [B, output_dim, N]
