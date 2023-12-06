import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from script.utility import Align


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1,
                 bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                                           dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result


class TemporalConvLayer(nn.Module):

    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * Residual Connection *
    #        |                                |
    #        |    |--->--- CasualConv2d ----- + -------|
    # -------|----|                                   ⊙ ------>
    #             |--->--- CasualConv2d --- Sigmoid ---|
    #

    # param x: tensor, [bs, c_in, ts, n_vertex] shape:[B, C, T, N]

    def __init__(self, kernel_size, c_in, c_out, dilation_t):
        super(TemporalConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.c_in = c_in
        self.c_out = c_out
        self.dilation_t = dilation_t
        self.align = Align(c_in, c_out)
        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(self.kernel_size, 1),
                                        enable_padding=True, dilation=(self.dilation_t, 1))
        # self.causal_conv_7 = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(7, 1), enable_padding=True,
        #                                   dilation=(self.dilation_t, 1))
        self.causal_conv_5 = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(5, 1), enable_padding=
                                          True, dilation=(self.dilation_t, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_in = self.align(x)  # 左边shape:[B, C, T, N]
        # g = torch.sigmoid(self.causal_conv(x) + self.causal_conv_5(x))
        x_causal_conv = self.causal_conv(x) + self.causal_conv_5(x)

        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out:, :, :]
        x = torch.mul((x_p + x_in), self.sigmoid(x_q))

        return x  # 输出形状：[B, C, T, N]
