import torch.nn as nn
import torch
from math import sqrt

class Asymmetric_Gaussian_Mse(nn.Module):
    def __init__(self, a=0.1):
        super(Asymmetric_Gaussian_Mse, self).__init__()
        self.a = a

    def gaussian_function(self, output, target):
        sigma = sqrt(2) / 2
        temp = -((output - target)**2) / 2*(sigma**2)
        value = 1 - torch.exp(temp)
        return value

    def forward(self, outputs, targets):
        return torch.mean(torch.where(torch.greater(outputs, targets), (2*self.a) * self.gaussian_function(outputs,
                                                                                                           targets),
                                      (2-2*self.a)* self.gaussian_function(outputs, targets)))

# class Asymmetric_Mse(nn.Module):
#     def __init__(self, a=0.45):
#         super(Asymmetric_Mse, self).__init__()
#         self.a = a
#
#     def forward(self, outputs, targets):
#         # return torch.mean(torch.where(torch.greater(outputs, targets), 2*self.a*(outputs - targets)**2,
#         #                               2*(self.a+(1-(2*self.a)))*(outputs - targets)**2))
#         return torch.mean(torch.where(torch.greater(outputs, targets), (outputs - targets) ** 2,
#                                       2 * (self.a + 0.5) * (outputs - targets) ** 2))


