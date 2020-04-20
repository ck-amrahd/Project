import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules import Module


class GraphConvolution(Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.bias = Parameter(torch.FloatTensor(self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inp, adj):
        x = torch.mm(inp, self.weight)
        output = torch.mm(adj, x) + self.bias
        return output
