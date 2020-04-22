import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, adj_tensor, n_input, n_hidden, n_output):
        super(GCN, self).__init__()
        self.adj = adj_tensor
        self.gc1 = GraphConvolution(n_input, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_output)

    def forward(self, inp):
        x = F.relu(self.gc1(inp, self.adj))
        x = self.gc2(x, self.adj)
        return x
