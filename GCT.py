from __future__ import absolute_import
import sys
from collections import OrderedDict
import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchsummary import summary
from ChebConv import ChebConv, _ResChebGC
from sem_graph_conv import SemGraphConv

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    # edges = torch.as_tensor(edges, dtype=torch.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)

    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))


    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)

    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)

    return adj_mx


#gan_edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4],
#                          [0, 5], [5, 6], [6, 7], [7, 8],
#                          [0, 9], [9, 10], [10, 11], [11, 12],
#                          [0, 13], [13, 14], [14, 15], [15, 16],
#                          [0, 17], [17, 18], [18, 19], [19, 20]], dtype=torch.long)

gan_edges = (np.array([[1, 2], [1, 6], [2, 3], [3, 4], [4, 5], [5, 6], [7, 8],
                       [7, 12], [8, 9], [9, 10], [10, 11], [11, 12], [1, 7],
                       [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]]) - 1)



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class GraAttenLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GraAttenLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(Q, K, V, mask=None, dropout=None):
    # Query=Key=Value: [batch_size, 8, max_len, 64]
    d_k = Q.size(-1)
    # Q * K.T = [batch_size, 8, max_len, 64] * [batch_size, 8, 64, max_len]
    # scores: [batch_size, 8, max_len, max_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, V), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        Q, K, V = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True,
                           True]]])


class LAM_Gconv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(LAM_Gconv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X


class GraphNet(nn.Module):

    def __init__(self, in_features=2, out_features=2, n_pts=12):
        super(GraphNet, self).__init__()

        self.A_hat = Parameter(torch.eye(n_pts).float(), requires_grad=True)
        self.gconv1 = LAM_Gconv(in_features, in_features * 2)
        self.gconv2 = LAM_Gconv(in_features * 2, out_features, activation=None)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1


class ChebGAC(nn.Module):

    def __init__(self, adj1, adj2, adj3, in_features=3, out_features=12):
        super(ChebGAC, self).__init__()
        self.hop1 = adj1
        self.hop2 = adj2
        self.hop3 = adj3
        self.gconv1 = ChebConv(in_c=in_features, out_c=out_features, K=2)
        self.gconv2 = ChebConv(in_c=in_features, out_c=out_features, K=2)
        self.gconv3 = ChebConv(in_c=in_features, out_c=out_features, K=2)
        self.fc = nn.Linear(in_features=12*12*3, out_features=12*out_features)
        self.act_fn = nn.ReLU()
    def forward(self, X):
        X_0 = self.gconv1(X, self.hop1)
        X_1 = self.gconv2(X, self.hop2)
        X_2 = self.gconv3(X, self.hop2)
        x = torch.cat([X_0, X_1, X_2], dim=1)
        x = x.view(x.shape[0], 12 * 12 * 3)
        z0 = self.fc(x)
        z0 = z0.view(z0.shape[0], 12, 12)
        z1 = X_0*X_1
        z2 = X_1*X_2
        x = self.act_fn(z0+z1+z2)
        return x.view(x.shape[0], 12, 12)

class _GraphsConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphsConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))
        x = self.relu(x)
        return x

class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphsConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphsConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class GCT(nn.Module):
    def __init__(self, adj, adj1, adj2, hid_dim=96, coords_dim=(3, 3), num_layers=4,
                 n_head=4, dropout=0.1, n_pts=12):
        super(GCT, self).__init__()
        self.n_layers = num_layers
        self.adj = adj


        _gconv_input = ChebConv(in_c=coords_dim[0], out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)
        semg = SemGraphConv(in_features=12, out_features=96, adj=self.adj)

        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adj=self.adj, input_dim=96, output_dim=hid_dim,
                                            hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_input = _gconv_input
        self.semg = semg
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output1 = ChebConv(in_c=dim_model, out_c=3, K=2)
        self.gconv_output2 = ChebConv(in_c=dim_model, out_c=4, K=2)
        self.chebgca = ChebGAC(adj1=adj, adj2=adj1, adj3=adj2)


        self.DLayer1 = torch.nn.Sequential(OrderedDict([
            ('Fc1', torch.nn.Linear(36, 24)),
            ('relu1', torch.nn.LeakyReLU()),
            ('Batch1', torch.nn.BatchNorm1d(24)),
            ('Fc2', torch.nn.Linear(24, 3))]))

        self.DLayer2 = torch.nn.Sequential(OrderedDict([
            ('Fc1', torch.nn.Linear(48, 24)),
            ('relu1', torch.nn.LeakyReLU()),
            ('Batch1', torch.nn.BatchNorm1d(24)),
            ('Fc4', torch.nn.Linear(24, 9))]))

    def forward(self, x, mask):
        x = self.chebgca(x)
        x = self.semg(x)
        res1 = x
        out = x
        for i in range(self.n_layers):
            out = self.gconv_layers[i](out)
            out = self.atten_layers[i](out, mask)
            if i == 1:
                out = self.atten_layers[i](out, mask) + res1
                res2 = out
            if i == 3:
                out = self.atten_layers[i](out, mask) + res1 + res2

        out1 = self.gconv_output1(out, self.adj)
        out2 = self.gconv_output2(out, self.adj)

        out1 = out1.view(out1.shape[0], 36)
        out2 = out2.view(out2.shape[0], 48)
        # out1 = nn.Linear(3, 3, dtype=torch.double)(out1)
        # out2 = nn.Linear(9, 9, dtype=torch.double)(out2)
        out1 = self.DLayer1(out1)
        out2 = self.DLayer2(out2)
        out = torch.cat([out1, out2], dim=1)
        return out



if __name__ == '__main__':
    gan_edges = (np.array([[1, 2], [1, 6], [2, 3], [3, 4], [4, 5], [5, 6], [7, 8],
                           [7, 12], [8, 9], [9, 10], [10, 11], [11, 12], [1, 7],
                           [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]]) - 1)

    gan_edges1 = (np.array([[1, 8], [1, 12], [1, 3], [1, 5], [2, 9], [2, 7], [2, 4],
                            [2, 6], [3, 10], [3, 8], [3, 5], [3, 1], [4, 11],
                            [4, 9], [4, 6], [4, 2], [5, 12], [5, 10], [5, 1], [5, 3], [6, 7], [6, 11], [6, 2],
                            [6, 4]]) - 1)

    gan_edges2 = (np.array([[1, 9], [1, 11], [1, 4],
                            [2, 10], [2, 12], [2, 5],
                            [3, 11], [3, 7], [3, 6],
                            [4, 12], [4, 8], [4, 1],
                            [5, 7], [5, 9], [5, 2],
                            [6, 8], [6, 10], [6, 3]]) - 1)
    adj = adj_mx_from_edges(num_pts=12, edges=gan_edges, sparse=False).cuda()
    adj1 = adj_mx_from_edges(num_pts=12, edges=gan_edges1, sparse=False).cuda()
    adj2 = adj_mx_from_edges(num_pts=12, edges=gan_edges2, sparse=False).cuda()
    net = GCT(adj, adj1, adj2).double()
    src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True,
                               True]]]).cuda()
    device = torch.device("cuda:0")
    net.to(device)

    # generate some example data on the GPU
    x = torch.randn((1000, 12, 3), dtype=torch.double).to(device)
    print(x)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params}")


