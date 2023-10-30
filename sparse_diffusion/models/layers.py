import math

import torch
import torch.nn as nn

import torch_geometric.nn.pool as pool


class SparseXtoy(nn.Module):
    def __init__(self, dx, dy):
        """Map node features to global features"""
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, batch):
        """X: N, dx."""
        batch = batch.long()
        m = pool.global_mean_pool(X, batch)
        mi = -pool.global_max_pool(-X, batch)
        ma = pool.global_max_pool(X, batch)
        std = (X - m[batch]) * (X - m[batch])
        std = pool.global_mean_pool(std, batch)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class SparseEtoy(nn.Module):
    def __init__(self, d, dy):
        """Map edge features to global features."""
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, edge_index, edge_attr, batch, top_triu=False):
        """E: M, de
        Features relative to the diagonal of E could potentially be added.
        """
        batch = batch.long()
        if not top_triu:
            batchE = batch[edge_index[0]]
            m = pool.global_mean_pool(edge_attr, batchE)
            mi = -pool.global_max_pool(-edge_attr, batchE)
            ma = pool.global_max_pool(edge_attr, batchE)
            std = (edge_attr - m[batchE]) * (edge_attr - m[batchE])
            std = pool.global_mean_pool(std, batchE)
            z = torch.hstack((m, mi, ma, std))
        else:
            dy = edge_attr.shape[-1]
            batchE1 = batch[edge_index[0]]
            batchE2 = batch[edge_index[1]]
            batchE = torch.hstack([batchE1, batchE2])
            edge_attr_rep = edge_attr.repeat((2, 1))
            m = pool.global_mean_pool(edge_attr_rep, batchE)
            mi = -pool.global_max_pool(-edge_attr_rep, batchE)
            ma = pool.global_max_pool(edge_attr_rep, batchE)
            std = (edge_attr_rep - m[batchE]) * (edge_attr_rep - m[batchE])
            std = pool.global_mean_pool(std, batchE)

            len_m = len(m)
            z = torch.zeros((batch.max() + 1, 4 * dy)).to(edge_index.device)
            z[:len_m, :dy] = m
            z[:len_m, dy : 2 * dy] = mi
            z[:len_m, 2 * dy : 3 * dy] = ma
            z[:len_m, 3 * dy :] = std

        out = self.lin(z)
        return out


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.type_as(x)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
