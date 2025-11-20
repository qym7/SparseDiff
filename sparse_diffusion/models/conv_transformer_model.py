import os, sys
import pathlib
import os.path as osp
RootPath = pathlib.Path(osp.realpath(__file__)).parents[1]
sys.path.append(f'{RootPath}')
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.nn import functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm


import torch_geometric.nn.pool as pool
from torch_geometric.utils import softmax, sort_edge_index

import utils
from models.transconv_layer import TransformerConv
from models.layers import SparseXtoy, SparseEtoy


class XEyTransformerLayer(nn.Module):
    """Transformer that updates node, edge and global features
    d_x: node features
    d_e: edge features
    dz : global features
    n_head: the number of heads in the multi_head_attention
    dim_feedforward: the dimension of the feedforward network model after self-attention
    dropout: dropout probablility. 0 to disable
    layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        last_layer: bool = True,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        super().__init__()

        self.last_layer = last_layer
        self.self_attn = TransformerConv(
            dx=dx,
            de=de,
            dy=dy,
            heads=n_head,
            concat=True,
            dropout=dropout,
            bias=True,
            last_layer=last_layer,
        )

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)  # TODO: set norm
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)  # TODO: set norm
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)

        if self.last_layer:
            self.lin_y1 = Linear(dy, dim_ffy, **kw)
            self.lin_y2 = Linear(dim_ffy, dy, **kw)
            self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)  # TODO: set norm
            self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)

        self.activation = F.relu

    def forward(
        self, X: Tensor, edge_index: Tensor, edge_attr: Tensor, y: Tensor, batch: Tensor
    ):
        """Pass the input through the encoder layer.
        X: (N, d)
        edge_index: (M, 2)
        edge_attr: (M, d)
        batch: (n)
        y: (n)
        """
        new_x, new_edge_attr, new_y = self.self_attn(X, edge_index, edge_attr, y, batch)

        X = self.normX1(X + new_x)

        edge_attr = self.normE1(edge_attr + new_edge_attr)

        if self.last_layer:
            y = self.norm_y1(y + new_y)
        else:
            y = new_y

        ff_outputX = self.linX2(self.activation(self.linX1(X)))
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.activation(self.linE1(edge_attr)))
        edge_attr = self.normE2(edge_attr + ff_outputE)

        if self.last_layer:
            ff_output_y = self.lin_y2(self.activation(self.lin_y1(y)))
            y = self.norm_y2(y + ff_output_y)

        return X, edge_attr, y


class GraphTransformerConv(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dims: utils.PlaceHolder,
        hidden_dims: dict,
        output_dims: utils.PlaceHolder,
        dropout: 0.1,
        sn_hidden_dim: int,
        output_y: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims.X
        self.out_dim_E = output_dims.E
        self.out_dim_y = output_dims.y
        self.out_dim_charge = output_dims.charge
        self.output_y = output_y
        self.dropout = dropout

        self.lin_in_X = nn.Linear(
            input_dims.X + input_dims.charge + sn_hidden_dim, hidden_dims["dx"]
        )
        self.lin_in_E = nn.Linear(input_dims.E, hidden_dims["de"])
        self.lin_in_y = nn.Linear(input_dims.y, hidden_dims["dy"])

        # last layer is True when we keep the last output layers of y
        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                    last_layer=True if output_y else (i < n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )
        self.out_ln_X = nn.LayerNorm(hidden_dims["dx"])
        self.out_ln_E = nn.LayerNorm(hidden_dims["de"])
        self.lin_out_X = nn.Linear(
            hidden_dims["dx"], output_dims.X + output_dims.charge
        )
        self.lin_out_E = nn.Linear(hidden_dims["de"], output_dims.E)

        if self.output_y:
            self.out_ln_y = nn.LayerNorm(hidden_dims["dy"])
            self.lin_out_y = nn.Linear(hidden_dims["dy"], output_dims.y)

    def forward(self, X, edge_attr, edge_index, y, batch):
        # Save for residual connection
        X0 = X.clone()
        edge_attr0 = edge_attr.clone()
        y0 = y.clone()

        # Input block
        X = self.lin_in_X(X)
        edge_attr = self.lin_in_E(edge_attr)
        y = self.lin_in_y(y)

        # Transformer layers
        for layer in self.tf_layers:
            X, edge_attr, y = layer(X, edge_index, edge_attr, y, batch)

        # Output block
        X = self.lin_out_X(self.out_ln_X(X))
        edge_attr = self.lin_out_E(self.out_ln_E(edge_attr))
        if self.output_y:
            y = self.lin_out_y(self.out_ln_y(y))

        # make results symmetrical
        top_edge_index, top_edge_attr = sort_edge_index(edge_index, edge_attr)
        _, bot_edge_attr = sort_edge_index(edge_index[[1, 0]], edge_attr)

        charges = (
            X[:, self.out_dim_X : self.out_dim_X + self.out_dim_charge]
            + X0[:, self.out_dim_X : self.out_dim_X + self.out_dim_charge]
        )
        X = X[:, : self.out_dim_X] + X0[:, : self.out_dim_X]
        edge_attr = top_edge_attr + bot_edge_attr + edge_attr0[:, : self.out_dim_E]

        if self.output_y:
            y = y + y0[:, : self.out_dim_y]

        return utils.SparsePlaceHolder(
            node=X,
            edge_attr=edge_attr,
            edge_index=top_edge_index,
            y=y,
            batch=batch,
            charge=charges,
        )
