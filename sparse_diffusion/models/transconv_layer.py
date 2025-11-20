import os, sys
import pathlib
import os.path as osp
RootPath = pathlib.Path(osp.realpath(__file__)).parents[1]
sys.path.append(f'{RootPath}')
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch.nn import ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils import softmax
from models.layers import SparseXtoy, SparseEtoy

compress_alpha = True
compress_edge_attr = True


class TransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    multi-head dot product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax} \left(
        \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
        {\sqrt{d}} \right)
    """
    _alpha: OptTensor

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        last_layer: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / heads)
        self.heads = heads
        self.concat = concat
        self.last_layer = last_layer
        self.dropout = dropout

        self.lin_key = Linear(dx, heads * self.df)
        self.lin_query = Linear(dx, heads * self.df)
        self.lin_value = Linear(dx, heads * self.df)

        if concat:
            self.lin_skip = Linear(dx, heads * self.df, bias=bias)
        else:
            self.lin_skip = Linear(dx, self.df, bias=bias)

        # FiLM E to X: de = dx here as defined in lin_edge

        self.e_add = Linear(de, heads)
        self.e_mul = Linear(de, heads)

        # FiLM y to E
        self.y_e_mul = Linear(dy, de)
        self.y_e_add = Linear(dy, de)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        if self.last_layer:
            self.y_y = Linear(dy, dy)
            self.x_y = SparseXtoy(dx, dy)
            self.e_y = SparseEtoy(de, dy)
            self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        # self.e_front = Linear(de, de)     # use this when we put the y information in front

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        y: Tensor = None,
        batch: Tensor = None,
    ):
        r"""Runs the forward pass of the module.
        Add ReLU after integrate information

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.df

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out, new_edge_attr = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            index=edge_index[1],
            size=None,
        )

        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.df)
        else:
            out = out.mean(dim=1)

        x_r = self.lin_skip(x)
        out = out + x_r

        # Incorporate y to edge_attr
        batch = batch.long()
        batchE = batch[edge_index[0]]
        # Output _edge_attr
        new_edge_attr = new_edge_attr.flatten(start_dim=1)  # M, h * df (dx)
        new_edge_attr = self.e_out(new_edge_attr)  # M, de
        ye1 = self.y_e_add(y)  # M, de
        ye2 = self.y_e_mul(y)  # M, de
        new_edge_attr = ye1[batchE] + (ye2[batchE] + 1) * new_edge_attr  # M, de

        # Incorporate y to X
        # new_x = self.x_out(x)     # use this when we want new_x and new_edge_attr symmetric
        yx1 = self.y_x_add(y)
        yx2 = self.y_x_mul(y)
        new_x = yx1[batch] + (yx2[batch] + 1) * out
        # Output X
        new_x = self.x_out(new_x)

        if self.last_layer:
            new_y = self.predict_graph(y, x, edge_index, edge_attr, batch)
        else:
            new_y = y

        return (new_x, new_edge_attr, new_y)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages."""
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)

        msg_kwargs = self.inspector.distribute("message", coll_dict)
        out, edge_attr = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)

        out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.inspector.distribute("update", coll_dict)
        out = self.update(out, **update_kwargs)

        return (out, edge_attr)

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        Y = (query_i * key_j) / math.sqrt(self.df)  # M, H, C

        edge_attr_mul = self.e_mul(edge_attr)  # M, H
        edge_attr_add = self.e_add(edge_attr)  # M, H
        Y = Y * (edge_attr_mul.unsqueeze(-1) + 1) + edge_attr_add.unsqueeze(-1)

        alpha = softmax(Y.sum(-1), index, ptr, size_i)  # M, H

        # dropout or not
        self._alpha = alpha  # alpha = attn
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # M, H

        out = value_j  # M, H, C
        out = out * alpha.view(-1, self.heads, 1)  # M, H, C, out = weighted_V

        return (out, Y)

    def predict_graph(self, y, x, edge_index, edge_attr, batch):
        y = self.y_y(y)
        e_y = self.e_y(edge_index, edge_attr, batch, top_triu=True)
        x_y = self.x_y(x, batch)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)  # bs, dy

        return new_y

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.dx}, " f"{self.df}, heads={self.heads})"
        )
