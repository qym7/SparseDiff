import os
import numpy as np
import wandb
from omegaconf import OmegaConf

import torch_geometric.utils
from torch_geometric.utils import (
    to_dense_adj,
    to_dense_batch,
    dense_to_sparse,
    coalesce,
    remove_self_loops,
)
import torch_geometric.nn.pool as pool
import torch
import torch.nn.functional as F



def setup_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    name = cfg.dataset.name
    if name == "qm9" and cfg.dataset.remove_h == False:
        name = "qm9_h"
    kwargs = {
        "name": cfg.general.name,
        "project": f"sparse_{name}",
        "config": config_dict,
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": True,
        "mode": cfg.general.wandb,
    }
    wandb.init(**kwargs)
    wandb.save("*.txt")
    return cfg


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs("graphs")
        os.makedirs("chains")
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs("graphs/" + args.general.name)
        os.makedirs("chains/" + args.general.name)
    except OSError:
        pass


def to_dense(x, edge_index, edge_attr, batch, charge):
    batch = batch.to(torch.int64)
    X, node_mask = to_dense_node(x=x, batch=batch)
    # node_mask = node_mask.float()
    max_num_nodes = X.size(1)
    E = to_dense_edge(edge_index, edge_attr, batch, max_num_nodes)

    if charge.numel() > 0:
        charge, _ = to_dense_node(x=charge, batch=batch)

    return PlaceHolder(X=X, E=E, y=None, charge=charge), node_mask


def to_dense_node(x, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)

    return X, node_mask


def to_dense_edge(edge_index, edge_attr, batch, max_num_nodes):
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
        edge_index, edge_attr
    )
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)
    return E


def encode_no_edge(E):
    assert len(E.shape) == 4, f"E should be 4D tensor but is {E.shape}"
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0
    return E


def to_sparse(X, E, y, node_mask, charge=None):
    """
    This function will only used for development, thus is not efficient
    """
    bs, n, n, de = E.shape
    device = X.device
    node = []
    charge_list = []
    edge_index = []
    edge_attr = []
    batch = []
    n_nodes = [0]
    for i in range(bs):
        Xi = X[i]
        Ei = E[i]
        mask_i = node_mask[i]
        n_node = sum(mask_i)

        node.append(Xi[:n_node])
        if charge.numel() > 0:
            charge_list.append(charge[i][:n_node])

        batch.append(torch.ones(sum(mask_i)) * i)
        Ei = Ei[:n_node, :n_node]
        Ei = torch.argmax(Ei, -1)
        # Ei = torch.triu(Ei, diagonal=0)
        edge_index_i, edge_attr_i = dense_to_sparse(Ei)
        edge_index.append(edge_index_i + n_nodes[-1])
        edge_attr.append(edge_attr_i)
        n_nodes.append(n_nodes[-1] + int(n_node))

    node = torch.vstack(node).float()
    edge_index = torch.hstack(edge_index).to(device)
    edge_attr = torch.hstack(edge_attr)
    edge_attr = F.one_hot(edge_attr, num_classes=de).to(device).float()
    batch = torch.hstack(batch).long().to(device)

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    ptr = torch.unique(batch, sorted=True, return_counts=True)[1]
    ptr = torch.hstack([torch.tensor([0]).to(device), ptr.cumsum(-1)]).long()

    if charge.numel() > 0:
        charge_list = torch.vstack(charge_list).float()
    else:
        charge_list = node.new_zeros((*node.shape[:-1], 0))

    sparse_noisy_data = {
        "node_t": node,
        "edge_index_t": edge_index,
        "edge_attr_t": edge_attr,
        "batch": batch,
        "y_t": y,
        "ptr": ptr,
        "charge_t": charge_list,
    }

    return sparse_noisy_data


class PlaceHolder:
    def __init__(self, X, E, y, charge=None, t_int=None, t=None, node_mask=None):
        self.X = X
        self.charge = charge
        self.E = E
        self.y = y
        self.t_int = t_int
        self.t = t
        self.node_mask = node_mask

    def device_as(self, x: torch.Tensor):
        self.X = self.X.to(x.device) if self.X is not None else None
        # self.charge = self.charge.to(x.device) if self.charge.numel() > 0 else None
        self.E = self.E.to(x.device) if self.E is not None else None
        self.y = self.y.to(x.device) if self.y is not None else None
        return self

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask=None, collapse=False):
        if node_mask is None:
            assert self.node_mask is not None
            node_mask = self.node_mask
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        diag_mask = (
            ~torch.eye(n, dtype=torch.bool, device=node_mask.device)
            .unsqueeze(0)
            .expand(bs, -1, -1)
            .unsqueeze(-1)
        )  # bs, n, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)
            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            if self.X is not None:
                self.X = self.X * x_mask
            if self.charge.numel() > 0:
                self.charge = self.charge * x_mask
            if self.E is not None:
                self.E = self.E * e_mask1 * e_mask2 * diag_mask
        try:
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        except:
            import pdb
            pdb.set_trace()
        return self

    def collapse(self, collapse_charge=None):
        copy = self.copy()
        copy.X = torch.argmax(self.X, dim=-1)
        copy.E = torch.argmax(self.E, dim=-1)
        x_mask = self.node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        copy.X[self.node_mask == 0] = -1
        copy.charge[self.node_mask == 0] = 1000
        copy.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        return copy

    def __repr__(self):
        return (
            f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- "
            + f"charge: {self.charge.shape if type(self.charge) == torch.Tensor else self.charge} -- "
            + f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- "
            + f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}"
        )

    def copy(self):
        return PlaceHolder(
            X=self.X,
            charge=self.charge,
            E=self.E,
            y=self.y,
            t_int=self.t_int,
            t=self.t,
            node_mask=self.node_mask,
        )


class SparsePlaceHolder:
    def __init__(
        self, node, edge_index, edge_attr, y, ptr=None, batch=None, charge=None
    ):
        self.node = node  # (N, dx)
        self.edge_index = edge_index  # (2, M)
        self.edge_attr = edge_attr  # (M, de)
        self.y = y  # (n, dy)
        self.batch = batch  # (n)
        self.ptr = ptr
        self.charge = charge

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.node = self.node.type_as(x)
        self.edge_index = self.edge_index.type_as(x)
        self.edge_attr = self.edge_attr.type_as(x)
        self.y = self.y.type_as(x)

        self.ptr = self.ptr if self.ptr is None else self.ptr.type_as(x)
        self.batch = self.batch if self.batch is None else self.batch.type_as(x)
        self.charge = self.charge if self.charge is None else self.charge.type_as(x)

        return self

    def to_device(self, device: str):
        """Changes the device and device of X, E, y."""
        self.node = self.node.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.y = self.y.to(device)

        self.ptr = self.ptr if self.ptr is None else self.ptr.to(device)
        self.batch = self.batch if self.batch is None else self.batch.to(device)
        self.charge = self.charge if self.charge is None else self.charge.to(device)

        return self

    def coalesce(self):
        self.edge_index, self.edge_attr = coalesce(
            self.edge_index.long(), self.edge_attr
        )
        return self

    def symmetry(self):
        """ecover directed graph to undirected graph"""
        self.edge_index, self.edge_attr = to_undirected(self.edge_index, self.edge_attr)
        return self

    def collapse(self, collapse_charge=None):
        self.node = torch.argmax(self.node, dim=-1)
        self.edge_attr = torch.argmax(self.edge_attr, dim=-1)


class SparseChainPlaceHolder:
    def __init__(self, keep_chain):
        # node_list/edge_index_list/edge_attr_list is a list of length (keep_chain)
        self.node_list = []
        self.edge_index_list = []
        self.edge_attr_list = []
        self.batch = None
        self.ptr = None
        self.keep_chain = keep_chain

    def append(self, data):
        node = data.node
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        atom_pos_i = data.batch < self.keep_chain
        bond_pos_i = data.batch[edge_index[0]] < self.keep_chain

        self.node_list.append(torch.argmax(node, -1)[atom_pos_i])
        self.edge_attr_list.append(torch.argmax(edge_attr, -1)[bond_pos_i])
        self.edge_index_list.append(edge_index[:, bond_pos_i])
        self.batch = data.batch[atom_pos_i]
        self.ptr = data.ptr[: self.keep_chain + 1]


def delete_repeated_twice_edges(edge_index, edge_attr):    
    min_edge_index, min_edge_attr = coalesce(
            edge_index, edge_attr, reduce="min"
        )
    max_edge_index, max_edge_attr = coalesce(
            edge_index, edge_attr, reduce="max"
        )
    rand_pos = torch.randint(0, 2, (len(edge_attr),))
    max_edge_attr[rand_pos] = min_edge_attr[rand_pos]

    return max_edge_index, max_edge_attr


def to_undirected(edge_index, edge_attr=None):
    row, col = edge_index[0], edge_index[1]
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.vstack([row, col])

    if edge_attr is not None:
        if len(edge_attr.shape) > 1:
            edge_attr = torch.vstack([edge_attr, edge_attr])
        else:
            edge_attr = torch.hstack([edge_attr, edge_attr])
        return edge_index, edge_attr
    else:
        return edge_index


def undirected_to_directed(edge_index, edge_attr=None):
    top_index = edge_index[0] < edge_index[1]
    if edge_attr is not None:
        return edge_index[:, top_index], edge_attr[top_index]
    else:
        return edge_index[:, top_index]


def ptr_to_node_mask(ptr, batch, n_node):
    device = ptr.device
    node_mask = torch.ones((int(batch.max() + 1), int(n_node))).cumsum(-1)  # bs, n_node
    node_mask = node_mask.to(device) <= (ptr.diff()).unsqueeze(-1).repeat(1, n_node)

    return node_mask


def concat_sparse_graphs(graphs):
    """Concatenate several sparse placeholders into a single one."""
    graph = graphs[0]
    graph.node = torch.hstack([g.node for g in graphs])
    graph.edge_attr = torch.hstack([g.edge_attr for g in graphs])
    graph.y = torch.vstack([g.y for g in graphs])
    num_node_ptr = [0] + [len(g.batch) for g in graphs]
    num_node_ptr = torch.tensor(num_node_ptr).cumsum(-1)
    graph.edge_index = torch.hstack(
        [g.edge_index + num_node_ptr[i] for i, g in enumerate(graphs)]
    )
    num_graph_ptr = [0] + [int(g.batch.max()) + 1 for g in graphs]
    num_graph_ptr = torch.tensor(num_graph_ptr).cumsum(-1)
    graph.batch = torch.hstack(
        [g.batch + num_graph_ptr[i] for i, g in enumerate(graphs)]
    )
    ptr = torch.unique(graph.batch, sorted=True, return_counts=True)[1]
    graph.ptr = torch.hstack(
        [torch.tensor([0]).to(graphs[0].batch.device), ptr.cumsum(-1)]
    ).long()
    if graph.charge.numel() > 0:
        graph.charge = torch.hstack([g.charge for g in graphs])
    else:
        # when the charge size is [N, 0], we can not apply argmax thus can not reduce the dimension to 1 and use hstack
        graph.charge = torch.vstack([g.charge for g in graphs])

    return graph

def split_samples(samples, start_idx, end_idx):
    y = samples.y
    node = samples.node
    charge = samples.charge
    edge_attr = samples.edge_attr
    edge_index = samples.edge_index
    ptr = samples.ptr
    batch = samples.batch
    
    node_mask = torch.logical_and(batch < end_idx, batch >= start_idx)
    node = node[node_mask]
    charge = charge[node_mask]

    edge_batch = batch[edge_index[0]]
    edge_mask = torch.logical_and(edge_batch < end_idx, edge_batch >= start_idx)
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask]

    batch = (batch - start_idx).long()
    ptr = ptr[start_idx:end_idx+1] - ptr[start_idx]
    y = y[start_idx:end_idx+1]

    return SparsePlaceHolder(
        node=node, edge_index=edge_index, edge_attr=edge_attr,
        y=y, ptr=ptr, batch=batch, charge=charge
        )