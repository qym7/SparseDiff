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


def to_sparse_all_edges(X, E, y, node_mask):
    """
    This function will only used for development, thus is not efficient
    """
    bs, n, n, de = E.shape
    device = X.device
    node = []
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
        batch.append(torch.ones(sum(mask_i)) * i)
        Ei = Ei[:n_node, :n_node]
        Ei = torch.argmax(Ei, -1)
        Ei = Ei + 1  # n, n
        # Ei = torch.triu(Ei, diagonal=0)
        edge_index_i, edge_attr_i = dense_to_sparse(Ei)
        edge_index.append(edge_index_i + n_nodes[-1])
        edge_attr.append(edge_attr_i - 1)
        n_nodes.append(n_nodes[-1] + int(n_node))

    node = torch.vstack(node).float()
    edge_index = torch.hstack(edge_index).to(device)
    edge_attr = torch.hstack(edge_attr)
    edge_attr = F.one_hot(edge_attr, num_classes=de).to(device).float()
    batch = torch.hstack(batch).to(device)

    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    ptr = torch.unique(batch, sorted=True, return_counts=True)[1]
    ptr = torch.hstack([torch.tensor([0]).to(self.device), ptr.cumsum(-1)]).long()

    sparse_noisy_data = {
        "node_t": node,
        "edge_index_t": edge_index,
        "edge_attr_t": edge_attr,
        "batch": batch,
        "y_t": y,
        "ptr": ptr,
    }

    return sparse_noisy_data


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
        # copy.charge = collapse_charge.to(self.charge.device)[torch.argmax(self.charge, dim=-1)]
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
        # copy.charge = collapse_charge.to(self.charge.device)[torch.argmax(self.charge, dim=-1)]
        self.edge_attr = torch.argmax(self.edge_attr, dim=-1)


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


def mask_node(node, node_mask):
    x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
    return node * x_mask


def densify_noisy_data(sparse_noisy_data):
    noisy_data = dict()
    for i in ["t_float", "y_t", "t_int", "y_t", "alpha_t_bar", "alpha_s_bar", "beta_t"]:
        if i in sparse_noisy_data.keys():
            noisy_data[i] = sparse_noisy_data[i]

    dense_noisy_data, node_mask = to_dense(
        sparse_noisy_data["node_t"],
        sparse_noisy_data["edge_index_t"],
        sparse_noisy_data["edge_attr_t"],
        sparse_noisy_data["batch"],
        charge=sparse_noisy_data["charge_t"],
    )

    noisy_data["X_t"] = dense_noisy_data.X
    noisy_data["E_t"] = dense_noisy_data.E
    noisy_data["charge_t"] = dense_noisy_data.charge
    noisy_data["node_mask"] = node_mask

    return noisy_data


def sample_rand_edges(
    ptr,
    batch,
    edge_index,
    edge_attr,
    n_rand_edges=None,
    rand_nodes=None,
    n_rand_edges_batch=None,
    rand_edges_dist_batch=None,
):
    """
    :param ptr:
    :param batch: (n) LongTensor containing the graph index of each node
    :param edge_index: (2, E) LongTensor
    :param edge_attr: (E) LongTensor
    :param n_rand_edges: (int) number of random edges to sample
    :param rand_nodes:
    :param rand_edge_index: (2, E) LongTensor -- nothing to sample in this case
    :return:
    """
    n_node = len(batch)
    device = batch.device
    de = edge_attr.shape[-1]
    bs = batch.max() + 1
    true_rand_edge_attr = None

    if n_rand_edges is not None:
        # weighted sampling for each graph, so that the number of edges sampled in graph i ~ n_node i
        weight = ptr.diff()[batch].detach().cpu().numpy() - 1
        weight = weight / weight.sum()
        rand_edge_index1 = torch.tensor(
            np.random.choice(  # Sample the first node of the edge
                n_node, size=(n_rand_edges,), p=weight
            ),
            dtype=torch.long,
        ).to(device)
        rand_edge_batch = batch[rand_edge_index1]  # Graph index of each edge
        # create random index 2 for the corresponding batch
        rand_edge_index2 = torch.rand(n_rand_edges).to(device)
        n_rand_node = ptr.diff()[rand_edge_batch]
        # random sample the edge index 2 within the graph defined by edge index 1
        rand_edge_index2 = torch.floor(n_rand_node * rand_edge_index2).long()
        rand_edge_index2 = rand_edge_index2 + ptr[rand_edge_batch]
        # TODO: check if this works for larger dataset, if not, need to change the sampling mechanism
        # Everything work if the graphs don't have the same size, but for large graphs and large batch size
        # this sampling might not be very accurate
        rand_edge_index = torch.vstack([rand_edge_index1, rand_edge_index2])

    elif n_rand_edges_batch is not None:
        # build node mask
        n_node = ptr.diff().max()
        node_mask = ptr_to_node_mask(ptr, batch, n_node)
        # sample edge index 1
        max_n_nodes_batch = n_rand_edges_batch.max()
        rand_edge_index1 = torch.multinomial(
            node_mask.float(), max_n_nodes_batch, replacement=True
        )
        rand_edge_mask = torch.ones_like(rand_edge_index1).cumsum(-1)
        rand_edge_mask[
            rand_edge_mask
            > n_rand_edges_batch.unsqueeze(-1).repeat(1, max_n_nodes_batch)
        ] = 0
        rand_edge_mask[rand_edge_mask > 0] = 1
        rand_edge_batch = torch.where(rand_edge_mask)[0]
        rand_edge_index1 = (
            rand_edge_index1.flatten()[rand_edge_mask.flatten().bool()]
            + ptr[rand_edge_batch]
        )
        # sample edge index 2
        n_rand_node = ptr.diff()[rand_edge_batch]
        rand_edge_index2 = torch.rand(n_rand_edges_batch.sum()).to(device)
        rand_edge_index2 = torch.floor(n_rand_node * rand_edge_index2).long()
        rand_edge_index2 = rand_edge_index2 + ptr[rand_edge_batch]
        rand_edge_index = torch.vstack([rand_edge_index1, rand_edge_index2])
        # sample edge attr
        if rand_edges_dist_batch is not None:
            true_rand_edge_attr = (
                torch.multinomial(
                    rand_edges_dist_batch, max_n_nodes_batch, replacement=True
                )
                + 1
            )
            true_rand_edge_attr = true_rand_edge_attr.flatten()[
                rand_edge_mask.flatten().bool()
            ]

    elif rand_nodes is not None:
        n_nodes = ptr.diff()
        b_nodes = batch[rand_nodes].long()
        n_edges = n_nodes[b_nodes]
        rand_edge_index1 = [
            torch.ones(n_edges[i]).to(device) * n for i, n in enumerate(rand_nodes)
        ]
        rand_edge_index2 = [
            torch.arange(n_edges[i]).to(device) + ptr[b] for i, b in enumerate(b_nodes)
        ]
        rand_edge_index1 = torch.hstack(rand_edge_index1)
        rand_edge_index2 = torch.hstack(rand_edge_index2)
        rand_edge_index = torch.vstack([rand_edge_index1, rand_edge_index2])
    else:
        raise ("n_rand_edges and rand_nodes can not be both None.")

    # delete self-loop
    # TODO: This is not great, we lose control on the exact number of edges that are sampled
    #       YM: This problem can be partially solved by save the number of rand edges, and using this number for sampling
    # TODO: We should try to think of a way to sample edges without self-loops

    rand_edge_pos, max_comp_edge_index, max_comp_edge_attr = coalesce_all_graphs(
        rand_edge_index, edge_index, edge_attr, true_rand_edge_attr
    )

    # it is wierd because rand_edge_pos sum is not equal to rand_edge_index after remove self-loop / to_undirected / coalesce

    return rand_edge_pos, max_comp_edge_index, max_comp_edge_attr


def coalesce_all_graphs(
    query_edge_index, clean_edge_index, clean_edge_attr, true_rand_edge_attr=None
):
    # get dimension information
    de = clean_edge_attr.shape[-1]
    device = query_edge_index.device

    if true_rand_edge_attr is None:
        rand_edge_attr = torch.zeros((query_edge_index.shape[1], de)).to(device)
        rand_edge_attr[:, 0] = 1
        # delete self-loop
        query_edge_index, rand_edge_attr = remove_self_loops(
            query_edge_index, rand_edge_attr
        )
        # make random edges symmetrical
        query_edge_index, rand_edge_attr = to_undirected(
            query_edge_index, rand_edge_attr
        )
    else:
        # delete self-loop
        query_edge_index, true_rand_edge_attr = remove_self_loops(
            query_edge_index, true_rand_edge_attr
        )
        # make random edges symmetrical
        query_edge_index, true_rand_edge_attr = to_undirected(
            query_edge_index, true_rand_edge_attr
        )
        # coalesce: reduce with random label
        sort_attr = torch.arange(query_edge_index.shape[1]).to(device)
        query_edge_index, true_sort_edge_attr = coalesce(
            query_edge_index.long(), sort_attr, reduce="min"
        )
        true_rand_edge_attr = true_rand_edge_attr[true_sort_edge_attr.long()]
        # define rand edge attr
        rand_edge_attr = torch.zeros((query_edge_index.shape[1], de)).to(device)
        rand_edge_attr[:, 0] = 1

    # find the attributes for random edges
    comp_edge_index = torch.hstack([clean_edge_index, query_edge_index])
    comp_edge_attr = torch.vstack([clean_edge_attr, rand_edge_attr])

    # get the computational graph: positive edges + random edges
    max_comp_edge_index, max_comp_edge_attr_label = coalesce(
        comp_edge_index.long(), torch.argmax(comp_edge_attr, -1), reduce="max"
    )

    # get the prediction graph: random edges
    _, min_comp_edge_attr = coalesce(
        comp_edge_index.long(), torch.argmax(comp_edge_attr, -1), reduce="min"
    )
    rand_edge_pos = min_comp_edge_attr == 0

    # pass attr to one-hot
    max_comp_edge_attr = F.one_hot(max_comp_edge_attr_label, num_classes=de).to(device)

    if true_rand_edge_attr is not None:
        comp_true_edge_attr = torch.hstack(
            [torch.argmax(clean_edge_attr, -1), true_rand_edge_attr]
        )
        _, true_comp_edge_attr = coalesce(
            comp_edge_index.long(), comp_true_edge_attr, reduce="max"
        )
        true_comp_edge_attr = F.one_hot(true_comp_edge_attr, num_classes=de).to(device)
        # position for random edges which is originally negative
        rand_edge_pos = torch.logical_and(
            max_comp_edge_attr_label == min_comp_edge_attr, rand_edge_pos
        )
        # attribute the new label to those edges
        max_comp_edge_attr[rand_edge_pos] = true_comp_edge_attr[rand_edge_pos]

    return rand_edge_pos, max_comp_edge_index, max_comp_edge_attr


def ptr_to_node_mask(ptr, batch, n_node):
    device = ptr.device
    node_mask = torch.ones((int(batch.max() + 1), int(n_node))).cumsum(-1)  # bs, n_node
    node_mask = node_mask.to(device) <= (ptr.diff()).unsqueeze(-1).repeat(1, n_node)

    return node_mask



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

def split_samples(samples, num_split):
    y = samples.y
    node = samples.node
    charge = samples.charge
    edge_attr = samples.edge_attr
    edge_index = samples.edge_index
    ptr = samples.ptr
    batch = samples.batch
    edge_batch = batch[edge_index[0]]
    
    num_sample = batch.max().item() + 1
    samples_lst = []
    
    for i in range(num_split):
        start_idx = int(num_sample * i / num_split)
        end_idx = int(num_sample * (i+1) / num_split)
    
        node_mask = torch.logical_and(batch < end_idx, batch >= start_idx)
        cur_node = node[node_mask]
        cur_charge = charge[node_mask]

        edge_mask = torch.logical_and(edge_batch < end_idx, edge_batch >= start_idx)
        cur_edge_index = edge_index[:, edge_mask]
        cur_edge_index = (cur_edge_index - (batch < start_idx).sum()).long()
        cur_edge_attr = edge_attr[edge_mask]

        cur_batch = batch[node_mask] - start_idx
        cur_ptr = ptr[start_idx:end_idx] - ptr[start_idx]
        cur_y = y[start_idx:end_idx]

        cur_samples =  SparsePlaceHolder(
            node=cur_node, edge_index=cur_edge_index, edge_attr=cur_edge_attr,
            y=cur_y, ptr=cur_ptr, batch=cur_batch, charge=cur_charge
            )

        samples_lst.append(cur_samples)

    return samples_lst