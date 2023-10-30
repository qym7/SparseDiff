import torch
from sparse_diffusion import utils

import torch_geometric.nn.pool as pool


class ExtraMolecularFeatures:
    def __init__(self, dataset_infos):
        self.charge = SparseChargeFeature(
            remove_h=dataset_infos.remove_h, valencies=dataset_infos.valencies
        )
        self.valency = SparseValencyFeature()
        self.weight = SparseWeightFeature(
            max_weight=dataset_infos.max_weight, atom_weights=dataset_infos.atom_weights
        )

    def __call__(self, noisy_data):
        charge = self.charge(noisy_data).unsqueeze(-1)  # (nx, 1)
        valency = self.valency(noisy_data).unsqueeze(-1)  # (nx, 1)
        weight = self.weight(noisy_data)  # (bs, 1)

        if "comp_edge_attr_t" not in noisy_data.keys():
            noisy_data["comp_edge_attr_t"] = noisy_data["edge_attr_t"]
        edge_attr_t = noisy_data["comp_edge_attr_t"]
        extra_edge_attr = torch.zeros((*edge_attr_t.shape[:-1], 0)).type_as(edge_attr_t)
        extra_node = torch.cat((charge, valency), dim=-1)
        extra_y = weight

        return utils.SparsePlaceHolder(
            node=extra_node, edge_attr=extra_edge_attr, edge_index=None, y=extra_y
        )


class SparseChargeFeature:
    def __init__(self, remove_h, valencies):
        self.remove_h = remove_h
        self.valencies = torch.tensor(valencies)

    def __call__(self, noisy_data):
        batch = noisy_data["batch"]
        edge_index = noisy_data["edge_index_t"]
        device = noisy_data["edge_attr_t"].device
        bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=device).reshape(1, -1)
        weighted_E = noisy_data["edge_attr_t"] * bond_orders  # (ne, de)
        current_valencies = weighted_E.argmax(dim=-1)  # (ne, )
        complete_valencies = torch.zeros(batch.shape[0]).to(device)  # (nx, )
        complete_index = torch.arange(batch.shape[0]).to(device)  # (nx, )
        current_valencies = pool.global_add_pool(
            torch.hstack([current_valencies, complete_valencies]),
            torch.hstack([edge_index[0], complete_index]),
        )  # (nx, )

        valencies = self.valencies.to(device).reshape(1, -1)
        X = noisy_data["node_t"] * valencies  # (nx, dx)
        normal_valencies = torch.argmax(X, dim=-1)  # (nx, )

        return (normal_valencies - current_valencies).type_as(noisy_data["node_t"])


class SparseValencyFeature:
    def __init__(self):
        pass

    def __call__(self, noisy_data):
        batch = noisy_data["batch"]
        edge_index = noisy_data["edge_index_t"]
        device = noisy_data["edge_attr_t"].device
        bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=device).reshape(1, -1)
        weighted_E = noisy_data["edge_attr_t"] * bond_orders  # (ne, de)
        current_valencies = weighted_E.argmax(dim=-1)  # (ne, )
        complete_valencies = torch.zeros(batch.shape[0]).to(device)  # (nx, )
        complete_index = torch.arange(batch.shape[0]).to(device)  # (nx, )
        current_valencies = pool.global_add_pool(
            torch.hstack([current_valencies, complete_valencies]),
            torch.hstack([edge_index[0], complete_index]),
        )  # (nx, )
        return current_valencies.type_as(noisy_data["node_t"])


class SparseWeightFeature:
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight
        self.atom_weight_list = torch.tensor(atom_weights)

    def __call__(self, noisy_data):
        X = torch.argmax(noisy_data["node_t"], dim=-1)  # (nx, )
        X_weights = self.atom_weight_list.to(X.device)[X]  # (nx, )
        X_weights = pool.global_add_pool(X_weights, noisy_data["batch"])  # (bs, )

        return (X_weights.type_as(noisy_data["node_t"]) / self.max_weight).unsqueeze(
            -1
        )  # (bs, 1)
