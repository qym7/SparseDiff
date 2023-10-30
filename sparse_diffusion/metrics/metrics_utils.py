import math
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data
import torch.nn.functional as F
from sparse_diffusion.datasets.dataset_utils import Statistics


def molecules_to_datalist(molecules):
    data_list = []
    for molecule in molecules:
        x = molecule.node_types.long()
        bonds = molecule.bond_types.long()
        positions = molecule.positions
        charge = molecule.charge
        edge_index = bonds.nonzero().contiguous().T
        bond_types = bonds[edge_index[0], edge_index[1]]
        edge_attr = bond_types.long()
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=positions,
            charge=charge,
        )
        data_list.append(data)

    return data_list


def compute_all_statistics(data_list, atom_encoder, charge_dic):
    num_nodes = node_counts(data_list)
    node_types = atom_type_counts(data_list, num_classes=len(atom_encoder))
    print(f"Atom types: {node_types}")
    bond_types = edge_counts(data_list)
    print(f"Bond types: {bond_types}")
    charge_types = charge_counts(
        data_list, num_classes=len(atom_encoder), charge_dic=charge_dic
    )
    print(f"Charge types: {charge_types}")
    valency = valency_count(data_list, atom_encoder)
    print("Valency: ", valency)
    return Statistics(
        num_nodes=num_nodes,
        node_types=node_types,
        bond_types=bond_types,
        charge_types=charge_types,
        valencies=valency,
    )


def node_counts(data_list):
    print("Computing node counts...")
    all_node_counts = Counter()
    for i, data in tqdm(enumerate(data_list)):
        num_nodes = data.num_nodes
        all_node_counts[num_nodes] += 1
    print("Done.")
    return all_node_counts


def graph_counts(data_list, num_graph_types):
    print("Computing node counts...")
    counts = np.zeros(num_graph_types)
    for i, data in tqdm(enumerate(data_list)):
        counts[data.y.argmax().item()] += 1
    print("Done.")
    return counts / counts.sum()


def atom_type_counts(data_list, num_classes):
    print("Computing node types distribution...")
    counts = np.zeros(num_classes)
    if num_classes == 1:
        print("Done.")
        return torch.tensor([1.0], dtype=torch.float)
    else:
        for data in tqdm(data_list):
            x = torch.nn.functional.one_hot(data.x, num_classes=num_classes)
            counts += x.sum(dim=0).cpu().numpy()
        counts = counts / counts.sum()
    print("Done.")
    return counts


def edge_counts(data_list, num_bond_types=5):
    print("Computing edge counts...")
    d = np.zeros(num_bond_types)

    for data in tqdm(data_list):
        total_pairs = data.num_nodes * (data.num_nodes - 1)

        num_edges = data.edge_attr.shape[0]
        num_non_edges = total_pairs - num_edges
        assert num_non_edges >= 0

        if len(data.edge_attr.shape) == 1:
            edge_types = (
                torch.nn.functional.one_hot(
                    data.edge_attr - 1, num_classes=num_bond_types - 1
                )
                .sum(dim=0)
                .cpu()
                .numpy()
            )
        else:
            edge_types = data.edge_attr.argmax(-1)
            edge_types = (
                torch.nn.functional.one_hot(
                    edge_types - 1, num_classes=num_bond_types - 1
                )
                .sum(dim=0)
                .cpu()
                .numpy()
            )

        d[0] += num_non_edges
        d[1:] += edge_types

    d = d / d.sum()
    return d


def charge_counts(data_list, num_classes, charge_dic):
    print("Computing charge counts...")
    d = np.zeros((num_classes, len(charge_dic)))

    for data in tqdm(data_list):
        for atom, charge in zip(data.x, data.charge):
            try:
                assert charge in [-2, -1, 0, 1, 2, 3]
                d[atom.item(), charge_dic[charge.item()]] += 1
            except:
                import pdb

                pdb.set_trace()

    s = np.sum(d, axis=1, keepdims=True)
    s[s == 0] = 1
    d = d / s
    print("Done.")
    return d


def valency_count(data_list, atom_encoder):
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    print("Computing valency counts...")
    valencies = {atom_type: Counter() for atom_type in atom_encoder.keys()}

    for data in tqdm(data_list):
        edge_attr = data.edge_attr.clone()
        edge_attr[edge_attr == 4] = 1.5
        bond_orders = edge_attr

        for atom in range(data.num_nodes):
            edges = bond_orders[data.edge_index[0] == atom]
            valency = edges.sum(dim=0)
            valencies[atom_decoder[data.x[atom].item()]][valency.item()] += 1

    # Normalizing the valency counts
    for atom_type in valencies.keys():
        s = sum(valencies[atom_type].values())
        for valency, count in valencies[atom_type].items():
            valencies[atom_type][valency] = count / s
    print("Done.")
    return valencies


def counter_to_tensor(c: Counter):
    max_key = max(c.keys())
    assert type(max_key) == int
    arr = torch.zeros(max_key + 1, dtype=torch.float)
    for k, v in c.items():
        arr[k] = v
    arr / torch.sum(arr)
    return arr


def wasserstein1d(preds, target, step_size=1):
    """preds and target are 1d tensors. They contain histograms for bins that are regularly spaced"""
    target = normalize(target) / step_size
    preds = normalize(preds) / step_size
    max_len = max(len(preds), len(target))
    preds = F.pad(preds, (0, max_len - len(preds)))
    target = F.pad(target, (0, max_len - len(target)))

    cs_target = torch.cumsum(target, dim=0)
    cs_preds = torch.cumsum(preds, dim=0)
    return torch.sum(torch.abs(cs_preds - cs_target))


def total_variation1d(preds, target):
    assert (
        target.dim() == 1 and preds.shape == target.shape
    ), f"preds: {preds.shape}, target: {target.shape}"
    target = normalize(target)
    preds = normalize(preds)
    return torch.sum(torch.abs(preds - target)).item(), torch.abs(preds - target)


def normalize(tensor):
    s = tensor.sum()
    assert s > 0
    return tensor / s
