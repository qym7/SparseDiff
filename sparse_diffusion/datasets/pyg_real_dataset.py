import inspect
import os, sys, pathlib
import os.path as osp

from sklearn.model_selection import train_test_split

from sparse_diffusion.metrics.metrics_utils import atom_type_counts, edge_counts, node_counts
from sparse_diffusion.utils import PlaceHolder
RootPath = pathlib.Path(os.path.realpath(__file__)).parents[1]
sys.path.append(f'{RootPath}')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import networkx_temporal as tx
from itertools import combinations

import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
from hydra.utils import get_original_cwd
from datasets.dataset_utils import (
    RemoveYTransform,
    load_pickle,
    Statistics,
    save_pickle,
    to_list,
)

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

def attribute_label(interaction_dict, snapshot_list):
    labeled_snapshots = []
    interaction_id = {b:i for i,b in enumerate(interaction_dict.keys())}
    node_labels = torch.eye(2)
    edge_labels = torch.eye(3)
    cross_edge_labels = torch.eye(3)
    for _, g in enumerate(snapshot_list):
        # intra-layer
        layers = {l: nx.Graph() for l in range(len(interaction_dict))}
        for l in range(len(interaction_dict)):
            for u in g.nodes():
                layers[l].add_node((u, l), x=torch.tensor(0, dtype=torch.long), active=0, nid=u)
            for u, v in g.edges():
                layers[l].add_edge((u, l), (v, l), edge_attr=torch.tensor(1, dtype=torch.long))
        for u, v, d in g.edges(data=True):
            interaction = d['interaction']
            l = interaction_id[interaction]
            layers[l].add_edge((u, l), (v, l), edge_attr=torch.tensor(2, dtype=torch.long))
            layers[l].nodes[(u, l)]['x'] = torch.tensor(1, dtype=torch.long)
            layers[l].nodes[(u, l)]['active'] = 1
            layers[l].nodes[(v, l)]['x'] = torch.tensor(1, dtype=torch.long)
            layers[l].nodes[(v, l)]['active'] = 1
        # inter-layer
        cross_layer_links = {(interaction_id[i1], interaction_id[i2]): nx.Graph() for i1, i2 in combinations(interaction_id.keys(),2)}
        for u in g.nodes():
            for i1, i2 in combinations(interaction_id, 2):
                l1, l2 = interaction_id[i1], interaction_id[i2]
                cross_layer_links[(l1, l2)].add_node((u, l1), nid=u, layer=l1, x=torch.tensor(0, dtype=torch.long))
                cross_layer_links[(l1, l2)].add_node((u, l2), nid=u, layer=l2, x=torch.tensor(0, dtype=torch.long))
                cross_layer_links[(l1, l2)].add_edge((u, l1),(u, l2), edge_attr=torch.tensor(1, dtype=torch.long))
                if layers[l1].nodes[(u, l1)]['active'] == layers[l2].nodes[(u, l2)]['active']:
                    cross_layer_links[(l1, l2)].nodes[(u, l1)]['x'] = torch.tensor(1, dtype=torch.long)
                    cross_layer_links[(l1, l2)].nodes[(u, l2)]['x'] = torch.tensor(1, dtype=torch.long)
                    cross_layer_links[(l1, l2)].edges[(u, l1),(u, l2)]['edge_attr'] = torch.tensor(2, dtype=torch.long)
        labeled_snapshots.append({'intra': layers, 'inter': cross_layer_links})
    return labeled_snapshots

class RealDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, num_bins: int, seed=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        
        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2
        
        self.num_bins = num_bins
        self.seed = seed

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(
            self.processed_paths[0],
            **({"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {})
        )
        
        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
        )
    
    @property
    def raw_file_names(self):
        return ["train.csv", "val.csv", "test.csv", "actions.csv"]
    
    @property
    def split_file_name(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.split == "train":
            return [
                f"train.pt",
                f"train_n.pickle",
                f"train_node_types.npy",
                f"train_bond_types.npy",
            ]
        elif self.split == "val":
            return [
                f"val.pt",
                f"val_n.pickle",
                f"val_node_types.npy",
                f"val_bond_types.npy",
            ]
        else:
            return [
                f"test.pt",
                f"test_n.pickle",
                f"test_node_types.npy",
                f"test_bond_types.npy",
            ]
    
    def download(self):
        df = pd.read_csv(self.raw_paths[-1])

        train_df, temp_df = train_test_split(df, test_size=0.30, shuffle=True, random_state=42)

        val_df, test_df = train_test_split(temp_df, test_size=0.50, shuffle=True, random_state=42)

        # 儲存
        train_df.to_csv(self.raw_paths[0], index=False)
        val_df.to_csv(self.raw_paths[1], index=False)
        test_df.to_csv(self.raw_paths[2], index=False)
    
    def process(self):
        print(f'Loading csv data = {self.split}, file_idx = {self.file_idx}...')
        df = pd.read_csv(self.raw_paths[self.file_idx])
        df = df[['source', 'target', 'interaction', 'datetime']]
        print('Building networkx graph')
        G = nx.from_pandas_edgelist(df, 'source', 'target', edge_attr=['interaction','datetime'])

        interactions = df['interaction'].unique().tolist()
        interaction_id = {name: i for i, name in enumerate(interactions)}

        print('Temporal sequence construction...')
        TG = tx.from_static(G)
        TG = TG.slice(bins=self.num_bins)
        snapshot_list = TG.to_snapshots()

        # node_number_list = [len(G.nodes) for G in sequence_list]
        # edge_number_list = [len(G.edges) for G in sequence_list]
        # N_max = max(node_number_list)
        # E_max = max(edge_number_list)

        print('Labeling snapshots...')
        labeled = attribute_label(interaction_id, snapshot_list)
        print(f'len(labeled_snapshots)={len(labeled)}')

        # flatten temporal graphs
        print('Flatten the multi-layer snapshots...')
        flatten_nx = []
        for s in labeled:
            G_flat = nx.Graph()

            for l in s['intra']:
                G_flat.add_nodes_from(s['intra'][l].nodes(data=True))
                G_flat.add_edges_from(s['intra'][l].edges(data=True))

            for (l1, l2) in s['inter']:
                for (u_node, v_node, attrs) in s['inter'][(l1, l2)].edges(data=True):
                    u, _layer1 = u_node
                    v, _layer2 = v_node

                    if u == v:
                        continue

                    G_flat.add_edge(u_node, v_node, **attrs)

            flatten_nx.append(G_flat)

        print("Flattened snapshots:", len(flatten_nx))
        
        print('Relabeling snapshots...')
        relabeled = []
        for Gf in flatten_nx:
            mapping = {n: i for i, n in enumerate(Gf.nodes())}
            relabeled.append(nx.relabel_nodes(Gf, mapping))
        print(f'len(relabeled_nx_snapshots)={len(relabeled)}')

        print('Converting to PyG format...')
        data_list = [from_networkx(snapshot) for _, snapshot in enumerate(relabeled)]
        
        for _, snapshot in enumerate(data_list):
            snapshot.y = torch.zeros(1,2)
        print(f'len(flatten_pyg_snapshots)={len(data_list)}')
            
        num_nodes = node_counts(data_list)
        node_types = atom_type_counts(data_list, num_classes=2)
        bond_types = edge_counts(data_list, num_bond_types=3)
        torch.save(self.collate(data_list), self.processed_paths[0])
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], node_types)
        np.save(self.processed_paths[3], bond_types)

class RealGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        self.dataset_name = cfg.dataset.name
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)
        pre_transform = RemoveYTransform()
        
        datasets = {
            "train": RealDataset(
                dataset_name=self.cfg.dataset.name,
                pre_transform=pre_transform,
                split="train",
                root=root_path,
                num_bins=cfg.general.num_bins
            ),
            "val": RealDataset(
                dataset_name=self.cfg.dataset.name,
                pre_transform=pre_transform,
                split="val",
                root=root_path,
                num_bins=cfg.general.num_bins
            ),
            "test": RealDataset(
                dataset_name=self.cfg.dataset.name,
                pre_transform=pre_transform,
                split="test",
                root=root_path,
                num_bins=cfg.general.num_bins
            ),
        }
        
        self.statistics = {
            "train": datasets["train"].statistics,
            "val": datasets["val"].statistics,
            "test": datasets["test"].statistics,
        }

        super().__init__(cfg, datasets)
        super().prepare_dataloader()
        self.inner = self.train_dataset
    
    def save_datasets(self):
        """
        Save train/val/test datasets
        """
        output_dir = os.path.join(self.root_path, "processed")
        os.makedirs(output_dir, exist_ok=True)

        splits = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset
        }

        for split, dataset in splits.items():
            path = os.path.join(output_dir, f"{split}_dataset.pt")
            data_list = list(dataset)  # Ensure it's serializable
            torch.save(data_list, path)
            print(f"Saved {split} dataset with {len(data_list)} graphs to: {path}")

class RealDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.is_molecular = False
        self.spectre = False
        self.use_charge = False
        self.dataset_name = datamodule.dataset_name
        self.node_types = datamodule.inner.statistics.node_types
        self.bond_types = datamodule.inner.statistics.bond_types
        super().complete_infos(
            datamodule.statistics, len(datamodule.inner.statistics.node_types)
        )
        self.input_dims = PlaceHolder(
            X=len(self.node_types), E=len(self.bond_types), y=0, charge=0
        )
        self.output_dims = PlaceHolder(
            X=len(self.node_types), E=len(self.bond_types), y=0, charge=0
        )
        self.statistics = {
            'train': datamodule.statistics['train'],
            'val': datamodule.statistics['val'],
            'test': datamodule.statistics['test']
        }  