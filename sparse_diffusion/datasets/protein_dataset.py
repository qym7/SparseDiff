import os, sys
import pathlib
import os.path as osp
RootPath = pathlib.Path(osp.realpath(__file__)).parents[1]
sys.path.append(f'{RootPath}')

import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import InMemoryDataset, download_url
from hydra.utils import get_original_cwd

from utils import PlaceHolder
from datasets.abstract_dataset import (
    AbstractDataModule,
    AbstractDatasetInfos,
)
from datasets.dataset_utils import (
    load_pickle,
    save_pickle,
    Statistics,
    to_list,
    RemoveYTransform,
)
from metrics.metrics_utils import (
    node_counts,
    atom_type_counts,
    edge_counts,
    graph_counts,
)

class ProteinDataset(InMemoryDataset):
    '''
    Implementation based on https://github.com/KarolisMart/SPECTRE/blob/main/data.py
    '''
    def __init__(
        self,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_name = 'protein'
        root = root

        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
        )

    @property
    def raw_file_names(self):
        return ["train_indices.pt", "val_indices.pt", "test_indices.pt"]

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
        """
        Download raw files.
        """
        raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/DD"
        for name in ['DD_A.txt', 'DD_graph_indicator.txt', 'DD_graph_labels.txt', 'DD_node_labels.txt']:
            download_url(f'{raw_url}/{name}', self.raw_dir)

        # read
        path = os.path.join(self.root, 'raw')
        data_graph_indicator = np.loadtxt(os.path.join(path, 'DD_graph_indicator.txt'), delimiter=',').astype(int)

        # split data
        g_cpu = torch.Generator()
        g_cpu.manual_seed(1234)
        
        min_num_nodes=100
        max_num_nodes=500
        available_graphs = []
        for idx in np.arange(1, data_graph_indicator.max()+1):
            node_idx = data_graph_indicator == idx
            if node_idx.sum() >= min_num_nodes and node_idx.sum() <= max_num_nodes:
                available_graphs.append(idx)
        available_graphs = torch.Tensor(available_graphs)

        self.num_graphs = len(available_graphs)
        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        # # OLD IMPLEMENTATION
        # indices = torch.randperm(self.num_graphs, generator=g_cpu)
        # train_indices = available_graphs[indices][:train_len]
        # val_indices = available_graphs[indices][train_len : train_len + val_len]
        # test_indices = available_graphs[indices][train_len + val_len :]
        # SPECTRE IMPLEMENTATION
        train_indices, val_indices, test_indices = random_split(available_graphs,
                                                                [train_len, val_len, test_len],
                                                                generator=torch.Generator().manual_seed(1234))
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
        
        print(f"Train indices: {train_indices}")
        print(f"Val indices: {val_indices}")
        print(f"Test indices: {test_indices}")

        torch.save(train_indices, self.raw_paths[0])
        torch.save(val_indices, self.raw_paths[1])
        torch.save(test_indices, self.raw_paths[2])

    def process(self):
        indices = torch.load(os.path.join(self.raw_dir, "{}_indices.pt".format(self.split)))
        data_adj = torch.Tensor(np.loadtxt(os.path.join(self.raw_dir, 'DD_A.txt'), delimiter=',')).long() - 1
        data_node_label = torch.Tensor(np.loadtxt(os.path.join(self.raw_dir, 'DD_node_labels.txt'), delimiter=',')).long() - 1
        data_graph_indicator = torch.Tensor(np.loadtxt(os.path.join(self.raw_dir, 'DD_graph_indicator.txt'), delimiter=',')).long()
        data_graph_types = torch.Tensor(np.loadtxt(os.path.join(self.raw_dir, 'DD_graph_labels.txt'), delimiter=',')).long() - 1
        data_list = []

        # get information
        self.num_node_type = data_node_label.max() + 1
        self.num_edge_type = 2
        self.num_graph_type = data_graph_types.max() + 1
        print(f"Number of node types: {self.num_node_type}")
        print(f"Number of edge types: {self.num_edge_type}")
        print(f"Number of graph types: {self.num_graph_type}")

        for idx in indices:
            offset = torch.where(data_graph_indicator == idx)[0].min()
            node_idx = data_graph_indicator == idx
            perm = torch.randperm(node_idx.sum()).long()
            reverse_perm = torch.sort(perm)[1]
            nodes = data_node_label[node_idx][perm].long()
            edge_idx = node_idx[data_adj[:, 0]]
            edge_index = data_adj[edge_idx] - offset
            edge_index[:, 0] = reverse_perm[edge_index[:, 0]]
            edge_index[:, 1] = reverse_perm[edge_index[:, 1]]
            edge_attr = torch.ones_like(edge_index[:, 0]).long()
            edge_index, edge_attr = remove_self_loops(edge_index.T, edge_attr)
            data = torch_geometric.data.Data(
                x=nodes,
                edge_index=edge_index,
                edge_attr=edge_attr,
                n_nodes=nodes.shape[0],
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            print(node_idx.sum())

        num_nodes = node_counts(data_list)
        node_types = atom_type_counts(data_list, num_classes=self.num_node_type)
        bond_types = edge_counts(data_list, num_bond_types=self.num_edge_type)
        torch.save(self.collate(data_list), self.processed_paths[0])
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], node_types)
        np.save(self.processed_paths[3], bond_types)


class ProteinDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, cfg.dataset.datadir)
        transform = RemoveYTransform()

        datasets = {
            "train": ProteinDataset(
                root=root_path,
                transform=transform,
                split="train",
            ),
            "val": ProteinDataset(
                root=root_path,
                transform=transform,
                split="val",
            ),
            "test": ProteinDataset(
                root=root_path,
                transform=transform,
                split="test",
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


class ProteinInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.is_molecular = False
        self.spectre = True
        self.use_charge = False
        self.dataset_name = datamodule.inner.dataset_name
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

    def to_one_hot(self, data):
        """
        call in the beginning of data
        get the one_hot encoding for a charge beginning from -1
        """
        data.charge = data.x.new_zeros((*data.x.shape[:-1], 0))
        data.x = F.one_hot(data.x, num_classes=self.num_node_types).float()
        data.edge_attr = F.one_hot(data.edge_attr, num_classes=self.num_edge_types).float()

        return data
