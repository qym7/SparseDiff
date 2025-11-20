import inspect
import os, sys
import pathlib
import os.path as osp
RootPath = pathlib.Path(osp.realpath(__file__)).parents[1]
sys.path.append(f'{RootPath}')

import numpy as np
from tqdm import tqdm
import networkx as nx
import torch
import pickle as pkl
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from hydra.utils import get_original_cwd
from networkx import to_numpy_array

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
)


class SpectreGraphDataset(InMemoryDataset):
    def __init__(
        self,
        dataset_name,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.sbm_file = "sbm_200.pt"
        self.planar_file = "planar_64_200.pt"
        self.comm20_file = "community_12_21_100.pt"
        self.dataset_name = dataset_name

        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], **({"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}))

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
        )

    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt", 'ego.pkl', 'ego_ns.pkl']
        # return ["train.pkl", "val.pkl", "test.pkl"]

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
        Download raw qm9 files. Taken from PyG QM9 class
        """
        if self.dataset_name == "sbm":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt"
        elif self.dataset_name == "planar":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt"
        elif self.dataset_name == "comm20":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt"
        elif self.dataset_name == "ego":        
            raw_url = "https://raw.githubusercontent.com/tufts-ml/graph-generation-EDGE/main/graphs/Ego.pkl"
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
        file_path = download_url(raw_url, self.raw_dir)

        if self.dataset_name == 'ego':
            networks = pkl.load(open(file_path, 'rb'))
            adjs = [torch.Tensor(to_numpy_array(network)).fill_diagonal_(0) for network in networks]
        else:
            (
                adjs,
                eigvals,
                eigvecs,
                n_nodes,
                max_eigval,
                min_eigval,
                same_sample,
                n_max,
            ) = torch.load(file_path)
            
        g_cpu = torch.Generator()
        g_cpu.manual_seed(1234)
        self.num_graphs = len(adjs)

        if self.dataset_name == 'ego':
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round(self.num_graphs * 0.8))
            val_len = int(round(self.num_graphs * 0.2))
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
            train_indices = indices[:train_len]
            val_indices = indices[:val_len]
            test_indices = indices[train_len:]
        else:
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round((self.num_graphs - test_len) * 0.8))
            val_len = self.num_graphs - train_len - test_len
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
            train_indices = indices[:train_len]
            val_indices = indices[train_len : train_len + val_len]
            test_indices = indices[train_len + val_len :]

        print(f"Train indices: {train_indices}")
        print(f"Val indices: {val_indices}")
        print(f"Test indices: {test_indices}")
        train_data = []
        val_data = []
        test_data = []
        train_data_nx = []
        val_data_nx = []
        test_data_nx = []

        for i, adj in enumerate(adjs):
            # permute randomly nodes as for molecular datasets
            random_order = torch.randperm(adj.shape[-1])
            adj = adj[random_order, :]
            adj = adj[:, random_order]
            net = nx.from_numpy_array(adj.numpy()).to_undirected()

            if i in train_indices:
                train_data.append(adj)
                train_data_nx.append(net)
            if i in val_indices:
                val_data.append(adj)
                val_data_nx.append(net)
            if i in test_indices:
                test_data.append(adj)
                test_data_nx.append(net)

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

        # import pdb; pdb.set_trace()
        all_data = {'train': train_data_nx, 'val': val_data_nx, 'test': test_data_nx}
        import pickle
        with open(self.raw_paths[3], 'wb') as handle:
            pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.raw_paths[4], 'wb') as handle:
            pickle.dump(train_data_nx+val_data_nx+test_data_nx, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def process(self):
        raw_dataset = torch.load(os.path.join(self.raw_dir, "{}.pt".format(self.split)))
        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.long)
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            n_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(
                x=X.float(), edge_index=edge_index, edge_attr=edge_attr.float(), n_nodes=n_nodes
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        num_nodes = node_counts(data_list)
        node_types = atom_type_counts(data_list, num_classes=1)
        bond_types = edge_counts(data_list, num_bond_types=2)
        torch.save(self.collate(data_list), self.processed_paths[0])
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], node_types)
        np.save(self.processed_paths[3], bond_types)


class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)
        pre_transform = RemoveYTransform()

        datasets = {
            "train": SpectreGraphDataset(
                dataset_name=self.cfg.dataset.name,
                pre_transform=pre_transform,
                split="train",
                root=root_path,
            ),
            "val": SpectreGraphDataset(
                dataset_name=self.cfg.dataset.name,
                pre_transform=pre_transform,
                split="val",
                root=root_path,
            ),
            "test": SpectreGraphDataset(
                dataset_name=self.cfg.dataset.name,
                pre_transform=pre_transform,
                split="test",
                root=root_path,
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


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.is_molecular = False
        self.spectre = True
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

    def to_one_hot(self, data):
        """
        call in the beginning of data
        get the one_hot encoding for a charge beginning from -1
        """
        data.charge = data.x.new_zeros((*data.x.shape[:-1], 0))
        if data.y is None:
            data.y = data.x.new_zeros((data.batch.max().item()+1, 0))

        return data


class Comm20DataModule(SpectreGraphDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class SBMDataModule(SpectreGraphDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class PlanarDataModule(SpectreGraphDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class EgoDataModule(SpectreGraphDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
