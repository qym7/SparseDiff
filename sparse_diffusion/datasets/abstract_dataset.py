import abc
import numpy as np
import os, sys
import pathlib
import os.path as osp
RootPath = pathlib.Path(osp.realpath(__file__)).parents[1]
sys.path.append(f'{RootPath}')
from diffusion.distributions import DistributionNodes
import utils
import torch
import torch.nn.functional as F
from torch_geometric.data.lightning import LightningDataset


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(
            train_dataset=datasets["train"],
            val_dataset=datasets["val"],
            test_dataset=datasets["test"],
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            pin_memory=getattr(cfg.dataset, "pin_memory", False),
        )
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None
        
        self.dataset_stat()

    def dataset_stat(self):
        dataset = self.train_dataset + self.val_dataset + self.test_dataset
        nodes = []
        edges = []
        sparsity = []
        
        for data in dataset:
            nodes.append(data.x.shape[0])
            edges.append(data.edge_attr.shape[0])
            sparsity.append(data.edge_attr.shape[0] / (data.x.shape[0] * data.x.shape[0]))
            
        print('n graph', len(nodes))
        print("nodes: ", np.min(nodes), np.max(nodes))
        print("edges: ", np.min(edges), np.max(edges))
        print("sparsity: ", np.min(sparsity), np.max(sparsity))

    def prepare_dataloader(self):
        self.dataloaders = {}
        self.dataloaders["train"] = self.train_dataloader()
        self.dataloaders["val"] = self.val_dataloader()
        self.dataloaders["test"] = self.test_dataloader()

    def node_counts(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ["train", "val", "test"]:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for data in self.dataloaders["train"]:
            num_classes = data.x.shape[1]

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.dataloaders["train"]):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.dataloaders["train"]:
            num_classes = data.edge_attr.shape[1]

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.dataloaders["train"]):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(
            3 * max_n_nodes - 2
        )  # Max valency possible if everything is connected

        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for split in ["train", "val", "test"]:
            for i, data in enumerate(self.dataloaders[split]):
                n = data.x.shape[0]

                for atom in range(n):
                    edges = data.edge_attr[data.edge_index[0] == atom]
                    edges_total = edges.sum(dim=0)
                    valency = (edges_total * multiplier).sum()
                    valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    @abc.abstractmethod
    def to_one_hot(self, data):
        """
        call in the beginning of data
        get the one_hot encoding for a charge beginning from -1
        """
        one_hot_data = data.clone()
        one_hot_data.x = F.one_hot(data.x, num_classes=self.num_node_types).float()
        one_hot_data.edge_attr = F.one_hot(data.edge_attr, num_classes=self.num_edge_types).float()

        if not self.use_charge:
            one_hot_data.charge = data.x.new_zeros((*data.x.shape[:-1], 0))
        else:
            one_hot_data.charge = F.one_hot(data.charge + 1, num_classes=self.num_charge_types).float()

        return one_hot_data

    def one_hot_charge(self, charge):
        """
        get the one_hot encoding for a charge beginning from -1
        """
        if not self.use_charge:
            charge = charge.new_zeros((*charge.shape[:-1], 0))
        else:
            charge = F.one_hot(
                charge + 1, num_classes=self.num_charge_types
            ).float()
        return charge

    def complete_infos(self, statistics, node_types):
        # atom and edge type information
        self.node_types = statistics["train"].node_types
        self.edge_types = statistics["train"].bond_types
        self.charge_types = statistics["train"].charge_types
        self.num_node_types = len(self.node_types)
        self.num_edge_types = len(self.edge_types)
        self.num_charge_types = self.charge_types.shape[-1] if self.use_charge else 0

        # Train + val + test for n_nodes
        train_n_nodes = statistics["train"].num_nodes
        val_n_nodes = statistics["val"].num_nodes
        test_n_nodes = statistics["test"].num_nodes
        max_n_nodes = max(
            max(train_n_nodes.keys()), max(val_n_nodes.keys()), max(test_n_nodes.keys())
        )
        n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
        for c in [train_n_nodes, val_n_nodes, test_n_nodes]:
            for key, value in c.items():
                n_nodes[key] += value
        self.n_nodes = n_nodes / n_nodes.sum()

        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_dims(self, datamodule, extra_features, domain_features):
        data = next(iter(datamodule.train_dataloader()))
        print(data)
        example_batch = self.to_one_hot(data)
        ex_dense, node_mask = utils.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch,
            example_batch.charge,
        )

        self.input_dims = utils.PlaceHolder(
            X=example_batch.x.size(1),
            E=example_batch.edge_attr.size(1),
            y=example_batch.y.size(1) + 1 if example_batch.y is not None else 1,
            charge=self.num_charge_types,
        )

        example_data = {
            "node_t": example_batch.x,
            "edge_index_t": example_batch.edge_index,
            "edge_attr_t": example_batch.edge_attr,
            "batch": example_batch.batch,
            "y_t": example_batch.y,
            "charge_t": example_batch.charge,
        }

        ex_extra_feat = extra_features(example_data)
        if type(ex_extra_feat) == tuple:
            ex_extra_feat = ex_extra_feat[0]
        try:
            self.input_dims.X += ex_extra_feat.X.size(-1)
            self.input_dims.E += ex_extra_feat.E.size(-1)
            self.input_dims.y += ex_extra_feat.y.size(-1)
        except:
            self.input_dims.X += ex_extra_feat.node.size(-1)
            self.input_dims.E += ex_extra_feat.edge_attr.size(-1)
            self.input_dims.y += ex_extra_feat.y.size(-1)
            
        mol_extra_feat = domain_features(example_data)
        if type(mol_extra_feat) == tuple:
            mol_extra_feat = mol_extra_feat[0]
        self.input_dims.X += mol_extra_feat.node.size(-1)
        self.input_dims.E += mol_extra_feat.edge_attr.size(-1)
        self.input_dims.y += mol_extra_feat.y.size(-1)
