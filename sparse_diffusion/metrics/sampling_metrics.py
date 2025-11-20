import os, sys
import pathlib
import os.path as osp
RootPath = pathlib.Path(osp.realpath(__file__)).parents[1]
sys.path.append(f'{RootPath}')
from collections import Counter

import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import wandb
from torchmetrics import MeanMetric, MaxMetric, Metric, MeanAbsoluteError
import torch
from torch import Tensor
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric as pyg

import utils
from metrics.metrics_utils import (
    counter_to_tensor,
    wasserstein1d,
    total_variation1d,
)

class SamplingMetrics(nn.Module):
    def __init__(self, dataset_infos, test, dataloaders=None):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.test = test

        self.disconnected = MeanMetric()
        self.mean_components = MeanMetric()
        self.max_components = MaxMetric()
        self.num_nodes_w1 = MeanMetric()
        self.node_types_tv = MeanMetric()
        self.edge_types_tv = MeanMetric()

        self.domain_metrics = None
        if dataset_infos.is_molecular:
            from sparse_diffusion.metrics.molecular_metrics import (
                SamplingMolecularMetrics,
            )

            self.domain_metrics = SamplingMolecularMetrics(
                dataset_infos.train_smiles,
                dataset_infos.test_smiles if test else dataset_infos.val_smiles,
                dataset_infos,
                test
            )

        elif dataset_infos.spectre:
            from sparse_diffusion.metrics.spectre_utils import (
                Comm20SamplingMetrics,
                PlanarSamplingMetrics,
                SBMSamplingMetrics,
                ProteinSamplingMetrics,
                PointCloudSamplingMetrics,
                EgoSamplingMetrics
            )

            if dataset_infos.dataset_name == "comm20":
                self.domain_metrics = Comm20SamplingMetrics(dataloaders=dataloaders, test=test)
            elif dataset_infos.dataset_name == "planar":
                self.domain_metrics = PlanarSamplingMetrics(dataloaders=dataloaders, test=test)
            elif dataset_infos.dataset_name == "sbm":
                self.domain_metrics = SBMSamplingMetrics(dataloaders=dataloaders, test=test)
            elif dataset_infos.dataset_name == "protein":
                self.domain_metrics = ProteinSamplingMetrics(dataloaders=dataloaders, test=test)
            elif dataset_infos.dataset_name == "point_cloud":
                self.domain_metrics = PointCloudSamplingMetrics(dataloaders=dataloaders, test=test)
            elif dataset_infos.dataset_name == "ego":
                self.domain_metrics = EgoSamplingMetrics(dataloaders=dataloaders, test=test)
            else:
                raise ValueError(
                    "Dataset {} not implemented".format(dataset_infos.dataset_name)
                )

    def reset(self):
        for metric in [
            self.mean_components,
            self.max_components,
            self.disconnected,
            self.num_nodes_w1,
            self.node_types_tv,
            self.edge_types_tv,
        ]:
            metric.reset()
        if self.domain_metrics is not None:
            self.domain_metrics.reset()

    def compute_all_metrics(self, generated_graphs: list, current_epoch, local_rank):
        """Compare statistics of the generated data with statistics of the val/test set"""
        stat = (
            self.dataset_infos.statistics["test"]
            if self.test
            else self.dataset_infos.statistics["val"]
        )

        # Number of nodes
        self.num_nodes_w1(number_nodes_distance(generated_graphs, stat.num_nodes))

        # Node types
        node_type_tv, node_tv_per_class = node_types_distance(
            generated_graphs, stat.node_types, save_histogram=True
        )
        self.node_types_tv(node_type_tv)

        # Edge types
        edge_types_tv, edge_tv_per_class = bond_types_distance(
            generated_graphs, stat.bond_types, save_histogram=True
        )
        self.edge_types_tv(edge_types_tv)

        # Components
        device = self.disconnected.device
        connected_comp = connected_components(generated_graphs).to(device)
        self.disconnected(connected_comp > 1)
        self.mean_components(connected_comp)
        self.max_components(connected_comp)

        key = "val" if not self.test else "test"
        to_log = {
            f"{key}/NumNodesW1": self.num_nodes_w1.compute().item(),
            f"{key}/NodeTypesTV": self.node_types_tv.compute().item(),
            f"{key}/EdgeTypesTV": self.edge_types_tv.compute().item(),
            f"{key}/Disconnected": self.disconnected.compute().item() * 100,
            f"{key}/MeanComponents": self.mean_components.compute().item(),
            f"{key}/MaxComponents": self.max_components.compute().item(),
        }

        if self.domain_metrics is not None:
            do_metrics = self.domain_metrics.forward(
                generated_graphs, current_epoch, local_rank
            )
            to_log.update(do_metrics)

        if wandb.run:
            wandb.log(to_log, commit=False)
        if local_rank == 0:
            print(
                f"Sampling metrics", {key: round(val, 5) for key, val in to_log.items()}
            )

        return to_log, edge_tv_per_class


def number_nodes_distance(generated_graphs, dataset_counts):
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(
        max_number_nodes + 1, device=generated_graphs.batch.device
    )
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for i in range(generated_graphs.batch.max() + 1):
        c[int((generated_graphs.batch == i).sum())] += 1

    generated_n = counter_to_tensor(c).to(reference_n.device)
    return wasserstein1d(generated_n, reference_n)


def node_types_distance(generated_graphs, target, save_histogram=True):
    generated_distribution = torch.zeros_like(target)

    for node in generated_graphs.node:
        generated_distribution[node] += 1

    if save_histogram:
        if wandb.run:
            data = [[k, l] for k, l in zip(target, generated_distribution/generated_distribution.sum())]
            table = wandb.Table(data=data, columns=["target", "generate"])
            wandb.log({'node distribution': wandb.plot.histogram(table, 'types', title="node distribution")})

        np.save("generated_node_types.npy", generated_distribution.cpu().numpy())

    return total_variation1d(generated_distribution, target)


def bond_types_distance(generated_graphs, target, save_histogram=True):
    device = generated_graphs.batch.device
    generated_distribution = torch.zeros_like(target).to(device)
    edge_index, edge_attr = utils.undirected_to_directed(
        generated_graphs.edge_index, generated_graphs.edge_attr
    )
    for edge in edge_attr:
        generated_distribution[edge] += 1

    # get the number of non-existing edges
    n_nodes = pyg.nn.pool.global_add_pool(
        torch.ones_like(generated_graphs.batch).unsqueeze(-1), generated_graphs.batch
    ).flatten()
    generated_distribution[0] = (n_nodes * (n_nodes - 1) / 2).sum()
    generated_distribution[0] = (
        generated_distribution[0] - generated_distribution[1:].sum()
    )

    if save_histogram:
        if wandb.run:
            data = [[k, l] for k, l in zip(target, generated_distribution/generated_distribution.sum())]
            table = wandb.Table(data=data, columns=["target", "generate"])
            wandb.log({'edge distribution': wandb.plot.histogram(table, 'types', title="edge distribution")})

        np.save("generated_bond_types.npy", generated_distribution.cpu().numpy())

    tv, tv_per_class = total_variation1d(generated_distribution, target.to(device))
    return tv, tv_per_class


def connected_components(generated_graphs):
    num_graphs = int(generated_graphs.batch.max() + 1)
    all_num_components = torch.zeros(num_graphs)
    batch = generated_graphs.batch
    edge_batch = batch[generated_graphs.edge_index[0]]
    for i in range(num_graphs):
        # get the graph
        node_mask = batch == i
        edge_mask = edge_batch == i
        node = generated_graphs.node[node_mask]
        edge_index = generated_graphs.edge_index[:, edge_mask] - generated_graphs.ptr[i]
        # DENSE OPERATIONS
        sp_adj = to_scipy_sparse_matrix(edge_index, num_nodes=len(node))
        num_components, component = sp.csgraph.connected_components(sp_adj.toarray())
        all_num_components[i] = num_components

    return all_num_components


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        target = self.target_histogram.to(pred.device)
        super().update(pred, target)


class CEPerClass(Metric):
    full_state_update = True

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.0).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples


class MeanNumberEdge(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("total_edge", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples
