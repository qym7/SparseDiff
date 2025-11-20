import torch.nn as nn
import wandb
import os, sys
import pathlib
import os.path as osp
RootPath = pathlib.Path(osp.realpath(__file__)).parents[1]
sys.path.append(f'{RootPath}')
from metrics.abstract_metrics import CrossEntropyMetric


class TrainLossDiscrete(nn.Module):
    """Train with Cross entropy"""

    def __init__(self, lambda_train, edge_fraction):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.charge_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train
        self.lambda_train[0] = self.lambda_train[0] / edge_fraction

    def forward(self, pred, true_data, log: bool):
        loss_X = (
            self.node_loss(pred.node, true_data.node)
            if true_data.node.numel() > 0
            else 0.0
        )
        loss_E = self.edge_loss(pred.edge_attr, true_data.edge_attr)
        loss_y = 0.0
        loss_charge = self.charge_loss(pred.charge, true_data.charge) if pred.charge.numel() > 0 else 0.0

        if log:
            to_log = {
                "train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                "train_loss/X_CE": self.node_loss.compute()
                if true_data.node.numel() > 0
                else -1,
                "train_loss/E_CE": self.edge_loss.compute()
                if true_data.edge_attr.numel() > 0
                else -1,
                "train_loss/y_CE": -1,
                "train_loss/charge_CE": loss_charge if pred.charge.numel() > 0
                else -1,
            }
            if wandb.run:
                wandb.log(to_log, commit=True)

        return (
            loss_X
            + self.lambda_train[0] * loss_E
            + self.lambda_train[1] * loss_y
            + self.lambda_train[2] * loss_charge
        )

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = (
            self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        )
        epoch_edge_loss = (
            self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        )
        epoch_y_loss = (
            self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1
        )
        epoch_charge_loss = (
            self.charge_loss.compute() if self.charge_loss.total_samples > 0 else -1
        )

        to_log = {
            "train_epoch/x_CE": epoch_node_loss,
            "train_epoch/E_CE": epoch_edge_loss,
            "train_epoch/y_CE": epoch_y_loss,
            "train_epoch/charge_CE": epoch_charge_loss,
        }
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log
