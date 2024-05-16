import os
from collections import Counter
import pandas as pd

# from fcd import get_fcd, load_ref_model
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
# from moses.metrics.metrics import get_all_metrics
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch_geometric as pyg
import torch_geometric.nn.pool as pool
from torchmetrics import (
    MeanMetric,
    MaxMetric,
    Metric,
    MetricCollection,
    MeanAbsoluteError,
)
from torchmetrics.utilities.data import _flatten_dict, allclose
from fcd_torch import FCD

import utils
from sparse_diffusion.metrics.metrics_utils import (
    counter_to_tensor,
    wasserstein1d,
    total_variation1d,
)


def canonicalize_smiles(smiles):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]

class TrainMolecularMetricsDiscrete(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        self.train_bond_metrics = BondMetricsCE()

    def reset(self):
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log["train_epoch/" + key] = val.item()
        for key, val in epoch_bond_metrics.items():
            to_log["train_epoch/" + key] = val.item()
        if wandb.run:
            wandb.log(to_log, commit=False)

        for key, val in epoch_atom_metrics.items():
            epoch_atom_metrics[key] = val.item()
        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = val.item()

        return epoch_atom_metrics, epoch_bond_metrics

    def forward(self, pred, true_data, log):
        self.train_atom_metrics(pred.node, true_data.node)
        self.train_bond_metrics(pred.edge_attr, true_data.edge_attr)

        if log:
            to_log = {}
            for key, val in self.train_atom_metrics.compute().items():
                to_log['train/' + key] = val.item()
            for key, val in self.train_bond_metrics.compute().items():
                to_log['train/' + key] = val.item()
            if wandb.run:
                wandb.log(to_log, commit=False)


class SamplingMolecularMetrics(nn.Module):
    def __init__(self, train_smiles, test_smiles, dataset_infos, test):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.atom_decoder = dataset_infos.atom_decoder

        self.test_smiles = test_smiles
        self.train_smiles = train_smiles
        self.test = test

        self.atom_stable = MeanMetric()
        self.mol_stable = MeanMetric()

        # Retrieve dataset smiles only for qm9 currently.
        self.train_smiles = set(train_smiles)
        self.validity_metric = MeanMetric()
        self.valid_mols = []
        self.charge_w1 = MeanMetric()
        self.valency_w1 = MeanMetric()

    def reset(self):
        for metric in [
            self.atom_stable,
            self.mol_stable,
            self.validity_metric,
            self.charge_w1,
            self.valency_w1,
        ]:
            metric.reset()
        self.valid_mols = []

    def compute_validity(self, generated):
        """generated: list of couples (positions, node_types)"""
        valid = []
        all_smiles = []
        error_message = Counter()
        for mol in generated:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        rdmol, asMols=True, sanitizeFrags=True
                    )
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                    error_message[-1] += 1
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                    all_smiles.append('error')
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                    all_smiles.append('error')
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
                    all_smiles.append('error')
        print(
            f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
            f" -- No error {error_message[-1]}"
        )
        self.validity_metric.update(
            value=len(valid) / len(generated), weight=len(generated)
        )

        return valid, all_smiles, error_message

    def evaluate(self, generated):
        """generated: list of pairs (positions: n x 3, node_types: n [int])
        the positions and atom types should already be masked."""
        # Validity
        valid, all_smiles, error_message = self.compute_validity(generated)
        validity = self.validity_metric.compute().item()
        uniqueness, novelty, fcd = 0, 0, 0
        self.valid_mols.extend(valid)

        # FCD
        FCD_eval = FCD(device='cuda:0', n_jobs=8)
        # test_smiles = [for i in test_smiles]
        test_smiles_no_h = []
        total_count = 0
        valid_count = 0
        for smile in list(self.test_smiles):
            total_count += 1
            # mol = Chem.MolFromSmiles(smile, isomericSmiles=False)
            # mol = Chem.MolFromSmiles(smile, isomericSmiles=False, canonical=True)
            mol = Chem.MolFromSmiles(smile, sanitize=True)
            # print('TEST SMILE', mol)
            if mol is not None:  # wierd thing happens to test_smiles
                valid_count += 1
                for a in mol.GetAtoms():
                    if a.GetNumImplicitHs():
                        a.SetNumRadicalElectrons(a.GetNumImplicitHs())
                        a.SetNoImplicit(True)
                        a.UpdatePropertyCache()
                molRemoveAllHs = Chem.RemoveAllHs(mol)
                test_smiles_no_h.append(Chem.MolToSmiles(molRemoveAllHs))

        print('among {} test smiles, {} are valid'.format(total_count, valid_count))
        # import pdb; pdb.set_trace()
        # fcd = FCD_eval(test_smiles_no_h, test_smiles_no_h)
        fcd = FCD_eval(test_smiles_no_h, valid)
        # fcd = FCD_eval(list(self.test_smiles), list(self.test_smiles))
        # fcd = FCD_eval(list(self.train_smiles), list(self.train_smiles))

        # Uniqueness
        if len(self.valid_mols) > 0:
            unique = list(set(self.valid_mols))
            uniqueness = len(unique) / len(valid)

            if self.train_smiles is not None:
                novel = []
                for smiles in unique:
                    if smiles not in self.train_smiles:
                        novel.append(smiles)
                novelty = len(novel) / len(unique)
            else:
                print("Train smiles not provided, can't compute novelty")
        else:
            uniqueness = -1
            novelty = -1
            fcd = -1

        num_molecules = int(self.validity_metric.weight.item())
        print(f"Validity over {num_molecules} molecules: {validity * 100 :.4f}%")
        print(f"Uniqueness: {uniqueness * 100 :.4f}%")
        print(f"Novelty: {novelty * 100 :.4f}%")
        print(f"FCD: {fcd * 100 :.4f}%")

        key = "val" if not self.test else "test"
        dic = {
            f"{key}/Validity": validity * 100,
            f"{key}/Uniqueness": uniqueness * 100 if uniqueness != 0 else 0,
            f"{key}/Novelty": novelty * 100 if novelty != 0 else 0,
            f"{key}/FCD": fcd if fcd != 0 else 0,
        }

        if wandb.run:
            wandb.log(dic, commit=False)

        print(fcd)

        return all_smiles, dic

    def forward(self, generated_graphs: list, current_epoch, local_rank):
        molecules = []
        num_graphs = max(generated_graphs.batch) + 1

        for i in range(num_graphs):
            node_mask = generated_graphs.batch == i
            edge_mask = generated_graphs.batch[generated_graphs.edge_index[0]] == i
            charge = (
                generated_graphs.charge[node_mask].long()
                if generated_graphs.charge.numel() > 0
                else None
            )
            molecule = SparseMolecule(
                node_types=generated_graphs.node[node_mask].long(),
                bond_index=generated_graphs.edge_index[:, edge_mask].long()
                - generated_graphs.ptr[i],
                bond_types=generated_graphs.edge_attr[edge_mask].long(),
                atom_decoder=self.dataset_infos.atom_decoder,
                charge=charge,
            )
            molecules.append(molecule)

        if not self.dataset_infos.remove_h and self.dataset_infos.use_charge:
            print(f"Analyzing molecule stability on {local_rank}...")
            for i, mol in enumerate(molecules):
                mol_stable, at_stable, num_bonds = mol.check_stability()
                self.mol_stable.update(value=mol_stable)
                self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)

            stability_dict = {
                "mol_stable": self.mol_stable.compute().item(),
                "atm_stable": self.atom_stable.compute().item(),
            }
            if local_rank == 0:
                print("Stability metrics:", stability_dict)
                if wandb.run:
                    wandb.log(stability_dict, commit=False)

        # Validity, uniqueness, novelty
        all_generated_smiles, metrics = self.evaluate(molecules)
        if len(all_generated_smiles) > 0 and local_rank == 0:
            print("Some generated smiles: " + " ".join(all_generated_smiles[:10]))

        # Save in any case in the graphs folder
        os.makedirs("graphs", exist_ok=True)
        textfile = open(
            f"graphs/valid_unique_molecules_e{current_epoch}_GR{local_rank}.txt", "w"
        )
        textfile.writelines(all_generated_smiles)
        textfile.close()

        # save in csv for moses evaluation
        df = pd.DataFrame({
            'SMILES': all_generated_smiles,
            'SPLIT': ['generated'] * len(all_generated_smiles)
        })

        # Define the CSV file name
        from datetime import datetime
        time_str = datetime.now().strftime("%M%S")
        csv_filename_pandas = f"smiles_data_{time_str}.csv"
        # Save the DataFrame to a CSV file
        df.to_csv(csv_filename_pandas, index=False)

        # Save in the root folder if test_model
        if self.test:
            filename = f"final_smiles_GR{local_rank}_{0}.txt"
            for i in range(2, 10):
                if os.path.exists(filename):
                    filename = f"final_smiles_GR{local_rank}_{i}.txt"
                else:
                    break
            with open(filename, "w") as fp:
                for smiles in all_generated_smiles:
                    # write each item on a new line
                    fp.write("%s\n" % smiles)
                print(f"All smiles saved on rank {local_rank}")

        # Compute statistics
        stat = (
            self.dataset_infos.statistics["test"]
            if self.test
            else self.dataset_infos.statistics["val"]
        )
        charge_w1, charge_w1_per_class = charge_distance(
            molecules, stat.charge_types, stat.node_types, self.dataset_infos
        )
        self.charge_w1(charge_w1)
        valency_w1, valency_w1_per_class = valency_distance(
            molecules, stat.valencies, stat.node_types, self.dataset_infos.atom_encoder
        )
        self.valency_w1(valency_w1)
        key = "val" if not self.test else "test"
        metrics[f"{key}/ChargeW1"] = self.charge_w1.compute().item()
        metrics[f"{key}/ValencyW1"] = self.valency_w1.compute().item()
        if local_rank == 0:
            print(f"Sampling metrics", {k: round(val, 3) for k, val in metrics.items()})

        if wandb.run:
            wandb.log(metrics, commit=False)
        if local_rank == 0:
            print(f"Sampling metrics done.")

        return metrics


def number_nodes_distance(molecules, dataset_counts):
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(max_number_nodes + 1)
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for molecule in molecules:
        c[molecule.num_nodes] += 1

    generated_n = counter_to_tensor(c).to(molecules[0])
    return wasserstein1d(generated_n, reference_n)


def node_types_distance(molecules, target, save_histogram=False):
    generated_distribution = torch.zeros_like(target)
    for molecule in molecules:
        for atom_type in molecule.node_types:
            generated_distribution[atom_type] += 1
    if save_histogram:
        np.save("generated_node_types.npy", generated_distribution.cpu().numpy())
    return total_variation1d(generated_distribution, target)


def bond_types_distance(molecules, target, save_histogram=False):
    device = molecules[0].bond_types.device
    generated_distribution = torch.zeros_like(target).to(device)
    for molecule in molecules:
        bond_types = molecule.bond_types
        mask = torch.ones_like(bond_types)
        mask = torch.triu(mask, diagonal=1).bool()
        bond_types = bond_types[mask]
        unique_edge_types, counts = torch.unique(bond_types, return_counts=True)
        for type, count in zip(unique_edge_types, counts):
            generated_distribution[type] += count
    if save_histogram:
        np.save("generated_bond_types.npy", generated_distribution.cpu().numpy())
    sparsity_level = generated_distribution[0] / torch.sum(generated_distribution)
    tv, tv_per_class = total_variation1d(generated_distribution, target.to(device))
    return tv, tv_per_class, sparsity_level


def charge_distance(molecules, target, node_types_probabilities, dataset_infos):
    device = molecules[0].bond_types.device
    generated_distribution = torch.zeros_like(target).to(device)
    for molecule in molecules:
        for atom_type in range(target.shape[0]):
            mask = molecule.node_types == atom_type
            if mask.sum() > 0:
                at_charge = dataset_infos.one_hot_charge(molecule.charge[mask])
                at_charge = molecule.charge[mask]
                generated_distribution[atom_type] += at_charge.sum(dim=0)

    s = generated_distribution.sum(dim=1, keepdim=True)
    s[s == 0] = 1
    generated_distribution = generated_distribution / s

    cs_generated = torch.cumsum(generated_distribution, dim=1)
    cs_target = torch.cumsum(target, dim=1).to(device)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1)

    w1 = torch.sum(w1_per_class * node_types_probabilities.to(device)).item()

    return w1, w1_per_class


def valency_distance(
    molecules, target_valencies, node_types_probabilities, atom_encoder
):
    # Build a dict for the generated molecules that is similar to the target one
    num_node_types = len(node_types_probabilities)
    generated_valencies = {i: Counter() for i in range(num_node_types)}
    for molecule in molecules:
        edge_types = molecule.bond_types
        edge_index = molecule.bond_index
        node_types = molecule.node_types
        edge_types[edge_types == 4] = 1.5
        valencies = pool.global_add_pool(
            edge_types.repeat(2), edge_index.flatten(), size=node_types.shape[0]
        )  # (nx, )
        # print('valencies for {} atoms is {}'.format(molecule.node_types.shape, valencies))
        for atom, val in zip(molecule.node_types, valencies):
            generated_valencies[atom.item()][val.item()] += 1

    # Convert the valencies to a tensor of shape (num_node_types, max_valency)
    max_valency_target = max(
        max(vals.keys()) if len(vals) > 0 else -1 for vals in target_valencies.values()
    )
    max_valency_generated = max(
        max(vals.keys()) if len(vals) > 0 else -1
        for vals in generated_valencies.values()
    )
    max_valency = int(max(max_valency_target, max_valency_generated))

    valencies_target_tensor = torch.zeros(num_node_types, max_valency + 1)
    for atom_type, valencies in target_valencies.items():
        for valency, count in valencies.items():
            valencies_target_tensor[int(atom_encoder[atom_type]), int(valency)] = count

    valencies_generated_tensor = torch.zeros(num_node_types, max_valency + 1)
    for atom_type, valencies in generated_valencies.items():
        for valency, count in valencies.items():
            valencies_generated_tensor[int(atom_type), int(valency)] = count

    # Normalize the distributions
    s1 = torch.sum(valencies_target_tensor, dim=1, keepdim=True)
    s1[s1 == 0] = 1
    valencies_target_tensor = valencies_target_tensor / s1

    s2 = torch.sum(valencies_generated_tensor, dim=1, keepdim=True)
    s2[s2 == 0] = 1
    valencies_generated_tensor = valencies_generated_tensor / s2

    cs_target = torch.cumsum(valencies_target_tensor, dim=1)
    cs_generated = torch.cumsum(valencies_generated_tensor, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_target - cs_generated), dim=1)

    # print('debugging for molecular_metrics - valency_distance')
    # print('cs_target', cs_target)
    # print('cs_generated', cs_generated)

    total_w1 = torch.sum(w1_per_class * node_types_probabilities).item()
    return total_w1, w1_per_class


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
        target_one_hot = torch.nn.functional.one_hot(target.long(), num_classes=preds.shape[-1]).float()
        mask = (target_one_hot != 0.0).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target_one_hot = target_one_hot[:, self.class_id]
        target_one_hot = target_one_hot[mask]

        output = self.binary_cross_entropy(prob, target_one_hot)
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


class HydrogenCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetricsCE(MetricCollection):
    def __init__(self, dataset_infos):
        atom_decoder = dataset_infos.atom_decoder

        class_dict = {
            "H": HydrogenCE,
            "C": CarbonCE,
            "N": NitroCE,
            "O": OxyCE,
            "F": FluorCE,
            "B": BoronCE,
            "Br": BrCE,
            "Cl": ClCE,
            "I": IodineCE,
            "P": PhosphorusCE,
            "S": SulfurCE,
            "Se": SeCE,
            "Si": SiCE,
        }

        metrics_list = []
        for i, atom_type in enumerate(atom_decoder):
            metrics_list.append(class_dict[atom_type](i))
        super().__init__(metrics_list)


class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],
        -1: 2,
    },  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


class Molecule:
    def __init__(self, node_types: Tensor, bond_types: Tensor, atom_decoder, charge):
        """node_types: n      LongTensor
        charge: n         LongTensor
        bond_types: n x n  LongTensor
        atom_decoder: extracted from dataset_infos."""

        assert node_types.dim() == 1 and node_types.dtype == torch.long, (
            f"shape of atoms {node_types.shape} " f"and dtype {node_types.dtype}"
        )
        assert bond_types.dim() == 2 and bond_types.dtype == torch.long, (
            f"shape of bonds {bond_types.shape} --" f" {bond_types.dtype}"
        )
        assert len(node_types.shape) == 1
        assert len(bond_types.shape) == 2

        self.node_types = node_types.long()
        self.bond_types = bond_types.long()
        self.charge = charge if charge is not None else torch.zeros_like(node_types)
        self.charge = charge.long()
        self.rdkit_mol = self.build_molecule(atom_decoder)
        self.atom_decoder = atom_decoder
        self.num_nodes = len(node_types)
        self.num_node_types = len(atom_decoder)
        self.device = self.node_types.device

    def build_molecule(self, atom_decoder):
        """If positions is None,"""
        mol = Chem.RWMol()
        for atom, charge in zip(self.node_types, self.charge):
            if atom == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom.item())])
            if charge.numel() > 0:
                a.SetFormalCharge(charge.item())
            mol.AddAtom(a)

        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(
                    bond[0].item(),
                    bond[1].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None

        return mol

    def check_stability(self, debug=False):
        e = self.bond_types.clone()
        e[e == 4] = 1.5
        e[e < 0] = 0
        valencies = torch.sum(e, dim=-1).long()

        n_stable_at = 0
        mol_stable = True
        for i, (atom_type, valency, charge) in enumerate(
            zip(self.node_types, valencies, self.charge)
        ):
            atom_type = atom_type.item()
            valency = valency.item()
            charge = charge.item()
            possible_bonds = allowed_bonds[self.atom_decoder[atom_type]]
            if type(possible_bonds) == int:
                is_stable = possible_bonds == valency
            elif type(possible_bonds) == dict:
                expected_bonds = (
                    possible_bonds[charge]
                    if charge in possible_bonds.keys()
                    else possible_bonds[0]
                )
                is_stable = (
                    expected_bonds == valency
                    if type(expected_bonds) == int
                    else valency in expected_bonds
                )
            else:
                is_stable = valency in possible_bonds
            if not is_stable:
                mol_stable = False
            if not is_stable and debug:
                print(
                    f"Invalid atom {self.atom_decoder[atom_type]}: valency={valency}, charge={charge}"
                )
                print()
            n_stable_at += int(is_stable)

        return mol_stable, n_stable_at, len(self.node_types)

        # TODO: converting to tensor might be necessary
        # return torch.tensor([mol_stable], dtype=torch.float, device=self.device), \
        #        torch.tensor([n_stable_at], dtype=torch.float, device=self.device), \
        #        len(self.node_types)


class SparseMolecule:
    def __init__(
        self,
        node_types: Tensor,
        bond_index: Tensor,
        bond_types: Tensor,
        atom_decoder,
        charge,
    ):
        """node_types: nx      LongTensor
        charge: nx         LongTensor
        bond_types: ne      LongTensor  # triu edges
        bond_types: 2 * ne  LongTensor
        atom_decoder: extracted from dataset_infos."""

        assert node_types.dim() == 1 and node_types.dtype == torch.long, (
            f"shape of atoms {node_types.shape} " f"and dtype {node_types.dtype}"
        )
        assert bond_types.dim() == 1 and bond_types.dtype == torch.long, (
            f"shape of bonds {bond_types.shape} --" f" {bond_types.dtype}"
        )

        bond_index, bond_types = utils.undirected_to_directed(bond_index, bond_types)
        self.node_types = node_types.long()
        self.bond_types = bond_types.long()
        self.bond_index = bond_index.long()
        self.charge = charge if charge is not None else torch.zeros_like(node_types)
        self.charge = self.charge.long()
        self.rdkit_mol = self.build_molecule(atom_decoder)
        self.atom_decoder = atom_decoder
        self.num_nodes = len(node_types)
        self.num_node_types = len(atom_decoder)
        self.device = self.node_types.device

    def build_molecule(self, atom_decoder):
        """If positions is None,"""
        mol = Chem.RWMol()
        for atom, charge in zip(self.node_types, self.charge):
            if atom == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom.item())])
            if charge.numel() > 0:
                a.SetFormalCharge(charge.item())
            mol.AddAtom(a)

        for i, bond in enumerate(self.bond_index.T):
            if bond[0].item() != bond[1].item():
                try:
                    mol.AddBond(
                        bond[0].item(),
                        bond[1].item(),
                        bond_dict[self.bond_types[i].item()],
                    )
                except:
                    import pdb
                    pdb.set_trace()

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None

        return mol

    def calculate_valency(self):
        nx = self.node_types.shape[0]
        edge_index = self.bond_index
        device = self.bond_index.device
        bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=device)
        valency_types = bond_orders[self.bond_types.repeat(2)]
        current_valencies = pool.global_add_pool(
            valency_types, edge_index.flatten(), size=nx
        )

        return current_valencies

    def check_stability(self, debug=False):
        valencies = self.calculate_valency()
        n_stable_at = 0
        mol_stable = True
        for i, (atom_type, valency, charge) in enumerate(
            zip(self.node_types, valencies, self.charge)
        ):
            atom_type = atom_type.item()
            valency = valency.item()
            charge = charge.item()
            possible_bonds = allowed_bonds[self.atom_decoder[atom_type]]
            if type(possible_bonds) == int:
                is_stable = possible_bonds == valency
            elif type(possible_bonds) == dict:
                expected_bonds = (
                    possible_bonds[charge]
                    if charge in possible_bonds.keys()
                    else possible_bonds[0]
                )
                is_stable = (
                    expected_bonds == valency
                    if type(expected_bonds) == int
                    else valency in expected_bonds
                )
            else:
                is_stable = valency in possible_bonds
            if not is_stable:
                mol_stable = False
            if not is_stable and debug:
                print(
                    f"Invalid atom {self.atom_decoder[atom_type]}: valency={valency}, charge={charge}"
                )
                print()
            n_stable_at += int(is_stable)

        return mol_stable, n_stable_at, len(self.node_types)

        # TODO: converting to tensor might be necessary
        # return torch.tensor([mol_stable], dtype=torch.float, device=self.device), \
        #        torch.tensor([n_stable_at], dtype=torch.float, device=self.device), \
        #        len(self.node_types)
