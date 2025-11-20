import inspect
import os, sys
import pathlib
import os.path as osp
RootPath = pathlib.Path(osp.realpath(__file__)).parents[1]
sys.path.append(f'{RootPath}')

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from hydra.utils import get_original_cwd

from utils import PlaceHolder
from datasets.abstract_dataset import (
    MolecularDataModule,
    AbstractDatasetInfos,
)
from datasets.dataset_utils import (
    load_pickle,
    save_pickle,
    mol_to_torch_geometric,
    Statistics,
    remove_hydrogens,
    to_list,
    files_exist,
)
from metrics.metrics_utils import compute_all_statistics


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectMuTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data


class SelectHOMOTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data


atom_encoder = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
atom_decoder = [key for key in atom_encoder.keys()]


class QM9Dataset(InMemoryDataset):
    raw_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    processed_url = "https://data.pyg.org/datasets/qm9_v3.zip"

    def __init__(
        self,
        split,
        root,
        remove_h: bool,
        target_prop=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        self.target_prop = target_prop

        self.atom_encoder = atom_encoder
        if remove_h:
            self.atom_encoder = {
                k: v - 1 for k, v in self.atom_encoder.items() if k != "H"
            }

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], **({"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}))

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])).float(),
            valencies=load_pickle(self.processed_paths[5]),
        )
        self.smiles = load_pickle(self.processed_paths[6])

    @property
    def raw_file_names(self):
        return ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]

    @property
    def split_file_name(self):
        return ["train.csv", "val.csv", "test.csv"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        h = "noh" if self.remove_h else "h"
        if self.split == "train":
            return [
                f"train_{h}.pt",
                f"train_n_{h}.pickle",
                f"train_node_types_{h}.npy",
                f"train_bond_types_{h}.npy",
                f"train_charge_{h}.npy",
                f"train_valency_{h}.pickle",
                "train_smiles.pickle",
            ]
        elif self.split == "val":
            return [
                f"val_{h}.pt",
                f"val_n_{h}.pickle",
                f"val_node_types_{h}.npy",
                f"val_bond_types_{h}.npy",
                f"val_charge_{h}.npy",
                f"val_valency_{h}.pickle",
                f"val_smiles_{h}.pickle",
            ]
        else:
            return [
                f"test_{h}.pt",
                f"test_n_{h}.pickle",
                f"test_node_types_{h}.npy",
                f"test_bond_types_{h}.npy",
                f"test_charge_{h}.npy",
                f"test_valency_{h}.pickle",
                f"test_smiles_{h}.pickle",
            ]

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa

            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)
            _ = download_url(self.raw_url2, self.raw_dir)
            os.rename(
                osp.join(self.raw_dir, "3195404"),
                osp.join(self.raw_dir, "uncharacterized.txt"),
            )
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(
            dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train]
        )

        train.to_csv(os.path.join(self.raw_dir, "train.csv"))
        val.to_csv(os.path.join(self.raw_dir, "val.csv"))
        test.to_csv(os.path.join(self.raw_dir, "test.csv"))

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        target_df.drop(columns=["mol_id"], inplace=True)

        with open(self.raw_paths[-1], "r") as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        data_list = []
        all_smiles = []
        num_errors = 0
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            if smiles is None:
                num_errors += 1
            else:
                all_smiles.append(smiles)

            data = mol_to_torch_geometric(mol, atom_encoder, smiles)
            if self.remove_h:
                data = remove_hydrogens(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if data.edge_index.numel() > 0:
                data_list.append(data)

        statistics = compute_all_statistics(
            data_list, self.atom_encoder, charge_dic={-1: 0, 0: 1, 1: 2}
        )
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.node_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(set(all_smiles), self.processed_paths[6])
        torch.save(self.collate(data_list), self.processed_paths[0])
        print("Number of molecules that could not be mapped to smiles: ", num_errors)


class QM9DataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)

        target = getattr(cfg.general, "guidance_target", None)
        regressor = getattr(self, "regressor", None)
        if regressor and target == "mu":
            transform = SelectMuTransform()
        elif regressor and target == "homo":
            transform = SelectHOMOTransform()
        elif regressor and target == "both":
            transform = None
        else:
            transform = RemoveYTransform()

        self.remove_h = cfg.dataset.remove_h
        datasets = {
            "train": QM9Dataset(
                split="train",
                root=root_path,
                remove_h=self.cfg.dataset.remove_h,
                target_prop=target,
                transform=RemoveYTransform(),
            ),
            "val": QM9Dataset(
                split="val",
                root=root_path,
                remove_h=self.cfg.dataset.remove_h,
                target_prop=target,
                transform=RemoveYTransform(),
            ),
            "test": QM9Dataset(
                split="test",
                root=root_path,
                remove_h=self.cfg.dataset.remove_h,
                target_prop=target,
                transform=transform,
            ),
        }

        self.statistics = {
            "train": datasets["train"].statistics,
            "val": datasets["val"].statistics,
            "test": datasets["test"].statistics,
        }
        super().__init__(cfg, datasets)


class QM9Infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        # basic settings
        self.name = "qm9"
        self.is_molecular = True
        self.remove_h = cfg.dataset.remove_h
        self.use_charge = cfg.model.use_charge
        self.collapse_charges = torch.Tensor([-1, 0, 1]).int()

        # statistics
        self.statistics = datamodule.statistics
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.collapse_charge = torch.Tensor([-1, 0, 1]).int()
        self.train_smiles = datamodule.train_dataset.smiles
        self.val_smiles = datamodule.val_dataset.smiles
        self.test_smiles = datamodule.test_dataset.smiles
        if self.remove_h:
            self.atom_encoder = {
                k: v - 1 for k, v in self.atom_encoder.items() if k != "H"
            }
            self.atom_decoder = [key for key in self.atom_encoder.keys()]
        super().complete_infos(datamodule.statistics, self.atom_encoder)

        # dimensions settings
        self.output_dims = PlaceHolder(X=self.num_node_types, charge=self.num_charge_types, E=5, y=0)
        if not self.use_charge:
            self.output_dims.charge = 0

        # special settings
        # atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        self.valencies = [4, 3, 2, 1] if self.remove_h else [1, 4, 3, 2, 1]
        self.atom_weights = [12, 14, 16, 19] if self.remove_h else [1, 12, 14, 16, 19]
        self.max_weight = 40 * 19  # Quite arbitrary
