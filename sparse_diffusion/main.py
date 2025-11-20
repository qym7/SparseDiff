import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pathlib

# graph_tool need to be imported before torch
try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    print("Graph tool not found.")
import torch
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from diffusion_model_sparse import DiscreteDenoisingDiffusion
from metrics.molecular_metrics import TrainMolecularMetricsDiscrete
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures
from metrics.sampling_metrics import SamplingMetrics

# debug for multi-gpu
import resource

resource.setrlimit(
    resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    pl.seed_everything(cfg.train.seed)

    print("creating datasets")
    if dataset_config["name"] in ["sbm", "comm20", "planar", "ego"]:
        from datasets.spectre_dataset_pyg import (
            SBMDataModule,
            Comm20DataModule,
            EgoDataModule,
            PlanarDataModule,
            SpectreDatasetInfos,
        )

        if dataset_config["name"] == "sbm":
            datamodule = SBMDataModule(cfg)
        elif dataset_config["name"] == "comm20":
            datamodule = Comm20DataModule(cfg)
        elif dataset_config["name"] == "ego":
            datamodule = EgoDataModule(cfg)
        else:
            datamodule = PlanarDataModule(cfg)

        dataset_infos = SpectreDatasetInfos(datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        domain_features = DummyExtraFeatures()
        dataloaders = datamodule.dataloaders

    elif dataset_config["name"] == "protein":
        from datasets import protein_dataset

        datamodule = protein_dataset.ProteinDataModule(cfg)
        dataset_infos = protein_dataset.ProteinInfos(datamodule=datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        domain_features = DummyExtraFeatures()
        dataloaders = datamodule.dataloaders

    elif dataset_config["name"] == "point_cloud":
        from datasets import point_cloud_dataset

        datamodule = point_cloud_dataset.PointCloudDataModule(cfg)
        dataset_infos = point_cloud_dataset.PointCloudInfos(datamodule=datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        domain_features = DummyExtraFeatures()
        dataloaders = datamodule.dataloaders

    elif dataset_config["name"] in ["qm9", "guacamol", "moses"]:
        if dataset_config["name"] == "qm9":
            from datasets import qm9_dataset

            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9Infos(datamodule=datamodule, cfg=cfg)

        elif dataset_config["name"] == "guacamol":
            from datasets import guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.GuacamolInfos(datamodule, cfg)

        elif dataset_config.name == "moses":
            from datasets import moses_dataset

            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MosesInfos(datamodule, cfg)
        else:
            raise ValueError("Dataset not implemented")

        dataloaders = None

        if cfg.model.extra_features is not None:
            # domain_features = DummyExtraFeatures()
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            domain_features = DummyExtraFeatures()

        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    elif dataset_config["name"] in ["wiki-vote", "my_random_data"]:
        from datasets.pyg_real_dataset import (RealGraphDataModule, RealDatasetInfos)
        datamodule = RealGraphDataModule(cfg)
        dataset_infos = RealDatasetInfos(datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        domain_features = DummyExtraFeatures()
        dataloaders = datamodule.dataloaders
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    ef = cfg.model.extra_features
    edge_f = cfg.model.edge_features
    extra_features = (
        ExtraFeatures(
            eigenfeatures=cfg.model.eigenfeatures,
            edge_features_type=edge_f,
            dataset_info=dataset_infos,
            num_eigenvectors=cfg.model.num_eigenvectors,
            num_eigenvalues=cfg.model.num_eigenvalues,
            num_degree=cfg.model.num_degree,
            dist_feat=cfg.model.dist_feat,
            use_positional=cfg.model.positional_encoding,
        )
        if ef is not None
        else DummyExtraFeatures()
    )

    dataset_infos.compute_input_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    val_sampling_metrics = SamplingMetrics(
        dataset_infos, test=False, dataloaders=dataloaders
    )
    test_sampling_metrics = SamplingMetrics(
        dataset_infos, test=True, dataloaders=dataloaders
    )
    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "val_sampling_metrics": val_sampling_metrics,
        "test_sampling_metrics": test_sampling_metrics,
    }

    utils.create_folders(cfg)

    print("creating model")
    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="{epoch}",
            monitor="val/epoch_NLL",
            save_top_k=15,
            mode="min",
            every_n_epochs=1,
            save_last=True,
        )
        last_ckpt_save = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="last",
            every_n_epochs=1,
        )
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    name = cfg.general.name
    if name == "debug":
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = pl.Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp",
        # strategy="ddp_find_unused_parameters_true",
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        val_check_interval=cfg.general.val_check_interval,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == "debug",
        callbacks=callbacks,
        log_every_n_steps=50 if name != "debug" else 1,
        enable_progress_bar=False,
        logger=[],
    )

    if not cfg.general.test_only and not cfg.general.generated_path:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name != "debug":
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        pl.seed_everything(1000)
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if ".ckpt" in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    utils.setup_wandb(cfg)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
