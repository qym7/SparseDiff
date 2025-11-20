# Sparse denoising diffusion for large graph generation
Forked from the official code for the paper, "Sparse Training of Discrete Diffusion Models for Graph Generation," available [here](https://arxiv.org/abs/2311.02142).
Checkpoints to reproduce the results can be found at [this link](https://drive.switch.ch/index.php/s/1hHNVCb0ylbYPoQ). 
Please refer to the updated version [here](https://arxiv.org/abs/2311.02142).

## Environment installation (Modified from README.md of [SparseDiff](https://github.com/vincenttsai2015/SparseDiff/blob/main/README.md))
This code was tested with PyTorch 2.4.1, cuda 12.1 and torch_geometrics 2.4.0
* Download anaconda/miniconda if needed
* Conda environment building: ```conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9```
* Activate the environment: ```conda activate digress```
* Install graph-tool: ```conda install -c conda-forge graph-tool=2.45```
* Verify the installation:
  * ```python3 -c 'from rdkit import Chem'```
  * ```python3 -c 'import graph_tool as gt'```
* Install the nvcc drivers: ```conda install -c "nvidia/label/cuda-12.1.0" cuda```
* Install Pytorch: ```(python -m) pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121```
* Install PyG related packages: ```(python -m) pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html```
* Install DGL (for SparseDiff): ```conda install -c dglteam/label/th24_cu121 dgl```
* Please ensure the synchronization of the versions of *nvcc drivers, Pytorch, PyG, and DGL*!
* Install the rest packages: ```pip install -r requirements.txt```
* Install mini-moses (optional): ```pip install git+https://github.com/igor-krawczuk/mini-moses```
* Navigate to the directory ```./sparse_diffusion/analysis/orca``` and compile orca.cpp: ```g++ -O2 -std=c++11 -o orca orca.cpp```

## Main execution file usage
* Use config files in folder ```config/experiments```.
* Example command for execution: ```CUDA_VISIBLE_DEVICES=0 python main.py +experiment=ego.yaml```
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 main.py +experiment=debug.yaml`. We advise to try to run the debug mode first
    before launching full experiments.
  - To run a code on only a few batches: `python3 main.py general.name=test`.
  - You can specify the dataset with `python3 main.py dataset=guacamol`. Look at `configs/dataset` for the list of datasets that are currently available
  - You can specify the edge fraction (denoted as $\lambda$ in the paper) with `python3 main.py model.edge_fraction=0.2` to control the GPU-usage

## Cite the paper
```
@misc{qin2023sparse,
      title={Sparse Training of Discrete Diffusion Models for Graph Generation}, 
      author={Yiming Qin and Clement Vignac and Pascal Frossard},
      year={2023},
      eprint={2311.02142},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Troubleshooting 
`PermissionError: [Errno 13] Permission denied: 'SparseDiff/sparse_diffusion/analysis/orca/orca'`: You probably did not compile orca.
