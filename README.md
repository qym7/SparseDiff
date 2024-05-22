# `Sparse denoising diffusion for large graph generation`

Official code of [Sparse Training of Discrete Diffusion Models for Graph Generation]

Note (November 7th): the code is currently incomplete as we running experiments again with a slightly improved architecture. Checkpoints will be added progressively when they are available.

## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n sparse rdkit=2023.03.2 python=3.9```
  - `conda activate sparse`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install graph-tool (https://graph-tool.skewed.de/):  
    
    ```conda install -c conda-forge graph-tool=2.45```
  - Check that this line does not return an error:
    
    ```python3 -c 'import graph_tool as gt' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```
  - Install mini-moses: 
    
    ```pip install git+https://github.com/igor-krawczuk/mini-moses```
  - Run:
    
    ```pip install -e .```

  - Navigate to the ./sparse_diffusion/analysis/orca directory and compile orca.cpp: 
    
     ```g++ -O2 -std=c++11 -o orca orca.cpp```


## Run the code
  
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 main.py +experiment=debug.yaml`. We advise to try to run the debug mode first
    before launching full experiments.
  - To run a code on only a few batches: `python3 main.py general.name=test`.
  - You can specify the dataset with `python3 main.py dataset=guacamol`. Look at `configs/dataset` for the list of datasets that are currently available
  - You can specify the edge fraction (denoted as $\lambda$ in the paper) with `python3 main.py model.edge_fraction=0.2` to control the GPU-usage


<!-- If you have retrained a model from scratch for which the samples are not available yet, we would be very happy if you could send them to us! -->

## Troubleshooting 

`PermissionError: [Errno 13] Permission denied: 'SparseDiff/sparse_diffusion/analysis/orca/orca'`: You probably did not compile orca.
