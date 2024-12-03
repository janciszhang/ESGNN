"""
nvidia - smi
nvcc --version
python --version
pip show torch
# cuda 11.8, python 3.10.6
torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
dgl==1.1.1+cu118

"""
"""
# CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

"""
"""
pip uninstall torch torchvision torchaudio
pip uninstall dgl
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118 --user
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --user
C:\Python312\python.exe -m pip install --upgrade pip setuptools --user
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118 --user
pip install dgl==1.1.1+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html --user
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 --user
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install pytorch_lightning
pip install yacs
"""


import torch
import torchvision
import torchaudio
import subprocess
import os
import gc


from torch_geometric.datasets import Planetoid


# from base_gnn import GNN
from load_data import get_data_info



def get_cuda_info():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    print(f'CUDA is available: {torch.cuda.is_available()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'Current device index: {torch.cuda.current_device()}')
    print(f'GPU total memory: {torch.cuda.get_device_properties(0).total_memory/(1024**3)} G')
    print(f'GPU allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB')
    print(f'GPU reserved: {torch.cuda.memory_reserved(0)/(1024**2):.2f} MB')


def get_gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return int(result.stdout.split()[0])


def set_gpu_memory(limit_gpu_size):
    original_total_gpu = torch.cuda.get_device_properties(0).total_memory/ (1024 ** 2) # MB
    torch.cuda.set_per_process_memory_fraction(limit_gpu_size/original_total_gpu, device=0)


def clean_gpu():
    # 清除缓存
    torch.cuda.empty_cache()

    # 删除未使用的变量
    # del variable_name  # 用已定义的变量替换

    # 强制进行垃圾回收
    gc.collect()

if __name__ == '__main__':
    get_cuda_info()

    # set_gpu_memory(18)
    # print(f'GPU allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB')
    # print(f'GPU reserved: {torch.cuda.memory_reserved(0)/(1024**2):.2f} MB')
    #
    #
    #
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # data = dataset[0]
    # input_dim = dataset[0].x.size(1)  # Feature dimension from the first subgraph
    # output_dim = len(torch.unique(dataset[0].y))  # Number of classes based on the labels in the first subgraph
    # model = GNN(input_dim, output_dim)
    #
    # print(data)
    # get_data_info(dataset)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    #
    # # Move model and data to the chosen device (GPU or CPU)
    # data = data.to(device)
    # model = model.to(device)
    # print(f'GPU allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB')
    # print(f'GPU reserved: {torch.cuda.memory_reserved(0)/(1024**2):.2f} MB')
    #
    # clean_gpu()
    # del data
    # # data.to("cpu")
    # # model = model.to("cpu")
    #
    # print(f'GPU allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB')
    # print(f'GPU reserved: {torch.cuda.memory_reserved(0) / (1024 ** 2):.2f} MB')





