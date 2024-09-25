"""
Metis Partition
export METIS_DLL=/opt/homebrew/opt/metis/lib/libmetis.dylib
export METIS_DLL=/opt/home/s4069853/lib/metis/libmetis.dylib
mkdir -p /opt/home/s4069853/lib/metis
pip install metis==0.2a5 --target /opt/home/s4069853/lib/metis
python metis_calculation_job_GPU.py
"""
import os
import sys
import time
from random import random

import networkx as nx
import metis
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import PPI
from torch_geometric.utils import to_networkx
# from torch_geometric.datasets import OGB
from torch_geometric.datasets import OGB_MAG
from ogb.nodeproppred import NodePropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import TUDataset
import csv
import pandas as pd


# 定义Metis分割函数
def metis_partition(G, num_partitions):
    _, parts = metis.part_graph(G, nparts=num_partitions)
    return parts


def partition_K(data, K):
    # 将图数据转换为NetworkX图
    G = to_networkx(data, to_undirected=True)
    subgraphs = []
    if K == 1:
        subgraphs = [data]
    else:
        membership = metis_partition(G, K)
        # 创建子图
        for i in range(K):
            nodes = [n for n in range(len(membership)) if membership[n] == i]
            subgraph_nodes = torch.tensor(nodes, dtype=torch.long)
            subgraph = data.subgraph(subgraph_nodes)
            subgraphs.append(subgraph)
    return subgraphs


def save_subgraphs(subgraphs, file_prefix):
    # Create the directory if it doesn't exist
    dir_path = f'subgraph_data/{file_prefix}'
    os.makedirs(dir_path, exist_ok=True)

    # Save each subgraph
    K = len(subgraphs)
    for i, subgraph in enumerate(subgraphs):
        torch.save(subgraph, f'{dir_path}/subgraph_{K}_{i}.pt')


def load_subgraphs(file_prefix, num_subgraphs):
    subgraphs = []
    dir_path = f'subgraph_data/{file_prefix}'

    for i in range(num_subgraphs):
        subgraph = torch.load(f'{dir_path}/subgraph_{num_subgraphs}_{i}.pt')
        subgraphs.append(subgraph)

    return subgraphs


def metis_main(dataset, K):
    name = dataset.__class__.__name__
    if hasattr(dataset, 'name'):
        name = name + '/' + dataset.name

    # 获取图数据和标签
    data = dataset[0]
    # K = 4
    subgraphs = partition_K(data, K)
    for subgraph in subgraphs:
        print(subgraph)
    save_subgraphs(subgraphs, name)
    subgraphs = load_subgraphs(name, K)


if __name__ == '__main__':
    # 对dataset进行分割，并保存子图数据在对应路径
    K_values = [8,16]
    for K_value in K_values:
        # metis_main(dataset=Planetoid(root='/tmp/Cora', name='Cora'), K=K_value)
        # metis_main(dataset=Planetoid(root='/tmp/Citeseer', name='Citeseer'), K=K_value)
        # metis_main(dataset=Planetoid(root='/tmp/Pubmed', name='Pubmed'), K=K_value)
        metis_main(dataset=Reddit(root='/tmp/Reddit'), K=K_value)
        # metis_main(dataset=PPI(root='/tmp/PPI'), K=K_value)
        # metis_main(dataset=Flickr(root='/tmp/Flickr'), K=K_value)
        # metis_main(dataset=Amazon(root='/tmp/Amazon', name='Computers'), K=K_value)
        # metis_main(dataset=Amazon(root='/tmp/Amazon', name='Photo'), K=K_value)
        # metis_main(dataset=TUDataset(root='/tmp/TUDataset', name='PROTEINS'), K=K_value)
        # metis_main(dataset=TUDataset(root='/tmp/TUDataset', name='ENZYMES'), K=K_value)
        # metis_main(dataset=TUDataset(root='/tmp/TUDataset', name='IMDB-BINARY'), K=K_value)

        # metis_main(dataset=PygNodePropPredDataset(name='ogbn-products'), K=K_value)
        # metis_main(dataset=PygNodePropPredDataset(name='ogbn-proteins'), K=K_value)
        # metis_main(dataset=PygNodePropPredDataset(name='ogbn-arxiv'), K=K_value)
