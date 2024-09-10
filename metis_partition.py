"""
Metis Partition
export METIS_DLL=/opt/homebrew/opt/metis/lib/libmetis.dylib

"""
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
import csv
import pandas as pd


# 定义Metis分割函数
def metis_partition(G, num_partitions):
    _, parts = metis.part_graph(G, nparts=num_partitions)
    return parts

def partition_K(data,K):
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


if __name__ == '__main__':
    # 加载Cora数据集
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # 获取图数据和标签
    data = dataset[0]
    subgraphs=partition_K(data,K=4)
    for subgraph in subgraphs:
        print(subgraph)



