"""
Metis Partition: metis_main(dataset, K, target_ratios=None, is_save=True)

brew install metis
export METIS_DLL=/opt/homebrew/opt/metis/lib/libmetis.dylib

pip install metis
$env METIS_DLL=C:/Program Files/metis_dll_x86-64/metis.dll
$env METIS_DLL

SOURCE_FILE="metis_dll_x86-64/metis.dll"
ls "SOURCE_FILE"
mkdir -p "C:\metis"
METIS_DLL_PATH="C:/metis/metis.dll"
cp "$SOURCE_FILE" "$METIS_DLL_PATH"
ls "$METIS_DLL_PATH"
export METIS_DLL=$METIS_DLL_PATH
echo $METIS_DLL

python metis_partition.py
"""
import os
import time
from math import gcd
from functools import reduce

import dgl
import networkx as nx
import metis
# import pymetis
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import PPI
from torch_geometric.utils import to_networkx, from_networkx
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import TUDataset

from load_data import load_dataset_by_name



def calculate_min_integer_ratios(ratios):
    # 如果输入是小数，则转换为整数比例
    if any(isinstance(ratio, float) for ratio in ratios):
        try:
            # 将小数转换为整数比例
            lcm_value = reduce(lambda x, y: x * y // gcd(x, y), [int(1 / r) for r in ratios])
            integer_ratios = [int(r * lcm_value) for r in ratios]
        except:
            integer_ratios=[int(ratio) for ratio in ratios]
    else:
        integer_ratios = ratios

    # 计算最大公约数
    gcd_value = reduce(gcd, integer_ratios)

    # 将每个比例除以最大公约数
    min_integer_ratios = [ratio // gcd_value for ratio in integer_ratios]

    return min_integer_ratios


def initial_metis_partition(G, num_partitions):
    try:
        _, parts = metis.part_graph(G, nparts=num_partitions)

    except Exception as e:
        print(f"Error with metis: {e}. Trying pymetis.")
        # 如果 metis 失败，使用 pymetis
        try:
            adjacency = nx.to_numpy_array(G).tolist()
            _, parts = pymetis.part_graph(num_partitions, adjacency=adjacency)
        except Exception as e2:
            print(f"Error with pymetis: {e2}. No partitioning performed.")
            parts = None  # 处理失败情况
    return parts


def combine_partitions(parts, target_ratios, num_final_partitions):
    # 计算每个子图的节点数量
    num_partitions = max(parts) + 1
    partition_counts = [0] * num_partitions
    for part in parts:
        partition_counts[part] += 1

    # 目标数量计算
    total_nodes = sum(partition_counts)
    target_counts = [round(ratio / sum(target_ratios) * total_nodes) for ratio in target_ratios]

    # 创建新的分区
    new_parts = [-1] * total_nodes
    current_partition = 0
    count_in_current_partition = 0

    for i in range(num_partitions):
        for _ in range(partition_counts[i]):
            if count_in_current_partition < target_counts[current_partition]:
                new_parts[sum(partition_counts[:i]) + _] = current_partition
                count_in_current_partition += 1
            else:
                current_partition += 1
                count_in_current_partition = 1
                new_parts[sum(partition_counts[:i]) + _] = current_partition

            if current_partition >= num_final_partitions:
                break

        if current_partition >= num_final_partitions:
            break

    return new_parts


def metis_partition(G, num_partitions, target_ratios=None):
    if target_ratios is None:
        # 如果没有传入target_ratios，默认按等比例分配
        parts = initial_metis_partition(G, num_partitions)
    else:
        # 目标数量计算
        target_integer_ratios = calculate_min_integer_ratios(target_ratios)
        # print(target_integer_ratios)
        num_initial_partitions = len(target_integer_ratios)
        initial_parts = initial_metis_partition(G, num_initial_partitions)
        if initial_parts is None:
            return None  # 处理失败情况
        # 合并分区以符合目标比例
        parts = combine_partitions(initial_parts, target_integer_ratios, num_partitions)
    return parts



def partition_K(data, K, target_ratios=None):
    # 将图数据转换为NetworkX图
    G = to_networkx(data, to_undirected=True)
    subgraphs = []
    if K == 1:
        subgraphs = [data]
    else:
        membership = metis_partition(G, K, target_ratios=target_ratios)
        # 创建子图
        try:
            for i in range(K):
                nodes = [n for n in range(len(membership)) if membership[n] == i]
                subgraph_nodes = torch.tensor(nodes, dtype=torch.long)
                subgraph = data.subgraph(subgraph_nodes)
                subgraphs.append(subgraph)
        except Exception as e:
            print(f"Error with partition process: {e}")
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
    if file_prefix.startswith('subgraph_data'):
        dir_path = f'{file_prefix}'
    else:
        dir_path = f'subgraph_data/{file_prefix}'

    for i in range(num_subgraphs):
        subgraph = torch.load(f'{dir_path}/subgraph_{num_subgraphs}_{i}.pt')
        subgraphs.append(subgraph)

    return subgraphs


def metis_main(dataset, K, target_ratios=None, is_save=False):
    name = dataset.__class__.__name__
    if hasattr(dataset, 'name'):
        name = name + '/' + dataset.name

    # 获取图数据和标签
    data = dataset[0]
    # K = 4
    print(name)
    subgraphs = partition_K(data, K, target_ratios=target_ratios)
    for subgraph in subgraphs:
        print(subgraph)
    if is_save:
        save_subgraphs(subgraphs, name)
        subgraphs = load_subgraphs(name, K)


if __name__ == '__main__':
    calculate_min_integer_ratios([400, 46.40625])
    dataset = load_dataset_by_name('ogbn-proteins')
    data = dataset[0]
    start_time=time.time()
    # G = to_networkx(data, to_undirected=True)
    # # 将 PyTorch Geometric 的图数据直接转换为 DGL 图
    # G_dgl = dgl.graph((data.edge_index[0], data.edge_index[1]))
    # # 如果需要从 DGL 图转换回 PyTorch Geometric 格式
    # data_from_dgl = from_networkx(G_dgl.to_networkx())
    G_dgl = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes) # 0.14s
    # data_from_dgl = from_networkx(G_dgl.to_networkx()) #MemoryError
    num_partitions = 4
    from dgl import metis_partition
    partitioned_graphs = metis_partition(G_dgl, num_partitions)

    # partitioned_graphs 是一个包含多个子图的字典
    for part_id, subgraph in partitioned_graphs.items():
        print(f"Subgraph {part_id} has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")

    print(f'Time: {time.time() - start_time}') # 91s
    # metis_main(dataset=dataset, K=10)
    # 对dataset进行分割，并保存子图数据在对应路径
    # K_values = [1,2,4,8,16]
    # ratios = None
    # # ratios = [2, 4, 6, 8]
    # is_save = False
    # for K_value in K_values:
    #     # metis_main(dataset=Planetoid(root='/tmp/Cora', name='Cora'), K=K_value, target_ratios=ratios, is_save=is_save)
    #     # metis_main(dataset=Planetoid(root='/tmp/Citeseer', name='Citeseer'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     # metis_main(dataset=Planetoid(root='/tmp/Pubmed', name='Pubmed'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     # metis_main(dataset=Reddit(root='/tmp/Reddit'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     # metis_main(dataset=PPI(root='/tmp/PPI'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     # metis_main(dataset=Flickr(root='/tmp/Flickr'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     # metis_main(dataset=Amazon(root='/tmp/Amazon', name='Computers'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     # metis_main(dataset=Amazon(root='/tmp/Amazon', name='Photo'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     # metis_main(dataset=TUDataset(root='/tmp/TUDataset', name='PROTEINS'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     # metis_main(dataset=TUDataset(root='/tmp/TUDataset', nasme='ENZYMES'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     # metis_main(dataset=TUDataset(root='/tmp/TUDataset', name='IMDB-BINARY'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     # metis_main(dataset=PygNodePropPredDataset(name='ogbn-products'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     metis_main(dataset=PygNodePropPredDataset(name='ogbn-proteins'), K=K_value,target_ratios=ratios,is_save=is_save)
    #     metis_main(dataset=PygNodePropPredDataset(name='ogbn-arxiv'), K=K_value,target_ratios=ratios,is_save=is_save)
