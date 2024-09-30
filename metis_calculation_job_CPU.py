"""
export METIS_DLL=/opt/homebrew/opt/metis/lib/libmetis.dylib
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import metis
import networkx as nx
import time

from get_path import get_all_file_paths
from metis_partition import partition_K
from base_gnn import GNN


# def load_subgraphs(file_prefix, num_subgraphs):
#     subgraphs = []
#     for i in range(num_subgraphs):
#         subgraph = torch.load(f'{file_prefix}_subgraph_{i}.pt')
#         subgraphs.append(subgraph)
#     return subgraphs
def load_subgraphs(dir_path, num_subgraphs):
    subgraphs = []
    for i in range(num_subgraphs):
        subgraph = torch.load(f'{dir_path}/subgraph_{num_subgraphs}_{i}.pt')
        subgraphs.append(subgraph)
    return subgraphs

def estimate_task_cpu(model, task_graph):
    # 开始计时
    start_time = time.time()
    # 前向传播
    model.eval()
    with torch.no_grad():
        out = model(task_graph)
    # 结束计时
    end_time = time.time()
    model_time = end_time - start_time

    # 计算内存使用情况（以MB为单位）
    cpu_memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    cpu_memory_MB = cpu_memory_bytes / 1024 ** 2

    return cpu_memory_MB, model_time


def estimate_tasks_cpu(model, subgraphs,is_save=True):
    sizes = []
    times = []
    estimate_info=[]
    # estimate_info.append(subgraphs)
    
    for i, subgraph in enumerate(subgraphs):
        cpu_memory_MB, model_time = estimate_task_cpu(model, subgraph)

        times.append(model_time)
        sizes.append(cpu_memory_MB)

        print(f"Partition {i + 1}:")
        print(f"Time: {model_time:.4f} seconds")
        print(f"CPU Memory: {cpu_memory_MB:.4f} MB")
        estimate_info.append(f"Partition {i + 1}:")
        estimate_info.append(f"Time: {model_time:.4f} seconds")
        estimate_info.append(f"CPU Memory: {cpu_memory_MB:.2f} MB")

    # 保存
    # print(subgraphs)
    # print(times)
    # print(sizes)
    if is_save:
        # 将输出内容写入文件
        with open('cpu_estimate_result.txt', 'a') as f:
            for line in estimate_info:
                line = "".join(map(str, line)) 
                f.write(line + '\n')
            f.write('--------------------------------------\n')
    return times, sizes


if __name__ == '__main__':
    # # 加载Cora数据集
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # # 获取图数据和标签
    # data = dataset[0]
    # # subgraphs = partition_K(data, K=4)
    # subgraphs = load_subgraphs('subgraph', num_subgraphs=4)
    # # 初始化模型
    # model = GNN(data.num_node_features, len(torch.unique(data.y)))
    # times, sizes = estimate_tasks_cpu(model,subgraphs)
    directory = 'subgraph_data'
    all_pt_folder_paths = get_all_file_paths(directory)
    print(all_pt_folder_paths)
    
    for K in [1,2,4,8,16]:
        n=1
        m=0
        for pt_folder_path in all_pt_folder_paths:
            print(pt_folder_path)
            subgraphs = load_subgraphs(pt_folder_path, K)
            # print(subgraphs)
            # 初始化模型
            try:
                input_dim = subgraphs[0].x.size(1)  # Feature dimension from the first subgraph
                output_dim = len(torch.unique(subgraphs[0].y))  # Number of classes based on the labels in the first subgraph
                model = GNN(input_dim, output_dim)
                with open('cpu_estimate_result.txt', 'a') as f:
                    f.write(pt_folder_path+ '\n')
                times, sizes = estimate_tasks_cpu(model, subgraphs)
                print(n)
                n=n+1
            except:
                print(subgraphs)
                m=m+1
                print(m)
            



