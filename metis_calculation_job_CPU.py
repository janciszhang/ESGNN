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
from ESGNN11.metis_partition import partition_K
from base_gnn import GNN



def estimate_task_cpu(model,task_graph):
    # 开始计时
    start_time = time.time()
    # 前向传播
    model.eval()
    with torch.no_grad():
        out = model(task_graph)
    # 结束计时
    end_time = time.time()
    model_time = end_time - start_time
    # 计算内存使用情况（这里简化为模型参数的大小）
    model_size = sum(p.numel() for p in model.parameters())
    return model_size, model_time

def estimate_tasks_cpu(model,subgraphs):
    # 初始化模型
    # model = GNN()
    # 评估每个子任务的内存大小和预计计算时间
    sizes=[]
    times=[]
    for i, subgraph in enumerate(subgraphs):
        model_size, model_time = estimate_task_cpu(model,subgraph)

        times.append(model_time)
        sizes.append(model_size)

        print(f"Partition {i + 1}:")
        print(f"Time: {model_time:.4f} seconds")
        print(f"Model Size: {model_size} parameters")
        print()

    # save
    print(subgraphs)
    print(times)
    print(sizes)
    return times, sizes

if __name__ == '__main__':
    # 加载Cora数据集
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # 获取图数据和标签
    data = dataset[0]
    subgraphs = partition_K(data, K=4)
    # 初始化模型
    model = GNN(dataset)
    times, sizes = estimate_tasks_cpu(model,subgraphs)
