"""
export METIS_DLL=/opt/homebrew/opt/metis/lib/libmetis.dylib
"""
import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import metis
import networkx as nx
import time
# from ESGNN.metis_partition import partition_K
from base_gnn import GNN


def load_subgraphs(file_prefix, num_subgraphs):
    subgraphs = []
    for i in range(num_subgraphs):
        subgraph = torch.load(f'{file_prefix}_subgraph_{i}.pt')
        subgraphs.append(subgraph)
    return subgraphs

def estimate_tasks_gpu(model,subgraphs):
    # 初始化模型和优化器
    # model = GNN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # 评估每个子任务的GPU资源和运行时间
    for i, subgraph in enumerate(subgraphs):
        # 将子任务数据移到GPU
        subgraph = subgraph.cuda()

        # 开始计时和内存跟踪
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

        # 前向传播
        model.train()
        optimizer.zero_grad()
        out = model(subgraph)
        loss = criterion(out[subgraph.train_mask], subgraph.y[subgraph.train_mask])

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录时间和内存
        end_time = time.time()
        gpu_memory = torch.cuda.max_memory_allocated()

        print(f"Partition {i + 1}:")
        print(f"Time: {end_time - start_time:.4f} seconds")
        print(f"GPU Memory: {gpu_memory / 1024 ** 2:.2f} MB")




if __name__ == '__main__':
    # 加载Cora数据集
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # 获取图数据和标签
    data = dataset[0]
    # subgraphs = partition_K(data, K=4)
    subgraphs = load_subgraphs('subgraph', num_subgraphs=4)
    print(subgraphs)
    # 初始化模型
    model = GNN(dataset).cuda()
    times, sizes = estimate_tasks_gpu(model, subgraphs)