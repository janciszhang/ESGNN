"""
export METIS_DLL=/opt/homebrew/opt/metis/lib/libmetis.dylib
"""
import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import time

from ESGNN.get_path import get_all_file_paths
# from ESGNN.metis_partition import partition_K
from base_gnn import GNN


def load_subgraphs(file_prefix, num_subgraphs):
    subgraphs = []
    dir_path = f'subgraph_data/{file_prefix}'

    for i in range(num_subgraphs):
        subgraph = torch.load(f'{dir_path}/subgraph_{num_subgraphs}_{i}.pt')
        subgraphs.append(subgraph)

    return subgraphs

def load_subgraphs(dir_path, num_subgraphs):
    subgraphs = []
    for i in range(num_subgraphs):
        subgraph = torch.load(f'{dir_path}/subgraph_{num_subgraphs}_{i}.pt')
        subgraphs.append(subgraph)
    return subgraphs

def estimate_tasks_gpu(model,subgraphs,is_save=True):
    # 初始化模型和优化器
    # model = GNN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    estimate_info=[]
    estimate_info.append(subgraphs)

    sizes = []
    times = []

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
        model_time = end_time - start_time
        gpu_memory = torch.cuda.max_memory_allocated()
        gpu_memory_MB = gpu_memory / 1024 ** 2

        times.append(model_time)
        sizes.append(gpu_memory_MB)

        print(f"Partition {i + 1}:")
        print(f"Time: {model_time:.4f} seconds")
        print(f"GPU Memory: {gpu_memory_MB:.2f} MB")
        estimate_info.append(f"Partition {i + 1}:")
        estimate_info.append(f"Time: {model_time:.4f} seconds")
        estimate_info.append(f"GPU Memory: {gpu_memory_MB:.2f} MB")

    # 保存
    print(subgraphs)
    print(times)
    print(sizes)

    if is_save:
        # 将输出内容写入文件
        with open('gpu_estimate_result.txt', 'a') as f:
            for line in estimate_info:
                f.write(line + '\n')
            f.write('--------------------------------------\n')
    return times, sizes




if __name__ == '__main__':
    # # 加载Cora数据集
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # name = dataset.__class__.__name__
    # if hasattr(dataset, 'name'):
    #     name = name + '/' + dataset.name

    directory = 'subgraph_data'
    K = 4
    all_pt_folder_paths = get_all_file_paths(directory)
    for pt_folder_path in all_pt_folder_paths:
        print(pt_folder_path)
        subgraphs = load_subgraphs(pt_folder_path, K)
        print(subgraphs)

        # # 获取图数据和标签
        # data = dataset[0]
        # K = 4
        # # subgraphs = partition_K(data, K)
        # subgraphs = load_subgraphs(name, K)
        # print(subgraphs)
        # 初始化模型
        # model = GNN(data.num_node_features, len(torch.unique(data.y))).cuda()
        # times, sizes = estimate_tasks_gpu(model, subgraphs)

        input_dim = subgraphs[0].num_node_features  # 假设每个子图的节点特征数相同
        hidden_dim = 64
        output_dim = subgraphs[0].num_classes  # 假设每个子图的类别数相同

        model = GNN(input_dim, hidden_dim, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)