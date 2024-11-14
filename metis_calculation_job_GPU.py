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
import os
# from get_path import get_all_file_paths
# from metis_partition import partition_K
# from base_gnn import GNN


# def load_subgraphs(file_prefix, num_subgraphs):
#     subgraphs = []
#     dir_path = f'subgraph_data/{file_prefix}'
#     for i in range(num_subgraphs):
#         subgraph = torch.load(f'{dir_path}/subgraph_{num_subgraphs}_{i}.pt')
#         subgraphs.append(subgraph)
#     return subgraphs

def get_all_file_paths(directory):
    pt_file_paths = []
    pt_folder_paths = set()  # 使用 set 防止重复
    # os.walk 会递归遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pt'):  # 检查文件是否以 .pt 结尾
                # 获取每个文件的完整路径
                pt_file_paths.append(os.path.join(root, file))
                pt_folder_paths.add(root)  # 只保存文件夹路径
    return pt_folder_paths

def load_subgraphs(dir_path, num_subgraphs):
    subgraphs = []
    for i in range(num_subgraphs):
        subgraph = torch.load(f'{dir_path}/subgraph_{num_subgraphs}_{i}.pt')
        subgraphs.append(subgraph)
    return subgraphs


def measure_task_gpu(model, task_graph,optimizer,loss_fn):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    model.to(device)
    task_graph = task_graph.to(device)
    
    torch.cuda.synchronize()
    # create CUDA event
    start_event=torch.cuda.Event(enable_timing=True)
    end_event=torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    # 开始计时
    start_time = time.time()
    
    
    model.train()
    optimizer.zero_grad()
    output=model(task_graph)
    loss=loss_fn(output,task_graph.y)
    
    loss.backward()
    optimizer.step()
    
    # 结束计时
    end_time = time.time()
    model_time = end_time - start_time

    # 计算内存使用情况（以MB为单位）
    gpu_memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    gpu_memory_MB = gpu_memory_bytes / 1024 ** 2

    return gpu_memory_MB, model_time

def estimate_task_gpu(model, task_graph):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    model.to(device)
    task_graph = task_graph.to(device)
    
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
    gpu_memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    gpu_memory_MB = gpu_memory_bytes / 1024 ** 2

    return gpu_memory_MB, model_time


def estimate_tasks_gpu(model,subgraphs,is_save=True):
    # 初始化模型和优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    estimate_info=[]
    # estimate_info.append(subgraphs)

    sizes = []
    times = []

    # 评估每个子任务的GPU资源和运行时间
    for i, subgraph in enumerate(subgraphs):
        # # 将子任务数据移到GPU
        # subgraph = subgraph.cuda()

        # # 开始计时和内存跟踪
        # start_time = time.time()
        # torch.cuda.reset_peak_memory_stats()

        # # 前向传播
        # model.train()
        # optimizer.zero_grad()
        # # out = model(subgraph)
        # # loss = criterion(out[subgraph.train_mask], subgraph.y[subgraph.train_mask])
        # # 计算loss时不使用train_mask
        # out = model(subgraph)
        # loss = criterion(out, subgraph.y)

        # # 反向传播
        # loss.backward()
        # optimizer.step()

        # # 记录时间和内存
        # end_time = time.time()
        # model_time = end_time - start_time
        # gpu_memory = torch.cuda.max_memory_allocated()
        # gpu_memory_MB = gpu_memory / 1024 ** 2
        
        gpu_memory_MB, model_time = estimate_task_gpu(model, subgraph)

        times.append(model_time)
        sizes.append(gpu_memory_MB)

        print(f"Partition {i + 1}:")
        print(f"Time: {model_time:.4f} seconds")
        print(f"GPU Memory: {gpu_memory_MB:.2f} MB")
        estimate_info.append(f"Partition {i + 1}:")
        estimate_info.append(f"Time: {model_time:.4f} seconds")
        estimate_info.append(f"GPU Memory: {gpu_memory_MB:.2f} MB")

    # 保存
    # print(subgraphs)
    # print(times)
    # print(sizes)

    if is_save:
        # 将输出内容写入文件
        with open('gpu_estimate_result.txt', 'a') as f:
            for line in estimate_info:
                line = "".join(map(str, line)) 
                f.write(line + '\n')
            f.write('--------------------------------------\n')
    return times, sizes





if __name__ == '__main__':
    directory = 'subgraph_data'
    all_pt_folder_paths = get_all_file_paths(directory)
    print(all_pt_folder_paths)
    
    # for K in [1,2,4,8,16]:
    #     n=1
    #     m=0
    #     for pt_folder_path in all_pt_folder_paths:
    #         print(pt_folder_path)
    #         subgraphs = load_subgraphs(pt_folder_path, K)
    #         # print(subgraphs)
    #         # 初始化模型
    #         try:
    #             input_dim = subgraphs[0].x.size(1)  # Feature dimension from the first subgraph
    #             output_dim = len(torch.unique(subgraphs[0].y))  # Number of classes based on the labels in the first subgraph
    #             model = GNN(input_dim, output_dim).cuda()
    #             with open('gpu_estimate_result.txt', 'a') as f:
    #                 f.write(pt_folder_path+ '\n')
    #             times, sizes = estimate_tasks_gpu(model, subgraphs,is_save=True)
    #             print(n)
    #             n=n+1
    #         except:
    #             print(subgraphs)
    #             m=m+1
    #             print(f'Unsuccess {f}')
            


        