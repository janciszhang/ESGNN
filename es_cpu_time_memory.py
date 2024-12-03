import tracemalloc

import torch
import time
import torch
import time
import psutil
import os
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Reddit, TUDataset, Amazon, Flickr, PPI
from torch_geometric.utils import subgraph

from base_gnn import GNN, train, split_dataset, train_each_epoch, train_model
from torch_geometric.loader import DataLoader, NeighborLoader, GraphSAINTNodeSampler, ClusterData, ClusterLoader, \
    NeighborSampler

from load_data import get_data_info, load_dataset_by_name
from metis_partition import metis_partition, partition_K
import os

# from test_cuda import clean_gpu, set_gpu_memory

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 訓練模型
def train_model(model,data):

    train_loader = GraphSAINTNodeSampler(data, batch_size=6000, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    # for batch_size, n_id, adjs in train_loader:
    for batch in train_loader:
        # print(batch)
        batch = batch.to('cpu')
        train(model, batch)

    return




def load_sub_data(data,num_sub_nodes=10000,k=10):
    # 檢查 data.x 是否為 None
    if data.x is not None:
        total_nodes = data.x.size(0)
    else:
        print("Warning: 'data.x' is None. No node features are available.")
        total_nodes = data.num_nodes  # 如果沒有特徵，可以使用節點總數來代替

    if total_nodes//k > num_sub_nodes:
        subset_size = num_sub_nodes
        k = total_nodes // subset_size
    else:
        subset_size = total_nodes // k

    # 隨機選擇節點
    subset_nodes = torch.randperm(total_nodes)[:subset_size]
    subset_edge_index, subset_edge_mask = subgraph(subset_nodes, data.edge_index, relabel_nodes=True)
    # 創建子集數據
    sub_data = data.clone()
    # 檢查是否有其他屬性可作為特徵
    if data.x is None:
        print("No node features available, using node_species or other attributes as default features.")
        if hasattr(data, 'node_species'):
            data.x = data.node_species  # 使用 node_species 作為特徵
        else:
            data.x = torch.ones((data.num_nodes, 1))  # 如果不存在其他屬性，創建默認特徵

    # 現在 data.x 不再為 None，可以安全地對其進行下標操作
    sub_data.x = data.x[subset_nodes]  # 保留子集的節點特徵
    sub_data.y = data.y[subset_nodes]  # 保留子集的節點標籤
    sub_data.edge_index = subset_edge_index  # 保留子集的邊

    # 重新設置掩碼（這裡重置為全 False，因為我們不知道哪些節點屬於訓練/驗證/測試集）
    sub_data.train_mask = torch.zeros(subset_size, dtype=torch.bool)
    sub_data.val_mask = torch.zeros(subset_size, dtype=torch.bool)
    sub_data.test_mask = torch.zeros(subset_size, dtype=torch.bool)

    # 如果需要，可以自定義一些節點作為訓練/驗證/測試節點
    train_ratio = 0.6  # 訓練集比例
    val_ratio = 0.2  # 驗證集比例

    num_train = int(train_ratio * subset_size)
    num_val = int(val_ratio * subset_size)

    # 隨機選擇訓練、驗證和測試節點
    sub_data.train_mask[:num_train] = True
    sub_data.val_mask[num_train:num_train + num_val] = True
    sub_data.test_mask[num_train + num_val:] = True
    return sub_data,k


def measure_cpu_usage():
    """Measures CPU memory usage during model training."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Convert to MB


def es_cpu(dataset, num_sub_nodes=10000, k=10):
    data = dataset[0]
    # print(data)
    train_subset, k = load_sub_data(data, num_sub_nodes, k)
    # print(train_subset)

    # 創建 DataLoader
    train_loader = DataLoader([train_subset], batch_size=32, shuffle=True)

    # 模型、優化器和損失函數
    input_dim = train_subset.x.size(1)  # 1433
    output_dim = len(torch.unique(train_subset.y))  # 根據 y 的唯一值計算類別數
    model = GNN(input_dim, output_dim)   # Keep the model on CPU

    tracemalloc.start()  # Start tracking memory usage
    start_time = time.time()
    memory_before = measure_cpu_usage()
    train_model(model, train_subset)
    memory_after = measure_cpu_usage()
    end_time = time.time()
    print(f"CPU Memory: {memory_after} - {memory_before} = {memory_after - memory_before}")
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 ** 2:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 ** 2:.2f} MB")
    tracemalloc.stop()  # Stop tracking memory usage


    # print(f"訓練時間: {end_time - start_time:.2f} 秒")
    # print(f"訓練後的 GPU 記憶體佔用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    # print(f"訓練中最大 GPU 記憶體佔用: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

    es_time = (end_time - start_time) * k
    es_max_memory = (memory_after - memory_before)* k
    # es_memory = (torch.cuda.memory_allocated() / 1024 ** 2) * k
    # es_max_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2) * k
    # print(f"es Runtime on GPU: {es_memory:.2f} 秒")
    # print(f"es GPU Memory Used: {es_memory:.2f} MB")
    # print(f"es Max GPU Memory Used: {es_max_memory:.2f} MB")
    del model
    # print(f'GPU allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB')
    return [es_time, es_max_memory, k]

if __name__ == '__main__':
    """
es Runtime on GPU: 16.33 秒
es GPU Memory Used: 16.33 MB
es Max GPU Memory Used: 4746.01 MB
    """
    # set_gpu_memory(3000)
    # 加载数据集
    # datasets = [
    #     'Cora', 'Citeseer', 'Pubmed', 'Reddit', 'PPI', 'Flickr', 'Amazon-Computers',
    #     'Amazon-Photo', 'PROTEINS', 'ENZYMES', 'IMDB-BINARY', 'ogbn-products', 'ogbn-proteins', 'ogbn-arxiv'
    # ]
    # datasets = ['PPI', 'PROTEINS', 'ENZYMES', 'IMDB-BINARY', 'ogbn-proteins'] # have problem
    # datasets = ['ogbn-proteins']  # have problem
    datasets = ['Cora', 'Citeseer', 'Pubmed', 'Flickr', 'Amazon-Computers', 'Amazon-Photo']
    # datasets = ['Reddit']
    # datasets = ['PPI', 'PROTEINS', 'ENZYMES', 'IMDB-BINARY']
    # datasets = ['ogbn-products', 'ogbn-proteins', 'ogbn-arxiv']
    # datasets = ['ogbn-arxiv','ogbn-products']

    # datasets = ['Cora']


    for dataset_name in datasets:
        dataset = load_dataset_by_name(dataset_name)
        # get_data_info(dataset)
        # print(dataset[0])
        [es_time, es_max_memory, k] = es_cpu(dataset, num_sub_nodes=30000, k=10)
        print([es_time, es_max_memory, k])
        # try:
        #     [es_time, es_max_memory, k] = es_gpu(dataset, num_sub_nodes=30000, k=10)
        # except Exception as e:
        #     print(f"Error running es_gpu on {dataset}: {e}")


    # data = dataset[0]
    # input_dim = data.x.size(1)  # 1433
    # output_dim = len(torch.unique(data.y))  # 根據 y 的唯一值計算類別數
    # model = GNN(input_dim, output_dim).cuda()
    # train_model(model, data)


    # [es_time, es_max_memory, k]=es_gpu(dataset, num_sub_nodes=30000000, k=2)
