import random

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Reddit, PPI, Amazon, Flickr, TUDataset

from es_cpu_time_memory import es_cpu
from es_gpu_time_memory import es_gpu
from task import Task


# 定義數據集加載函數
def load_cora():
    return Planetoid(root='../dataset/Cora', name='Cora')

def load_citeseer():
    return Planetoid(root='../dataset/Citeseer', name='Citeseer')

def load_pubmed():
    return Planetoid(root='../dataset/Pubmed', name='Pubmed')

def load_reddit():
    return Reddit(root='../dataset/Reddit')

def load_ppi():
    return PPI(root='../dataset/PPI')

def load_flickr():
    return Flickr(root='../dataset/Flickr')

def load_amazon_computers():
    return Amazon(root='../dataset/Amazon/Computers', name='Computers')

def load_amazon_photo():
    return Amazon(root='../dataset/Amazon/Photo', name='Photo')

def load_tu_proteins():
    return TUDataset(root='../dataset/TUDataset/PROTEINS', name='PROTEINS')

def load_tu_enzymes():
    return TUDataset(root='../dataset/TUDataset/ENZYMES', name='ENZYMES')

def load_tu_imdb():
    return TUDataset(root='../dataset/TUDataset/IMDB-BINARY', name='IMDB-BINARY')

def load_ogbn_products():
    return PygNodePropPredDataset(root='../dataset/ogbn_products', name='ogbn-products')

def load_ogbn_proteins():
    return PygNodePropPredDataset(root='../dataset/ogbn_proteins', name='ogbn-proteins')

def load_ogbn_arxiv():
    return PygNodePropPredDataset(root='../dataset/ogbn_arxiv', name='ogbn-arxiv')

def load_ogbn_papers100m():
    return PygNodePropPredDataset(root='../dataset/ogbn_papers100M', name='ogbn-papers100M')

# 將數據集名稱和對應的加載函數存入字典
dataset_loaders = {
    'Cora': load_cora,
    'Citeseer': load_citeseer,
    'Pubmed': load_pubmed,
    'Reddit': load_reddit,
    # 'PPI': load_ppi, # 多標籤分類
    'Flickr': load_flickr,
    'Amazon_Computers': load_amazon_computers,
    'Amazon_Photo': load_amazon_photo,
    # 'TUDataset_PROTEINS': load_tu_proteins,
    # 'TUDataset_ENZYMES': load_tu_enzymes,
    # 'TUDataset_IMDB': load_tu_imdb, # no x
    'OGBN_products': load_ogbn_products,
    # 'OGBN_proteins': load_ogbn_proteins, # no x
    'OGBN_arxiv': load_ogbn_arxiv,
    # 'OGBN_papers100M': load_ogbn_papers100m
}


# 根據大小將數據集分為 "小"、"中"、"大" 三類
small_datasets = ['Cora', 'Citeseer', 'Pubmed']
medium_datasets = ['Amazon_Computers', 'Amazon_Photo', 'Flickr']
# large_datasets = ['Reddit', 'OGBN_arxiv','OGBN_products'] #, 'OGBN_papers100M'
large_datasets = ['OGBN_products'] #, 'OGBN_papers100M'


# 定義隨機名稱生成函數
def generate_random_name(category):
    """生成一個隨機名稱，如 S00001, M00023, L00145"""
    prefix = category  # S, M, L 對應小、中、大數據集
    number = random.randint(1, 99999)  # 隨機生成 1 到 99999 之間的數字
    name = f"{prefix}{number:05d}"  # 格式化為五位數字，前面補 0
    return name

def select_datasets(num_small=2,num_medium=2,num_large=2):
    # 從每一類中隨機選擇數據集(允許重複選擇)
    selected_small = random.choices(small_datasets, k=num_small)
    selected_medium = random.choices(medium_datasets, k=num_medium)
    selected_large = random.choices(large_datasets, k=num_large)

    dataset_category_list = []
    # 加載選擇的數據集，同時將分類（S, M, L）放進列表
    for dataset_name in selected_small:
        print(f"Loading small dataset: {dataset_name}")
        dataset = dataset_loaders[dataset_name]()  # 調用對應的加載函數
        dataset_category_list.append((dataset, 'S'))  # 'S' 表示小數據集

    for dataset_name in selected_medium:
        print(f"Loading medium dataset: {dataset_name}")
        dataset = dataset_loaders[dataset_name]()  # 調用對應的加載函數
        dataset_category_list.append((dataset, 'M'))  # 'M' 表示中等數據集

    for dataset_name in selected_large:
        print(f"Loading large dataset: {dataset_name}")
        dataset = dataset_loaders[dataset_name]()  # 調用對應的加載函數
        dataset_category_list.append((dataset, 'L'))  # 'L' 表示大數據集

    print(f"Loaded dataset_category_list: {dataset_category_list}")

    return dataset_category_list

def load_tasks(dataset_category_list):
    tasks=[]
    for dataset,category in dataset_category_list:
        # print(dataset[0])
        name=generate_random_name(category)
        if torch.cuda.is_available():
            [es_time, es_max_memory, k] = es_gpu(dataset)
        else:
            [es_time, es_max_memory, k] = es_cpu(dataset)
        print([es_time, es_max_memory, k])
        duration = es_time
        size = es_max_memory # 有问题？cpu m可能是复数？
        task = Task(name, size, duration)
        task.data = dataset[0]
        task.calculate_pmc()
        task.arrival_time = random.randint(0,10)
        tasks.append(task)

    for task in tasks:
        print(task)
    return tasks


if __name__ == '__main__':
    dataset_category_list = select_datasets(num_small=2,num_medium=1,num_large=1)
    tasks = load_tasks(dataset_category_list)