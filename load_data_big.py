"""
MariusGNN 使用的数据集:
Papers100M 文件大小:  ~250 GB
Mag240M-Cites 文件大小: ~200 GB
Freebase86M 文件大小: ~10 GB
Facebook15 文件大小: ~20 GB
WikiKG90Mv2 文件大小: ~50 GB
Hyperlink 2012 文件大小: ~60 GB

"""
import os
import sys
import torch_geometric
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
from ogb.lsc import MAG240MDataset
from dgl.data import FB15kDataset
from ogb.lsc import WikiKG90Mv2Dataset
from torch_geometric.datasets import SNAPDataset


def get_graph_size(num_nodes, num_edges, num_features):
    """
    图的size MB
    通过计算数据集中所有特征、节点、边等信息的占用内存来获取数据集的大致大小，单位为 MB。
    可以基于节点特征、边、标签等信息的大小来估算总的内存占用。
    """
    # 计算大小（单位：字节）
    # 节点特征的大小 (num_nodes * num_features * 4 bytes per float)
    node_feature_size = num_nodes * num_features * sys.getsizeof(float())
    # 边的大小 (num_edges * 2 for source and target nodes * 4 bytes per integer)
    edge_size = num_edges * 2 * sys.getsizeof(int())
    # 节点标签的大小 (num_nodes * 4 bytes per integer for labels)
    label_size = num_nodes * sys.getsizeof(int())

    # 总大小 (字节)
    total_size_bytes = node_feature_size + edge_size + label_size

    # 转换为 MB (1 MB = 1024 * 1024 字节)
    total_size_mb = (node_feature_size + edge_size) / (1024 * 1024)  # 图的大小 (MB)

    return total_size_mb
def get_folder_size(folder):
    """
    计算存储在磁盘上的文件的大小MB
    """
    dataset_folder_size_bytes = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            dataset_folder_size_bytes += os.path.getsize(file_path)
    dataset_folder_size_mb = dataset_folder_size_bytes / (1024 * 1024)
    return dataset_folder_size_mb

def get_size_in_mb(path):
    """计算文件或文件夹的大小"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)  # 文件大小 (MB)
    elif os.path.isdir(path):
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # 文件夹大小 (MB)
    return 0


def get_data_info(dataset,is_print=True,is_save=True):
    """
    获取data信息：数据集name，nodes数量，edges数量，features数量，classes数量，图的size MB，存储在磁盘上的文件的size MB
    """
    # 加载Cora数据集
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    # 数据集信息
    name = dataset.__class__.__name__
    if hasattr(dataset, 'name'):
        name = name + '/' + dataset.name
    num_nodes = data.num_nodes()
    num_edges = data.num_edges()
    num_features = data.ndata.get('feat', None)
    if num_features is not None:
        num_features = data.ndata['feat'].shape[1]
    else:
        num_features = 0  # 默认特征数量为0
    num_classes = 0
    if hasattr(dataset, 'num_classes'):
        num_classes=dataset.num_classes



    # 数据集信息
    dataset_info = [
        f'Dataset: {name}',
        f'Number of nodes: {num_nodes}',
        f'Number of edges: {num_edges}',
        f'Number of features: {num_features}',
        f'Number of classes: {num_classes}',
        f'Total dataset size: {get_graph_size(num_nodes,num_edges,num_features):.2f} MB',
        f'Total dataset disk size: {get_size_in_mb(dataset.save_path):.2f} MB',
    ]

    if is_print:
        print(dataset_info)
    if is_save:
        # 将输出内容写入文件
        with open('dataset_info.txt', 'a') as f:
            for line in dataset_info:
                f.write(line + '\n')
            f.write('--------------------------------------\n')



if __name__ == '__main__':
    # OGBN_papers100M 56.17GB GB
    # dataset_OGBN_papers100M = PygNodePropPredDataset(root='./dataset/ogbn_papers100M', name='ogbn-papers100M')

    # Freebase86M 73GB
    # dataset_FB15k = FB15kDataset(root='./dataset/FB15k')
    #
    #
    # # Hyperlink 2012 3.4k GB
    dataset_Hyperlink2012 = torch_geometric.datasets.Planetoid(root='./dataset/hyperlink2012', name='Hyperlink2012')
    #
    # # Facebook15 8.5k GB
    # Facebook15_dataset = SNAPDataset(root='./dataset/facebook', name='Facebook')
    #
    # # MAG240M - Cites 202GB
    # dataset_MAG240M = MAG240MDataset(root='./dataset/MAG240M')
    #
    # # WikiKG90Mv2
    # dataset_WikiKG90Mv2 = WikiKG90Mv2Dataset(root='./dataset/dataset')

    # get_data_info(dataset_OGBN_papers100M)
    # get_data_info(dataset_FB15k)





