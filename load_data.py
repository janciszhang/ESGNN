"""
Planetoid数据集：
Cora (Planetoid): 包含2708个节点（科学出版物）和5429条边（引用关系）。每个节点有1433维特征向量和7类标签。文件大小: ~500 KB
CiteSeer (Planetoid): 包含3327个节点和4732条边。每个节点有3703维特征向量和6类标签。文件大小: ~2 MB
Pubmed (Planetoid): 包含19717个节点和44338条边。每个节点有500维特征向量和3类标签。文件大小: ~20 MB

Reddit: 包含232965个节点和114615892条边。每个节点有602维特征向量。类别: 41。文件大小: ~300 MB
PPI (Protein-Protein Interaction): 蛋白质-蛋白质相互作用数据集，适用于图分类任务。包含56944个节点(总计多个图)和818716条边(总计多个图)。多标签分类任务。类别: 121。文件大小: ~1 GB

Amazon Computers (Amazon Dataset)：节点数: 13,752，边数: 245,861，类别: 10。文件大小: ~40 MB
Amazon Photo (Amazon Dataset)：节点数: 7,650，边数: 119,081，类别: 8。文件大小: ~30 MB
Flickr：包含 Flickr 社交网络中的图片与用户之间的关系。节点数: 89,250，边数: 899,756，类别: 7。文件大小: ~200 MB
PROTEINS (TUDataset)：图数量: 1,113，节点数: 平均每个图约 39 个节点，边数: 平均每个图约 72 条边。文件大小: ~2 MB

OGBN数据集：
OGBN-Products (OGB Benchmark): 节点数: 2,449,029，边数: 61,859,140，类别: 47。文件大小: ~2 GB
OGBN-Arxiv (OGB Benchmark): 节点数: 169,343，边数: 1,166,243，类别: 40。文件大小: ~100 MB
- ogbn-products 文件大小: ~3.2 GB
- ogbn-proteins 文件大小: ~0.8 GB
- ogbn-arxiv 文件大小: ~0.57 GB
- ogbn-papers100M 文件大小: ~250 GB
- ogbn-mag 文件大小: ~22 GB
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
# from dgl.data import FB15kDataset
from ogb.lsc import WikiKG90Mv2Dataset
from torch_geometric.datasets import SNAPDataset

def load_graph_data():
    # 加载Cora数据集
    dataset = Planetoid(root='data/Cora', name='Cora')

    # 加载Reddit数据集
    # dataset = Reddit(root='../dataset/Reddit')

    # 加载Pubmed数据集
    # dataset = Planetoid(root='../dataset/Pubmed', name='Pubmed')

    # 加载Reddit数据集
    # dataset = Reddit(root='../dataset/Reddit')

    # 加载PPI数据集
    # dataset = PPI(root='../dataset/PPI')

    data = dataset[0]

    # 将图数据转换为NetworkX图
    G = to_networkx(data, to_undirected=True)

    return G

def get_graph_size(num_nodes, num_edges, num_features):
    """
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
    total_size_mb = total_size_bytes / (1024 * 1024)
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



def get_data_info(dataset,is_print=True,is_save=False):
    """
    获取data信息：数据集name，nodes数量，edges数量，features数量，classes数量，图的size MB，存储在磁盘上的文件的size MB
    """
    data = dataset[0]
    # 数据集信息
    name = dataset.__class__.__name__
    if hasattr(dataset, 'name'):
        name = name+'/'+dataset.name

    num_features = 0
    if hasattr(data, 'num_features'):
        num_features=data.num_features

    # 数据集信息
    dataset_info = [
        f'Dataset: {name}',
        f'Number of nodes: {data.num_nodes}',
        f'Number of edges: {data.num_edges}',
        f'Number of features: {num_features}',
        f'Number of classes: {dataset.num_classes}',
        f'Total dataset size: {get_graph_size(data.num_nodes,data.num_edges,num_features):.2f} MB',
        f'Total dataset disk size: {get_folder_size(dataset.root):.2f} MB',
    ]

    if is_print:
        print(dataset_info)
    if is_save:
        # 将输出内容写入文件
        with open('dataset_info.txt', 'a') as f:
            for line in dataset_info:
                f.write(line + '\n')
            f.write('--------------------------------------\n')



def load_dataset_by_name(name):
    if name == 'Cora':
        return Planetoid(root='../dataset/Cora', name='Cora')
    elif name == 'Citeseer':
        return Planetoid(root='../dataset/Citeseer', name='Citeseer')
    elif name == 'Pubmed':
        return Planetoid(root='../dataset/Pubmed', name='Pubmed')
    elif name == 'Reddit':
        return Reddit(root='../dataset/Reddit')
    elif name == 'PPI':
        return PPI(root='../dataset/PPI')
    elif name == 'Flickr':
        return Flickr(root='../dataset/Flickr')
    elif name == 'Amazon-Computers':
        return Amazon(root='../dataset/Amazon/Computers', name='Computers')
    elif name == 'Amazon-Photo':
        return Amazon(root='../dataset/Amazon/Photo', name='Photo')
    elif name == 'PROTEINS':
        return TUDataset(root='../dataset/TUDataset/PROTEINS', name='PROTEINS')
    elif name == 'ENZYMES':
        return TUDataset(root='../dataset/TUDataset/ENZYMES', name='ENZYMES')
    elif name == 'IMDB-BINARY':
        return TUDataset(root='../dataset/TUDataset/IMDB-BINARY', name='IMDB-BINARY')
    elif name == 'ogbn-products':
        return PygNodePropPredDataset(root='../dataset/ogbn_products', name='ogbn-products')
    elif name == 'ogbn-proteins':
        return PygNodePropPredDataset(root='../dataset/ogbn_proteins', name='ogbn-proteins')
    elif name == 'ogbn-arxiv':
        return PygNodePropPredDataset(root='../dataset/ogbn_arxiv', name='ogbn-arxiv')
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def load_all_datasets():
    datasets = []

    try:
        dataset_Cora = Planetoid(root='../dataset/Cora', name='Cora') # 加载Cora数据集500 KB
        dataset_Citeseer = Planetoid(root='../dataset/Citeseer', name='Citeseer') # 加载Citeseer数据集2 MB
        dataset_Pubmed = Planetoid(root='../dataset/Pubmed', name='Pubmed') # 加载Pubmed数据集20 MB
        datasets.extend([dataset_Cora, dataset_Citeseer, dataset_Pubmed])
    except:
        pass
    try:
        dataset_PPI = PPI(root='../dataset/PPI') # 加载PPI数据集1 GB
        dataset_Flickr = Flickr(root='../dataset/Flickr') # Flickr
        dataset_Amazon_Computers = Amazon(root='../dataset/Amazon/Computers', name='Computers') # Amazon Computers 40MB
        dataset_Amazon_Photo = Amazon(root='../dataset/Amazon/Photo', name='Photo') # Amazon Photo 30 MB
        dataset_TU_PROTEINS = TUDataset(root='../dataset/TUDataset/PROTEINS',name='PROTEINS')  # 选择适合的名称 ENZYMES, PROTEINS, IMDB-BINARY
        dataset_TU_ENZYMES = TUDataset(root='../dataset/TUDataset/ENZYMES', name='ENZYMES') # TUDataset
        dataset_TU_IMDB = TUDataset(root='../dataset/TUDataset/IMDB-BINARY', name='IMDB-BINARY') # no x
        dataset_Reddit = Reddit(root='../dataset/Reddit')  # 加载Reddit数据集300 MB
        datasets.extend([dataset_PPI,dataset_Flickr,dataset_Amazon_Computers,dataset_Amazon_Photo,dataset_Reddit])
        # datasets.extend([dataset_TU_PROTEINS,dataset_TU_ENZYMES, dataset_TU_IMDB])
    except:
        pass
    try:
        dataset_OGBN_products = PygNodePropPredDataset(root='../dataset/ogbn_products', name='ogbn-products')
        dataset_OGBN_proteins = PygNodePropPredDataset(root='../dataset/ogbn_proteins', name='ogbn-proteins') # no x
        dataset_OGBN_arxiv = PygNodePropPredDataset(root='../dataset/ogbn_arxiv', name='ogbn-arxiv')
        # dataset_OGBN_papers100M = PygNodePropPredDataset(name='ogbn-papers100M') # 很大，文件大小:  ~250 GB
        # dataset_OGBN_mag = PygNodePropPredDataset(name='ogbn-mag') # 该数据结构包含多个图结构的复杂对象，不是单一图结构，不能之间get_data_info()
        datasets.extend([dataset_OGBN_products,dataset_OGBN_proteins,dataset_OGBN_arxiv])
    except:
        pass

    print(dataset_Cora[0])
    print(dataset_Citeseer[0])
    print(dataset_Pubmed[0])
    print(dataset_Reddit[0])
    print(dataset_PPI[0])
    print(dataset_Flickr[0])
    print(dataset_Amazon_Computers[0])
    print(dataset_Amazon_Photo[0])
    print(dataset_TU_PROTEINS[0])
    print(dataset_TU_ENZYMES[0])
    print(dataset_TU_IMDB[0])
    print(dataset_OGBN_products[0])
    print(dataset_OGBN_proteins[0])
    print(dataset_OGBN_arxiv[0])

    get_data_info(dataset_Cora)
    get_data_info(dataset_Citeseer)
    get_data_info(dataset_Pubmed)
    get_data_info(dataset_Reddit)
    get_data_info(dataset_PPI)
    get_data_info(dataset_Flickr)
    get_data_info(dataset_Amazon_Computers)
    get_data_info(dataset_Amazon_Photo)
    get_data_info(dataset_TU_PROTEINS)
    get_data_info(dataset_TU_ENZYMES)
    get_data_info(dataset_TU_IMDB)
    get_data_info(dataset_OGBN_products)
    get_data_info(dataset_OGBN_proteins)
    get_data_info(dataset_OGBN_arxiv)
    # get_data_info(dataset_OGBN_papers100M) # 很大，文件大小:  ~250 GB
    # get_data_info(dataset_OGBN_mag) # 该数据结构包含多个图结构的复杂对象，不是单一图结构，不能之间get_data_info()

    return datasets


if __name__ == '__main__':
    datasets = load_all_datasets()
    # dataset = load_dataset_by_name('Cora')
