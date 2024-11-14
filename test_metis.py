import torch
import dgl
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import k_hop_subgraph
import igraph as ig
from torch_geometric.utils import subgraph


# 随机节点采样函数
def random_node_sampling2(data, num_sub_nodes=100000):
    total_nodes = data.num_nodes
    subset_nodes = torch.randperm(total_nodes)[:num_sub_nodes]

    # 根据采样的节点提取子图
    subset_edge_index, _ = subgraph(subset_nodes, data.edge_index, relabel_nodes=True)

    # 构建子图数据
    sub_data = data.clone()
    sub_data.x = data.x[subset_nodes]
    sub_data.y = data.y[subset_nodes]
    sub_data.edge_index = subset_edge_index

    return sub_data

def random_node_sampling(data, num_sub_nodes=100000, num_partitions=5):
    total_nodes = data.num_nodes
    partitions = []

    for _ in range(num_partitions):
        # 随机采样一部分节点
        subset_nodes = torch.randperm(total_nodes)[:num_sub_nodes]

        # 根据采样的节点提取子图
        subset_edge_index, _ = subgraph(subset_nodes, data.edge_index, relabel_nodes=True)

        # 构建子图数据
        sub_data = data.clone()
        sub_data.x = data.x[subset_nodes]
        sub_data.y = data.y[subset_nodes]
        sub_data.edge_index = subset_edge_index

        # 将子图添加到分区列表中
        partitions.append(sub_data)

    return partitions

# 邻域扩展采样函数
def neighborhood_sampling2(data, start_node, num_hops=2, num_sub_nodes=100000):
    # 使用 k-hop 子图来扩展邻域
    subset_nodes, subset_edge_index, _, _ = k_hop_subgraph(
        start_node, num_hops, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

    # 截取最大 num_sub_nodes 的节点
    subset_nodes = subset_nodes[:num_sub_nodes]

    # 构建子图数据
    sub_data = data.clone()
    sub_data.x = data.x[subset_nodes]
    sub_data.y = data.y[subset_nodes]
    sub_data.edge_index = subset_edge_index[:, :subset_nodes.size(0)]

    return sub_data

def neighborhood_sampling(data, start_nodes, num_hops=2, num_sub_nodes=100000):
    partitions = []

    # 对于每个起始节点，执行 k-hop 子图采样
    for start_node in start_nodes:
        # 使用 k-hop 子图来扩展邻域
        subset_nodes, subset_edge_index, _, _ = k_hop_subgraph(
            start_node, num_hops, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

        # 截取最大 num_sub_nodes 的节点
        subset_nodes = subset_nodes[:num_sub_nodes]

        # 构建子图数据
        sub_data = data.clone()
        sub_data.x = data.x[subset_nodes]
        sub_data.y = data.y[subset_nodes]
        sub_data.edge_index = subset_edge_index[:, :subset_nodes.size(0)]

        # 将子图添加到分区列表中
        partitions.append(sub_data)

    return partitions



# 基于边的切分函数
def edge_cut_partition(data, num_partitions=10):
    total_edges = data.edge_index.size(1)
    edges_per_partition = total_edges // num_partitions

    partitions = []
    for i in range(num_partitions):
        start = i * edges_per_partition
        end = (i + 1) * edges_per_partition if i < num_partitions - 1 else total_edges
        partition_edge_index = data.edge_index[:, start:end]

        # 创建子图
        sub_data = data.clone()
        sub_data.edge_index = partition_edge_index

        # 提取子图中的节点
        sub_nodes = torch.unique(partition_edge_index)
        sub_data.x = data.x[sub_nodes]
        sub_data.y = data.y[sub_nodes]

        partitions.append(sub_data)

    return partitions

if __name__ == '__main__':
    # 1. 加载 ogbn-papers100M 数据集
    dataset = PygNodePropPredDataset(root='./dataset/ogbn_products', name='ogbn-products')
    data = dataset[0]

    # 采样子图
    # sub_data = random_node_sampling(data, num_sub_nodes=100000)
    # print(f"Subgraph has {sub_data.num_nodes} nodes and {sub_data.edge_index.size(1)} edges.")
    random_partitions = random_node_sampling(data, num_sub_nodes=100000, num_partitions=5)
    for i, sub_data in enumerate(random_partitions):
        print(f"Subgraph {i} has {sub_data.num_nodes} nodes and {sub_data.edge_index.size(1)} edges.")


    # 从一个随机节点开始，进行邻域扩展
    # start_node = torch.randint(0, data.num_nodes, (1,)).item()
    # sub_data = neighborhood_sampling(data, start_node, num_hops=2, num_sub_nodes=100000)
    # print(f"Subgraph has {sub_data.num_nodes} nodes and {sub_data.edge_index.size(1)} edges.")
    start_nodes = [torch.randint(0, data.num_nodes, (1,)).item() for _ in range(5)]  # 5 个随机起始节点
    neighborhood_partitions = neighborhood_sampling(data, start_nodes=start_nodes, num_hops=2, num_sub_nodes=100000)
    for i, sub_data in enumerate(neighborhood_partitions):
        print(f"Subgraph {i} has {sub_data.num_nodes} nodes and {sub_data.edge_index.size(1)} edges.")


    # 切分图成多个子图
    edge_cut_partitions = edge_cut_partition(data, num_partitions=10)
    for i, sub_data in enumerate(edge_cut_partitions):
        print(f"Subgraph {i} has {sub_data.num_nodes} nodes and {sub_data.edge_index.size(1)} edges.")