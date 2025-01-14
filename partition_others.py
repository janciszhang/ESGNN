import torch
import dgl
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import k_hop_subgraph
import igraph as ig
from torch_geometric.utils import subgraph

from metis_partition import calculate_min_integer_ratios, initial_metis_partition
from load_data import load_dataset_by_name
from torch_geometric.data import Data


def merge_two_subgraphs(subgraph1, subgraph2):
    """
    Merges two subgraphs into one.

    Args:
    - subgraph1 (Data): The first subgraph.
    - subgraph2 (Data): The second subgraph.

    Returns:
    - merged_data (Data): The merged graph containing nodes and edges from both subgraphs.
    """
    # Ensure subgraph2's edge index is adjusted with an offset for the node ids
    node_offset = subgraph1.num_nodes
    subgraph2.edge_index += node_offset  # Shift node ids of subgraph2 by node_offset

    # Merge the edge indexes of both subgraphs
    merged_edge_index = torch.cat([subgraph1.edge_index, subgraph2.edge_index], dim=1)

    # Merge the node features and labels
    merged_node_features = torch.cat([subgraph1.x, subgraph2.x], dim=0)
    merged_node_labels = torch.cat([subgraph1.y, subgraph2.y], dim=0)

    # Create the merged graph data object
    merged_data = Data(
        x=merged_node_features,
        edge_index=merged_edge_index,
        y=merged_node_labels,
        num_nodes=subgraph1.x.size(0) + subgraph2.x.size(0)  # Total number of nodes # num_nodes在分割時數據不對
    )

    return merged_data

def merge_all_subgraphs(subgraphs):
    """
    Merges multiple subgraphs back into the original graph.

    Args:
    - subgraphs (list of Data): The list of partitioned subgraphs to merge.

    Returns:
    - merged_data (Data): The combined graph containing all nodes and edges.
    """
    # Initialize lists to store the edge indexes and node features/labels
    all_edge_index = []
    all_node_features = []
    all_node_labels = []
    node_offset = 0  # To handle node renumbering

    # Loop through each subgraph and merge its edges, node features, and node labels
    for subgraph in subgraphs:
        # Adjust edge_index by adding an offset to node indices
        edge_index = subgraph.edge_index + node_offset
        all_edge_index.append(edge_index)

        # Append the node features and labels
        all_node_features.append(subgraph.x)
        all_node_labels.append(subgraph.y)

        # Update the node_offset for the next subgraph
        # node_offset += subgraph.num_nodes # num_nodes在分割時數據不對
        node_offset += subgraph.x.size(0) # num_nodes在分割時數據不對

    # Concatenate all edge indexes, node features, and node labels
    merged_edge_index = torch.cat(all_edge_index, dim=1)
    merged_node_features = torch.cat(all_node_features, dim=0)
    merged_node_labels = torch.cat(all_node_labels, dim=0)

    # Create the merged data object
    merged_data = Data(
        x=merged_node_features,
        edge_index=merged_edge_index,
        y=merged_node_labels,
        num_nodes=node_offset  # Total number of nodes
    )

    return merged_data



def merge_subgraphs_by_ratios(initial_parts, target_integer_ratios=[1,2,3]):
    merged_subgraphs = []
    start_index = 0  # To track the starting index for each set of subgraphs

    # Calculate the total ratio sum and the number of subgraphs to merge per ratio
    total_ratio = sum(target_integer_ratios)

    # Loop over each ratio and merge the corresponding number of subgraphs
    for ratio in target_integer_ratios:
        # print(ratio)
        # Calculate the end index based on the ratio
        end_index = start_index + ratio

        # Slice the initial parts to get the relevant subgraphs
        subgraph_part = initial_parts[start_index:end_index]

        # Merge the selected subgraphs
        merged_data = subgraph_part[0]  # Start with the first subgraph
        for subgraph in subgraph_part[1:]:
            merged_data = merge_two_subgraphs(merged_data, subgraph)  # Merge each subsequent subgraph

        # Add the merged subgraph to the list of merged subgraphs
        # print(merged_data)
        merged_subgraphs.append(merged_data)

        # Update the starting index for the next set of subgraphs to merge
        start_index = end_index

    return merged_subgraphs




def partition_K_plus(data, K, target_ratios=None):
    if target_ratios is None:
        # 如果没有传入target_ratios，默认按等比例分配
        sub_data = edge_cut_partition(data, num_partitions=K)
    else:
        # 目标数量计算
        target_integer_ratios = calculate_min_integer_ratios(target_ratios)
        # print(target_integer_ratios)
        num_initial_partitions = sum(target_integer_ratios)
        # print(num_initial_partitions)
        initial_parts = edge_cut_partition(data, num_partitions=num_initial_partitions)
        if initial_parts is None:
            return None  # 处理失败情况
        # 合并分区以符合目标比例
        sub_data = merge_subgraphs_by_ratios(initial_parts, target_integer_ratios)
    return sub_data





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

def neighborhood_sampling(data, start_nodes, num_hops=2, num_sub_nodes=100000): # 不均分割
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
def edge_cut_partition(data, num_partitions=10): #調整後可用於大圖比例分割
    total_edges = data.edge_index.size(1)
    edges_per_partition = total_edges // num_partitions

    partitions = []
    for i in range(num_partitions):
        start = i * edges_per_partition
        end = (i + 1) * edges_per_partition if i < num_partitions - 1 else total_edges
        partition_edge_index = data.edge_index[:, start:end]
        # print(partition_edge_index)

        # 创建子图
        sub_data = data.clone()
        # print(sub_data)
        sub_data.edge_index = partition_edge_index

        # 提取子图中的节点
        sub_nodes = torch.unique(partition_edge_index)
        sub_data.x = data.x[sub_nodes]
        sub_data.y = data.y[sub_nodes]

        partitions.append(sub_data)

    return partitions


def test_partition_methods(data, num_partitions):
    print(data)
    # 采样子图
    # sub_data = random_node_sampling(data, num_sub_nodes=100000)
    # print(f"Subgraph has {sub_data.num_nodes} nodes and {sub_data.edge_index.size(1)} edges.")
    random_partitions = random_node_sampling(data, num_sub_nodes=100000, num_partitions=num_partitions)
    for i, sub_data in enumerate(random_partitions):
        print(f"Subgraph {i} has {sub_data.num_nodes} nodes and {sub_data.edge_index.size(1)} edges.")
        print(sub_data)

    print(data)
    # 从一个随机节点开始，进行邻域扩展
    # start_node = torch.randint(0, data.num_nodes, (1,)).item()
    # sub_data = neighborhood_sampling(data, start_node, num_hops=2, num_sub_nodes=100000)
    # print(f"Subgraph has {sub_data.num_nodes} nodes and {sub_data.edge_index.size(1)} edges.")
    start_nodes = [torch.randint(0, data.num_nodes, (1,)).item() for _ in range(num_partitions)]  # num_partitions个随机起始节点
    neighborhood_partitions = neighborhood_sampling(data, start_nodes=start_nodes, num_hops=2, num_sub_nodes=100000)
    for i, sub_data in enumerate(neighborhood_partitions):
        print(f"Subgraph {i} has {sub_data.num_nodes} nodes and {sub_data.edge_index.size(1)} edges.")
        print(sub_data)

    print(data)
    # 切分图成多个子图
    edge_cut_partitions = edge_cut_partition(data, num_partitions=num_partitions)
    for i, sub_data in enumerate(edge_cut_partitions):
        print(f"Subgraph {i} has {sub_data.num_nodes} nodes and {sub_data.edge_index.size(1)} edges.")
        print(sub_data)




if __name__ == '__main__':
    # 1. 加载 ogbn-papers100M 数据集
    # dataset = PygNodePropPredDataset(root='./dataset/ogbn_products', name='ogbn-products')
    # dataset = load_dataset_by_name('ogbn-products')
    dataset = load_dataset_by_name('Reddit')
    # dataset = load_dataset_by_name('Cora')
    data = dataset[0]
    print(data.num_nodes)
    print(data)

    # test_partition_methods(data, num_partitions=5)

    # random_partitions = random_node_sampling(data, num_sub_nodes=10000000, num_partitions=5)
    #
    # for partition in random_partitions:
    #     print(partition)

    # edge_cut_partitions = edge_cut_partition(data, num_partitions=3)

    # for i, sub_data in enumerate(edge_cut_partitions):
    #     print(sub_data)

    # print(merge_edge_cut_partitions(edge_cut_partitions))

    # partitions = partition_K_plus(data, K=4, target_ratios=[2,4,6,8])
    # for partition in partitions:
    #     print(partition)





