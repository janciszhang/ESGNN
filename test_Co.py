from torch_geometric.datasets import Planetoid  # Example dataset
from torch_geometric.utils import to_dense_adj

from load_data import load_all_datasets

datasets = load_all_datasets()

print('PMC==========')
print(datasets)

for dataset in datasets:
    data = dataset[0]  # Access the graph object

    # Feature size from dataset metadata
    # feature_size = dataset.num_node_features
    # feature_size = len(data.x[0])
    # feature_size = data.x.shape[1]
    if data.x is None:
        print(data)
        # feature_size = dataset.num_node_features
        feature_size = data.node_species.size(1)
    else:
        feature_size = data.x.shape[1]

    # Adjacency size from dataset metadata
    adjacency_size = data.num_edges

    print(f"Feature size: {feature_size}")
    print(f"Adjacency size: {adjacency_size}")


