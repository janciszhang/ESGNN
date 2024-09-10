# 加载Cora数据集
from torch_geometric.datasets import Planetoid

from ESGNN11.base_gnn import GNN

dataset = Planetoid(root='/tmp/Cora', name='Cora')

data = dataset[0]

model = GNN(dataset)