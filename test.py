# 加载Cora数据集
from torch_geometric.datasets import Planetoid
import torch; 
from base_gnn import GNN
from metis_calculation_job_GPU import estimate_tasks_gpu
from metis_partition import partition_K


print(torch.cuda.is_available())

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
subgraphs = partition_K(data, 4)
model = GNN(data.num_node_features, len(torch.unique(data.y))).cuda()
times, sizes = estimate_tasks_gpu(model, subgraphs,is_save=True)