import torch
import time
from torch_geometric.datasets import Planetoid


def load_subgraphs(file_prefix, num_subgraphs):
    subgraphs = []
    for i in range(num_subgraphs):
        subgraph = torch.load(f'{file_prefix}_subgraph_{i}.pt')
        subgraphs.append(subgraph)
    return subgraphs


def estimate_network_overhead(subgraph):
    start_time = time.time()
    # Simulate moving subgraph to GPU
    subgraph = subgraph.cuda()
    torch.cuda.synchronize()
    end_time = time.time()
    network_overhead = end_time - start_time
    return network_overhead


def estimate_gpu_utilization():
    gpu_memory_used = torch.cuda.memory_allocated()
    gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_utilization = gpu_memory_used / gpu_total_memory
    return gpu_utilization


def estimate_aggregation_time(models):
    start_time = time.time()
    # Example of aggregation (e.g., averaging model parameters)
    aggregated_model = models[0]

