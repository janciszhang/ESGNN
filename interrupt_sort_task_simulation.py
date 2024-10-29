import torch
import torch.nn.functional as F
import torch_geometric
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Reddit, Flickr
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import PPI
# from torch_geometric.datasets import OGB
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv
import metis
import networkx as nx
import time
import heapq
import pandas as pd

from ESGNN.base_gnn import GNN, train_model
from ESGNN.flex_gnn import FlexibleGNN
from ESGNN.metis_calculation_job_CPU import estimate_tasks_cpu
from ESGNN.metis_partition import partition_K, load_subgraphs
from ESGNN.task import Task, new_task


def split_task2(task, available_size):
    sub_tasks = []
    remaining_size = task.remaining_size
    while remaining_size > 0:
        sub_size = min(remaining_size, available_size)
        sub_duration = task.duration * (sub_size / task.size)
        sub_task = Task(task.name, sub_size, sub_duration, task.data, is_sub=True)
        sub_task.name = f"{task.name}_sub"
        sub_task.remaining_size = sub_size
        sub_task.remaining_duration = sub_duration
        sub_tasks.append(sub_task)
        remaining_size -= sub_size
    return sub_tasks


def split_task(task, available_size):
    sub_tasks = []
    remaining_size = task.remaining_size
    data = task.data  # 假设data是一个可切片的结构
    sub_task_index = 1  # 初始化子任务序号

    while remaining_size > 0:
        sub_size = min(remaining_size, available_size)
        sub_duration = task.original_duration * (sub_size / task.original_size)  # 使用原始持续时间

        # 切分数据
        # sub_data = data_chunks[:sub_size]  # 示例：简单切片
        data_chunks = partition_K(data, K=2, target_ratios=[sub_size, remaining_size - sub_size])
        sub_data = data_chunks[0]

        # 使用原始任务的模型
        sub_task = Task(data=sub_data, name=f"{task.name}_sub_{sub_task_index}", duration=sub_duration,size=sub_size, is_sub=True)
        sub_task.model = task.model
        sub_task.original_size = task.original_size
        sub_task.original_duration = task.original_duration
        sub_tasks.append(sub_task)

        remaining_size -= sub_size
        data = data_chunks[1]
        sub_task_index += 1
        break

    return sub_tasks


def schedule_tasks(tasks,available_size,  borrow_schedule=[]):
    # 定义权重
    weight_size = 1
    weight_queue_time = 1
    weight_is_running = 1
    weight_is_sub = 1

    # 初始化队列
    task_queue = tasks[:]
    current_time = 0
    borrowed_applied = set()  # Track when borrowed space is applied
    returned_applied = set()  # Track when space is returned
    running_tasks = []
    completed_tasks = []
    utilization_time = 0


    # 记录任务等待时间、完成时间和利用率
    total_remaining_size = sum(task.size for task in tasks)
    schedule = []

    borrowed_spaces = {}
    borrow_events = []

    while task_queue or running_tasks:
        # 更新优先级
        for task in task_queue:
            task.calculate_run_priority(current_time, total_remaining_size, weight_size, weight_queue_time,
                                        weight_is_running, weight_is_sub)

        # 按优先级排序任务队列
        heapq.heapify(task_queue)

        # 检查并处理借出归还事件
        for time, space in list(borrowed_spaces.items()):
            if time <= current_time:
                available_size += space
                del borrowed_spaces[time]

        # 检查借出归还计划
        for start, end, space in borrow_schedule:
            print(start, end, space)

            if start <= current_time and end >= current_time:
                print(f"available_size: {available_size}")
                if available_size >= space:
                    available_size -= space
                    borrowed_spaces[end] = space
                    borrow_events.append((start, end, space))

                    # 检查当前运行的任务是否会被中断
                    if task_queue and task_queue[0].is_running and task_queue[0].remaining_size > available_size:
                        interrupted_task = heapq.heappop(task_queue)
                        interrupted_task.interruptions += 1
                        interrupted_task.is_running = False
                        interrupted_task.status = 'waiting'

                        schedule.append({
                            'Task': interrupted_task.name,
                            'Start Time': interrupted_task.start_time,
                            'End Time': current_time,
                            'Arrival Time': interrupted_task.arrival_time,
                            'Size': interrupted_task.size,
                            'Duration': interrupted_task.duration - interrupted_task.remaining_duration,
                            'Waiting Time': interrupted_task.waiting_time,
                            'Interruptions': interrupted_task.interruptions,
                            'Status': interrupted_task.status
                        })

                        interrupted_task.remaining_size = interrupted_task.original_size - (
                                    interrupted_task.size - interrupted_task.remaining_size)
                        interrupted_task.remaining_duration = interrupted_task.original_duration * (
                                    interrupted_task.remaining_size / interrupted_task.original_size)
                        interrupted_task.start_time = current_time
                        interrupted_task.waiting_time += (current_time - interrupted_task.start_time)

                        subtasks = split_task(interrupted_task, available_size)
                        for subtask in subtasks:
                            heapq.heappush(task_queue, subtask)

        # 运行任务
        if task_queue:
            running_task = heapq.heappop(task_queue)
            if running_task.remaining_size <= available_size:
                running_task.is_running = True
                running_task.status = 'doing'
                if running_task.start_time:
                    start_time = max(current_time, running_task.start_time)
                else:
                    start_time=current_time

                running_task.start_time = start_time
                running_task.waiting_time += start_time - running_task.arrival_time
                current_time = start_time + running_task.remaining_duration
                print(f'current_time: {current_time}')

                # 训练GNN
                # running_task.train_model()

                utilization_time += running_task.size * running_task.remaining_duration
                running_task.is_running = False
                running_task.status = 'done'
                running_task.end_time = current_time
                completed_tasks.append(running_task)

                schedule.append({
                    'Task': running_task.name,
                    'Start Time': running_task.start_time,
                    'End Time': current_time,
                    'Arrival Time': running_task.arrival_time,
                    'Size': running_task.size,
                    'Duration': running_task.remaining_duration,
                    'Waiting Time': running_task.waiting_time,
                    'Interruptions': running_task.interruptions,
                    'Status': running_task.status
                })
            else:
                running_task.interruptions += 1
                running_task.remaining_size = running_task.original_size
                running_task.remaining_duration = running_task.original_duration
                subtasks = split_task(running_task, available_size)
                current_time += 0
                for subtask in subtasks:
                    heapq.heappush(task_queue, subtask)

                schedule.append({
                    'Task': running_task.name,
                    'Start Time': running_task.start_time,
                    'End Time': current_time,
                    'Arrival Time': running_task.arrival_time,
                    'Size': running_task.size,
                    'Duration': running_task.duration - running_task.remaining_duration,
                    'Waiting Time': running_task.waiting_time,
                    'Interruptions': running_task.interruptions,
                    'Status': running_task.status
                })
                running_task.start_time = current_time

            total_remaining_size = sum(task.remaining_size for task in task_queue)

    # 计算任务的等待时间和完成时间
    total_waiting_time = sum(task.waiting_time for task in completed_tasks)
    total_completion_time = current_time
    utilization_rate = utilization_time / (total_completion_time * available_size)

    # 吞吐量
    throughput = len(completed_tasks) / total_completion_time


    # 输出结果
    for task in completed_tasks:
        print(
            f"Task {task.name} completed in {task.duration:.2f} minutes with waiting time {task.waiting_time:.2f} and {task.interruptions} interruptions.")

    print(f"Total waiting time: {total_waiting_time:.2f}")
    print(f"Total completion time: {total_completion_time:.2f}")
    print(f"Utilization rate: {utilization_rate:.2f}")
    print(f"Throughput: {throughput:.2f} tasks per minute")

    # 计算中断时间和完成时间的数学期望
    for task in tasks:
        task_completion_times = [subtask.remaining_duration for subtask in split_task(task, available_size)]
        average_completion_time = sum(task_completion_times) / len(task_completion_times)
        print(f"Task {task.name} average completion time: {average_completion_time:.2f}")

    # 记录和保存调度计划
    df_schedule = pd.DataFrame(schedule)
    df_schedule.to_csv('schedule.csv', index=False)





if __name__ == '__main__':
    tasks=[]
    # 加载Cora数据集500 KB
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    tasks.append(new_task(dataset, duration=3, size=3)) # 4 1
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    tasks.append(new_task(dataset, duration=15, size=5)) # 4 4 2
    # dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    # dataset = Flickr(root='/tmp/Flickr')

    # dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    # dataset = Reddit(root='/tmp/Reddit')
    data = dataset[0]
    # print(tasks[0])
    # print(tasks[1])


    available_size = 4
    # sub_tasks = split_task(tasks[0], available_size)
    # for sub_task in sub_tasks:
    #     print(sub_task.data)

    # # 将子图分成多个子任务
    # num_partitions = 4
    # # subgraphs = partition_K(data, K=num_partitions)
    #
    # name = dataset.__class__.__name__
    # if hasattr(dataset, 'name'):
    #     name = name + '/' + dataset.name
    # subgraphs= load_subgraphs(file_prefix=name, num_subgraphs=num_partitions)

    # 初始化模型
    # input_dim = data.x.shape[1]
    # output_dim = dataset.num_classes
    # hidden_dim = 64
    # num_layers = 3
    # epochs = 200
    # # model = FlexibleGNN(input_dim, hidden_dim, output_dim, num_layers)
    #
    # model = GNN(input_dim=data.num_node_features, output_dim=len(torch.unique(data.y)))
    #
    # times, sizes = estimate_tasks_cpu(model, subgraphs,is_save=False)
    # 初始化任务
    # tasks = [Task(f"Subgraph_{i}",  data) for i in range(num_partitions)]
    # tasks = [Task(f"Subgraph_{i}",  data) for i in range(num_partitions)]
    # print(tasks)
    # # 定义可用空间
    # available_size = 5
    #
    # # 定义借出归还空间的计划：start_time, end_time, space
    borrow_schedule = [
        (5, 6, 2)
        # (15, 17, 3),
        # (20, 24, 1)
    ]
    schedule_tasks(tasks,available_size, borrow_schedule)
