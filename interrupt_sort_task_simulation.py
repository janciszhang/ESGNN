import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid, Reddit
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

from ESGNN.base_gnn import GNN
from ESGNN.metis_calculation_job_CPU import estimate_tasks_cpu
from ESGNN.metis_partition import partition_K
from ESGNN.task import Task



def split_task(task, available_size):
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

def train_gnn(task):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task.model.to(device)
    task.data = task.data.to(device)

    optimizer = torch.optim.Adam(task.model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    task.model.train()
    optimizer.zero_grad()
    out = task.model(task.data)

    loss = criterion(out[task.data.train_mask], task.data.y[task.data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()

def scheduler(available_size,tasks,borrow_schedule):
    # 定义权重
    weight_size = 1
    weight_queue_time = 1
    weight_is_running = 1
    weight_is_sub = 1

    # 初始化队列
    task_queue = tasks[:]
    current_time = 0
    completed_tasks = []
    utilization_time = 0

    # 记录任务等待时间、完成时间和利用率
    total_remaining_size = sum(task.size for task in tasks)
    schedule = []


    borrowed_spaces = {}
    borrow_events = []

    while task_queue:
        # 更新优先级
        for task in task_queue:
            task.calculate_run_priority(current_time, total_remaining_size, weight_size, weight_queue_time, weight_is_running, weight_is_sub)

        # 按优先级排序任务队列
        heapq.heapify(task_queue)

        # 检查并处理借出归还事件
        for time, space in list(borrowed_spaces.items()):
            if time <= current_time:
                available_size += space
                del borrowed_spaces[time]

        # 检查借出归还计划
        for start, end, space in borrow_schedule:
            if start == current_time:
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

                        interrupted_task.remaining_size = interrupted_task.original_size - (interrupted_task.size - interrupted_task.remaining_size)
                        interrupted_task.remaining_duration = interrupted_task.original_duration * (interrupted_task.remaining_size / interrupted_task.original_size)
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
                start_time = max(current_time, running_task.start_time)
                running_task.start_time = start_time
                running_task.waiting_time += start_time - running_task.arrival_time
                current_time = start_time + running_task.remaining_duration

                # 训练GNN
                loss = train_gnn(running_task)

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
    print(f"Throughput: {throughput:.2f} tasks per minute")

    # 输出结果
    for task in completed_tasks:
        print(
            f"Task {task.name} completed in {task.duration:.2f} minutes with waiting time {task.waiting_time:.2f} and {task.interruptions} interruptions.")

    print(f"Total waiting time: {total_waiting_time:.2f}")
    print(f"Total completion time: {total_completion_time:.2f}")
    print(f"Utilization rate: {utilization_rate:.2f}")

    # 计算中断时间和完成时间的数学期望
    for task in tasks:
        task_completion_times = [subtask.remaining_duration for subtask in split_task(task, available_size)]
        average_completion_time = sum(task_completion_times) / len(task_completion_times)
        print(f"Task {task.name} average completion time: {average_completion_time:.2f}")

    # 记录和保存调度计划
    df_schedule = pd.DataFrame(schedule)
    df_schedule.to_csv('schedule.csv', index=False)

if __name__ == '__main__':
    # 加载Cora数据集500 KB
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    # 将子图分成多个子任务
    num_partitions = 4
    subgraphs = partition_K(data, K=num_partitions)
    # 初始化模型
    model = GNN(data.num_node_features, len(torch.unique(data.y)))
    times, sizes = estimate_tasks_cpu(model, subgraphs)
    # 初始化任务
    tasks = [Task(f"Subgraph_{i}", len(subgraphs[i]), times[i], data) for i in range(num_partitions)]
    # 定义可用空间
    available_size = 5

    # 定义借出归还空间的计划：start_time, end_time, space
    borrow_schedule = [
        (7, 10, 2),
        (15, 17, 3),
        (20, 24, 1)
    ]
    scheduler(available_size,tasks, borrow_schedule)