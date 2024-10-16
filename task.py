"""
RUN prioritization sort index：
1. size（该任务大小或该任务剩余未运行部分的大小）
2. waiting_time（任务等待时间为地方开始时间到该任务开始运行的时间，比如从0时刻开始，如果先运行C花了5min，再运行部分B 6min，再运行A，那么此时A的等待时间为11min）
3. is_running（是否在运行）
4. is_sub （任务的分割情况：尽量中断子任务）

总体策略：
1. 尽量先运行大小最小的任务，如果地方有空余，可以根据空余地方大小分割其他任务塞进来同时运行。
2. 如果已经在运行某分割任务，则后面尽可能先完成该任务。
3. 另外，如果一个任务等待时间太久，也可以提前运行。

评估：
1. GPU利用率
2. 任务的总完成时间
3. 每个任务（包括多个子任务）的等待时间
"""
import time

import torch
from sklearn.metrics import classification_report
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from ESGNN.flex_gnn import FlexibleGNN, combine_FlexibleGNN_models
from ESGNN.load_data import get_folder_size
from ESGNN.metis_calculation_job_CPU import estimate_task_cpu
from ESGNN.metis_partition import partition_K

"""
STOP prioritization sort index：
1. size（该任务/子任务大小：尽量小）
2. remaining_duration（任务的剩余持续时间：任务即将完成时尽量不中断）
3. waiting_time（任务等待时间：等待时间较长的任务被中断会增加额外的等待时间）
4. is_sub（任务的分割情况：尽量中断子任务）

总体策略：
1. 任务大小（Size）: 优先选择较小的任务或子任务中断，以减少中断的开销。
2. 剩余持续时间（Remaining Duration）: 尽量避免中断即将完成的任务，以防止已经投入的计算资源浪费。
3. 等待时间（Waiting Time）: 避免中断等待时间较长的任务，因为这会进一步增加其等待时间。
4. 任务分割情况（Is Sub）: 优先中断子任务，而不是整个任务，以减少对整体调度的影响。

评估：
1. GPU利用率
2. 任务的总完成时间
3. 每个任务（包括多个子任务）的等待时间
4. 任务完成期望/中断期望
10min 15min(中断浪费的时间5min)
"""


from ESGNN.base_gnn import GNN, train_model, evaluate_model


class Task:
    def __init__(self, name, size=None, duration=None,data=None,is_sub=False):
        self.name = name
        self.data = data
        self.model = None
        if not is_sub and data is not None:
            self.create_model()

        if self.model is not None and (duration is None or size is None):
            duration, size = estimate_task_cpu(self.model, self.data) # 单位：秒，MB
        self.size = size # 任务大小，用于调度和资源分配(eg.原始10，分割出去4，这里还有6)
        self.remaining_size = size # 剩余大小，表示尚未完成的部分
        self.original_size = size # 原始预计size（创建时）
        self.duration = duration # 任务持续时间
        self.remaining_duration = duration # 剩余持续时间
        self.original_duration = duration # 原始预计持续时间（创建时）

        self.arrival_time = 0
        self.start_time = None
        self.end_time=None
        self.estimated_end_time = None
        self.waiting_time = 0

        self.run_priority = 0
        self.interrupt_priority = 0
        self.is_sub = is_sub
        self.is_running = False
        self.interruptions = 0

        self.status = 'waiting'  # waiting，doing，done，interrupted
        self.subtasks = []  # 子任务列表

    def start_doing(self,current_time):
        self.status = 'doing'
        self.is_running = True
        self.start_time = current_time
        self.waiting_time = current_time - self.arrival_time
        self.estimated_end_time = self.start_time + self.remaining_duration

    def create_model(self,num_layers = 2):
        input_dim = self.data.x.size(1)  # Feature dimension from the first subgraph
        output_dim = len(torch.unique(self.data.y))  # Number of classes based on the labels in the first subgraph
        # model = GNN(input_dim, output_dim)
        self.model = FlexibleGNN(input_dim, output_dim, num_layers=num_layers)


    def train_model(self,epochs=200, patience=20, early_stopping=True,split_ratios=[6, 3, 2]):
        self.status = 'doing'
        train_model(self.model, self.data, epochs=epochs, patience=patience, early_stopping=early_stopping,split_ratios=split_ratios)
        # self.end_time = time.time()
        self.status = 'done'

    def calculate_run_priority(self, current_time, total_remaining_size, weight_size=0.4, weight_waiting_time=0.3,weight_is_running=0.2, weight_is_sub=0.1):
        self.waiting_time = current_time - self.arrival_time
        normalized_size = self.remaining_size / total_remaining_size if total_remaining_size > 0 else 0
        normalized_waiting_time = self.waiting_time / current_time if current_time > 0 else 0
        self.run_priority = (
                weight_size * (1/(normalized_size+1e-8)) +
                weight_waiting_time * normalized_waiting_time +
                weight_is_running * (1 if self.is_running else 0) +
                weight_is_sub * (1 if self.is_sub else 0)
        )


    def calculate_interrupt_priority(self, current_time, total_remaining_size,weight_size=0.4, weight_remaining_duration=0.3, weight_waiting_time=0.2,weight_is_sub=0.1):
        self.waiting_time = current_time - self.arrival_time
        normalized_size = self.remaining_size / total_remaining_size if total_remaining_size > 0 else 0
        normalized_remaining_duration = self.remaining_duration / self.original_duration if self.original_duration > 0 else 0
        normalized_waiting_time = 1 / (self.waiting_time + 1)

        self.interrupt_priority = (
                weight_size * normalized_size +
                weight_remaining_duration * normalized_remaining_duration +
                weight_waiting_time * normalized_waiting_time +
                weight_is_sub * (1 if self.is_sub else 0)
        )
        print(normalized_waiting_time)

    def __lt__(self, other):
        return self.run_priority > other.run_priority
        # Sort by remaining duration and size
        # return (self.remaining_duration, self.remaining_size) < (other.remaining_duration, other.remaining_size)

    def __str__(self):
        # task_info=""
        # task_info+=f"Task(Name: {self.name} "
        # task_info+=f"\nSize: {self.remaining_size}/{self.size}/{self.original_size}, Duration: {self.remaining_duration}/{self.duration}/{self.original_duration}, Status: {self.status}"
        # task_info+=f"\nTime: {self.arrival_time}/{self.start_time}/{self.end_time}/{self.estimated_end_time}, Waiting Time: {self.waiting_time}"
        # task_info+=f"\nRun Priority: {self.run_priority}, Interrupt Priority: {self.interrupt_priority}"
        # task_info+=f"\nIs Subtask: {self.is_sub}, Is Running: {self.is_running}, Interruptions: {self.interruptions}"
        # if self.data:
        #     task_info+=f"\nData: {self.data}"
        #     if self.subtasks:
        #         task_info+=f"\nSubtasks: {self.subtasks}"
        #     if self.model:
        #         task_info+=f"\nModel: {self.model}"
        # task_info +=f')'
        # return task_info
        return f"Task({self.name}, {self.remaining_size}/{self.size}/{self.original_size}, {self.remaining_duration}/{self.duration}/{self.original_duration}, {self.status}),{self.start_time},{self.end_time},{self.estimated_end_time}"

        # return (f"Task Name: {self.name}\n"
        #         f"Status: {self.status}\n"
        #         f"Size: {self.remaining_size}/{self.original_size}\n"
        #         f"Duration: {self.remaining_duration}/{self.original_duration}\n"
        #         f"Run Priority: {self.run_priority}\n"
        #         f"Interrupt Priority: {self.interrupt_priority}\n"
        #         f"Interruptions: {self.interruptions}\n"
        #         f"Is Subtask: {self.is_sub}\n"
        #         f"Start Time: {self.start_time}, End Time: {self.end_time}\n")

def new_task(dataset,duration=20, size=10):
    data = dataset[0]

    name = dataset.__class__.__name__
    if hasattr(dataset, 'name'):
        name = name + '/' + dataset.name

    task = Task(name=name,data=data,duration=duration, size=size)
    return task


def split_task(task, available_size):
    sub_tasks = []
    # remaining_size = task.remaining_size
    index = 1
    while task.remaining_size > 0:
        sub_size = min(task.remaining_size, available_size)
        sub_duration = task.duration * (sub_size / task.size)

        sub_task = Task(f"{task.name}_sub", sub_size, sub_duration,is_sub=True)
        if task.data:
            data_chunks = partition_K(task.data, K=2, target_ratios=[sub_size, task.remaining_size - sub_size])
            sub_task.data = data_chunks[0]

            task.remaining_size -= sub_task.remaining_size
            task.remaining_duration -= sub_task.remaining_duration
            task.size -= sub_task.size
            task.duration -= sub_task.duration
            task.data = data_chunks[1]
            task.subtasks.append(sub_task)

        sub_tasks.append(sub_task)
        # remaining_size -= sub_size
        index += 1
        break
    # print(len(sub_tasks),sub_tasks[0])
    return sub_tasks



if __name__ == '__main__':
    tasks=[]
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    # task = new_task(dataset, duration=3, size=3)
    task= new_task(dataset,duration=None, size=None)
    print(task)
    tasks.append(task)

    # dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    # task = new_task(dataset, duration=15, size=5)
    # tasks.append(task)
    #
    # current_time=0
    # total_remaining_size = sum(task.remaining_size for task in tasks)
    # available_size=4
    #
    # for task in tasks:
    #     task.calculate_run_priority(current_time, total_remaining_size, weight_size=0.4, weight_waiting_time=0.3,weight_is_running=0.2, weight_is_sub=0.1)
    #     task.calculate_interrupt_priority(current_time, total_remaining_size,weight_size=0.4, weight_remaining_duration=0.3,weight_waiting_time=0.2, weight_is_sub=0.1)
    # print(tasks[0])
    # print(tasks[1])
    #
    # tasks[1].create_model(num_layers=2)
    # # tasks[1].train_model()
    # # evaluate_model(tasks[1].model,tasks[1].data)
    #
    # sub_tasks = split_task(tasks[1], available_size)
    # print(sub_tasks[0])
    # print(tasks[1])

    # sub_tasks[0].create_model(num_layers=2)
    # sub_tasks[0].train_model()
    # evaluate_model(sub_tasks[0].model, sub_tasks[0].data)
    #
    # tasks[1].create_model(num_layers=2)
    # tasks[1].train_model()
    # evaluate_model(tasks[1].model, tasks[1].data)
    #
    # combined_model=combine_FlexibleGNN_models(sub_tasks[0].model, tasks[1].model)
    # evaluate_model(combined_model, sub_tasks[0].data)


