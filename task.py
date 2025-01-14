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
4. 吞吐量
"""
import copy
import re
import time
from math import ceil

import torch
from sklearn.metrics import classification_report
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from partition_others import partition_K_plus
from flex_gnn import FlexibleGNN, combine_FlexibleGNN_models
from load_data import get_folder_size
from metis_calculation_job_CPU import estimate_task_cpu
from metis_partition import partition_K

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
5. 吞吐量
10min 15min(中断浪费的时间5min)
"""

from base_gnn import GNN, train_model, evaluate_model


class Task:
    def __init__(self, name, size=None, duration=None, data=None, is_sub=False, is_main=True, arrival_time=0):
        # size = round(size if size is not None else None)
        self.name = name
        self.data = data
        self.original_data = data
        self.model = None
        if not is_sub and data is not None:
            self.create_model()

        if self.model is not None and (duration is None or size is None):
            duration, size = estimate_task_cpu(self.model, self.data)  # 单位：秒，MB
        self.size = size  # 任务大小，用于调度和资源分配(eg.原始10，分割出去4，这里还有6)
        self.remaining_size = size  # 剩余大小，表示尚未完成的部分
        self.original_size = size  # 原始预计size（创建时）
        self.duration = duration  # 任务持续时间
        self.remaining_duration = duration  # 剩余持续时间
        self.original_duration = duration  # 原始预计持续时间（创建时）

        self.arrival_time = arrival_time
        self.start_time = None
        self.end_time = None
        # self.estimated_end_time = None
        # self.waiting_time = 0

        self.run_priority = 0
        self.interrupt_priority = 0

        self.is_sub = is_sub
        self.is_main = is_main

        # self.is_running = False
        self.interruptions = 0
        self.interruption_time = 0

        self.status = 'waiting'  # waiting，doing，done，interrupted
        self.subtasks = []  # 子任务列表

        self.pmc = self.calculate_pmc()  # Peak Memory Consumption placeholder(CoGNN)

    def calculate_pmc(self):
        """
        Estimate the Peak Memory Consumption (PMC) for the task.

        - feature_size (int): The number of features per node.
        - batch_size (int): The number of nodes in a batch.
        - adjacency_size (int): The number of edges (size of adjacency matrix).

        Returns:
        - pmc (int): Estimated peak memory consumption in bytes.
        """
        if self.data is not None:
            # Feature size from dataset metadata
            # feature_size = dataset.num_node_features
            # feature_size = len(data.x[0])
            # feature_size = data.x.shape[1]
            if self.data.x is None:
                # feature_size = dataset.num_node_features
                feature_size = self.data.node_species.size(1)
            else:
                feature_size = self.data.x.shape[1]

            # Adjacency size from dataset metadata
            adjacency_size = self.data.num_edges

            print(f"Feature size: {feature_size}")
            print(f"Adjacency size: {adjacency_size}")

            # Memory for features: nodes × features × data type size
            feature_memory = self.size * feature_size * 4  # Assume float32, 4 bytes per value

            # Memory for adjacency matrix (sparse format)
            adjacency_memory = adjacency_size * 4  # Assume integer indices, 4 bytes per value

            # Model parameters or temporary buffers (dummy value, depends on model size)
            buffer_memory = 100 * 1024 * 1024  # Example: 100 MB for intermediate buffers

            # Total memory consumption
            self.pmc = feature_memory + adjacency_memory + buffer_memory
        else:
            self.pmc = 0
        print(f"PMC: {self.pmc}")
        return self.pmc

    def get_waiting_time(self, current_time=time.time()):
        if self.start_time is None:
            return current_time - self.arrival_time
        else:
            return self.start_time - self.arrival_time

    def get_estimated_end_time(self):
        if self.start_time is not None:
            return self.start_time + self.duration
        else:
            return None

    def get_completion_time(self):
        if self.end_time==None or self.start_time==None:
            return None
        else:
            return self.end_time - self.start_time

    def get_is_running(self):
        if self.status == 'doing':  # waiting，doing，done，interrupted
            return True
        else:
            return False

    def combine_subtasks(self):
        print(f'Combining subtasks1: {self.__str__()}')
        if self.is_sub or self.subtasks:
        # if self.status == 'done' and self.subtasks and self.is_sub:
            if self.status != 'interrupted':
                print(f'Combining subtasks2: {self.__str__()}')
                if self.data != self.original_data or self.start_time!=self.subtasks[0].start_time:
                    sub_task = copy.copy(self)
                    sub_task.name = self.name + '_sub'
                    sub_task.original_size = self.size
                    sub_task.original_duration = self.duration
                    sub_task.original_data = self.data
                    sub_task.is_main = False
                    self.subtasks.append(sub_task)
                    self.interruptions = 0

                # update task
                start_times = [self.start_time]
                end_times = [self.end_time]
                if self.subtasks:
                    self.interruptions = 0
                for subtask in self.subtasks:
                    start_times.append(subtask.start_time)
                    end_times.append(subtask.end_time)
                    # if subtask.start_time < start_time:
                    #     start_time = subtask.start_time
                    # if subtask.end_time > end_time:
                    #     end_time = subtask.end_time
                    if subtask.status == 'interrupted':
                        self.interruptions += subtask.interruptions
                        # print(f'interruption_time : {subtask.get_completion_time()}')
                        subtask.interruption_time = subtask.get_completion_time()
                        self.interruption_time += subtask.get_completion_time()
                        # print(f'interruption_time : {self.interruption_time}')
                    if self.status != 'done' and subtask.status != 'waiting':
                        self.status = 'doing'

                self.data = self.original_data
                self.size = self.original_size
                self.duration = self.original_duration
                print(start_times, end_times)
                self.start_time = min([start_time for start_time in start_times if start_time is not None])
                self.end_time = None if None in end_times else max([end_time for end_time in end_times if end_time is not None])
                print(self.start_time, self.end_time)
                self.is_sub = False
                # self.estimated_end_time = self.start_time + self.duration
                # self.waiting_time = self.start_time-self.arrival_time
                self.run_priority = 0
                self.interrupt_priority = 0


    def start_doing(self, current_time):
        self.status = 'doing'
        # self.is_running = True
        self.start_time = current_time
        # self.waiting_time = current_time - self.arrival_time
        # self.estimated_end_time = self.start_time + self.remaining_duration

    def create_model(self, num_layers=2):
        input_dim = self.data.x.size(1)  # Feature dimension from the first subgraph
        output_dim = len(torch.unique(self.data.y))  # Number of classes based on the labels in the first subgraph
        # model = GNN(input_dim, output_dim)
        self.model = FlexibleGNN(input_dim, output_dim, num_layers=num_layers)

    def train_model(self, epochs=200, patience=20, early_stopping=True, split_ratios=[6, 3, 2]):
        self.status = 'doing'
        losses, train_accuracies,test_accuracies,gpu_usage=train_model(self.model, self.data, epochs=epochs, patience=patience, early_stopping=early_stopping,
                    split_ratios=split_ratios)
        # self.end_time = time.time()
        # self.status = 'done'
        return gpu_usage

    def calculate_run_priority(self, current_time, total_remaining_size, weight_size=0.4, weight_waiting_time=0.3,
                               weight_is_running=0.2, weight_is_sub=0.1):
        waiting_time = self.get_waiting_time(current_time)
        normalized_size = self.remaining_size / total_remaining_size if total_remaining_size > 0 else 0
        normalized_waiting_time = waiting_time / current_time if current_time > 0 else 0

        cannot_run = 0
        if current_time < self.arrival_time:
            cannot_run = -1000000

        self.run_priority = (
                weight_size * (1 / (normalized_size + 1e-8)) +
                weight_waiting_time * normalized_waiting_time +
                weight_is_running * (1 if self.get_is_running() else 0) +
                weight_is_sub * (1 if self.is_sub else 0) +
                cannot_run
        )

    def calculate_interrupt_priority(self, current_time, total_remaining_size, weight_size=0.4,
                                     weight_remaining_duration=0.3, weight_waiting_time=0.2, weight_is_sub=0.1):
        waiting_time = self.get_waiting_time(current_time)
        normalized_size = self.remaining_size / total_remaining_size if total_remaining_size > 0 else 0
        normalized_remaining_duration = self.remaining_duration / self.original_duration if self.original_duration > 0 else 0
        normalized_waiting_time = 1 / (waiting_time + 1)

        self.interrupt_priority = (
                weight_size * normalized_size +
                weight_remaining_duration * normalized_remaining_duration +
                weight_waiting_time * normalized_waiting_time +
                weight_is_sub * (1 if self.is_sub else 0)
        )
        # print(normalized_waiting_time)

    def __lt__(self, other):
        return self.run_priority > other.run_priority
        # Sort by remaining duration and size
        # return (self.remaining_duration, self.remaining_size) < (other.remaining_duration, other.remaining_size)

    def __str__(self, option=1):
        end_time = self.end_time
        if end_time is not None:
            end_time = round(end_time, 2)
        start_time = self.start_time
        if start_time is not None:
            start_time = round(start_time, 2)
        estimated_end_time = self.get_estimated_end_time()
        if estimated_end_time is not None:
            estimated_end_time = round(estimated_end_time, 2)
        if option == 1:
            return (f"Task({self.name}, size: {self.remaining_size:.2f}/{self.size:.2f}/{self.original_size:.2f}, "
                    f"duration: {self.remaining_duration:.2f}/{self.duration:.2f}/{self.original_duration:.2f}, {self.status}, "
                    f"{self.arrival_time:.2f}/{start_time}/{end_time}/{estimated_end_time}, "
                    f"Is Subtask: {self.is_sub}, Is Main task: {self.is_main}, Interruptions: {self.interruptions}/{self.interruption_time}), PMC: {self.pmc}")
        if option == 2:
            task_info = ""
            task_info += f"Task(Name: {self.name} "
            task_info += f"\nSize: {self.remaining_size}/{self.size}/{self.original_size}, Duration: {self.remaining_duration}/{self.duration}/{self.original_duration}, Status: {self.status}"
            task_info += f"\nTime: {self.arrival_time}/{start_time}/{end_time}/{estimated_end_time}, Waiting Time: {self.get_waiting_time()}"
            task_info += f"\nRun Priority: {self.run_priority}, Interrupt Priority: {self.interrupt_priority}, PMC: {self.pmc}"
            task_info += f"\nIs Subtask: {self.is_sub}, Is Main task: {self.is_main}, Interruptions: {self.interruptions}/{self.interruption_time}"
            if self.data:
                task_info += f"\nData: {self.data}"
                if self.subtasks:
                    task_info += f"\nSubtasks: {self.subtasks}"
                if self.model:
                    task_info += f"\nModel: {self.model}"
            task_info += f')'
            return task_info

        # if option == 3:
        #     return (f"Task Name: {self.name}\n"
        #             f"Status: {self.status}\n"
        #             f"Size: {self.remaining_size}/{self.original_size}\n"
        #             f"Duration: {self.remaining_duration}/{self.original_duration}\n"
        #             f"Run Priority: {self.run_priority}\n"
        #             f"Interrupt Priority: {self.interrupt_priority}\n"
        #             f"Interruptions: {self.interruptions}\n"
        #             f"Is Subtask: {self.is_sub}\n"
        #             f"Start Time: {self.start_time}, End Time: {self.end_time}\n")


def new_task(dataset, duration=20, size=10):
    data = dataset[0]

    name = dataset.__class__.__name__
    if hasattr(dataset, 'name'):
        name = name + '/' + dataset.name

    task = Task(name=name, data=data, duration=duration, size=size)
    return task


def get_task_from_task_str(task_str):
    # 正则表达式模式
    # pattern1 = r"Task\(([^,]+), size: ([\d.]+)/([\d.]+)/([\d.]+), duration: ([\d.]+)/([\d.]+)/([\d.]+), ([^,]+), ([\d.]+)/([\d.]+)/([^/]+)/([\d.]+), Is Subtask: (True|False), Is Main task: (True|False), Interruptions: (\d+)\)"
    # pattern1 = r"Task\(([^,]+), size: ([\d.]+)/([\d.]+)/([\d.]+), duration: ([\d.]+)/([\d.]+)/([\d.]+), ([^,]+), ([\d.]+)/([^/]+)/([^/]+)/([^/]+), Is Subtask: (True|False), Is Main task: (True|False), Interruptions: (\d+)\)"
    pattern1 = r"Task\(([^,]+), size: ([\d.-]+)/([\d.-]+)/([\d.-]+), duration: ([\d.-]+)/([\d.-]+)/([\d.-]+), ([^,]+), ([\d.-]+)/([^/]+)/([^/]+)/([^/]+), Is Subtask: (True|False), Is Main task: (True|False), Interruptions: (\d+)\)"
    pattern2 = r"Task\(Name: (?P<name>[\w\s]+).*?Size: (?P<remaining_size>\d+)/(?P<size>\d+)/(?P<original_size>\d+).*?" \
               r"Duration: (?P<remaining_duration>[\d.]+)/(?P<duration>[\d.]+)/(?P<original_duration>[\d.]+).*?" \
               r"Status: (?P<status>\w+).*?Time: (?P<arrival_time>[\d.]+)/(?P<start_time>[\d.]+)/(?P<end_time>[\d.]+)/" \
               r"(?P<estimated_end_time>[\d.]+).*?Waiting Time: (?P<waiting_time>[\d.]+).*?Run Priority: (?P<run_priority>[\d.]+).*?" \
               r"Interrupt Priority: (?P<interrupt_priority>[\d.]+).*?Is Subtask: (?P<is_sub>True|False).*?" \
               r"Is Main task: (?P<is_main>True|False).*?Interruptions: (?P<interruptions>\d+)(?:.*?Data: (?P<data>Data\(.*?\)))?" \
               r"(?:.*?Model: (?P<model>FlexibleGNN\(.*?\)\))?)?"

    # 进行匹配
    match1 = re.search(pattern1, task_str.strip())
    match2 = re.search(pattern2, task_str, re.DOTALL)

    if match1:
        name = match1.group(1).strip()
        remaining_size = float(match1.group(2))
        size = float(match1.group(3))
        original_size = float(match1.group(4))
        remaining_duration = float(match1.group(5))
        duration = float(match1.group(6))
        original_duration = float(match1.group(7))
        status = match1.group(8)
        arrival_time = float(match1.group(9))
        start_time = float(match1.group(10)) if match1.group(10)!='None' else None
        end_time = float(match1.group(11)) if match1.group(11)!='None' else None
        estimated_end_time = float(match1.group(12)) if match1.group(12)!='None' else None
        is_sub = match1.group(13) == "True"
        is_main = match1.group(14) == "True"
        interruptions = int(match1.group(15))

        data = None

        task = Task(name, size, duration, data, is_sub, is_main, arrival_time=arrival_time)
        task.remaining_size = remaining_size
        task.original_size = original_size
        task.remaining_duration = remaining_duration
        task.original_duration = original_duration
        task.status = status
        task.start_time = start_time
        task.end_time = end_time
        task.interruptions = interruptions
        return task
    else:
        if match2:
            # 提取匹配的字段并将它们转换为合适的类型
            name = match2.group('name').strip()
            remaining_size = int(match2.group('remaining_size'))
            size = int(match2.group('size'))
            original_size = int(match2.group('original_size'))
            remaining_duration = float(match2.group('remaining_duration'))
            duration = float(match2.group('duration'))
            original_duration = float(match2.group('original_duration'))
            status = match2.group('status')
            arrival_time = float(match2.group('arrival_time'))
            start_time = float(match2.group('start_time'))
            end_time = float(match2.group('end_time'))
            estimated_end_time = float(match2.group('estimated_end_time'))
            waiting_time = float(match2.group('waiting_time'))
            run_priority = float(match2.group('run_priority'))
            interrupt_priority = float(match2.group('interrupt_priority'))
            is_sub = match2.group('is_sub') == 'True'
            is_main = match2.group('is_main') == 'True'
            interruptions = int(match2.group('interruptions'))
            # data = match2.group('data')  # 数据可以直接作为字符串
            # model = match2.group('model')  # 模型也直接作为字符串

            data = None

            task = Task(name, size, duration, data, is_sub, is_main, arrival_time)
            task.remaining_size = remaining_size
            task.original_size = original_size
            task.remaining_duration = remaining_duration
            task.original_duration = original_duration
            task.status = status
            task.start_time = start_time
            task.end_time = end_time
            task.run_priority = run_priority
            task.interrupt_priority = interrupt_priority
            task.interruptions = interruptions

            # Return a Task instance
            return task
        else:
            raise ValueError("Task information string is not in the correct format.")

def get_tasks_from_str(tasks_str):
    # 找到所有的任务字符串
    task_pattern = r"Task\((.*?)\)"  # 匹配Task(开头到)结束的字符串，(.*?) 允许多行
    matches = re.findall(task_pattern, tasks_str.strip(), re.DOTALL)  # 使用re.DOTALL匹配多行内容

    # 存储任务的列表
    tasks = []
    for match in matches:
        match = f"Task({match})"
        try:
            task = get_task_from_task_str(match)
            tasks.append(task)
        except ValueError as e:
            print(e)
    return tasks

def split_task(task, available_size):
    sub_tasks = []
    # remaining_size = task.remaining_size
    index = 1
    while task.remaining_size > 0:
        sub_size = min(task.remaining_size, available_size)
        sub_duration = task.duration * (sub_size / task.size)

        sub_task = Task(f"{task.name}_sub", sub_size, sub_duration, is_sub=True, is_main=False)
        if task.data:
            # if task.data.num_nodes < 20000:
            data_chunks = partition_K(task.data, K=2,target_ratios=[ceil(sub_size), ceil(task.remaining_size - sub_size)])  # partition_K分割比例数必须是整数
            # else:
            #     data_chunks = partition_K_plus(task.data, K=2, target_ratios=[ceil(sub_size), ceil(task.remaining_size - sub_size)])  # partition_K分割比例数必须是整数(for big data)
            #     print('HHHHHHHHHHHHHHHHHHHHHHHHHH')
            sub_task.data = data_chunks[0]
            sub_task.arrival_time = task.arrival_time

            task.remaining_size -= sub_task.remaining_size
            task.remaining_duration -= sub_task.remaining_duration
            task.size -= sub_task.size
            task.duration -= sub_task.duration
            task.data = data_chunks[1]
            task.is_sub = True

        sub_tasks.append(sub_task)
        # remaining_size -= sub_size
        index += 1
        break
    # print(len(sub_tasks),sub_tasks[0])
    return sub_tasks

def split_task_K(task, K):
    sub_tasks = []
    if task.data:
        data_chunks = partition_K(task.data, K=K, target_ratios=None)  # partition_K分割比例数必须是整数
    for i in range(K-1):
        sub_task = Task(f"{task.name}_sub_{i}", task.size/K, task.duration/K, is_sub=True, is_main=False, arrival_time = task.arrival_time)
        if task.data:
            sub_task.data = data_chunks[i]
        sub_tasks.append(sub_task)

    task.remaining_size = task.remaining_size/K
    task.remaining_duration = task.remaining_duration/K
    task.size = task.size/K
    task.duration = task.duration/K
    task.data = data_chunks[K-1]
    task.is_sub = True
    sub_tasks.append(task)
    return sub_tasks


def merge_subtasks(tasks):
    print(f'===================================final_tasks BEFORE=======================================')
    for task in tasks:
        print(task)
    task_dict = {}

    # Step 1: 根据任务名称建立任务的主任务和子任务的关系
    for task in tasks:
        # 获取主任务名称（去掉 '_sub' 或 '_sub_sub' 等后缀）
        main_task_name = task.name.split('_')[0]
        if main_task_name not in task_dict:
            task_dict[main_task_name] = []
        # 将任务加入到对应的主任务列表中
        task_dict[main_task_name].append(task)
    print(f'task_dict: {task_dict}')

    # Step 2: 合并子任务到主任务
    merged_tasks = []
    for main_task_name, task_list in task_dict.items():
        print(main_task_name, task_list)
        # 找到主任务
        main_task = None
        subtasks = []

        for task in task_list:
            print(task)
            # 如果是主任务 (没有 '_sub' 后缀)
            if task.name == main_task_name and task.status != 'interrupted':
                print('is main')
                main_task = task
            else:
                print('is sub')
                subtasks.append(task)

        # 如果存在主任务，则合并子任务
        if main_task:
            for subtask in subtasks:
                main_task.subtasks.append(subtask)
            print(f'BEFORE combine_subtasks, Main task: {main_task}')
            for subtask in main_task.subtasks:
                print(subtask)
            print('Do combine_subtasks')
            main_task.combine_subtasks()
            # main_task.subtasks = sorted(main_task.subtasks, key=lambda task: task.start_time, reverse=False)
            main_task.subtasks = sorted(main_task.subtasks, key=lambda task: (task.start_time is None, task.start_time))
            merged_tasks.append(main_task)
        merged_tasks = sorted(merged_tasks, key=lambda task: (task.start_time is None, task.start_time))
    return merged_tasks


if __name__ == '__main__':
    tasks = []
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    # task = new_task(dataset, duration=3, size=3)
    task = new_task(dataset, duration=None, size=None)
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
