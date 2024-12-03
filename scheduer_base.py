"""
有时候会报错，可以多运行几次，会有成功的
"""
import re

import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# from test_cuda import clean_gpu
from viz import plot_tasks
from task import Task, split_task, merge_subtasks
import copy
import heapq
import random
from torch_geometric.datasets import Planetoid, Flickr
from metis_partition import partition_K
from task import Task, split_task, new_task
import time

import math


def generate_sine_borrow_schedule(start_time, end_time, amplitude, frequency, interval_duration):
    """
    :param start_time: The start time of the schedule
    :param end_time: The end time of the schedule
    :param amplitude: Amplitude of the sine wave (max borrow space)
    :param frequency: Frequency of the sine wave
    :param interval_duration: Duration of each interval in time units
    :return: borrow_schedule
    """
    borrow_schedule = []
    time = start_time

    while time <= end_time:
        borrow_space = int(amplitude * math.sin(frequency * time))
        borrow_end_time = time + interval_duration

        borrow_schedule.append((time, borrow_end_time, borrow_space))
        time += interval_duration

    return borrow_schedule


def generate_available_size_schedule(time_range=[0, 10], available_size=4, borrow_schedule=[(5, 6, 2)]):
    available_size_schedule = []
    start_time = time_range[0]
    end_time = time_range[1]

    # 按 borrow_schedule 的顺序构建 available_size_schedule
    for borrow_start, borrow_end, borrowed_size in borrow_schedule:
        # 在借出之前的时间段，保持默认的可用容量
        if start_time < borrow_start <= end_time:
            available_size_schedule.append((start_time, borrow_start, available_size))
        if borrow_start <= start_time <= end_time:
            # 在借出时间段，减少借出的容量
            if borrow_end > end_time:
                available_size_schedule.append((borrow_start, end_time, available_size - borrowed_size))
            else:
                available_size_schedule.append((borrow_start, borrow_end, available_size - borrowed_size))

        # 更新下一个时间段的开始时间
        start_time = borrow_end

    # 借出计划后的时间段，恢复原始可用容量
    if start_time < end_time:
        available_size_schedule.append((start_time, end_time, available_size))

    return available_size_schedule


def find_minimum_end_time_indexes(end_times):
    indexes = [0]
    minimum_end_time = end_times[0]
    i = 0
    while True:
        if end_times[i] is not None:
            indexes = [i]
            minimum_end_time = end_times[i]
            break
        if i >= len(end_times) - 1:
            break
        i += 1

    for j in range(i + 1, len(end_times)):
        if end_times[j] is not None:
            if end_times[j] < minimum_end_time:
                minimum_end_time = end_times[j]
                indexes = [j]
            else:
                if minimum_end_time == end_times[j]:
                    indexes.append(j)
    if end_times[indexes[0]] == None:
        indexes = []
    return indexes


def borrow_handler(current_time, available_size, borrow_schedule, borrowed_applied, returned_applied):
    # print(f'=================Current time: {current_time}==================')
    next_borrow_end_times = []
    next_borrow_start_times = []
    for (borrow_start_time, borrow_end_time, borrow_space) in borrow_schedule:
        # print(f'borrow_start_time, borrow_end_time, borrow_space: {borrow_start_time}, {borrow_end_time}, {borrow_space}')
        # print(f'{borrow_start_time, borrow_space} not in {borrowed_applied}:{(borrow_start_time, borrow_space) not in borrowed_applied}')
        if borrow_start_time <= current_time < borrow_end_time:
            if (borrow_start_time, borrow_space) not in borrowed_applied:
                available_size -= borrow_space  # Borrowed space decreases available size once
                borrowed_applied.add((borrow_start_time, borrow_space))  # Mark this time as processed
                print(f"Borrowed {borrow_space} space, available size now: {available_size}")
            if (borrow_end_time, borrow_space) not in returned_applied:
                next_borrow_end_times.append(borrow_end_time)
        else:
            if current_time < borrow_start_time:
                next_borrow_start_times.append(borrow_start_time)
                next_borrow_end_times.append(borrow_end_time)
                # next_borrow_start_time = borrow_start_time
                # next_borrow_end_time = borrow_end_time
            else:
                if borrow_end_time <= current_time and (borrow_end_time, borrow_space) not in returned_applied:
                    available_size += borrow_space  # Return the space once at the end time
                    returned_applied.add((borrow_end_time, borrow_space))  # Mark this time as processed
                    print(f"Returned {borrow_space} space, available size now: {available_size}")
    next_borrow_start_time = min(next_borrow_start_times) if next_borrow_start_times else None
    next_borrow_end_time = min(next_borrow_end_times) if next_borrow_end_times else None

    # print(f"borrowed_applied,returned_applied: {borrowed_applied}, {returned_applied}")
    # print(f'next_borrow_start_time, next_borrow_end_time: {next_borrow_start_time}, {next_borrow_end_time}')
    # print(f'available_size: {available_size}')
    return next_borrow_start_time, next_borrow_end_time, available_size, borrowed_applied, returned_applied


def interrupt_handler(current_time, available_size, task_queue, running_tasks, interrupt_tasks,
                      key_func=lambda task: task.size, reverse=True):
    while available_size < 0 and len(running_tasks) >= 1:
        # 按照指定的 key_func 对 running_tasks 排序
        running_tasks = sorted(running_tasks, key=key_func, reverse=reverse)
        task_to_interrupt = running_tasks.pop()
        task_to_interrupt.remaining_duration = task_to_interrupt.duration - (
                    current_time - task_to_interrupt.start_time)
        task_to_interrupt.remaining_size = task_to_interrupt.size - (
                task_to_interrupt.size / task_to_interrupt.duration * (current_time - task_to_interrupt.start_time))

        print(f"Task {task_to_interrupt.name} is interrupted.")
        task_to_interrupt_copy = copy.copy(task_to_interrupt)

        task_to_interrupt.status = 'interrupted'
        task_to_interrupt.interruptions += 1
        task_to_interrupt.end_time = current_time
        task_to_interrupt.is_sub = True
        task_to_interrupt.is_main = False
        print(task_to_interrupt)
        interrupt_tasks.append(copy.copy(task_to_interrupt))  # 记录中断任务副本

        # Reset task to original state before interrupt
        # task_to_interrupt_copy = copy.copy(task_to_interrupt)
        task_to_interrupt_copy.remaining_size = task_to_interrupt_copy.size
        task_to_interrupt_copy.remaining_duration = task_to_interrupt_copy.duration
        task_to_interrupt_copy.status = 'waiting'
        task_to_interrupt_copy.start_time = None
        task_to_interrupt_copy.end_time = None
        task_to_interrupt_copy.interruptions = 0
        print(f'Reset this interrupted task {task_to_interrupt_copy.name}: ', end="")
        print(task_to_interrupt_copy)
        heapq.heappush(task_queue, task_to_interrupt_copy)
        available_size += task_to_interrupt_copy.size
        print(f'After {task_to_interrupt_copy.name} interrupt, available_size: {available_size}')
    return available_size, task_queue, running_tasks, interrupt_tasks


def execute_handler(current_time, available_size, running_tasks, completed_tasks, n):
    # 运行Execute running tasks
    end_times = []
    print(end_times)
    for task in running_tasks[:]:
        task.remaining_duration = task.duration - (current_time - task.start_time)
        # print(f'task.remaining_duration: {task.name}: {task.duration} - {(current_time - task.start_time)} {task.remaining_duration<=0}')
        # task.remaining_size = task.size-(task.size/task.duration * (current_time - task.start_time))
        if round(task.remaining_duration) <= 0 or n > 0:
            if n > 0:
                n -= 1  # 调控当task.remaining_duration>0，没有办法通过train model更新新的duration，而陷入循环
            task.model = None
            try:
                print(end_times)
                print(f'GPU allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB')
                print(f'train model: {task.name}')
                start_time = time.time()
                task.create_model()
                gpu_usage=task.train_model()
                real_model_time = (time.time() - start_time) * 19
                print(f'Task {task.name} real_model_time: {real_model_time}')
                end_time = task.start_time + real_model_time
                print(f'Task {task.name} real_end_time: {end_time}')
                print(f'GPU allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB')
                print(f'Task {task.name} GPU usage: {gpu_usage:.2f} MB')
                if gpu_usage > task.size:
                    available_size += (task.size-gpu_usage)
                    task.size = gpu_usage
                if end_time <= current_time:
                    end_times.append(end_time)
                else:
                    end_times.append(None)
                    task.duration = end_time - task.start_time
            except torch.cuda.OutOfMemoryError as e:
                print(f'CUDA out of memory: {e}')
                match = re.search(r"allocate (\d+\.\d+)", str(e))
                if match:
                    allocated_memory = float(match.group(1))
                    print(allocated_memory)
                    task.size += (allocated_memory + available_size)
                    available_size = - allocated_memory

                end_times.append(None)
            except Exception as e:
                print(f'Exception: {e}')
                print(task.data)
                print(f'{{"CUDA error" in str(e)}}')
                if 'CUDA error' in str(e):
                    print(f'aaaaa GPU allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB')
                    del task.data
                    del task.model
                    print(f'hhhhhh GPU allocated: {torch.cuda.memory_allocated(0) / (1024 ** 2):.2f} MB')
                    return 0
                end_time = task.start_time + task.duration
                end_times.append(None)
                print(end_times)
            # if end_time <= current_time:
            #     end_times.append(end_time)
            # else:
            #     end_times.append(None)
            #     task.duration = end_time - task.start_time
        else:
            end_times.append(None)

    if end_times:
        indexes = find_minimum_end_time_indexes(end_times)
        print(end_times, indexes)
        if len(indexes) > 0 and end_times[indexes[0]] <= current_time:
            running_tasks_copy = copy.copy(running_tasks)
            for i in indexes:
                task = running_tasks_copy[i]
                end_time = end_times[i]
                available_size += task.size
                print(f"Task {task.name} finished.")
                task.status = 'done'
                # task.end_time = current_time + task.remaining_duration # 任务结束时间
                task.end_time = end_time  # 任务结束时间
                task.remaining_size = 0
                task.remaining_duration = 0
                # print(current_time, task.end_time-task.start_time, task.end_time)
                print(task)
                print(f'After {task.name} finished, available_size: {available_size}')
                completed_tasks.append(task)
                running_tasks.remove(task)
                current_time = end_time
    return current_time, available_size, running_tasks, completed_tasks, n


def advance_time_handler(current_time, next_time, next_borrow_start_time, next_borrow_end_time, task_queue,
                         running_tasks, n):
    if len(running_tasks) > 0 and current_time < next_time:
        next_time = current_time
        print(f'next_time --- {next_time}')
    else:
        next_times = [next_borrow_start_time, next_borrow_end_time]
        for task in running_tasks[:]:
            next_times.append(task.get_estimated_end_time())
        try:
            next_time = min([time for time in next_times if isinstance(time, (int, float)) and time > current_time])
            current_time = next_time
            print(f'next_time:  {next_times} --- {next_time}')
        except:
            if len(running_tasks) == 0 and len(task_queue) == 0:
                print('ALl DONE')
            else:
                n += 1
                # if n >4:
                #     return running_tasks
        # print(f'n = {n}')
    return current_time, next_time, n


def get_result(running_tasks, completed_tasks, interrupt_tasks, task_queue, is_save=False):


    # Result
    print(
        f'=================================interrupt_tasks {len(interrupt_tasks)}=======================================')
    for task in interrupt_tasks[:]:
        print(task)
    print(
        f'=================================completed_tasks {len(completed_tasks)}=======================================')
    for task in completed_tasks[:]:
        print(task)
    print(
        f'===================================running_tasks {len(running_tasks)}=======================================')
    for task in running_tasks[:]:
        print(task)
    print(
        f'=======================================task_queue {len(task_queue)}=======================================')
    for task in task_queue[:]:
        print(task)

    final_tasks = completed_tasks + interrupt_tasks + running_tasks + task_queue
    final_tasks = merge_subtasks(final_tasks)
    print(f'===================================final_tasks {len(final_tasks)}=======================================')
    for task in final_tasks:
        print(task.__str__(option=2))
        if task.is_main and task.subtasks:
            print('---------subtasks---------')
            for subtask in task.subtasks:
                print(subtask)
        print('------------------')

    if is_save:
        # 将输出内容写入文件
        with open('final_tasks.txt', 'a') as f:
            f.write('\n===========================final_tasks===========================\n')
            for task in final_tasks[:]:
                f.write(task.__str__(option=2))  # detail str
                f.write('\n')
                if task.is_main and task.subtasks:
                    f.write('---------subtasks---------\n')
                    for subtask in task.subtasks:
                        f.write(subtask.__str__(option=1))
                        f.write('\n')
                f.write('------------------\n')

            f.write('\n----------------------------------------------------------------------\n')
    return final_tasks



def schedule_tasks_ESGNN(tasks, available_size, borrow_schedule=[], is_save=False):
    n = 0  # 调控当task.remaining_duration>0，没有办法通过train model更新新的duration，而陷入循环
    running_tasks = []
    interrupt_tasks = []
    completed_tasks = []
    task_queue = []

    # Initial task queue
    for task in tasks:
        heapq.heappush(task_queue, task)

    current_time = 0
    next_time = 0
    total_remaining_size = sum(task.size for task in tasks)

    borrowed_applied = set()  # Track when borrowed space is applied
    returned_applied = set()  # Track when space is returned

    while task_queue or running_tasks:
        next_borrow_start_time = None
        next_borrow_end_time = None
        print(f"------------ Time: {current_time} min ---------------")

        # Check for borrowed schedule
        next_borrow_start_time, next_borrow_end_time, available_size, borrowed_applied, returned_applied = borrow_handler(
            current_time, available_size, borrow_schedule, borrowed_applied, returned_applied)

        # 中断Interrupt mechanism
        available_size, task_queue, running_tasks, interrupt_tasks = interrupt_handler(current_time, available_size,
                                                                                       task_queue, running_tasks,
                                                                                       interrupt_tasks,
                                                                                       key_func=lambda task: task.size,
                                                                                       reverse=True)

        # 运行Execute running tasks
        current_time, available_size, running_tasks, completed_tasks, n = execute_handler(current_time, available_size,
                                                                                          running_tasks,
                                                                                          completed_tasks, n)

        # 安排将要运行的任务Schedule new tasks based on current available size
        while task_queue and available_size > 0:
            # 更新优先级
            for task in task_queue:
                task.calculate_run_priority(current_time, total_remaining_size, weight_size=0.4,
                                            weight_waiting_time=0.3, weight_is_running=0.2, weight_is_sub=0.1)
            # 按优先级排序任务队列
            heapq.heapify(task_queue)
            task = heapq.heappop(task_queue)
            if task.remaining_size > available_size:
                sub_tasks = split_task(task, available_size)
                running_tasks.append(sub_tasks[0])
                heapq.heappush(task_queue, task)
                available_size -= sub_tasks[0].remaining_size
                sub_tasks[0].start_doing(current_time)
                print(sub_tasks[0])
                # print(task)
            else:
                running_tasks.append(task)
                available_size -= task.remaining_size
                task.start_doing(current_time)
                print(task)
            print(available_size)

        # Advance time
        current_time, next_time, n = advance_time_handler(current_time, next_time, next_borrow_start_time,
                                                          next_borrow_end_time, task_queue, running_tasks, n)

        print(f"Running tasks: {[task.name for task in running_tasks]}")
        print(f"Available size: {available_size}")

        task_list = running_tasks + completed_tasks + interrupt_tasks + task_queue
        for task in task_list:
            print(task)
        print('--------------------------------------------------------')

    # Result
    final_tasks = get_result(running_tasks, completed_tasks, interrupt_tasks, task_queue, is_save)

    return final_tasks


if __name__ == "__main__":
    dataset1 = Planetoid(root='/tmp/Cora', name='Cora')
    dataset2 = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    dataset3 = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    dataset4 = Flickr(root='/tmp/Flickr')

    # Define tasks
    tasks = []
    task_A = Task("A", 3, 3)  # size 3, duration 5min 3
    task_B = Task("B", 5, 15)  # size 10, duration 20min 1/4
    task_C = Task("C", 10, 20)  # size 10, duration 20min 4/4/2
    task_D = Task("D", 10, 20)

    task_A.data = dataset1[0]
    task_B.data = dataset2[0]
    task_C.data = dataset3[0]
    task_D.data = dataset4[0]

    task_A.arrival_time = 0

    tasks.append(task_A)
    tasks.append(task_B)
    # tasks.append(task_C)
    # tasks.append(task_D)

    # Define borrow schedule (borrow_start_time, borrow_end_time, borrow_space)
    available_size = 4
    borrow_schedule = [(2, 3, -1), (5, 6, 2)]  # Borrow 1 unit of space between time 4 and 6

    # Schedule tasks
    final_tasks = schedule_tasks_ESGNN(tasks, available_size=available_size, borrow_schedule=borrow_schedule,
                                       is_save=False)

    # plot_tasks(final_tasks)
    # evaluation_tasks_scheduler(tasks,available_gpu_size=available_size)
