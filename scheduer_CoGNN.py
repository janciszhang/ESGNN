"""
有时候会报错，可以多运行几次，会有成功的
"""
from scheduler_evaluation import evaluation_tasks_scheduler
from task import split_task_K
from scheduer_base import find_minimum_end_time_indexes, get_result, borrow_handler, interrupt_handler, execute_handler, \
    advance_time_handler
from task import Task, split_task, merge_subtasks
from viz import plot_tasks
import copy
import heapq
import random
from torch_geometric.datasets import Planetoid, Flickr
from metis_partition import partition_K
from task import Task, split_task, new_task
import time


def schedule_tasks_CoGNN(tasks, available_size, m_max = 50, is_save=False):
    n = 0  # 调控当task.remaining_duration>0，没有办法通过train model更新新的duration，而陷入循环
    m = 0 # 调控无法处理的大任务，避免没有足够GPU而陷入循环(m_max最多循环多少次，return)
    running_tasks = []
    interrupt_tasks = []
    completed_tasks = []
    task_queue = []

    # Initial task queue
    for task in tasks:
        sub_tasks = split_task_K(task, K=4)
        for sub_task in sub_tasks:
            print(sub_task)
            heapq.heappush(task_queue, sub_task)


    current_time = 0
    next_time = 0
    total_remaining_size = sum(task.size for task in tasks)

    borrowed_applied = set()  # Track when borrowed space is applied
    returned_applied = set()  # Track when space is returned
    while task_queue or running_tasks:
        next_borrow_start_time = None
        next_borrow_end_time = None
        print(f"------------ Time: {current_time} min ---------------")

        # # Check for borrowed schedule
        # next_borrow_start_time, next_borrow_end_time, available_size, borrowed_applied, returned_applied = borrow_handler(
        #     current_time, available_size, borrow_schedule, borrowed_applied, returned_applied)
        #
        # # 中断Interrupt mechanism (Lyra根据size)
        # available_size, task_queue, running_tasks, interrupt_tasks = interrupt_handler(current_time, available_size,
        #                                                                                task_queue, running_tasks,
        #                                                                                interrupt_tasks,
        #                                                                                key_func=lambda task: task.size,
        #                                                                                reverse=True)


        # 运行Execute running tasks
        current_time, available_size, running_tasks, completed_tasks, n = execute_handler(current_time, available_size,
                                                                                          running_tasks,
                                                                                          completed_tasks, n)

        # 安排将要运行的任务Schedule new tasks based on current available size
        while task_queue and available_size > 0:
            # 更新优先级
            # task_queue.sort(key=lambda task: task.size, reverse=False)
            task_queue.sort(key=lambda task: (task.arrival_time > current_time, task.pmc)) # CoGNN按照pmc
            task = heapq.heappop(task_queue)
            if task.arrival_time > current_time:
                heapq.heappush(task_queue, task)
                break  # run priority already consider arrival_time
            else:
                if task.remaining_size > available_size:
                    print(f'Cannot handle this task: {task.name} now. Out of available GPU size.')
                    heapq.heappush(task_queue, task)
                    m += 1
                    print(f'm = {m}')
                    if m > m_max * len(running_tasks+completed_tasks+interrupt_tasks+task_queue):
                        return get_result(running_tasks, completed_tasks, interrupt_tasks, task_queue, is_save)
                    break
                else:
                    running_tasks.append(task)
                    available_size -= task.remaining_size
                    task.start_doing(current_time)
                    print(task)
                print(f'After {task.name} schedule to running_tasks, available_size: {available_size}')
        print(f'After all schedule, available_size: {available_size}')


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
    final_tasks = get_result(running_tasks, completed_tasks, interrupt_tasks, task_queue,is_save)

    return final_tasks





if __name__ == "__main__":
    dataset1 = Planetoid(root='/tmp/Cora', name='Cora')
    dataset2 = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    dataset3 = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    dataset4 = Flickr(root='/tmp/Flickr')

    # Define tasks
    tasks=[]
    task_A = Task("A", 3, 3, arrival_time=0)  # size 3, duration 5min 3
    task_B = Task("B", 4, 15, arrival_time=2)  # size 10, duration 20min 1/4
    task_C = Task("C", 7, 20, arrival_time=4)  # size 10, duration 20min 4/4/2
    task_D = Task("D", 2, 20, arrival_time=5)

    task_A.data=dataset1[0]
    task_B.data=dataset2[0]
    task_C.data=dataset3[0]
    task_D.data=dataset4[0]


    tasks.append(task_A)
    tasks.append(task_B)
    tasks.append(task_C)
    tasks.append(task_D)



    # Define borrow schedule (borrow_start_time, borrow_end_time, borrow_space)
    available_size = 4
    borrow_schedule = [(2,3,-1),(5, 6, 2)]  # Borrow 1 unit of space between time 4 and 6


    # Schedule tasks
    final_tasks = schedule_tasks_CoGNN(tasks, available_size=available_size,is_save=False)
    plot_tasks(final_tasks)


    evaluation_tasks_scheduler(tasks,available_gpu_size=available_size)

