"""
有时候会报错，可以多运行几次，会有成功的
"""
from ESGNN.es_gpu_time_memory import es_gpu
from ESGNN.load_task import select_datasets, load_tasks
from load_data import get_folder_size
from test_cuda import set_gpu_memory
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


def schedule_tasks_ESGNN(tasks, available_size, borrow_schedule=[],is_save=False):
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
        print(f"------------ Time: {current_time} seconds ---------------")

        # Check for borrowed schedule
        next_borrow_start_time, next_borrow_end_time, available_size, borrowed_applied, returned_applied = borrow_handler(
            current_time, available_size, borrow_schedule, borrowed_applied, returned_applied)

        # 中断Interrupt mechanism (ESGNN根据interrupt_priority)
        for task in running_tasks:
            task.calculate_interrupt_priority(current_time, total_remaining_size, weight_size=0.4,
                                              weight_remaining_duration=0.3, weight_waiting_time=0.2, weight_is_sub=0.1)
        available_size, task_queue, running_tasks, interrupt_tasks = interrupt_handler(current_time, available_size,task_queue, running_tasks,interrupt_tasks, key_func=lambda task: task.interrupt_priority, reverse=False)


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
            if task.arrival_time > current_time:
                print(f'Task {task.name} arrival_time({task.arrival_time}) < current_time({current_time}) not arrived: {task}')
                heapq.heappush(task_queue, task)
                break # calculate_run_priority already consider arrival_time
            else:
                if task.remaining_size > available_size:
                    sub_tasks = split_task(task, available_size)
                    running_tasks.append(sub_tasks[0])
                    heapq.heappush(task_queue, task)
                    available_size -=sub_tasks[0].remaining_size
                    sub_tasks[0].start_doing(current_time)
                    print(sub_tasks[0])
                    print(f'After {sub_tasks[0].name} schedule to running_tasks, available_size: {available_size}')
                else:
                    running_tasks.append(task)
                    available_size -= task.remaining_size
                    task.start_doing(current_time)
                    print(task)
                    print(f'After {task.name} schedule to running_tasks, available_size: {available_size}')
        print(f'After all schedule, available_size: {available_size}')


        # Advance time
        current_time, next_time, n = advance_time_handler(current_time, next_time, next_borrow_start_time, next_borrow_end_time, task_queue,running_tasks, n)

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
    # dataset1 = Planetoid(root='/tmp/Cora', name='Cora')
    # dataset2 = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    # dataset3 = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    # dataset4 = Flickr(root='/tmp/Flickr')
    # duration1 = es_gpu(dataset1)[0]
    # duration2 = es_gpu(dataset2)[0]
    # duration3 = es_gpu(dataset3)[0]
    # duration4 = es_gpu(dataset4)[0]
    # size1 = es_gpu(dataset1)[1]
    # size2 = es_gpu(dataset2)[1]
    # size3 = es_gpu(dataset3)[1]
    # size4 = es_gpu(dataset4)[1]
    # print(f'Dataset Size: {size1:.2f}, {size2:.2f}, {size3:.2f}, {size4:.2f}') # 15.54, 48.16, 47.49, 529.25
    #
    # # Define tasks
    # tasks=[]
    # task_A = Task("A", size1, duration1)
    # task_B = Task("B", size2, duration2)
    # task_C = Task("C", size3, duration3)
    # task_D = Task("D", size4, duration4)
    # # task_A = Task("A", 3, 3)  # size 3, duration 5min 3
    # # task_B = Task("B", 5, 15)  # size 10, duration 20min 1/4
    # # task_C = Task("C", 10, 20)  # size 10, duration 20min 4/4/2
    # # task_D = Task("D", 10, 20)
    #
    # task_A.data=dataset1[0]
    # task_B.data=dataset2[0]
    # task_C.data=dataset3[0]
    # task_D.data=dataset4[0]
    # # task_A.size=size1
    # # task_B.size=size2
    # # task_C.size=size3
    # # task_D.size=size4
    #
    # task_A.arrival_time=0
    #
    # tasks.append(task_A)
    # tasks.append(task_B)
    # # tasks.append(task_C)
    # # tasks.append(task_D)
    dataset_category_list = select_datasets(num_small=0, num_medium=0, num_large=1)
    tasks = load_tasks(dataset_category_list)



    # Define borrow schedule (borrow_start_time, borrow_end_time, borrow_space)
    available_size = 200 # MB
    set_gpu_memory(200)
    borrow_schedule = [(2,3,-10),(15, 16, 20)]  # Borrow 1 unit of space between time 4 and 6


    # Schedule tasks
    final_tasks = schedule_tasks_ESGNN(tasks, available_size=available_size, borrow_schedule=borrow_schedule,is_save=False)
    plot_tasks(final_tasks)


    # evaluation_tasks_scheduler(tasks,available_gpu_size=available_size)

