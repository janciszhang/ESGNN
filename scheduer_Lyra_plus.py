"""
有时候会报错，可以多运行几次，会有成功的
"""
from scheduer_base import find_minimum_end_time_indexes, get_result, borrow_handler, interrupt_handler, execute_handler, \
    advance_time_handler, plot_tasks
from task import Task, split_task, merge_subtasks
import copy
import heapq
import random
from torch_geometric.datasets import Planetoid, Flickr
from metis_partition import partition_K
from task import Task, split_task, new_task
import time



def schedule_tasks_Lyra_plus(tasks, available_size, borrow_schedule=[],is_save=False):
    n=0 # 调控当task.remaining_duration>0，没有办法通过train model更新新的duration，而陷入循环
    running_tasks = []
    interrupt_tasks = []
    completed_tasks = []
    task_queue = []

    # Initial task queue
    # 构建基于 size 的临时优先队列# Lyra根据size
    temp_queue = [(task.size, task) for task in tasks]

    # 使用 heapify 构建堆
    heapq.heapify(temp_queue)

    # 弹出任务时，只获取实际的 Task 对象
    task_queue = [heapq.heappop(temp_queue)[1] for _ in range(len(temp_queue))]
    # for task in tasks:
    #     heapq.heappush(task_queue, (task.size, task))

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
            # task_queue.sort(key=lambda task: task.size, reverse=False)
            task_queue.sort(key=lambda task: (task.arrival_time > current_time, task.size))
            # for task in task_queue:
            #     task.calculate_run_priority(current_time, total_remaining_size, weight_size=0.4,
            #                                 weight_waiting_time=0.3, weight_is_running=0.2, weight_is_sub=0.1)
            # 按优先级排序任务队列
            # heapq.heapify(task_queue)
            # 构建基于 size 的临时优先队列
            # temp_queue = [(task.size, task) for task in task_queue]
            # # 使用 heapify 构建堆
            # heapq.heapify(temp_queue)
            # # 弹出任务时，只获取实际的 Task 对象
            # task_queue = [heapq.heappop(temp_queue)[1] for _ in range(len(temp_queue))]
            task = heapq.heappop(task_queue)
            if task.remaining_size > available_size:
                if len(running_tasks) == 0:
                    sub_tasks = split_task(task, available_size)
                    running_tasks.append(sub_tasks[0])
                    print(f'main task: {task.__str__()}')
                    heapq.heappush(task_queue, task)

                    available_size -=sub_tasks[0].remaining_size
                    sub_tasks[0].start_doing(current_time)
                    print(f'sub task: {sub_tasks[0]}')
                else:
                    print(f'Cannot handle this task: {task.name} now. Out of available GPU size.')
                    heapq.heappush(task_queue, task)
                    break
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

        task_list = running_tasks + completed_tasks+ interrupt_tasks+ task_queue
        for task in task_list:
            print(task)
        print('--------------------------------------------------------')


    # Result
    final_tasks = get_result(running_tasks, completed_tasks, interrupt_tasks,task_queue,is_save)

    return final_tasks





if __name__ == "__main__":
    dataset1 = Planetoid(root='/tmp/Cora', name='Cora')
    dataset2 = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    # dataset3 = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    # dataset4 = Flickr(root='/tmp/Flickr')

    # Define tasks
    tasks = []
    task_A = Task("A", size=3.414, duration=7.071, arrival_time=0)
    task_B = Task("B", size=24.677, duration=8.083, arrival_time=0)
    task_C = Task("C", size=7.175, duration=17.649, arrival_time=4)
    task_D = Task("D", size=6.768, duration=5.908, arrival_time=5)

    task_A.data = dataset1[0]
    task_B.data = dataset2[0]
    task_C.data = dataset1[0]
    task_D.data = dataset2[0]

    tasks.append(task_A)
    tasks.append(task_B)
    tasks.append(task_C)
    tasks.append(task_D)

    # Define borrow schedule (borrow_start_time, borrow_end_time, borrow_space)
    available_size = 10
    borrow_schedule = [(2, 3, 9), (5, 6, 2), (8, 9, -1)]  # Borrow 1 unit of space between time 4 and 6

    # Schedule tasks
    is_save = False
    final_tasks = schedule_tasks_Lyra_plus(tasks, available_size=available_size, borrow_schedule=borrow_schedule,
                                           is_save=is_save)
    plot_tasks(final_tasks)