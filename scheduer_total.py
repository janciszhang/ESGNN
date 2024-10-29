from scheduler_evaluation import evaluation_tasks_scheduler
from load_data import get_data_info
from scheduer_Lyra_plus import schedule_tasks_Lyra_plus
from scheduer_base import generate_available_size_schedule, generate_sine_borrow_schedule, plot_tasks
from scheduer_ESGNN import schedule_tasks_ESGNN
from scheduer_Lyra import schedule_tasks_Lyra
import copy
import heapq
import time
from torch_geometric.datasets import Planetoid
import copy
import heapq
import random
import pandas as pd
from torch_geometric.datasets import Planetoid, Flickr
from metis_partition import partition_K
from task import Task, split_task, merge_subtasks


def schedule_total():
    # available_size = 10
    available_size = random.randint(3, 10)
    inferred_size = 5

    # Define borrow schedule (borrow_start_time, borrow_end_time, borrow_space)
    # borrow_schedule = [(0, 10, -5), (11, 13, 2), (15, 16, -2), (20, 30, -2)]
    space = random.randint(-inferred_size, available_size - 3)
    borrow_schedule = generate_sine_borrow_schedule(0, 50, space, frequency=0.5, interval_duration=2)

    print(available_size)
    print(borrow_schedule)

    dataset1 = Planetoid(root='/tmp/Cora', name='Cora')
    dataset2 = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    dataset3 = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    dataset4 = Flickr(root='/tmp/Flickr')
    get_data_info(dataset1, is_print=True, is_save=False)
    get_data_info(dataset2, is_print=True, is_save=False)

    # Schedule tasks
    is_save = True
    schedule_method = ['ESGNN', 'Lyra', 'Lyra Plus']
    for i in range(3):
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

        # tasks.append(task_A)
        # tasks.append(task_B)
        # tasks.append(task_C)
        # tasks.append(task_D)
        if random.random() < 0.5:
            tasks.append(task_A)
        if random.random() < 0.5:
            tasks.append(task_B)
        if random.random() < 0.5:
            tasks.append(task_C)
        if random.random() < 0.5:
            tasks.append(task_D)

        final_tasks = []
        with open('final_tasks.txt', 'a') as f1:
            print(f'///////////////////{schedule_method[i]}///////////////////')
            f1.write(f'///////////////////{schedule_method[i]}///////////////////\n')

        if i == 0:
            try:
                final_tasks = schedule_tasks_ESGNN(tasks, available_size=available_size,
                                                   borrow_schedule=borrow_schedule, is_save=is_save)
                plot_tasks(final_tasks)
            except Exception as e:
                print(e)
                f1.write(e)
                f1.write('\n')

        if i == 1:
            try:
                final_tasks = schedule_tasks_Lyra(tasks, available_size=available_size,
                                                  borrow_schedule=borrow_schedule, is_save=is_save)
                plot_tasks(final_tasks)
            except Exception as e:
                print(e)
                f1.write(e)
                f1.write('\n')

        if i == 2:
            try:
                final_tasks = schedule_tasks_Lyra_plus(tasks, available_size=available_size,
                                                       borrow_schedule=borrow_schedule, is_save=is_save)
                plot_tasks(final_tasks)
            except Exception as e:
                f1.write(e)
                f1.write('\n')

        with open('evaluation.txt', 'a') as f2:
            print(f'///////////////////{schedule_method[i]}///////////////////')
            f2.write(f'///////////////////{schedule_method[i]}///////////////////\n')
        try:
            evaluate_result = evaluation_tasks_scheduler(final_tasks, available_gpu_size=available_size,
                                                         borrow_schedule=borrow_schedule, is_save=is_save)
            print(evaluate_result)
        except Exception as e:
            print(e)
            # f2.write(str(e))
            # f2.write('\n')


if __name__ == '__main__':
    schedule_total()
