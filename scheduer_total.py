from ESGNN.scheduer_Baseline import schedule_tasks_Baseline
from ESGNN.scheduer_HongTu import schedule_tasks_HongTu
from viz import viz_evaluate_results, plot_tasks
from scheduler_evaluation import evaluation_tasks_scheduler
from load_data import get_data_info
from scheduer_Lyra_plus import schedule_tasks_Lyra_plus
from scheduer_base import generate_available_size_schedule, generate_sine_borrow_schedule
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
    available_size = random.randint(3, 10)
    # available_size = 3
    inferred_size = 5

    # Define borrow schedule (borrow_start_time, borrow_end_time, borrow_space)
    # space = random.choice([i for i in range(-inferred_size, available_size - 3) if i != 0])
    space = random.randint(-inferred_size, available_size - 3)
    borrow_schedule = generate_sine_borrow_schedule(0, 50, space, frequency=0.5, interval_duration=2)


    dataset1 = Planetoid(root='/tmp/Cora', name='Cora')
    dataset2 = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    dataset3 = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    dataset4 = Flickr(root='/tmp/Flickr')
    get_data_info(dataset1, is_print=True, is_save=False)
    get_data_info(dataset2, is_print=True, is_save=False)

    random_task_select=[]
    for i in range(4):
        random_task_select.append(random.random()<0.5)


    available_size = 7
    borrow_schedule = [(0, 2, 0), (2, 4, 1), (4, 6, 1), (6, 8, 0), (8, 10, -1), (10, 12, -1), (12, 14, 0), (14, 16, 1),
                      (16, 18, 1), (18, 20, 0), (20, 22, -1), (22, 24, -1), (24, 26, -1), (26, 28, 0), (28, 30, 1),
                      (30, 32, 1), (32, 34, 0), (34, 36, -1), (36, 38, -1), (38, 40, 0), (40, 42, 1), (42, 44, 1),
                      (44, 46, 0), (46, 48, -1), (48, 50, -1), (50, 52, 0)]
    random_task_select = [True, True, True, True]

    print(f'available_size = {available_size}')
    print(f'borrow_schedule = {borrow_schedule}')
    print(f'random_task_select = {random_task_select}')
    with open('test_sample.txt', 'a') as f0:
        f0.write(f'=================================Sample=======================================')
        f0.write(f'available_size = {available_size}\n')
        f0.write(f'borrow_schedule = {borrow_schedule}\n')
        f0.write(f'random_task_select = {random_task_select}\n')


    # Schedule tasks
    is_save = True
    schedule_method_name = ['Baseline','Lyra', 'Lyra Plus','HongTu','ESGNN']
    evaluate_results=[]
    for i in range(len(schedule_method_name)):
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

        if random_task_select[0]:
            tasks.append(task_A)
        if random_task_select[1]:
            tasks.append(task_B)
        if random_task_select[2]:
            tasks.append(task_C)
        if random_task_select[3]:
            tasks.append(task_D)

        with open('test_sample.txt', 'a') as f0:
            f0.write(f'///////////////////{schedule_method_name[i]}///////////////////\n')
            for task in tasks:
                print(task)
                f0.write(task.__str__())
                f0.write('\n')


        final_tasks = []
        with open('final_tasks.txt', 'a') as f1:
            print(f'///////////////////{schedule_method_name[i]}///////////////////')
            f1.write(f'///////////////////{schedule_method_name[i]}///////////////////\n')


        if i == 0:
            try:
                final_tasks = schedule_tasks_Baseline(tasks, available_size=available_size, is_save=is_save)
                plot_tasks(final_tasks)
            except Exception as e:
                print(e)
                f1.write(e.__str__())
                f1.write('\n')

        if i == 1:
            try:
                final_tasks = schedule_tasks_Lyra(tasks, available_size=available_size,
                                                  borrow_schedule=borrow_schedule, is_save=is_save)
                plot_tasks(final_tasks)
            except Exception as e:
                print(e)
                f1.write(e.__str__())
                f1.write('\n')

        if i == 2:
            try:
                final_tasks = schedule_tasks_Lyra_plus(tasks, available_size=available_size,
                                                       borrow_schedule=borrow_schedule, is_save=is_save)
                plot_tasks(final_tasks)
            except Exception as e:
                f1.write(e.__str__())
                f1.write('\n')

        if i == 3:
            try:
                final_tasks = schedule_tasks_HongTu(tasks, available_size=available_size, is_save=is_save)
                plot_tasks(final_tasks)
            except Exception as e:
                print(e)
                f1.write(e.__str__())
                f1.write('\n')

        if i == 4:
            try:
                final_tasks = schedule_tasks_ESGNN(tasks, available_size=available_size,
                                                   borrow_schedule=borrow_schedule, is_save=is_save)
                plot_tasks(final_tasks)
            except Exception as e:
                print(e)
                f1.write(e.__str__())
                f1.write('\n')


        with open('evaluation.txt', 'a') as f2:
            print(f'///////////////////{schedule_method_name[i]}///////////////////')
            f2.write(f'///////////////////{schedule_method_name[i]}///////////////////\n')

        try:
            if i==0 or i ==3:
                evaluate_result = evaluation_tasks_scheduler(final_tasks, available_gpu_size=available_size,borrow_schedule=[], is_save=is_save)
            else:
                evaluate_result = evaluation_tasks_scheduler(final_tasks, available_gpu_size=available_size,borrow_schedule=borrow_schedule, is_save=is_save)
            print(evaluate_result)
            evaluate_results.append(evaluate_result)
        except Exception as e:
            print(e)
            # f2.write(str(e))
            # f2.write('\n')
    viz_evaluate_results(evaluate_results,schedule_method_name)


if __name__ == '__main__':
    schedule_total()
