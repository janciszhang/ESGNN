import argparse
import ast
import os

from ESGNN.load_task import select_datasets, load_tasks, select_datasets_by_name
from ESGNN.test_cuda import set_gpu_memory
from scheduer_CoGNN import schedule_tasks_CoGNN
from scheduer_CoGNN_plus import schedule_tasks_CoGNN_plus
from scheduer_Baseline import schedule_tasks_Baseline
from scheduer_HongTu import schedule_tasks_HongTu
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


def schedule_prepare(num_small=4, num_medium=0, num_large=0, available_size = 800, space = 200 ,arrival_time_range =[0, 10]):
    # dataset_category_list = select_datasets_by_name(dataset_name_list=['Cora', 'Citeseer', 'Pubmed', 'Amazon_Computers', 'Amazon_Photo', 'Flickr', 'Reddit','OGBN_arxiv', 'OGBN_products'])
    # dataset_category_list = select_datasets_by_name(dataset_name_list=['Cora', 'Citeseer', 'Pubmed', 'Amazon_Computers','Amazon_Photo','Flickr', 'Reddit', 'OGBN_products'])
    # dataset_category_list = select_datasets_by_name(dataset_name_list=['Cora', 'Citeseer', 'Pubmed','DBLP', 'WikiCS', 'Cornell','Texas','Wisconsin']) # small
    # dataset_category_list = select_datasets_by_name(dataset_name_list=['Cora', 'Citeseer', 'Pubmed'])
    # dataset_category_list = select_datasets(num_small=4, num_medium=1, num_large=0)
    dataset_category_list = select_datasets(num_small=num_small, num_medium=num_medium, num_large=num_large)
    tasks_orginal = load_tasks(dataset_category_list, arrival_time_range)

    # Define available_size GPU size
    # available_size = 7
    available_size = available_size  # MB
    # available_size = random.randint(100, 200)
    # set_gpu_memory(1600)

    inferred_size = 400

    # Define borrow schedule (borrow_start_time, borrow_end_time, borrow_space)
    # borrow_schedule = [(2, 3, -10), (15, 16, 20)]  # Borrow 1 unit of space between time 4 and 6
    # space = random.choice([i for i in range(-inferred_size, available_size - 3) if i != 0])
    # space = random.randint(-inferred_size, available_size - 100)
    space = space
    # borrow_schedule = generate_sine_borrow_schedule(0, 200, space, frequency=0.5, interval_duration=2)
    borrow_schedule = generate_sine_borrow_schedule(0, 1000000, space, frequency=0.05, interval_duration=10)

    # dataset1 = Planetoid(root='/tmp/Cora', name='Cora')
    # dataset2 = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    # dataset3 = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    # dataset4 = Flickr(root='/tmp/Flickr')
    # get_data_info(dataset1, is_print=True, is_save=False)
    # get_data_info(dataset2, is_print=True, is_save=False)
    #
    # random_task_select=[]
    # for i in range(4):
    #     random_task_select.append(random.random()<0.5)


    # borrow_schedule = [(0, 2, 0), (2, 4, 1), (4, 6, 1), (6, 8, 0), (8, 10, -1), (10, 12, -1), (12, 14, 0), (14, 16, 1),
    #                    (16, 18, 1), (18, 20, 0), (20, 22, -1), (22, 24, -1), (24, 26, -1), (26, 28, 0), (28, 30, 1),
    #                    (30, 32, 1), (32, 34, 0), (34, 36, -1), (36, 38, -1), (38, 40, 0), (40, 42, 1), (42, 44, 1),
    #                    (44, 46, 0), (46, 48, -1), (48, 50, -1), (50, 52, 0)]
    # random_task_select = [True, True, True, True]

    print(f'available_size = {available_size}')
    print(f'borrow_schedule = {borrow_schedule}')
    print(f'random_task_select = {dataset_category_list}')
    with open('test_sample.txt', 'a') as f0:
        f0.write(f'=================================Sample=======================================\n')
        f0.write(f'available_size = {available_size}\n')
        f0.write(f'borrow_schedule = {borrow_schedule}\n')
        f0.write(f'random_task_select = {dataset_category_list}\n')

    return tasks_orginal,available_size,borrow_schedule

def schedule_total(para_list=[4,0,0,800,200,[0, 10]], schedule_method_names = ['Baseline','CoGNN','CoGNN Plus','Lyra', 'Lyra Plus','HongTu','ESGNN'], is_save = True):
    tasks_orginal, available_size, borrow_schedule = schedule_prepare(*para_list)

    # Schedule tasks
    # schedule_method_names = ['ESGNN','Baseline','CoGNN','CoGNN Plus','Lyra', 'Lyra Plus','HongTu']
    evaluate_results=[]
    number = random.randint(1, 99999)  # 隨機生成 1 到 99999 之間的數字
    folder_name = f"{int(time.time())}"  # 格式化為五位數字，前面補 0
    os.makedirs(f'../img/{folder_name}', exist_ok=True)  # 如果文件夹已存在，不会抛出错误
    for schedule_method_name in schedule_method_names:
        # Define tasks
        tasks=copy.deepcopy(tasks_orginal)

        # tasks = []
        # task_A = Task("A", size=3.414, duration=7.071, arrival_time=0)
        # task_B = Task("B", size=24.677, duration=8.083, arrival_time=0)
        # task_C = Task("C", size=7.175, duration=17.649, arrival_time=4)
        # task_D = Task("D", size=6.768, duration=5.908, arrival_time=5)
        #
        # task_A.data = dataset1[0]
        # task_B.data = dataset2[0]
        # task_C.data = dataset1[0]
        # task_D.data = dataset2[0]
        #
        # if random_task_select[0]:
        #     tasks.append(task_A)
        # if random_task_select[1]:
        #     tasks.append(task_B)
        # if random_task_select[2]:
        #     tasks.append(task_C)
        # if random_task_select[3]:
        #     tasks.append(task_D)

        with open('test_sample.txt', 'a') as f0:
            f0.write(f'folder_name: {folder_name}')
            f0.write(f'///////////////////{schedule_method_name}///////////////////\n')
            for task in tasks:
                print(task)
                f0.write(task.__str__())
                f0.write('\n')


        final_tasks = []
        with open('final_tasks.txt', 'a') as f1:
            f1.write(f'folder_name: {folder_name}')
            print(f'///////////////////{schedule_method_name}///////////////////')
            f1.write(f'///////////////////{schedule_method_name}///////////////////\n')


        if schedule_method_name == 'Baseline':
            try:
                final_tasks = schedule_tasks_Baseline(tasks, available_size=available_size, is_save=is_save)
                plot_tasks(final_tasks,schedule_method_name=schedule_method_name,folder_name=folder_name)
            except Exception as e:
                print(e)
                f1.write(e.__str__())
                f1.write('\n')

        if schedule_method_name == 'CoGNN':
            try:
                final_tasks = schedule_tasks_CoGNN(tasks, available_size=available_size, is_save=is_save)
                plot_tasks(final_tasks,schedule_method_name=schedule_method_name,folder_name=folder_name)
            except Exception as e:
                print(e)
                f1.write(e.__str__())
                f1.write('\n')

        if schedule_method_name == 'CoGNN Plus':
            try:
                final_tasks = schedule_tasks_CoGNN_plus(tasks, available_size=available_size, is_save=is_save)
                plot_tasks(final_tasks,schedule_method_name=schedule_method_name,folder_name=folder_name)
            except Exception as e:
                print(e)
                f1.write(e.__str__())
                f1.write('\n')

        if schedule_method_name == 'Lyra':
            try:
                final_tasks = schedule_tasks_Lyra(tasks, available_size=available_size,
                                                  borrow_schedule=borrow_schedule, is_save=is_save)
                plot_tasks(final_tasks,schedule_method_name=schedule_method_name,folder_name=folder_name)
            except Exception as e:
                print(e)
                f1.write(e.__str__())
                f1.write('\n')

        if schedule_method_name == 'Lyra Plus':
            try:
                final_tasks = schedule_tasks_Lyra_plus(tasks, available_size=available_size,
                                                       borrow_schedule=borrow_schedule, is_save=is_save)
                plot_tasks(final_tasks,schedule_method_name=schedule_method_name,folder_name=folder_name)
            except Exception as e:
                f1.write(e.__str__())
                f1.write('\n')

        if schedule_method_name == 'HongTu':
            try:
                final_tasks = schedule_tasks_HongTu(tasks, available_size=available_size, is_save=is_save)
                plot_tasks(final_tasks,schedule_method_name=schedule_method_name,folder_name=folder_name)
            except Exception as e:
                print(e)
                f1.write(e.__str__())
                f1.write('\n')

        if schedule_method_name == 'ESGNN':
            try:
                final_tasks = schedule_tasks_ESGNN(tasks, available_size=available_size,
                                                   borrow_schedule=borrow_schedule, is_save=is_save)
                plot_tasks(final_tasks,schedule_method_name=schedule_method_name,folder_name=folder_name)
            except Exception as e:
                print(e)
                f1.write(e.__str__())
                f1.write('\n')


        with open('evaluation.txt', 'a') as f2:
            f2.write(f'folder_name: {folder_name} - {schedule_method_name}')
            # print(f'///////////////////{schedule_method_name///////////////////')
            # f2.write(f'///////////////////{schedule_method_name}///////////////////\n')

        try:
            if schedule_method_name == 'Lyra' or schedule_method_name == 'Lyra Plus' or schedule_method_name == 'ESGNN':
                evaluate_result = evaluation_tasks_scheduler(final_tasks, available_gpu_size=available_size,
                                                             borrow_schedule=borrow_schedule, is_save=is_save,schedule_method_name=schedule_method_name) # 評估有借貸的方法
            else:
                evaluate_result = evaluation_tasks_scheduler(final_tasks, available_gpu_size=available_size,
                                                             borrow_schedule=[], is_save=is_save,schedule_method_name=schedule_method_name) # 評估無借貸的方法
            print(evaluate_result)
            evaluate_results.append(evaluate_result)
        except Exception as e:
            print(e)
            # f2.write(str(e))
            # f2.write('\n')

    viz_evaluate_results(evaluate_results,schedule_method_names=schedule_method_names,folder_name=folder_name)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run scheduler tasks.")
    parser.add_argument('para_list', type=str, help="Dataset number list for small medium large")

    # Parse arguments
    args = parser.parse_args()

    # Convert the string to a list
    para_list = ast.literal_eval(args.dataset_category_list)

    # Call the function with the parsed list
    schedule_total(para_list)

def test(available_size = 800, space = 200):
    for i in range(3):
        try:
            schedule_total([10, 0, 0, available_size, space, [0, 5]])
            i+=1
        except:
            pass
    for i in range(3):
        try:
            schedule_total([10, 0, 0, available_size, space, [0, 100]])
            i+=1
        except:
            pass

    for i in range(3):
        try:
            schedule_total([5, 5, 0, available_size, space, [0, 5]])
            i+=1
        except:
            pass
    for i in range(3):
        try:
            schedule_total([5, 5, 0, available_size, space, [0, 100]])
            i+=1
        except:
            pass

    for i in range(3):
        try:
            schedule_total([0, 10, 0, available_size, space, [0, 5]])
            i+=1
        except:
            pass
    for i in range(3):
        try:
            schedule_total([0, 10, 0, available_size, space, [0, 100]])
            i+=1
        except:
            pass



if __name__ == '__main__':
    # schedule_total([10, 0, 0, 800, 200, [0, 5]])
    # schedule_total([10, 0, 0, 800, 200, [0, 100]])
    # schedule_total([10, 0, 0, 800, 200, [0, 500]])
    # schedule_total([5, 5, 0, 800, 200, [0, 5]])
    # schedule_total([5, 5, 0, 800, 200, [0, 100]])
    # schedule_total([5, 5, 0, 800, 200, [0, 500]])
    # schedule_total([0, 10, 0, 800, 200, [0, 5]])
    # schedule_total([0, 10, 0, 800, 200, [0, 100]])
    # schedule_total([0, 10, 0, 800, 200, [0, 500]])
    #
    # schedule_total([10, 0, 0, 400, 100, [0, 5]])
    # schedule_total([10, 0, 0, 400, 100, [0, 100]])
    # schedule_total([10, 0, 0, 400, 100, [0, 500]])
    # schedule_total([5, 5, 0, 400, 100, [0, 5]])
    # schedule_total([5, 5, 0, 400, 100, [0, 100]])
    # schedule_total([5, 5, 0, 400, 100, [0, 500]])
    # schedule_total([0, 10, 0, 400, 100, [0, 5]])
    # schedule_total([0, 10, 0, 400, 100, [0, 100]])
    # schedule_total([0, 10, 0, 400, 100, [0, 500]])
    #
    # schedule_total([10, 0, 0, 200, 50, [0, 5]]) #error
    # schedule_total([10, 0, 0, 200, 50, [0, 100]])
    # schedule_total([10, 0, 0, 200, 50, [0, 500]]) #error
    # schedule_total([5, 5, 0, 200, 50, [0, 5]])
    # schedule_total([5, 5, 0, 200, 50, [0, 100]])
    # schedule_total([5, 5, 0, 200, 50, [0, 500]])
    # schedule_total([0, 10, 0, 200, 50, [0, 5]])
    # schedule_total([0, 10, 0, 200, 50, [0, 100]])
    # schedule_total([0, 10, 0, 200, 50, [0, 500]])
    #
    # schedule_total([10, 0, 0, 100, 25, [0, 5]]) #error
    # schedule_total([10, 0, 0, 100, 25, [0, 100]]) #error
    # schedule_total([10, 0, 0, 100, 25, [0, 500]]) #error
    # schedule_total([5, 5, 0, 100, 25, [0, 5]])
    # schedule_total([5, 5, 0, 100, 25, [0, 100]])
    # schedule_total([5, 5, 0, 100, 25, [0, 500]])
    # schedule_total([0, 10, 0, 100, 25, [0, 5]])
    # schedule_total([0, 10, 0, 100, 25, [0, 100]])
    # schedule_total([0, 10, 0, 100, 25, [0, 500]])

    # schedule_total([10, 0, 0, 20, 50, [0, 5]]) #error
    # schedule_total([10, 0, 0, 20, 50, [0, 100]]) #error
    # schedule_total([10, 0, 0, 20, 50, [0, 500]]) #error
    # schedule_total([5, 5, 0, 20, 5, [0, 5]])
    # schedule_total([5, 5, 0, 20, 5, [0, 100]])
    # schedule_total([5, 5, 0, 20, 5, [0, 500]])
    # schedule_total([0, 10, 0, 200, 50, [0, 5]])
    # schedule_total([0, 10, 0, 200, 50, [0, 100]])
    # schedule_total([0, 10, 0, 200, 50, [0, 500]])

    schedule_method_names = ['ESGNN', 'Baseline', 'CoGNN', 'CoGNN Plus', 'Lyra', 'Lyra Plus', 'HongTu']

    # schedule_total([5, 5, 0, 200, 50, [0, 5]],schedule_method_names)
    schedule_total([5, 0, 0, 100, 25, [0, 5]], schedule_method_names)


