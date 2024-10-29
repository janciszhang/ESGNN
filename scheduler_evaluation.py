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

def merge_intervals(intervals):
    # 步骤 1: 排序区间
    intervals.sort(key=lambda x: x[0])

    merged = []

    for interval in intervals:
        time_range=[interval[0],interval[1]]
        # 如果 merged 为空或者当前区间不重叠，添加到 merged
        if not merged or merged[-1][1] < time_range[0]:
            merged.append(time_range)
        else:
            # 合并重叠的区间
            merged[-1][1] = max(merged[-1][1], time_range[1])
    return merged
def get_net_gpu_time(total_gpu_time_ranges):
    net_gpu_time = 0
    # time_intervals=[]
    # for task in tasks:
    #     time_intervals.append([task.start_time,task.end_time])
    merged_time_ranges = merge_intervals(total_gpu_time_ranges)
    for time_range in merged_time_ranges:
        net_gpu_time += time_range[1] - time_range[0]
    return net_gpu_time

def total_available_gpu_capacity(work_time_range=[0,10],available_size=4, borrow_schedule=[(5, 6, 2)]):
    available_size_schedule = generate_available_size_schedule(work_time_range, available_size, borrow_schedule)
    total_available_size_capacity = 0

    for (start, end, available_size) in available_size_schedule:
        # 计算每个区间的容量
        duration = end - start
        capacity = duration * available_size

        # 累加到总容量中
        total_available_size_capacity += capacity
    return total_available_size_capacity

def evaluation_tasks_scheduler(tasks, available_gpu_size,borrow_schedule=[(5, 6, 2)],is_save=False):
    total_completion_time = 0
    total_waiting_time = 0
    total_net_gpu_time = 0
    total_gpu_utilization = 0
    total_gpu_communicated = 0
    total_interruptions = 0
    total_interruption_time = 0
    total_completion_duration = 0
    task_throughput = 0
    total_gpu_time_ranges =[]


    # Loop over each task, considering subtasks carefully
    for task in tasks:
        print(task)
        # 计算任务的等待时间和完成时间
        # Calculate waiting time for the main task, including any interruptions
        task_waiting_time = task.get_waiting_time()
        total_waiting_time += task_waiting_time

        # Sum completion time from start to end of the task (includes all subtasks)
        task_completion_time = task.get_completion_time()
        total_completion_time += task_completion_time

        # GPU utilization is based on the task's size (including all subtasks' sizes)
        if not task.subtasks:
            total_gpu_utilization += task.size * task_completion_time
            total_gpu_time_ranges.append([task.start_time, task.end_time,task.size,task.status,task.name])
            # Record total completion time (for expectation calculation)
            total_completion_duration += task_completion_time

        # Throughput is the total communication handled by the GPU
        total_gpu_communicated += task.size

        # Count the number of interruptions
        total_interruptions += task.interruptions
        total_interruption_time += task.interruption_time


        task_throughput += 1

        # For subtasks, add their contribution to the task's completion and waiting times
        for subtask in task.subtasks:
            subtask_waiting_time = subtask.start_time - subtask.arrival_time
            subtask_completion_time = subtask.get_completion_time()
            # total_completion_time += subtask_completion_time
            total_gpu_utilization += subtask.size * subtask.get_completion_time()
            total_gpu_communicated += subtask.size
            total_completion_duration += subtask_completion_time
            total_gpu_time_ranges.append([subtask.start_time, subtask.end_time,subtask.size,subtask.status,subtask.name])


    # Calculate overall metrics
    total_net_gpu_time = get_net_gpu_time(total_gpu_time_ranges)
    # total_time = max([task.end_time for task in tasks]) - min([task.arrival_time for task in tasks])
    work_time_range=[min([task.start_time for task in tasks]),max([task.end_time for task in tasks])]
    gpu_utilization_rate = (total_gpu_utilization / (total_available_gpu_capacity(work_time_range,available_gpu_size,borrow_schedule))) * 100  # Utilization as a percentage
    average_completion_time = total_completion_time / task_throughput if task_throughput > 0 else 0
    average_waiting_time = total_waiting_time / task_throughput if task_throughput > 0 else 0




    # 计算中断时间和完成时间的数学期望
    interrupt_expected_value = total_interruption_time / total_completion_time if task_throughput > 0 else 0

    # 吞吐量
    throughput = task_throughput / (work_time_range[1]-work_time_range[0])  # Tasks per second


    # Print the calculated metrics
    print('=================================Evaluation=======================================')
    print(f'total_gpu_time_ranges: [start_time, end_time, size, status, name]') # start_time, end_time, size, status, name
    for gpu_time_range in total_gpu_time_ranges:
        print(f'{gpu_time_range}')

    for task in tasks:
        print(f"Task {task.name} completed in {task.get_completion_time():.2f} seconds with waiting time {task.get_waiting_time():.2f} and {task.interruptions} interruptions ({task.interruption_time:.2f} seconds).")

    print(f"Total Waiting Time: {total_waiting_time:.2f} seconds")
    print(f"Average Waiting Time: {average_waiting_time:.2f} seconds")
    print(f"Total Completion Time: {total_completion_time:.2f} seconds")
    print(f"Average Completion Time: {average_completion_time:.2f} seconds")
    print(f"Total Net GPU Time: {total_net_gpu_time:.2f} seconds")
    print(f"GPU Utilization Rate: {gpu_utilization_rate:.2f}%")
    print(f"Total Interruptions: {total_interruptions:.0f} times")
    print(f"Total Interruption Time: {total_interruption_time:.2f} seconds")
    print(f"Interruption Expected Value: {interrupt_expected_value:.2f} per task")
    print(f"Throughput: {throughput:.2f} tasks per second")

    if is_save:
        # 将输出内容写入文件
        with open('evaluation.txt', 'a') as f:
            f.write('=============================Evaluation==============================\n')
            f.write(f'total_gpu_time_ranges: [start_time, end_time, size, status, name]\n')
            for gpu_time_range in total_gpu_time_ranges:
                f.write(f'{gpu_time_range}\n')

            for task in tasks:
                f.write(
                    f"Task {task.name} completed in {task.get_completion_time():.2f} seconds with waiting time {task.get_waiting_time():.2f} and {task.interruptions} interruptions ({task.interruption_time:.2f} seconds).\n")

            f.write(f"Total Waiting Time: {total_waiting_time:.2f} seconds\n")
            f.write(f"Average Waiting Time: {average_waiting_time:.2f} seconds\n")
            f.write(f"Total Completion Time: {total_completion_time:.2f} seconds\n")
            f.write(f"Average Completion Time: {average_completion_time:.2f} seconds\n")
            f.write(f"Total Net GPU Time: {total_net_gpu_time:.2f} seconds\n")
            f.write(f"GPU Utilization Rate: {gpu_utilization_rate:.2f}%\n")
            f.write(f"Total Interruptions: {total_interruptions:.0f} times\n")
            f.write(f"Total Interruption Time: {total_interruption_time:.2f} seconds\n")
            f.write(f"Interruption Expected Value: {interrupt_expected_value:.2f} per task\n")
            f.write(f"Throughput: {throughput:.2f} tasks per second\n")

            f.write('----------------------------------------------------------------------\n')


    # Return the metrics for further analysis if needed
    return [total_waiting_time,total_completion_time,gpu_utilization_rate,interrupt_expected_value,throughput]




if __name__ == "__main__":
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

    tasks = []
    task_A = Task("A", size=3.414, duration=7.071, arrival_time=0)
    task_B = Task("B", size=9.677, duration=8.083, arrival_time=0)
    task_A.data = dataset1[0]
    task_B.data = dataset2[0]

    tasks.append(task_A)
    tasks.append(task_B)

    is_save = True

    final_tasks = schedule_tasks_ESGNN(tasks, available_size=available_size, borrow_schedule=borrow_schedule,is_save=is_save)
    plot_tasks(final_tasks)
    evaluate_result = evaluation_tasks_scheduler(final_tasks, available_gpu_size=available_size, is_save=is_save)
    print(evaluate_result)



    # Schedule tasks
    # is_save = True
    # schedule_method=['ESGNN','Lyra','Lyra Plus']
    # for i in range(3):
    #     # Define tasks
    #     tasks = []
    #     task_A = Task("A", size=3.414, duration=7.071, arrival_time=0)
    #     task_B = Task("B", size=24.677, duration=8.083, arrival_time=0)
    #     task_C = Task("C", size=7.175, duration=17.649, arrival_time=4)
    #     task_D = Task("D", size=6.768, duration=5.908, arrival_time=5)
    #
    #     task_A.data = dataset1[0]
    #     task_B.data = dataset2[0]
    #     task_C.data = dataset1[0]
    #     task_D.data = dataset2[0]
    #
    #     # tasks.append(task_A)
    #     # tasks.append(task_B)
    #     # tasks.append(task_C)
    #     # tasks.append(task_D)
    #     if random.random() < 0.5:
    #         tasks.append(task_A)
    #     if random.random() < 0.5:
    #         tasks.append(task_B)
    #     if random.random() < 0.5:
    #         tasks.append(task_C)
    #     if random.random() < 0.5:
    #         tasks.append(task_D)
    #
    #     final_tasks=[]
    #     with open('final_tasks.txt', 'a') as f:
    #         f.write(f'///////////////////{schedule_method[i]}///////////////////\n')
    #         if i == 0:
    #             try:
    #                 final_tasks = schedule_tasks_ESGNN(tasks, available_size=available_size, borrow_schedule=borrow_schedule,is_save=is_save)
    #             except Exception as e:
    #                 f.write(e)
    #                 f.write('\n')
    #         if i == 1:
    #             try:
    #                 final_tasks = schedule_tasks_Lyra(tasks, available_size=available_size, borrow_schedule=borrow_schedule, is_save=is_save)
    #             except Exception as e:
    #                 f.write(e)
    #                 f.write('\n')
    #         if i == 2:
    #             try:
    #                 final_tasks = schedule_tasks_Lyra_plus(tasks, available_size=available_size, borrow_schedule=borrow_schedule,is_save=is_save)
    #             except Exception as e:
    #                 f.write(e)
    #                 f.write('\n')
    #     with open('evaluation.txt', 'a') as f:
    #         f.write(f'///////////////////{schedule_method[i]}///////////////////\n')
    #         try:
    #             total_waiting_time,total_completion_time,gpu_utilization_rate,interrupt_expected_value,throughput=evaluation_tasks_scheduler(final_tasks, available_gpu_size=available_size, is_save=is_save)
    #             print(total_waiting_time,total_completion_time,gpu_utilization_rate,interrupt_expected_value,throughput)
    #         except Exception as e:
    #             f.write(e)
    #             f.write('\n')


