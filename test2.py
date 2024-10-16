"""
有时候会报错，可以多运行几次，会有成功的
"""

import time
from datetime import datetime
import copy
import heapq
from torch_geometric.datasets import Planetoid
from ESGNN.metis_partition import partition_K
from ESGNN.task import Task, split_task, new_task


def schedule_tasks(tasks, available_size, borrow_schedule):
    running_tasks = []
    interrupt_tasks = []
    completed_tasks = []
    task_queue = []

    # Initial task queue
    for task in tasks:
        heapq.heappush(task_queue, task)

    current_time = time.time()
    total_remaining_size = sum(task.size for task in tasks)

    borrowed_applied = set()  # Track when borrowed space is applied
    returned_applied = set()  # Track when space is returned

    while task_queue or running_tasks:
        next_borrow_start_time = None
        next_borrow_end_time = None
        print(f"------------ Time: {datetime.now()} (Elapsed: {current_time} seconds) ---------------")

        # Check for borrowed schedule
        for (borrow_start_time, borrow_end_time, borrow_space) in borrow_schedule:
            if borrow_start_time <= current_time and (borrow_start_time, borrow_space) not in borrowed_applied:
                available_size -= borrow_space  # Borrowed space decreases available size once
                borrowed_applied.add((borrow_start_time, borrow_space))  # Mark this time as processed
                print(f"Borrowed {borrow_space} space, available size now: {available_size}")
            else:
                next_borrow_start_time = borrow_start_time
            if borrow_end_time <= current_time and (borrow_end_time, borrow_space) not in returned_applied:
                available_size += borrow_space  # Return the space once at the end time
                returned_applied.add((borrow_end_time, borrow_space))  # Mark this time as processed
                print(f"Returned {borrow_space} space, available size now: {available_size}")
            else:
                next_borrow_end_time = borrow_end_time

        # Interrupt mechanism
        while available_size < 0 and len(running_tasks) >= 1:
            for task in running_tasks:
                task.calculate_interrupt_priority(current_time, total_remaining_size, weight_size=0.4,
                                                  weight_remaining_duration=0.3, weight_waiting_time=0.2,
                                                  weight_is_sub=0.1)
            running_tasks = sorted(running_tasks, key=lambda task: task.interrupt_priority, reverse=False)
            task_to_interrupt = running_tasks.pop()
            task_to_interrupt.remaining_duration = task_to_interrupt.duration - (
                    current_time - task_to_interrupt.start_time)
            task_to_interrupt.remaining_size = task_to_interrupt.size - (
                    task_to_interrupt.original_size / task_to_interrupt.original_duration * (
                    current_time - task_to_interrupt.start_time))

            print(f"Task {task_to_interrupt.name} is interrupted.")
            task_to_interrupt.status = 'interrupted'
            task_to_interrupt.is_running = False
            task_to_interrupt.interruptions += 1
            task_to_interrupt.end_time = round(current_time, 2)
            interrupt_tasks.append(copy.copy(task_to_interrupt))

            # Reset task to original state before interrupt
            task_to_interrupt.remaining_size = task_to_interrupt.size
            task_to_interrupt.remaining_duration = task_to_interrupt.duration
            task_to_interrupt.status = 'waiting'
            print(task_to_interrupt)
            heapq.heappush(task_queue, task_to_interrupt)
            available_size += task_to_interrupt.size
            print(f'Interrput available_size: {available_size}')

        # Execute running tasks
        for task in running_tasks[:]:
            task.remaining_duration = task.duration - (current_time - task.start_time)
            # print(f"rate: {task.original_size/task.original_duration * 1}")
            task.remaining_size = task.size - (task.size / task.duration * (current_time - task.start_time))
            if task.remaining_duration <= 0:
                task.create_model()
                task.train_model()

                available_size += task.size
                print(f"Task {task.name} finished.")
                print(available_size)
                task.status = 'done'
                task.is_running = False
                task.end_time = round(current_time + task.remaining_duration, 2)
                task.remaining_size = 0
                task.remaining_duration = 0
                print(task)
                completed_tasks.append(task)
                running_tasks.remove(task)

        # Schedule new tasks based on current available size
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
        next_times = [next_borrow_start_time, next_borrow_end_time]
        for task in running_tasks[:]:
            next_times.append(task.estimated_end_time)
        try:
            next_time = min([time for time in next_times if isinstance(time, (int, float)) and time > current_time])
            current_time = next_time
            print(f'next_time:  {next_times} --- {next_time}')
        except:
            print('ALl DONE')
        print(f"Running tasks: {[task.name for task in running_tasks]}")
        print(f"Available size: {available_size}")

    print('========================================================================')
    for task in interrupt_tasks[:]:
        print(task)
    print('========================================================================')
    for task in completed_tasks[:]:
        print(task)


if __name__ == "__main__":
    dataset1 = Planetoid(root='/tmp/Cora', name='Cora')
    dataset2 = Planetoid(root='/tmp/Citeseer', name='Citeseer')

    # Define tasks
    tasks = []
    # task_A = Task("A", 3, 3)  # size 3, duration 5min 3
    # task_B = Task("B", 5, 15)  # size 10, duration 20min 1/4
    # task_C = Task("C", 10, 20)  # size 10, duration 20min 4/4/2

    task_A = new_task(dataset=dataset1, duration=None, size=None)
    task_B = new_task(dataset=dataset2, duration=None, size=None)

    # task_A.data=dataset1[0]
    # task_B.data=dataset2[0]

    # task_A.run_priority = 2
    # task_B.run_priority = 1
    #
    # task_A.interrupt_priority=-1
    # task_B.interrupt_priority=0
    tasks.append(task_B)
    tasks.append(task_A)

    # print(tasks[0])
    # print(tasks[1])
    tasks[0].arrival_time = time.time()
    tasks[1].arrival_time = time.time()

    # tasks.append(task_C)

    # Define borrow schedule (borrow_start_time, borrow_end_time, borrow_space)
    borrow_schedule = [(time.time()+5, time.time()+6, 350)]  # Borrow 1 unit of space between time 4 and 6

    # Schedule tasks
    schedule_tasks(tasks, available_size=400, borrow_schedule=borrow_schedule)
