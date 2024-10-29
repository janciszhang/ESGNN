from task import get_tasks_from_str

tasks_str="""
Running tasks: ['D']
Available size: 3
Task(D, size: 7.00/7.00/7.00, duration: -0.00/11.36/5.91, doing, 5.00/16.05/None/27.4, Is Subtask: False, Is Main task: True, Interruptions: 0)
Task(A_sub, size: 0.00/1.00/1.00, duration: 0.00/2.97/2.36, done, 0.00/2/4.47/4.97, Is Subtask: True, Is Main task: False, Interruptions: 0)
Task(A, size: 0.00/2.00/3.00, duration: 0.00/4.71/7.07, done, 0.00/6/10.38/10.71, Is Subtask: True, Is Main task: True, Interruptions: 0)
Task(C, size: 0.00/7.00/7.00, duration: 0.00/17.65/17.65, done, 4.00/4.36/16.05/22.01, Is Subtask: False, Is Main task: True, Interruptions: 0)
Task(A, size: 2.15/3.00/3.00, duration: 5.07/7.07/7.07, interrupted, 0.00/0/2/7.07, Is Subtask: True, Is Main task: False, Interruptions: 1)
Task(A, size: 1.15/2.00/3.00, duration: 2.71/4.71/7.07, interrupted, 0.00/3/5/7.71, Is Subtask: True, Is Main task: False, Interruptions: 1)
Task(B, size: 25.00/25.00/25.00, duration: 8.08/8.08/8.08, waiting, 0.00/None/None/None, Is Subtask: False, Is Main task: True, Interruptions: 0)
"""

final_tasks = get_tasks_from_str(tasks_str)
# 打印提取到的任务信息
for task in final_tasks:
    print(task)

# Task information string is not in the correct format.