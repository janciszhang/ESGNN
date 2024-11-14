import re

from viz import plot_tasks
from task import Task, get_tasks_from_str, merge_subtasks

if __name__ == '__main__':
    with open('execute_history.txt', 'r') as f:
        tasks_str_all = f.read()
        tasks_str = tasks_str_all

        final_tasks = get_tasks_from_str(tasks_str)

        # 打印提取到的任务信息
        # for task in tasks:
        #     print(task)

        final_tasks=merge_subtasks(final_tasks)
        print(f'===================================final_tasks {len(final_tasks)}=======================================')
        for task in final_tasks:
            print(task.__str__(option=2))
            if task.is_main and task.subtasks:
                print('---------subtasks---------')
                for subtask in task.subtasks:
                    print(subtask)
            print('------------------')

        plot_tasks(final_tasks)

