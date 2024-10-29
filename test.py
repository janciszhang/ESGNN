import re

from scheduer_base import plot_tasks
from task import Task, get_tasks_from_str, merge_subtasks

if __name__ == '__main__':
    with open('execute_history.txt', 'r') as f:
        tasks_str_all = f.read()
    # 正则表达式：匹配以 "Running tasks" 开头，到下一个 "------------ Time" 之间的所有内容
    pattern = r"(Running tasks.*?)(?=--------------------------------------------------------)"

    # 使用 re.findall() 找到所有匹配项
    matches = re.findall(pattern, tasks_str_all, re.DOTALL)
    i = 0
    for match in matches:
        print(match)
        print("##################################")
        tasks_str=match

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

        plot_tasks(final_tasks, file_name=f'task_execution_timeline{i}')
        i+=1
