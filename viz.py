import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def plot_tasks(tasks,file_name='task_execution_timeline'):
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 初始化 y 轴位置
    y_pos = 0
    y_labels = []  # 存储 y 轴标签及其位置
    y_positions = []
    labels_added = set()  # 用于追踪已添加的图例标签

    current_times=[]
    for task in tasks:
        print(task)
        current_times.append(task.start_time)
        current_times.append(task.end_time)
        for subtask in task.subtasks:
            current_times.append(subtask.start_time)
            current_times.append(subtask.end_time)
    print(current_times)
    current_time = max([current_time for current_time in current_times if current_time is not None])
    print(current_time)

    # 遍历任务并绘制每个任务的分段条形图
    for task in tasks:
        print(task)
        arrival_time = task.arrival_time
        start_time = task.start_time if task.start_time is not None else current_time
        end_time = task.end_time if task.end_time is not None else current_time
        print(f'arrival_time, start_time, end_time: {arrival_time}, {start_time}, {end_time}')
        if arrival_time <= current_time:
            # 绘制 arrival_time 到 start_time 的部分 (黄色)
            label = "Waiting Time" if "Waiting Time" not in labels_added else ""
            wait_bar = ax.barh(y_pos, start_time - arrival_time, left=arrival_time, color='yellow', edgecolor='black', label=label)
            labels_added.add("Waiting Time")
            # 添加标签数据
            ax.text(arrival_time + (start_time - arrival_time) / 2, y_pos, f"{start_time - arrival_time:.2f}",
                    ha='center', va='center', color="black")

            if start_time is not None:
                # 绘制 start_time 到 end_time 的部分 (done绿色)
                label = "Execution Time" if "Execution Time" not in labels_added else ""
                exec_bar = ax.barh(y_pos, end_time - start_time, left=start_time, color='green', edgecolor='black',
                                   label=label)
                labels_added.add("Execution Time")
                # 添加标签数据
                ax.text(start_time + (end_time - start_time) / 2, y_pos, f"{end_time - start_time:.2f}", ha='center',
                        va='center', color="black")
                # ax.text(end_time, y_pos, f"{end_time:.2f}", ha='center',
                #         va='center', color="black")

                if task.status=='doing':
                    # 绘制 current_time 到 es 的部分 (浅绿色)
                    label = "Task Estimated" if "Task Estimated" not in labels_added else ""
                    exec_bar = ax.barh(y_pos, task.get_estimated_end_time() - start_time,
                                            left=start_time,color="lightcyan", edgecolor="black", label=label)
                    labels_added.add("Task Estimated")
                    # 添加标签数据
                    ax.text(end_time + (task.get_estimated_end_time() - end_time) / 2, y_pos,
                            f"{task.get_estimated_end_time() - end_time:.2f}", ha='center', va='center',
                            color="black")

                    label = "Execution Time" if "Execution Time" not in labels_added else ""
                    exec_bar = ax.barh(y_pos, end_time - start_time, left=start_time, color='green', edgecolor='black',label=label)
                    labels_added.add("Execution Time")
                    # 添加标签数据
                    ax.text(start_time + (end_time - start_time) / 2, y_pos, f"{end_time - start_time:.2f}", ha='center',
                            va='center', color="black")

                if task.status=='done':
                    ax.text(end_time + 1, y_pos, f"{end_time:.2f}", ha='center',
                            va='center', color="black")

            # 记录主任务标签
            # y_labels.append((y_pos, task.name))
            y_labels.append(task.name)
            y_positions.append(y_pos)

            for subtask in task.subtasks:
                y_pos += 1  # 子任务绘制在主任务下方
                if subtask.status == 'interrupted':
                    label = "Interrupt Subtask Estimated" if "Interrupt Subtask Estimated" not in labels_added else ""
                    interrupt_bar = ax.barh(y_pos, subtask.get_estimated_end_time() - subtask.start_time,
                                            left=subtask.start_time,
                                            color="mistyrose", edgecolor="black", label=label)
                    labels_added.add("Interrupt Subtask Estimated")
                    label = "Interrupt Subtask" if "Interrupt Subtask" not in labels_added else ""
                    interrupt_bar = ax.barh(y_pos, subtask.end_time - subtask.start_time, left=subtask.start_time,
                                            color="red", edgecolor="black", label=label)
                    labels_added.add("Interrupt Subtask")
                    # 添加子任务标签数据arrival_time
                    ax.text(subtask.start_time + (subtask.end_time - subtask.start_time) / 2, y_pos,
                            f"{subtask.end_time - subtask.start_time:.2f}", ha='center', va='center', color="black")
                    ax.text(subtask.end_time + (subtask.get_estimated_end_time() - subtask.end_time) / 2, y_pos,
                            f"{subtask.get_estimated_end_time() - subtask.end_time:.2f}", ha='center', va='center',
                            color="black")
                else:
                    if subtask.status =='done':
                        label = "Subtask" if "Subtask" not in labels_added else ""
                        subtask_bar = ax.barh(y_pos, subtask.end_time - subtask.start_time, left=subtask.start_time,
                                              color="lightgreen", edgecolor="black", label=label)
                        labels_added.add("Subtask")
                        # 添加子任务标签数据arrival_time
                        ax.text(subtask.start_time + (subtask.end_time - subtask.start_time) / 2, y_pos,
                                f"{subtask.end_time - subtask.start_time:.2f}", ha='center', va='center', color="black")
                    if subtask.status =='doing':
                        label = "Subtask Estimated" if "Subtask Estimated" not in labels_added else ""
                        subtask_bar = ax.barh(y_pos, subtask.get_estimated_end_time() - subtask.start_time,
                                                left=subtask.start_time,
                                                color="mintcream", edgecolor="black", label=label)
                        labels_added.add("Subtask Estimated")
                        ax.text(subtask.start_time + (subtask.get_estimated_end_time() - subtask.start_time) / 2, y_pos,
                                f"{subtask.get_estimated_end_time() - subtask.start_time:.2f}", ha='center', va='center',
                                color="black")

                # 记录子任务标签
                y_labels.append(f"{task.name}_sub")
                y_positions.append(y_pos)

            # 每个主任务及其子任务绘制完后，y_pos 2，以在下一主任务和子任务间留出间隔
            y_pos += 2

    # 设置任务名称作为 y 轴标签
    # y_positions = list(range(len(y_labels)))  # y 轴位置从 0 开始，逐步递增
    ax.set_yticklabels(y_labels)
    ax.set_yticks(y_positions)  # 图例和标签对齐（要放后面，不然变y索引）

    # 在图表中添加 current_time 的垂直线
    # ax.axvline(x=current_time, color='red', linestyle='--', label=f'Current Time: {current_time:.2f}')
    # plt.vlines(current_time, -1, y_pos, colors='r', linestyles='--', label=f'Current Time = {current_time:.2f}')
    line = Line2D([current_time, current_time], [-1, y_pos], color='red', linestyle='--',
                  label=f'Current Time = {current_time:.2f}')
    ax.add_line(line)

    # 设置标签和图例
    ax.set_xlabel("Time/seconds")
    ax.set_ylabel("Task")
    ax.set_title("Task Execution Timeline")
    ax.legend(loc='best')

    # 显示图形
    plt.savefig(f"img/{file_name}.png", dpi=300, bbox_inches='tight')
    plt.show()



def viz_evaluate_results(evaluate_results, schedule_method_name):
    total_waiting_time = []
    total_completion_time = []
    gpu_utilization_rate = []
    interrupt_expected_value = []
    throughput = []
    for evaluate_result in evaluate_results:
        total_waiting_time.append(evaluate_result[0])
        total_completion_time.append(evaluate_result[1])
        gpu_utilization_rate.append(evaluate_result[2])
        interrupt_expected_value.append(evaluate_result[3])
        throughput.append(evaluate_result[4])

    # 数据准备
    metrics = [
        total_waiting_time,
        total_completion_time,
        gpu_utilization_rate,
        interrupt_expected_value,
        throughput
    ]

    metric_names = [
        'Total Waiting Time (seconds)',
        'Total Work Time (seconds)',
        'GPU Utilization Rate (%)',
        'Interrupt Expected Value',
        'Throughput (task/second)'
    ]

    colors = plt.cm.viridis(np.linspace(0, 1, len(schedule_method_name)))

    # 绘制每个指标的独立图
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(8, 5))
        x = np.arange(len(schedule_method_name))
        bars = plt.bar(x, metric, color=colors)
        plt.xticks(x, schedule_method_name)
        plt.ylabel(metric_names[i])
        plt.title(f'Comparison of {metric_names[i]}')

        # 添加图例
        plt.legend([bars[i] for i in range(len(bars))], schedule_method_name, title='Methods')

        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    evaluate_results = [
        [25.572810173034668, 57.981799840927124, 67.6614804257861, 0.4546301615757951, 0.03449358256361468],
        [25.572810173034668, 57.981799840927124, 67.6614804257861, 0.4546301615757951, 0.03449358256361468],
        [0, 33.40258026123047, 77.84535238269129, 0.45977273498147503, 0.06540214633226274]]

    schedule_method_name = ['Lyra', 'Lyra Plus', 'ESGNN']
    viz_evaluate_results(evaluate_results, schedule_method_name)