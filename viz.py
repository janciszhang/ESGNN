import matplotlib.pyplot as plt
import numpy as np

methods = ['Lyra','ESGNN']
# 方法1的计算结果
method2_results = [2, 51.5, 92.22222222222223, 0.0970873786407767, 0.17391304347826086]  # 示例数据

# 方法2的计算结果
method1_results = [23, 87, 45.794392523364486, 0.05747126436781609, 0.07407407407407407]   # 示例数据

total_waiting_time = [method1_results[0], method2_results[0]]         # 示例值
total_completion_time = [method1_results[1], method2_results[1]]       # 示例值
gpu_utilization_rate = [method1_results[2], method2_results[2]]    # 示例值
interrupt_expected_value = [method1_results[3], method2_results[3]]       # 示例值
throughput = [method1_results[4], method2_results[4]]

# 数据准备
metrics = [
    total_waiting_time,
    total_completion_time,
    gpu_utilization_rate,
    interrupt_expected_value,
    throughput
]

metric_names = [
    'Total Waiting Time',
    'Total Completion Time',
    'GPU Utilization Rate',
    'Interrupt Expected Value',
    'Throughput'
]

# 绘制每个指标的独立图
# 绘制每个指标的独立图
for i, metric in enumerate(metrics):
    plt.figure(figsize=(8, 5))
    x = np.arange(len(methods))
    bars = plt.bar(x, metric, color=['blue', 'orange'])
    plt.xticks(x, methods)
    plt.ylabel(metric_names[i])
    plt.title(f'Comparison of {metric_names[i]}')

    # 添加图例
    plt.legend([bars[0], bars[1]], [methods[0], methods[1]])

    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()