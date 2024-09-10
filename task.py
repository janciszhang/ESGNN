"""
RUN prioritization sort index：
1. size（该任务大小或该任务剩余未运行部分的大小）
2. waiting_time（任务等待时间为地方开始时间到该任务开始运行的时间，比如从0时刻开始，如果先运行C花了5min，再运行部分B 6min，再运行A，那么此时A的等待时间为11min）
3. is_running（是否在运行）

总体策略：
1. 尽量先运行大小最小的任务，如果地方有空余，可以根据空余地方大小分割其他任务塞进来同时运行。
2. 如果已经在运行某分割任务，则后面尽可能先完成该任务。
3. 另外，如果一个任务等待时间太久，也可以提前运行。

评估：
1. GPU利用率
2. 任务的总完成时间
3. 每个任务（包括多个子任务）的等待时间
"""
"""
STOP prioritization sort index：
1. size（该任务/子任务大小：尽量小）
2. remaining_duration（任务的剩余持续时间：任务即将完成时尽量不中断）
3. waiting_time（任务等待时间：等待时间较长的任务被中断会增加额外的等待时间）
4. is_sub（任务的分割情况：尽量中断子任务）

总体策略：
1. 任务大小（Size）: 优先选择较小的任务或子任务中断，以减少中断的开销。
2. 剩余持续时间（Remaining Duration）: 尽量避免中断即将完成的任务，以防止已经投入的计算资源浪费。
3. 等待时间（Waiting Time）: 避免中断等待时间较长的任务，因为这会进一步增加其等待时间。
4. 任务分割情况（Is Sub）: 优先中断子任务，而不是整个任务，以减少对整体调度的影响。

评估：
1. GPU利用率
2. 任务的总完成时间
3. 每个任务（包括多个子任务）的等待时间
4. 任务完成期望/中断期望
10min 15min(中断浪费的时间5min)
"""


from ESGNN11.base_gnn import GNN, dataset


class Task:
    def __init__(self, name, size, duration, data, is_sub=False):
        self.name = name
        self.size = size  # 初始大小，用于调度和资源分配
        self.original_size = size  # 原始大小，保持任务的初始需求
        self.remaining_size = size  # 剩余大小，表示尚未完成的部分
        self.duration = duration  # 初始预计持续时间
        self.original_duration = duration  # 原始预计持续时间（创建时）
        self.remaining_duration = duration  # 剩余持续时间
        self.queue_time = 0
        self.is_running = False
        self.start_time = 0
        self.end_time = 0
        self.arrival_time = 0
        self.waiting_time = 0
        self.run_priority = 0
        self.interrupt_priority = 0
        self.is_sub = is_sub
        self.interruptions = 0
        self.data = data
        self.model = self.create_model()
        self.status = 'not arrived' # not arrived，waiting，doing，done
        self.subtasks = []  # 子任务列表

    def create_model(self):
        return GNN(self.data.num_node_features, dataset.num_classes)

    def __lt__(self, other):
        return self.run_priority > other.run_priority

    def calculate_run_priority(self, current_time, total_remaining_size, weight_size, weight_queue_time,weight_is_running, weight_is_sub):
        self.queue_time = current_time - self.start_time
        normalized_size = self.remaining_size / total_remaining_size if total_remaining_size > 0 else 0
        normalized_queue_time = self.queue_time / current_time if current_time > 0 else 0
        self.run_priority = (
                weight_size * (-normalized_size) +
                weight_queue_time * normalized_queue_time +
                weight_is_running * (1 if self.is_running else 0) +
                weight_is_sub * (1 if self.is_sub else 0)
        )

    def calculate_interrupt_priority(self, current_time, weight_size, weight_remaining_duration, weight_queue_time,weight_is_sub):
        normalized_size = self.remaining_size / self.original_size if self.original_size > 0 else 0
        normalized_remaining_duration = self.remaining_duration / self.original_duration if self.original_duration > 0 else 0
        normalized_queue_time = self.queue_time / current_time if current_time > 0 else 0
        self.interrupt_priority = (
                weight_size * normalized_size +
                weight_remaining_duration * normalized_remaining_duration +
                weight_queue_time * normalized_queue_time +
                weight_is_sub * (1 if self.is_sub else 0)
        )
