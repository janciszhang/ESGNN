#!/bin/bash

#SBATCH --job-name=my_job_name         # 作业名称
#SBATCH --output=output_file_%j.out    # 标准输出文件（%j会自动替换为Job ID）
#SBATCH --error=error_file_%j.err      # 错误输出文件（可选）
#SBATCH --partition=compute            # 分区名称
#SBATCH --nodes=1                      # 使用的节点数量
#SBATCH --ntasks=1                     # 总的任务数量
#SBATCH --cpus-per-task=4              # 每个任务使用的CPU核心数量
#SBATCH --gres=gpu:1                   # 使用的GPU数量（可选）
#SBATCH --mem=16G                      # 内存需求（总量）
#SBATCH --time=02:00:00                # 运行的最长时间，格式为 hours:minutes:seconds
#SBATCH --mail-type=END                # 作业结束时邮件通知（可选）
#SBATCH --mail-user=your_email@domain.com  # 邮件地址（可选）

# 作业的实际工作内容：加载模块、运行Python脚本等
module load python/3.7               # 加载所需模块（例如 Python 环境）
source activate my_env               # 激活虚拟环境（如果使用的话）

# 运行脚本
python metis_calculation_job_GPU.py   # 执行你的 Python 脚本
