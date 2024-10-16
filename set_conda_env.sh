#!/bin/bash
#在 Linux 系统上安装 Anaconda（或 Miniconda）
# 创建目录
mkdir -p ~/miniconda3

# 下载 Anaconda 安装脚本并将其保存为 ~/miniconda3/miniconda.sh
wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -u -p ~/miniconda3

# 运行 Anaconda 安装脚本
#bash ~/miniconda3/miniconda.sh

# 以批处理模式安装 Miniconda
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# 配置 Linux 系统的环境变量，使得系统能够识别并使用 Miniconda 中的命令
echo 'export PATH="~/miniconda3/bin:$PATH"' >> ~/.bashrc


# 配置 Linux 系统的环境变量，使得系统能够识别并使用 Miniconda 中的命令
# 编辑 .bashrc 文件
nano ~/.bashrc
# 使配置生效
source ~/.bashrc

conda --version

echo "Anaconda (Miniconda) installation complete!"


conda create -n myenv python=3.9 -y
conda activate base
echo "conda activate myenv" >> ~/.bashrc
source ~/.bashrc

conda env list
conda env remove -n base