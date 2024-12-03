# ESGNN
cuda 11.8, python 3.10.6, torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
## Environment
1. Should install cuda 11.8 first.  
- nvidia - smi # check the version (may from environment variables setting)
- nvcc --version # check the version (actually used version)

2. Setting python 3.10.6 environment
- conda create -n my-env python=3.10.6 # 构建一个虚拟环境名为：my-env，Python版本为3.10.6
- conda init bash && source /root/.bashrc # 更新bashrc中的环境变量
- conda activate my-env # 切换到创建的虚拟环境：my-env
- python --version # 验证

3. install related libraries
- pip install -r requirements.txt
- If some lib cannot use 'requirements.txt' to install directly. Install it use pip or conda.
- pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118 --user
- pip install dgl==1.1.1+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html --user
- pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

4. Loading dataset and task
- python load_data.py
- python load_task.py

5. RUN scheduer
- python scheduer_ESGNN.py
- python scheduer_total.py


conda create -n torch_cpu python=3.10.6
conda activate torch_cpu
pip install torch==2.2.0