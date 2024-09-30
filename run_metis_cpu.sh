#! bash
cd C:/Users/ccloi/Downloads/ESGNN-main/ESGNN

# pip install metis
# copy metis_dll_x86-64 C:\path\to\target\directory
# python metis_partition.py

# python metis_calculation_job_CPU.py

# python -c "import torch; print(torch.__version__)"
# python -c "import torch; print(torch.cuda.is_available())"
python metis_calculation_job_GPU.py

