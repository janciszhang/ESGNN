#!/bin/bash
cd C:/Users/ccloi/Downloads/ESGNN-main/ESGNN

# pip install metis
SOURCE_FILE="metis_dll_x86-64/metis.dll"
ls "SOURCE_FILE"
mkdir -p "C:\metis"
METIS_DLL_PATH="C:/metis/metis"
cp "$SOURCE_FILE" "$METIS_DLL_PATH"
ls "$METIS_DLL_PATH"
export METIS_DLL=$METIS_DLL_PATH
echo $METIS_DLL
# python metis_partition.py
python -c "import pymetis; print(pymetis.__version__)"

# python metis_calculation_job_CPU.py

# python -c "import torch; print(torch.__version__)"
# python -c "import torch; print(torch.cuda.is_available())"

python metis_calculation_job_GPU.py

