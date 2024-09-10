#!/bin/bash
#SBATCH --partition=cybersecurity
#SBATCH --job-name=my_job
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

# Load modules or activate conda environment if needed
# module load ...

# Run your script
python metis_calculation_job_GPU.py
