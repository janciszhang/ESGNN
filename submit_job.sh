#!/bin/bash
#SBATCH --job-name=gnn-metis     # Job name
#SBATCH --output=gnn_metis_output.txt   # Standard output log file
#SBATCH --error=gnn_metis_error.txt    # Standard error log file
#SBATCH --partition=cybersecurity      # Partition to submit to

#SBATCH --gres=gpu:1                   # Request 1 GPU (if available)
#SBATCH --cpus-per-task=2              # Request 2 CPU cores
#SBATCH --mem=4G                       # Request 4 GB of memory

#SBATCH --time=01:00:00                # Maximum time for the job (1 hour)
#SBATCH --mail-type=ALL                # Send email on all events (begin, end, fail)
#SBATCH --mail-user=s4069853@rmit.edu.au  # Your email address

# Activate your virtual environment
source /opt/home/s4069853/miniconda3/bin/activate

# Change to the directory where your script is located
cd /opt/home/s4069853/ESGNN/

# Run your Python script
python metis_calculation_job_GPU.py
