#!/bin/bash
#SBATCH --job-name=gpu-audio
#SBATCH --output=submit_audio.txt
#SBATCH --error=submit_audio.err
#SBATCH --partition=SCT

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

#SBATCH --qos=normal

#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxxxx@rmit.edu.au

source virtualenv/lambda-stack-with-tensorflow-pytorch/bin/activate
cd virtualenv/
python audio.py