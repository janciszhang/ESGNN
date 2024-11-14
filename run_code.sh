#!/bin/bash

#pip install torch CUDA==version torchvision
#pip3 install torch==1.13.1+cu117 torchvision>=0.13.1+cu117 torchaudio>=0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir

# Loop 10 times
#for i in {1..2}
#do
#    echo "Running schedule_task.py (Attempt $i)..."
#
#    # Try to run the Python script
#    if python scheduler_evaluation.py; then
#        echo "Execution of schedule_task.py (Attempt $i) succeeded"
#    else
#        echo "Execution of schedule_task.py (Attempt $i) failed"
#        # Handle the failure (optional, e.g., log the error)
#    fi
#
#    # Optional: You can add a delay between each run if needed
#    # sleep 1
#done