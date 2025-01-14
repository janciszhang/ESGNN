#!/bin/bash
cd C:/Users/ccloi/Downloads/ESGNN-main/ESGNN
#pip install torch CUDA==version torchvision
#pip3 install torch==1.13.1+cu117 torchvision>=0.13.1+cu117 torchaudio>=0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir

# Loop 10 times
for i in {1..3}
do
    echo "Running schedule_prepare (Attempt $i)..."

    # Set your dataset list as a string
    num_list='[4,0,0]'  # Dataset list as a string
    if python scheduler_total.py "$num_list"; then
        echo "Execution of schedule_prepare (Attempt $i) succeeded"
    else
        echo "Execution of schedule_prepare (Attempt $i) failed"
    fi

    # Optional: You can add a delay between each run if needed
    # sleep 1
done