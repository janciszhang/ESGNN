#!/bin/bash

# Loop 10 times
for i in {1..2}
do
    echo "Running schedule_task.py (Attempt $i)..."

    # Try to run the Python script
    if python scheduler_evaluation.py; then
        echo "Execution of schedule_task.py (Attempt $i) succeeded"
    else
        echo "Execution of schedule_task.py (Attempt $i) failed"
        # Handle the failure (optional, e.g., log the error)
    fi

    # Optional: You can add a delay between each run if needed
    # sleep 1
done