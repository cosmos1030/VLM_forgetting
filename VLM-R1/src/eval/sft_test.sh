#!/bin/bash

for steps in {100..800..100}
do
    echo "Running step $steps"
    torchrun --nproc_per_node=10 test_rec_baseline.py --steps $steps
done
