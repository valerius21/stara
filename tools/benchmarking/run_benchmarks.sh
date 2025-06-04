#!/bin/bash

# Run benchmarks for all variations of mazes

poetry shell

for DATASET in `ls *.pkl`; do
    echo "================================================"
    echo "Running benchmarks for $DATASET"
    echo "================================================"
    benchmark-mazes --nuitka-bin ./astar_nuitka.bin --maze-file $DATASET
done