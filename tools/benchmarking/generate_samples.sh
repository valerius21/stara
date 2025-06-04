#!/bin/bash

# test, if batch-mazes is installed
if ! command -v batch-mazes &>/dev/null; then
    echo "batch-mazes is not installed"
    exit 1
fi

# 10x10, 1000 mazes, omit maze
batch-mazes --maze-size 10 --maze-count 1000 --omit-maze=true --output-path 10x10_1000_omit.pkl

# 100x100, 1000 mazes, omit maze
batch-mazes --maze-size 100 --maze-count 1000 --omit-maze=true --output-path 100x100_1000_omit.pkl

# 1000x1000, 1000 mazes, omit maze
batch-mazes --maze-size 1000 --maze-count 1000 --omit-maze=true --output-path 1000x1000_1000_omit.pkl

# 10000x10000, 100 mazes, omit maze
batch-mazes --maze-size 10000 --maze-count 100 --omit-maze=true --output-path 10000x10000_100_omit.pkl

# 100000x100000, 100 mazes, omit maze
batch-mazes --maze-size 100000 --maze-count 100 --omit-maze=true --output-path 100000x100000_100_omit.pkl

# 10x10, 1000 mazes, include maze
batch-mazes --maze-size 10 --maze-count 1000 --output-path 10x10_1000.pkl

# 100x100, 1000 mazes, include maze
batch-mazes --maze-size 100 --maze-count 1000 --output-path 100x100_1000.pkl

# 1000x1000, 1000 mazes, include maze
batch-mazes --maze-size 1000 --maze-count 1000 --output-path 1000x1000_1000.pkl
