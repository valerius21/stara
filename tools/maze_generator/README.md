# Stara Maze Generator

A Python package for generating, solving, and visualizing mazes using a modified version of Prim's algorithm. The mazes are represented as NumPy arrays and can be exported as HTML visualizations.

## Features

- Generate random mazes using a modified Prim's algorithm
- Configurable maze size and minimum number of valid paths
- Reproducible maze generation using seeds
- Pathfinding using Breadth-First Search (BFS)
- Export mazes as HTML visualizations with optional solution paths
- Efficient maze representation using NumPy arrays

## Installation

The package can be installed using pip:

```bash
pip install stara-maze-generator
```

Or using Poetry:

```bash
poetry add stara-maze-generator
```

## Usage

### Command Line Interface

The package provides a command-line interface for quick maze generation:

```bash
# Generate a default 40x40 maze
generate-maze

# Generate a 20x20 maze with custom start/goal positions
generate-maze --size 20 --start 0 0 --goal 19 19

# Generate a maze with a specific seed and show solution
generate-maze --seed 123 --draw-solution

# Generate a maze with more paths
generate-maze --min-valid-paths 5
```

### Python API

```python
from pathlib import Path
import numpy as np
from stara_maze_generator.vmaze import VMaze
from stara_maze_generator.pathfinder import Pathfinder

# Create a 20x20 maze
maze = VMaze(
    seed=42,              # Random seed for reproducibility
    size=20,              # Creates a 20x20 grid
    start=(1, 1),         # Starting position
    goal=(18, 18),        # Goal position
    min_valid_paths=3     # Minimum number of valid paths
)

# Generate the maze structure
maze.generate_maze(pathfinding_algorithm=Pathfinder.BFS)

# Find a path from start to goal
path = maze.find_path()

# Export as HTML visualization
maze.export_html(Path("maze.html"), draw_solution=True)
```

## Maze Representation

The maze is represented as a 2D NumPy array where:

- `0` represents walls
- `1` represents passages
- The maze is guaranteed to have at least `min_valid_paths` different paths from start to goal

## HTML Visualization

The exported HTML visualization includes:

- Color-coded cells (walls, passages, start, goal)
- Optional solution path highlighting
- Responsive grid layout
- Maze information (size, seed, algorithm used)

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/stara-maze-generator.git
cd stara-maze-generator

# Install dependencies with Poetry
poetry install

# Run tests
poetry run pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
