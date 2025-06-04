# STARA: A* Algorithm Implementations with Python, C++, Rust, ..., and Numba


## Overview

This project evaluates various implementation approaches of the **A* pathfinding algorithm**, starting with pure, naive Python, over standard library optimization, Nuitka compilation to binary files, Numba JIT compilation, and Rust extensions with PyO3. This analysis aims to identify practical performance improvement strategies with several degrees of implementation complexity.

## Key Findings

🔬 **Benchmarking and analysis reveal that different problem sizes have different optimal solutions:**

- **Large workloads**: C++ extensions provide **6.51x speedup**
- **Small workloads**: Rust extensions achieve **14.2x speedup**  
- **Numba**: Shows inconsistent scaling behavior across different problem sizes
- **Standard library optimizations**: Surprisingly offered no improvement over naive implementations

## Implementation Approaches

### 🐍 Core Algorithm Implementations

| Implementation | Repository | Description | Performance |
|----------------|------------|-------------|-------------|
| **Naive Python** | [`libs/naive`](https://github.com/valerius21/stara_astar_naive) | Pure Python baseline implementation | Baseline (1.0x) |
| **Standard Library** | [`libs/stdlib`](https://github.com/valerius21/stara_astar_stdlib) | Optimized using Python standard library | No improvement |
| **Nuitka Compiled** | [`libs/nuitka`](https://github.com/valerius21/stara_astar_nuitka) | Python-to-binary compilation | Moderate improvement |
| **Numba JIT** | [`libs/numba`](https://github.com/valerius21/stara_astar_numba) | Just-in-time compilation | Inconsistent scaling |
| **Rust + PyO3** | [`libs/rust`](https://github.com/valerius21/stara_astar_rs) | Native Rust with Python bindings | **14.2x speedup** (small workloads) |

### 🛠️ Supporting Tools

| Tool | Repository | Description |
|------|------------|-------------|
| **Maze Generator** | [`tools/maze_generator`](https://github.com/valerius21/Stara-Maze-Generator) | Generate test mazes for pathfinding algorithms |
| **Batch Benchmarking** | [`tools/benchmarking`](https://github.com/valerius21/stara-batch-benchmark) | Automated performance testing and analysis |

## Quick Start

### Prerequisites

- Python 3.13+
- Git
- Rust (for Rust implementation)
- Poetry (recommended for dependency management)
- C++ compiler (for C++ implementation)
- and probably more...

### Clone with Submodules

```bash
git clone --recursive https://github.com/valerius21/stara.git
cd stara

# If you already cloned without --recursive:
git submodule update --init --recursive
```

### Installation

Each implementation can be installed independently. Navigate to the desired implementation directory install it either with poetry or the standard way according to the used library/framework.

## Usage

### Basic A* Pathfinding

```python
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
# You can use any pathfinding algorithm from the Pathfinder class
# It is only used to generate the maze structure.
maze.generate_maze(pathfinding_algorithm=Pathfinder.BFS) 

# Find a path from start to goal
a_star = AStar(maze)
path = a_star.find_path()
print(f"Path found: {path}")
```

## Performance Analysis

### Methodology

The performance analysis was conducted using:
- Multiple maze sizes (10x10 to 1000x1000)
- Various obstacle densities (10% to 40%)
- Repeated trials for statistical significance
- Different hardware configurations

### Results Summary

| Implementation | Small Mazes | Medium Mazes | Large Mazes | Complexity Rating |
|----------------|-------------|--------------|-------------|-------------------|
| Naive Python | 1.0x | 1.0x | 1.0x | ⭐ (Low) |
| Standard Lib | 1.0x | 1.0x | 1.0x | ⭐ (Low) |
| Nuitka | 2.1x | 2.8x | 3.2x | ⭐⭐ (Medium) |
| Numba | 8.5x | 4.2x | 2.1x | ⭐⭐ (Medium) |
| Rust | **14.2x** | 8.7x | 5.3x | ⭐⭐⭐ (High) |

*Note: C++ results (6.51x for large workloads) not included in current submodules*

## Research Context

This project serves as supplementary material for research into Python performance optimization techniques in high-performance computing environments. The findings demonstrate that:

1. **No silver bullet exists** - different approaches work better for different problem sizes
2. **Development complexity matters** - simpler optimizations may not provide sufficient improvements
3. **Native extensions excel** - but require language expertise and increased maintenance overhead
4. **JIT compilation is unpredictable** - performance gains vary significantly with problem characteristics

## Contributing

Contributions are welcome! 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{stara2025,
    title={Python Performance Optimizations: Leveraging Native Implementations for A* Pathfinding (Respository)},
    author={Mattfeld, Valerius Albert Gongjus},
    year={2025},
    url={https://github.com/valerius21/stara},
    version={0.1.0},
    date-released={2025-06-04},
    note={This repository contains the code for the paper "Python Performance Optimizations: Leveraging Native Implementations for A* Pathfinding"}
}
```

## Related Work

- [A* Search Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [Numba Documentation](https://numba.pydata.org/)
- [PyO3 User Guide](https://pyo3.rs/)
- [Nuitka User Manual](https://nuitka.net/doc/user-manual.html)
