import pytest
import numpy as np
from stara_maze_generator.vmaze import VMaze


@pytest.fixture
def simple_maze():
    """Create a simple 4x4 maze for testing."""
    maze = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3), min_valid_paths=1)
    # Set up a simple maze layout (1 = passage, 0 = wall):
    # S 1 1 1
    # 0 0 1 1
    # 1 1 1 0
    # 1 0 1 G
    maze.maze_map = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1]])
    return maze
