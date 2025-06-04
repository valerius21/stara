import pytest
import numpy as np
from stara_maze_generator.vmaze import VMaze
from stara_maze_generator.pathfinder.bfs import BFS


class TestBFS:
    @pytest.fixture
    def simple_maze(self):
        """Create a simple 4x4 maze for testing."""
        maze = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3), min_valid_paths=1)
        maze.maze_map = np.array(
            [[1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1]]
        )
        return maze

    @pytest.fixture
    def bfs_pathfinder(self, simple_maze):
        """Create a BFS pathfinder instance with the simple maze."""
        return BFS(simple_maze)

    def test_find_path_exists(self, simple_maze, bfs_pathfinder):
        """Test that BFS finds a valid path when one exists."""
        path = bfs_pathfinder.find_path(simple_maze.start, simple_maze.goal)
        assert path is not None
        assert len(path) > 0
        assert path[0] == tuple(simple_maze.start)
        assert path[-1] == tuple(simple_maze.goal)

        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
            assert simple_maze.maze_map[curr] == 1
            assert simple_maze.maze_map[next_pos] == 1

    def test_find_path_no_path(self, simple_maze, bfs_pathfinder):
        """Test that BFS returns None when no path exists."""
        simple_maze.maze_map[0, 1:] = 0
        simple_maze.maze_map[1:, -1] = 0
        path = bfs_pathfinder.find_path(simple_maze.start, simple_maze.goal)
        assert path is None

    def test_find_path_shortest(self, simple_maze, bfs_pathfinder):
        """Test that BFS finds a shortest path."""
        simple_maze.maze_map = np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]]
        )
        path = bfs_pathfinder.find_path(simple_maze.start, simple_maze.goal)
        assert path is not None
        assert len(path) == 7

        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
            assert simple_maze.maze_map[curr] == 1
            assert simple_maze.maze_map[next_pos] == 1

    def test_path_updates_maze_path(self, simple_maze, bfs_pathfinder):
        """Test that finding a path updates the maze's path attribute."""
        path = bfs_pathfinder.find_path(simple_maze.start, simple_maze.goal)
        assert path is not None
        assert simple_maze.path == path

    def test_visited_cells_not_revisited(self, simple_maze, bfs_pathfinder):
        """Test that BFS doesn't revisit already visited cells."""
        simple_maze.maze_map = np.array(
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )
        path = bfs_pathfinder.find_path(simple_maze.start, simple_maze.goal)

        position_counts = {}
        for pos in path:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        assert all(count == 1 for count in position_counts.values())
