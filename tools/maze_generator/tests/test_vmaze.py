import json
from pathlib import Path

import numpy as np
import pytest

from stara_maze_generator.pathfinder import Pathfinder
from stara_maze_generator.vmaze import VMaze


class TestVMaze:
    @pytest.fixture
    def basic_maze(self):
        """Create a basic 4x4 maze for testing."""
        maze = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3))
        maze.maze_map = np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]]
        )
        return maze

    @pytest.fixture
    def simple_maze(self):
        """Create a simple maze with generated paths."""
        maze = VMaze(seed=42, size=4, start=(1, 1), goal=(2, 2))
        maze.generate_maze()
        return maze

    def test_initialization(self):
        """Test that VMaze initializes with correct parameters."""
        maze = VMaze(seed=42, size=10, start=(0, 0), goal=(9, 9))
        assert maze.rows == 10
        assert maze.cols == 10
        assert np.array_equal(maze.start, np.array([0, 0]))
        assert np.array_equal(maze.goal, np.array([9, 9]))
        assert maze.seed == 42
        assert maze.min_valid_paths == 3
        assert maze.pathfinding_algorithm == Pathfinder.BFS
        assert isinstance(maze.rng, np.random.Generator)

    def test_initialization_validation(self):
        """Test that initialization validates parameters."""
        with pytest.raises(ValueError, match="size must be at least 4"):
            VMaze(seed=42, size=3, start=(0, 0), goal=(2, 2))

        with pytest.raises(IndexError):
            maze = VMaze(seed=42, size=4, start=(0, 0), goal=(4, 4))
            maze.maze_map[4, 4] = 1  # This will raise IndexError

    def test_maze_connectivity(self, basic_maze):
        """Test that maze has a valid path from start to goal."""
        path = basic_maze.find_path()
        assert path is not None, "Maze should have a valid path"
        assert tuple(basic_maze.start) == path[0]
        assert tuple(basic_maze.goal) == path[-1]

        # Verify path is valid
        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
            assert basic_maze.maze_map[curr] == 1
            assert basic_maze.maze_map[next_pos] == 1

    def test_multiple_paths(self):
        """Test that maze can have multiple valid paths."""
        maze = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3))
        # Create a maze with multiple possible paths
        maze.maze_map = np.array(
            [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )

        # Find first path
        original_path = maze.find_path()
        assert original_path is not None

        # Block the first path and verify another exists
        for x, y in original_path[1:-1]:
            maze.maze_map[x, y] = 0

        # Should still find a different path
        new_path = maze.find_path()
        assert new_path is not None
        assert new_path != original_path

    def test_get_cell_neighbours_center(self, basic_maze):
        """Test getting neighbors for a center cell."""
        neighbors = basic_maze.get_cell_neighbours(1, 1)
        assert len(neighbors) == 4
        up, down, left, right = neighbors
        assert up == (0, 1, 1)
        assert down == (2, 1, 1)
        assert left == (1, 0, 0)
        assert right == (1, 2, 0)

    def test_get_cell_neighbours_corner(self, basic_maze):
        """Test getting neighbors for corner cells."""
        # Top-left corner
        neighbors = basic_maze.get_cell_neighbours(0, 0)
        assert len(neighbors) == 4
        assert neighbors[0] is None  # Up
        assert neighbors[2] is None  # Left
        assert neighbors[1] == (1, 0, 0)  # Down
        assert neighbors[3] == (0, 1, 1)  # Right

        # Bottom-right corner
        neighbors = basic_maze.get_cell_neighbours(3, 3)
        assert len(neighbors) == 4
        assert neighbors[1] is None  # Down
        assert neighbors[3] is None  # Right
        assert neighbors[0] == (2, 3, 1)  # Up
        assert neighbors[2] == (3, 2, 0)  # Left

    def test_get_cell_neighbours_edges(self, basic_maze):
        """Test getting neighbors for edge cells."""
        # Top edge
        neighbors = basic_maze.get_cell_neighbours(0, 1)
        assert len(neighbors) == 4
        assert neighbors[0] is None  # Up
        assert neighbors[1] is not None  # Down
        assert neighbors[2] is not None  # Left
        assert neighbors[3] is not None  # Right

        # Right edge
        neighbors = basic_maze.get_cell_neighbours(1, 3)
        assert len(neighbors) == 4
        assert neighbors[3] is None  # Right

    def test_export_html(self, basic_maze, tmp_path):
        """Test HTML export functionality."""
        export_path = tmp_path / "maze.html"
        basic_maze.export_html(export_path)

        assert export_path.exists()
        content = export_path.read_text()
        assert "<html>" in content
        assert "<body>" in content
        assert f"Maze #{basic_maze.seed}" in content
        assert "cell-start" in content
        assert "cell-goal" in content

    def test_export_html_with_solution(self, basic_maze, tmp_path):
        """Test HTML export with solution path."""
        path = basic_maze.find_path()
        assert path is not None
        export_path = tmp_path / "maze_solution.html"
        basic_maze.export_html(export_path, draw_solution=True)

        assert export_path.exists()
        content = export_path.read_text()
        assert "cell-path" in content

    def test_repr(self, basic_maze):
        """Test string representation of maze."""
        expected = "VMaze(rows=4, cols=4, start=[0 0], goal=[3 3])"
        assert repr(basic_maze) == expected

    def test_str(self, basic_maze):
        """Test the string representation of the maze."""
        expected_str = "VMaze(seed=42, rows=4, cols=4, start=[0, 0], goal=[3, 3])"
        assert str(basic_maze) == expected_str

    def test_to_dict(self, simple_maze):
        """Test dictionary conversion with original types."""
        maze_dict = simple_maze.to_dict()
        assert isinstance(maze_dict["seed"], (int, np.integer))
        assert isinstance(maze_dict["size"], (int, np.integer))
        assert isinstance(maze_dict["start"], list)
        assert isinstance(maze_dict["goal"], list)
        assert isinstance(maze_dict["min_valid_paths"], (int, np.integer))
        assert isinstance(maze_dict["maze_map"], list)
        assert isinstance(maze_dict["maze_map"][0], list)
        assert isinstance(maze_dict["pathfinding_algorithm"], str)

        # Verify values
        assert maze_dict["seed"] == simple_maze.seed
        assert maze_dict["size"] == simple_maze.rows
        assert maze_dict["start"] == simple_maze.start.tolist()
        assert maze_dict["goal"] == simple_maze.goal.tolist()
        assert maze_dict["min_valid_paths"] == simple_maze.min_valid_paths
        assert np.array_equal(maze_dict["maze_map"], simple_maze.maze_map.tolist())

    def test_to_json(self, simple_maze):
        """Test JSON string conversion with native Python types."""
        json_str = simple_maze.to_json()
        data = json.loads(json_str)  # Should not raise any JSON decode errors

        # Verify all numeric values are native Python types
        assert isinstance(data["seed"], int)
        assert isinstance(data["size"], int)
        assert all(isinstance(x, int) for x in data["start"])
        assert all(isinstance(x, int) for x in data["goal"])
        assert isinstance(data["min_valid_paths"], int)
        assert all(isinstance(x, int) for row in data["maze_map"] for x in row)

    def test_export_json(self, simple_maze, tmp_path):
        """Test JSON file export."""
        json_path = tmp_path / "maze.json"
        simple_maze.export_json(json_path)

        # Verify file exists and is valid JSON
        assert json_path.exists()
        with open(json_path) as f:
            maze_data = json.load(f)

        # Check key properties
        assert maze_data["seed"] == int(simple_maze.seed)
        assert maze_data["size"] == int(simple_maze.rows)
        assert maze_data["start"] == [int(x) for x in simple_maze.start.tolist()]
        assert maze_data["goal"] == [int(x) for x in simple_maze.goal.tolist()]
        assert np.array_equal(
            maze_data["maze_map"],
            [[int(cell) for cell in row] for row in simple_maze.maze_map.tolist()],
        )

    def test_generate_maze_with_min_paths(self):
        """Test maze generation with minimum valid paths requirement."""
        maze = VMaze(seed=42, size=6, start=(1, 1), goal=(4, 4), min_valid_paths=3)
        maze.generate_maze()

        # Block all but one path and verify we can still find a solution
        path = maze.find_path()
        assert path is not None

        # Block the found path
        for x, y in path[1:-1]:  # Don't block start/goal
            maze.maze_map[x, y] = 0

        # Should still find another path
        new_path = maze.find_path()
        assert new_path is not None

    def test_eq(self):
        """Test the equality operator for VMaze."""
        maze1 = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3))
        maze2 = VMaze(seed=42, size=4, start=(0, 0), goal=(3, 3))
        maze3 = VMaze(seed=43, size=4, start=(0, 0), goal=(3, 3))

        # Test equality between VMaze objects
        assert maze1 == maze2, "Mazes with the same attributes should be equal"
        assert maze1 != maze3, "Mazes with different seeds should not be equal"

        # Test comparison with non-VMaze objects
        assert maze1 != "not a maze"
        assert maze1 != 42
        assert maze1 != [1, 2, 3]
        assert maze1 != {"seed": 42}

    def test_from_json(self, simple_maze):
        """Test loading maze from JSON string."""
        # Convert maze to JSON and back
        json_str = simple_maze.to_json()
        loaded_maze = VMaze.from_json(json_str)

        # Verify basic properties match
        assert loaded_maze.seed == simple_maze.seed
        assert loaded_maze.rows == simple_maze.rows
        assert loaded_maze.cols == simple_maze.cols
        assert np.array_equal(loaded_maze.start, simple_maze.start)
        assert np.array_equal(loaded_maze.goal, simple_maze.goal)
        assert loaded_maze.min_valid_paths == simple_maze.min_valid_paths

    def test_from_json_invalid(self):
        """Test from_json with invalid JSON data."""
        # Test with missing required fields
        invalid_json = json.dumps({"seed": 42, "size": 4})  # Missing start/goal
        with pytest.raises(KeyError):
            VMaze.from_json(invalid_json)

        # Test with invalid JSON string
        with pytest.raises(json.JSONDecodeError):
            VMaze.from_json("invalid json")

        # Test with invalid values
        invalid_data = {
            "seed": 42,
            "size": 2,  # Too small
            "start": [0, 0],
            "goal": [1, 1],
            "min_valid_paths": 3,
        }
        with pytest.raises(ValueError, match="size must be at least 4"):
            VMaze.from_json(json.dumps(invalid_data))
