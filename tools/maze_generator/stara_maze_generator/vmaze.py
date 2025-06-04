"""
VMaze - A visual maze generator and solver using NumPy.

This module provides functionality to create, solve, and visualize mazes using a grid-based approach.
The mazes are generated using a modified version of the randomized Prim's algorithm and solved using BFS.
"""

from pathlib import Path
import json
from typing import Tuple, Dict, Any

import numpy as np
from numpy._typing import NDArray
from stara_maze_generator.pathfinder.types import Pathfinder
from stara_maze_generator.pathfinder.bfs import BFS
from stara_maze_generator.visualization import export_html


class VMaze:
    """
    A class representing a visual maze with generation and solving capabilities.

    The maze is represented as a 2D grid where 0 represents walls and 1 represents passages.
    The maze can be generated with a random seed for reproducibility and exported as an HTML visualization.
    """

    def __init__(
        self,
        seed: int,
        size: int,
        start: NDArray | Tuple[int],
        goal: NDArray | Tuple[int],
        min_valid_paths: int = 3,
    ):
        """
        Initialize a new maze with given dimensions and start/goal positions.

        Args:
            seed (int): Random seed for reproducible maze generation
            size (int): Size of the maze (creates a size x size grid)
            start (Tuple[int]): Starting position coordinates (row, col)
            goal (Tuple[int]): Goal position coordinates (row, col)
            min_valid_paths (int): Minimum number of valid paths to ensure connectivity (default: 3)

        Raises:
            ValueError: If size is less than 4
        """
        if size < 4:
            raise ValueError("size must be at least 4")
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.path = []
        self.min_valid_paths = min_valid_paths

        self.rows = size
        self.cols = size
        # 0 means wall
        self.maze_map = np.zeros((self.rows, self.cols), dtype=np.int64)
        self.start = np.array(start)
        self.goal = np.array(goal)
        # set start and goal as free from walls
        self.maze_map[self.start[0], self.start[1]] = 1
        self.maze_map[self.goal[0], self.goal[1]] = 1

        self.pathfinding_algorithm = Pathfinder.BFS

    def __repr__(self):
        return f"VMaze(rows={self.rows}, cols={self.cols}, start={self.start}, goal={self.goal})"

    def get_cell_neighbours(self, x: int, y: int) -> tuple[tuple[int, int, int]]:
        """
        Get the neighboring cells of a given position.

        Args:
            x (int): Row coordinate
            y (int): Column coordinate

        Returns:
            tuple[tuple[int, int, int]]: Tuple of neighbors, each containing (x, y, value)
                                     where value is 0 for wall, 1 for passage.
                                       None if neighbor would be out of bounds.
        """
        res = []
        if x - 1 >= 0:
            res.append((x - 1, y, self.maze_map[x - 1, y]))
        else:
            res.append(None)
        if x + 1 <= self.rows - 1:
            res.append((x + 1, y, self.maze_map[x + 1, y]))
        else:
            res.append(None)
        if y - 1 >= 0:
            res.append((x, y - 1, self.maze_map[x, y - 1]))
        else:
            res.append(None)
        if y + 1 <= self.cols - 1:
            res.append((x, y + 1, self.maze_map[x, y + 1]))
        else:
            res.append(None)

        return tuple(res)

    def find_path(
        self,
        pathfinding_algorithm: Pathfinder = Pathfinder.BFS,
    ):
        """
        Find a path from start to goal using Breadth-First Search.

        Args:
            pathfinding_algorithm (Pathfinder): Path finding algorithm to use (default: BFS)

        Returns:
            list[tuple[int, int]] | None: List of coordinates representing the path if found,
                                         None if no path exists.
        """
        if pathfinding_algorithm == Pathfinder.BFS:
            self.pathfinding_algorithm = Pathfinder.BFS
            return BFS(self).find_path(self.start, self.goal)
        raise NotImplementedError(
            f"Pathfinder {pathfinding_algorithm} not implemented"
        )  # pragma: no cover

    def to_dict(self) -> Dict[str, Any]:
        """Convert maze to dictionary preserving original types."""
        return {
            "seed": self.seed,
            "size": self.rows,
            "start": self.start.tolist(),
            "goal": self.goal.tolist(),
            "min_valid_paths": self.min_valid_paths,
            "maze_map": self.maze_map.tolist(),
            "path": self.path if self.path else None,
            "pathfinding_algorithm": self.pathfinding_algorithm.name,
        }

    def to_json(self) -> str:
        """Convert maze to JSON string with Python native types."""
        data = self.to_dict()
        return json.dumps(
            {
                "seed": int(data["seed"]),
                "size": int(data["size"]),
                "start": [int(x) for x in data["start"]],
                "goal": [int(x) for x in data["goal"]],
                "min_valid_paths": int(data["min_valid_paths"]),
                "maze_map": [[int(cell) for cell in row] for row in data["maze_map"]],
                "path": (
                    [[int(x) for x in pos] for pos in data["path"]]
                    if data["path"]
                    else None
                ),
                "pathfinding_algorithm": data["pathfinding_algorithm"],
            },
            indent=2,
        )

    @staticmethod
    def from_json(json_str: str) -> "VMaze":
        """Load maze from JSON string."""
        data = json.loads(json_str)
        maze = VMaze(
            seed=data["seed"],
            size=data["size"],
            start=data["start"],
            goal=data["goal"],
            min_valid_paths=data["min_valid_paths"],
        )
        # generate maze after init
        maze.generate_maze()
        return maze

    def export_json(self, dest_path: Path) -> None:
        """Export maze as JSON file."""
        with open(dest_path, "w") as f:
            f.write(self.to_json())

    def generate_maze(self, pathfinding_algorithm: Pathfinder = Pathfinder.BFS):
        """
        Generate a random maze using a modified version of Prim's algorithm.

        The generation ensures that there exists at least one valid path from start to goal.
        The maze is generated by starting with all walls and carving passages, ensuring
        connectivity between the start and goal positions.
        """
        # Start with both start and goal cells as passages
        self.maze_map[self.start[0], self.start[1]] = 1
        self.maze_map[self.goal[0], self.goal[1]] = 1

        # Add walls adjacent to both start and goal to the wall list
        start_neighbours = [
            (x, y)
            for (x, y, value) in self.get_cell_neighbours(self.start[0], self.start[1])
            if value is not None and value == 0
        ]
        goal_neighbours = [
            (x, y)
            for (x, y, value) in self.get_cell_neighbours(self.goal[0], self.goal[1])
            if value is not None and value == 0
        ]

        # Combine both sets of walls and convert to list for easier manipulation
        walls = list(
            set(start_neighbours + goal_neighbours)
        )  # Using set to remove duplicates

        while walls:  # Changed from len(walls) > 0 to just walls
            # Pick a random wall
            wall_idx = self.rng.integers(0, len(walls))
            current_wall = walls[wall_idx]

            wall_neighbours = self.get_cell_neighbours(current_wall[0], current_wall[1])
            wall_neighbours = [n for n in wall_neighbours if n is not None]
            passage_count = sum(1 for n in wall_neighbours if n[2] == 1)

            if passage_count == 1:
                self.maze_map[current_wall[0], current_wall[1]] = 1

                # Add new walls
                new_walls = [(x, y) for (x, y, value) in wall_neighbours if value == 0]
                walls.extend(new_walls)

            # Remove the processed wall
            walls.pop(wall_idx)

        # More efficient path finding - only break walls that create a valid path
        if self.find_path(pathfinding_algorithm) is None:
            wall_positions = list(zip(*np.nonzero(self.maze_map == 0)))
            self.rng.shuffle(wall_positions)  # Shuffle to try random walls
            valid_paths = 0
            walls_to_break = []

            for wall in wall_positions:
                # Temporarily break wall
                self.maze_map[wall[0], wall[1]] = 1

                # Check if this creates a valid path
                if self.find_path(pathfinding_algorithm) is not None:
                    valid_paths += 1
                    self.maze_map[wall[0], wall[1]] = 0  # Restore the wall
                    walls_to_break.append(wall)
                    if valid_paths >= self.min_valid_paths:
                        break
                else:
                    self.maze_map[wall[0], wall[1]] = 0  # Restore the wall

            # break all the marked walls
            for wall in walls_to_break:
                self.maze_map[wall[0], wall[1]] = 1

    def export_html(
        self, dest_path: Path = Path("./export.html"), draw_solution: bool = False
    ) -> None:
        return export_html(self, dest_path, draw_solution)

    def __str__(self) -> str:
        return f"VMaze(seed={self.seed}, rows={self.rows}, cols={self.cols}, start={self.start.tolist()}, goal={self.goal.tolist()})"

    def __eq__(self, other: object) -> bool:
        """Compare two VMaze objects for equality.
        NOTE: Pathfinding algorithm is not compared!

        Args:
            other (object): The other object to compare to

        Returns:
            bool: True if the objects are equal, False otherwise
        """
        if not isinstance(other, VMaze):
            return NotImplemented
        # Compare relevant attributes for equality
        return (
            self.rows == other.rows
            and self.cols == other.cols
            and self.seed == other.seed
            and self.min_valid_paths == other.min_valid_paths
            and np.array_equal(self.start, other.start)
            and np.array_equal(self.goal, other.goal)
            and np.array_equal(self.maze_map, other.maze_map)
        )
