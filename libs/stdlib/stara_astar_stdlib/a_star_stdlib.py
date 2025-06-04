import array
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from typing import Dict, List, NamedTuple, Optional, Tuple
import argparse
from time import time


from loguru import logger

from numpy.typing import NDArray
from stara_maze_generator.pathfinder.base import PathfinderBase
from stara_maze_generator.vmaze import VMaze


@dataclass(frozen=True)
class Position:
    """Immutable position representation for caching."""

    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y


class Node(NamedTuple):
    """Lightweight node representation with comparison based on f_score."""

    f_score: float
    g_score: float
    position: Position

    def __lt__(self, other):
        return self.f_score < other.f_score


class AStarStdLib(PathfinderBase):
    """
    Enhanced A* implementation with various optimizations:
    - LRU caching for distance calculations
    - Immutable position objects for better caching
    - Binary heap for priority queue
    - Manhattan Distance Heuristic
    - Early termination optimizations
    - Memory-efficient data structures
    """

    def __init__(self, maze):
        super().__init__(maze)
        self.rows = maze.rows
        self.cols = maze.cols
        # Pre-calculate grid dimensions for bounds checking
        self.max_x = self.rows - 1
        self.max_y = self.cols - 1
        # Direction vectors for neighbor calculation
        self.directions = array.array("b", [-1, 0, 1, 0, 0, -1, 0, 1])

    @lru_cache(maxsize=1024)
    def manhattan_distance(self, pos1: Position, pos2: Position) -> float:
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return abs(dx) + abs(dy)

    @lru_cache(maxsize=1024)
    def get_position(self, x: int, y: int) -> Position:
        """Create or retrieve cached Position instance."""
        return Position(x, y)

    def is_valid_position(self, x: int, y: int) -> bool:
        """Fast bounds checking."""
        return 0 <= x <= self.max_x and 0 <= y <= self.max_y

    def get_neighbors(self, pos: Position) -> List[Tuple[Position, float]]:
        """Get valid neighbors with their movement costs."""
        neighbors = []
        for i in range(4):  # Check all 4 directions
            nx = pos.x + self.directions[i]
            ny = pos.y + self.directions[i + 4]

            if not self.is_valid_position(nx, ny):
                continue

            cell = self.maze.maze_map[nx, ny]
            if cell == 1:  # Is passage
                neighbor_pos = self.get_position(nx, ny)
                # Could add different costs for different terrain types
                neighbors.append((neighbor_pos, 1.0))

        return neighbors

    def find_path(
        self, start: NDArray | Tuple[int, int], goal: NDArray | Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Enhanced A* pathfinding with optimizations.
        """
        # Convert to immutable positions for caching
        start_pos = self.get_position(start[0], start[1])
        goal_pos = self.get_position(goal[0], goal[1])

        # Priority queue for open set
        open_queue = []
        # Track nodes in open set for faster lookup
        open_set = set()
        # Track g_scores with defaultdict
        g_scores = defaultdict(lambda: float("inf"))
        # Track parents for path reconstruction
        came_from = {}

        # Initialize start node
        g_scores[start_pos] = 0
        start_node = Node(
            f_score=self.manhattan_distance(start_pos, goal_pos),
            g_score=0,
            position=start_pos,
        )
        heappush(open_queue, start_node)
        open_set.add(start_pos)

        while open_queue:
            current = heappop(open_queue)
            current_pos = current.position
            open_set.remove(current_pos)

            # Goal test
            if current_pos == goal_pos:
                return self._reconstruct_path(came_from, current_pos, start_pos)

            # Early termination if we've exceeded the best possible path
            if current.g_score > g_scores[current_pos]:
                continue

            # Process neighbors
            for next_pos, cost in self.get_neighbors(current_pos):
                # Calculate tentative g_score
                tentative_g = g_scores[current_pos] + cost

                if tentative_g < g_scores[next_pos]:
                    # Found better path, update scores
                    came_from[next_pos] = current_pos
                    g_scores[next_pos] = tentative_g
                    f_score = tentative_g + self.manhattan_distance(
                        next_pos, goal_pos
                    )

                    if next_pos not in open_set:
                        next_node = Node(f_score, tentative_g, next_pos)
                        heappush(open_queue, next_node)
                        open_set.add(next_pos)

        return None

    def _reconstruct_path(
        self, came_from: Dict[Position, Position], current: Position, start: Position
    ) -> List[Tuple[int, int]]:
        """Reconstruct the path from came_from map."""
        path = []
        while current in came_from:
            path.append((current.x, current.y))
            current = came_from[current]
        path.append((start.x, start.y))
        path.reverse()
        self.maze.path = path
        return path


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--file",
        type=str,
        help="Path to the maze file",
    )
    args = args.parse_args()
    with open(args.file) as f:
        maze = VMaze.from_json(f.read())
    pathfinder = AStarStdLib(maze)
    start_time = time()
    path = pathfinder.find_path(maze.start, maze.goal)
    end_time = time()
    if path is None:
        logger.error("No path found")
        exit(1)
    logger.info(f"Maze exported to {args.file}")
    logger.info([(int(x), int(y)) for (x, y) in path])
    logger.info(f"Path length: {len(path)}")
    logger.info(f"Time taken: {end_time - start_time} seconds")
