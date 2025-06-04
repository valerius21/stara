from typing import Tuple, List, Optional, Dict
import argparse
from loguru import logger
from time import time

from numpy.typing import NDArray

from stara_maze_generator.vmaze import VMaze
from stara_maze_generator.pathfinder.base import PathfinderBase


class AStar(PathfinderBase):
    """
    A* Search pathfinding implementation.

    A* is an informed search algorithm that uses a heuristic function to guide
    its search. It combines Dijkstra's algorithm with a heuristic estimate of
    the distance to the goal, making it more efficient than Dijkstra's algorithm
    while still guaranteeing the shortest path.
    """

    def __init__(self, maze: VMaze):
        """
        Initialize the A* pathfinder.

        Args:
            maze: VMaze instance
        """
        super().__init__(maze)

    @staticmethod
    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two points.

        https://en.wikipedia.org/wiki/Taxicab_geometry

        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)

        Returns:
            int: Manhattan distance between the points
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def get_lowest_f_score_node(
        f_scores: Dict[Tuple[int, int], int], open_set: set
    ) -> Tuple[int, int]:
        """
        Get the node with the lowest f_score from the open set.

        Args:
            f_scores: Dictionary of node positions to their f_scores
            open_set: Set of nodes to consider

        Returns:
            Tuple[int, int]: Position of the node with lowest f_score
        """
        return min(open_set, key=lambda pos: f_scores[pos])

    def find_path(
        self, start: NDArray | Tuple[int, int], goal: NDArray | Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to goal using A* search.

        Uses Manhattan distance as the heuristic function. The algorithm maintains
        both the actual distance from start (g_score) and the estimated total
        distance through each node to the goal (f_score = g_score + heuristic).

        Args:
            start: Starting position (row, col)
            goal: Target position (row, col)

        Returns:
            Optional[List[Tuple[int, int]]]: List of coordinates forming the path,
                                           or None if no path exists
        """
        start = tuple(start)
        goal = tuple(goal)

        # Track actual distance from start to each node
        g_scores = {start: 0}
        # Track estimated total distance through each node
        f_scores = {start: self.manhattan_distance(start, goal)}
        # For path reconstruction
        came_from = {}
        # Set of nodes to evaluate
        open_set = {start}
        # Keep track of visited nodes
        closed_set = set()

        while open_set:
            # Get node with lowest f_score
            current_pos = self.get_lowest_f_score_node(f_scores, open_set)
            open_set.remove(current_pos)

            if current_pos in closed_set:
                continue

            # If we reached the goal, reconstruct and return the path
            if current_pos == goal:
                path = []
                while current_pos in came_from:
                    path.append(current_pos)
                    current_pos = came_from[current_pos]
                path.append(start)
                path.reverse()
                self.maze.path = path
                return path

            closed_set.add(current_pos)

            # Check all neighbors
            neighbors = self.maze.get_cell_neighbours(*current_pos)
            for next_pos in neighbors:
                if next_pos is None:  # Skip if out of bounds
                    continue

                x, y, value = next_pos
                next_pos = (x, y)

                # Skip walls and already processed nodes
                if value == 0 or next_pos in closed_set:
                    continue

                # Calculate tentative g_score for this neighbor
                # All edges have weight 1 in this implementation
                tentative_g = g_scores[current_pos] + 1

                # If we found a better path to this neighbor
                if next_pos not in g_scores or tentative_g < g_scores[next_pos]:
                    # Update the path
                    came_from[next_pos] = current_pos
                    g_scores[next_pos] = tentative_g
                    f_scores[next_pos] = tentative_g + self.manhattan_distance(
                        next_pos, goal
                    )
                    open_set.add(next_pos)

        # If we get here, no path exists
        return None


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
    pathfinder = AStar(maze)
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
