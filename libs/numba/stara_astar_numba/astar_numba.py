from typing import Tuple, List, Optional
import argparse
from loguru import logger
from time import time
import numpy as np
from numpy.typing import NDArray
from numba import njit

from stara_maze_generator.vmaze import VMaze
from stara_maze_generator.pathfinder.base import PathfinderBase


@njit(cache=True)
def get_lowest_f_score_node(
    f_scores: np.ndarray,
    open_set_mask: np.ndarray,
) -> Tuple[int, int]:
    """
    Get the node with the lowest f_score from the open set.

    Args:
        f_scores: 2D array of f_scores
        open_set_mask: Boolean mask of open nodes

    Returns:
        Tuple[int, int]: Position of the node with lowest f_score
    """
    min_f = np.inf
    min_pos = (0, 0)

    rows, cols = f_scores.shape
    for i in range(rows):
        for j in range(cols):
            if open_set_mask[i, j] and f_scores[i, j] < min_f:
                min_f = f_scores[i, j]
                min_pos = (i, j)

    return min_pos


@njit(cache=True)
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


@njit(cache=True)
def reconstruct_path(
    came_from_row: np.ndarray,
    came_from_col: np.ndarray,
    start: Tuple[int, int],
    current: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from the came_from arrays.

    Args:
        came_from_row: Array storing parent row indices
        came_from_col: Array storing parent column indices
        start: Starting position
        current: Current (goal) position

    Returns:
        List[Tuple[int, int]]: The reconstructed path
    """
    path = []
    current_pos = current

    while current_pos != start:
        path.append(current_pos)
        row, col = current_pos
        new_row = came_from_row[row, col]
        new_col = came_from_col[row, col]
        current_pos = (new_row, new_col)

    path.append(start)
    return path[::-1]  # Reverse the path


@njit(cache=True)
def initialize_arrays(
    rows: int, cols: int, start: Tuple[int, int], goal: Tuple[int, int]
):
    """
    Initialize arrays needed for A* pathfinding.

    Args:
        rows: Number of rows in the maze
        cols: Number of columns in the maze
        start: Starting position (row, col)
        goal: Goal position (row, col)

    Returns:
        Tuple containing:
            g_scores: Array of actual distances from start
            f_scores: Array of estimated total distances
            open_set_mask: Boolean mask of nodes to evaluate
            closed_set_mask: Boolean mask of evaluated nodes
            came_from_row: Array storing parent row indices
            came_from_col: Array storing parent column indices
    """
    g_scores = np.full((rows, cols), np.inf)
    f_scores = np.full((rows, cols), np.inf)
    open_set_mask = np.zeros((rows, cols), dtype=np.bool_)
    closed_set_mask = np.zeros((rows, cols), dtype=np.bool_)
    came_from_row = np.full((rows, cols), -1)
    came_from_col = np.full((rows, cols), -1)

    # Initialize start position
    g_scores[start[0], start[1]] = 0
    f_scores[start[0], start[1]] = manhattan_distance(start, goal)
    open_set_mask[start[0], start[1]] = True

    return (
        g_scores,
        f_scores,
        open_set_mask,
        closed_set_mask,
        came_from_row,
        came_from_col,
    )


@njit(cache=True)
def get_valid_neighbors(
    current_pos: Tuple[int, int],
    maze_map: np.ndarray,
    closed_set_mask: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Get valid neighboring positions that aren't walls or already closed.

    Args:
        current_pos: Current position (row, col)
        maze_map: 2D array representing the maze (0 for walls, 1 for passages)
        closed_set_mask: Boolean mask of already evaluated nodes

    Returns:
        List[Tuple[int, int]]: List of valid neighbor positions
    """
    row, col = current_pos
    rows, cols = maze_map.shape
    valid_neighbors = []

    neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    for x, y in neighbors:
        if (
            0 <= x < rows
            and 0 <= y < cols
            and maze_map[x, y] != 0
            and not closed_set_mask[x, y]
        ):
            valid_neighbors.append((x, y))

    return valid_neighbors


@njit(cache=True)
def update_neighbor(
    current_pos: Tuple[int, int],
    neighbor: Tuple[int, int],
    goal: Tuple[int, int],
    g_scores: np.ndarray,
    f_scores: np.ndarray,
    came_from_row: np.ndarray,
    came_from_col: np.ndarray,
    open_set_mask: np.ndarray,
) -> None:
    """
    Update the neighbor's scores and parent if a better path is found.

    Args:
        current_pos: Current position being evaluated
        neighbor: Neighbor position to update
        goal: Target position
        g_scores: Array of actual distances from start
        f_scores: Array of estimated total distances
        came_from_row: Array storing parent row indices
        came_from_col: Array storing parent column indices
        open_set_mask: Boolean mask of nodes to evaluate

    Note:
        Updates are made in-place on the input arrays.
    """
    x, y = neighbor
    tentative_g = g_scores[current_pos] + 1

    if tentative_g < g_scores[x, y]:
        # This path is better than any previous one
        came_from_row[x, y] = current_pos[0]
        came_from_col[x, y] = current_pos[1]
        g_scores[x, y] = tentative_g
        f_scores[x, y] = tentative_g + manhattan_distance((x, y), goal)
        open_set_mask[x, y] = True


@njit(cache=True)
def find_path(
    maze_map: np.ndarray,
    start: NDArray | Tuple[int, int],
    goal: NDArray | Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    Find shortest path from start to goal using A* search.

    Args:
        maze_map: 2D array representing the maze (0 for walls, 1 for passages)
        start: Starting position (row, col)
        goal: Target position (row, col)

    Returns:
        Optional[List[Tuple[int, int]]]: List of coordinates forming the path,
                                        or None if no path exists
    """
    # Convert to int tuples in a Numba-compatible way
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    # Initialize all arrays
    rows, cols = maze_map.shape
    g_scores, f_scores, open_set_mask, closed_set_mask, came_from_row, came_from_col = (
        initialize_arrays(rows, cols, start, goal)
    )

    while np.any(open_set_mask):
        current_pos = get_lowest_f_score_node(f_scores, open_set_mask)

        if current_pos == goal:
            return reconstruct_path(came_from_row, came_from_col, start, current_pos)

        # Remove current from open set and add to closed set
        open_set_mask[current_pos] = False
        closed_set_mask[current_pos] = True

        # Process valid neighbors
        neighbors = get_valid_neighbors(current_pos, maze_map, closed_set_mask)
        for neighbor in neighbors:
            update_neighbor(
                current_pos,
                neighbor,
                goal,
                g_scores,
                f_scores,
                came_from_row,
                came_from_col,
                open_set_mask,
            )

    return None


class AStarNumba(PathfinderBase):
    """
    A* Search pathfinding implementation using Numba.

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

    def find_path(
        self, start: NDArray | Tuple[int, int], goal: NDArray | Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to goal using A* search.

        Args:
            start: Starting position (row, col)
            goal: Target position (row, col)

        Returns:
            Optional[List[Tuple[int, int]]]: List of coordinates forming the path,
                                           or None if no path exists
        """
        path = find_path(self.maze.maze_map, start, goal)
        if path is not None:
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
    pathfinder = AStarNumba(maze)
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
