from typing import Tuple, List, Optional

from numpy._typing import NDArray

from stara_maze_generator.pathfinder.base import PathfinderBase


class BFS(PathfinderBase):
    """
    Breadth-First Search pathfinding implementation.

    BFS explores the maze level by level, guaranteeing the shortest path
    in terms of number of steps when all edges have equal weight.
    """

    def find_path(
        self, start: NDArray | Tuple[int, int], goal: NDArray | Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find path from start to goal using Breadth-First Search.

        BFS explores nodes in order of their distance from the start,
        ensuring the first path found is the shortest possible.

        Args:
            start: Starting position (row, col)
            goal: Target position (row, col)

        Returns:
            Optional[List[Tuple[int, int]]]: List of coordinates forming the path,
                                           or None if no path exists
        """
        # Initialize visited set and queue for BFS
        visited = set()
        queue = [(tuple(start), [tuple(start)])]

        while queue:
            (current_pos, path) = queue.pop(0)  # FIFO - take first element

            # If we reached the goal, return the path
            if current_pos == tuple(goal):
                self.maze.path = path
                return path

            # Get all valid neighbors
            neighbors = self.maze.get_cell_neighbours(*current_pos)
            for next_pos in neighbors:
                if next_pos is None:  # Skip if out of bounds
                    continue

                x, y, value = next_pos
                next_pos = (x, y)

                # Skip if wall or already visited
                if value == 0 or next_pos in visited:
                    continue

                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))

        # If no path found, return None
        return None
