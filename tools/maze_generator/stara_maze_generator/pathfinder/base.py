from typing import Tuple, List, Optional, TYPE_CHECKING

from numpy._typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from stara_maze_generator.vmaze import VMaze


class PathfinderBase:
    """
    Base class for all pathfinding algorithms.

    Provides common initialization and defines the interface that all
    pathfinding implementations must follow.
    """

    def __init__(self, maze: "VMaze"):
        """
        Initialize pathfinder with a maze.

        Args:
            maze: A maze instance implementing the MazeProtocol
        """
        self.maze = maze

    def find_path(
        self, start: NDArray | Tuple[int, int], goal: NDArray | Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find a path from start to goal.

        Args:
            start: Starting position (row, col)
            goal: Target position (row, col)

        Returns:
            Optional[List[Tuple[int, int]]]: List of coordinates forming the path,
                                           or None if no path exists

        Raises:
            NotImplementedError: If the pathfinding algorithm is not implemented
        """
        raise NotImplementedError
