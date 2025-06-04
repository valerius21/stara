from typing import Tuple, List, Optional
import argparse
from time import time_ns
from loguru import logger

from stara_maze_generator.pathfinder.base import PathfinderBase
from stara_maze_generator.vmaze import VMaze
from stara_rs.stara_rs import MazeSolver


class AStarRust(PathfinderBase):
    def __init__(self, maze):
        super().__init__(maze)
        self.maze_solver = MazeSolver()
        self.maze_solver.load(self.maze.maze_map)

    def find_path(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        return self.maze_solver.astar(start, goal)


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
    pathfinder = AStarRust(maze)
    start = (maze.start[0], maze.start[1])
    goal = (maze.goal[0], maze.goal[1])
    start_time = time_ns()
    path = pathfinder.find_path(start, goal)
    end_time = time_ns()
    if path is None:
        logger.error("No path found")
        exit(1)
    logger.info(f"Maze exported to {args.file}")
    logger.info([(int(x), int(y)) for (x, y) in path])
    logger.info(f"Path length: {len(path)}")
    logger.info(f"Time taken: {end_time - start_time} ns")
