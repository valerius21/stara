#!/usr/bin/env python3

import argparse
from pathlib import Path
from time import time

from loguru import logger
import numpy as np

from stara_maze_generator.pathfinder import Pathfinder
from stara_maze_generator.vmaze import VMaze


def get_default_output(
    size: int,
    seed: int,
    min_valid_paths: int,
    pathfinding_algorithm: Pathfinder,
    format: str,
) -> Path:
    """Generate default output filename based on maze settings."""
    return Path(
        f"{size}x{size}_seed{seed}_paths{min_valid_paths}_{pathfinding_algorithm.name}_maze.{format}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and visualize mazes using Prim's algorithm"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=40,
        help="Size of the maze (creates a size x size grid)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible maze generation",
    )
    parser.add_argument(
        "--start",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("ROW", "COL"),
        help="Starting position coordinates (row, col)",
    )
    parser.add_argument(
        "--goal",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        help="Goal position coordinates (row, col). Defaults to (size-2, size-2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path. Defaults to size_seedX_pathsY_maze.[format]",
    )
    parser.add_argument(
        "--format",
        choices=["html", "json"],
        default="html",
        help="Output format (default: html)",
    )
    parser.add_argument(
        "--draw-solution",
        action="store_true",
        help="Draw the solution path in the HTML output",
    )
    parser.add_argument(
        "--min-valid-paths",
        type=int,
        default=3,
        help="Minimum number of valid paths to ensure connectivity",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If seed not specified, use random, print it
    if args.seed is None:
        rng = np.random.default_rng()
        args.seed = rng.integers(0, 10_000)
        logger.info(f"Seed: {args.seed}")

    # If goal not specified, use (size-2, size-2)
    if args.goal is None:
        args.goal = np.array([args.size - 2, args.size - 2])

    # Set default output path if not specified
    if args.output is None:
        args.output = get_default_output(
            args.size, args.seed, args.min_valid_paths, Pathfinder.BFS, args.format
        )

    # Create and generate maze
    maze = VMaze(
        seed=args.seed,
        size=args.size,
        start=args.start,
        goal=args.goal,
        min_valid_paths=args.min_valid_paths,
    )

    start_time = time()
    maze.generate_maze(pathfinding_algorithm=Pathfinder.BFS)
    end_time = time()

    path = maze.find_path()
    logger.info(
        f"{Pathfinder.BFS}, len(path): {len(path) if path else 'No path found'}"
    )
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")

    # Export in selected format
    if args.format == "json":
        maze.export_json(args.output)
    else:
        maze.export_html(args.output, draw_solution=args.draw_solution)
    logger.info(f"Maze exported to {args.output}")


if __name__ == "__main__":
    main()
