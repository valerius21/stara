import argparse
from stara_maze_generator.vmaze import VMaze
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

MAX_SEED = 10_000_000_000


def generate_maze(size: int, seed: int, min_valid_paths: int) -> VMaze:
    maze = VMaze(
        size=size,
        seed=seed,
        min_valid_paths=min_valid_paths,
        start=np.array([0, 0]),
        goal=np.array([size - 1, size - 1]),
    )
    return maze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze-size", type=int, default=10)
    parser.add_argument("--maze-count", type=int, default=10)
    parser.add_argument("--min-valid-paths", type=int, default=3)
    parser.add_argument("--omit-maze", type=bool, default=False)
    parser.add_argument("--output-path", type=str, default="mazes.pkl")
    args = parser.parse_args()

    if args.maze_count > MAX_SEED:
        logger.error("maze-count must be less than 10_000_000")
        return

    rng = np.random.default_rng()
    seeds = rng.choice(MAX_SEED, args.maze_count, replace=False)

    logger.info(f"Generating {args.maze_count} mazes")
    df = pd.DataFrame()

    for i in tqdm(range(args.maze_count)):
        seed = seeds[i]
        if args.omit_maze:
            maze = None
            data = {
                "seed": seed,
                "size": args.maze_size,
                "min_valid_paths": args.min_valid_paths,
                "start": [0, 0],
                "goal": [args.maze_size - 1, args.maze_size - 1],
            }
        else:
            maze = generate_maze(args.maze_size, seed, args.min_valid_paths)
            data = {
                "seed": seed,
                "maze": maze,
                "size": args.maze_size,
                "min_valid_paths": args.min_valid_paths,
                "start": maze.start.tolist(),
                "goal": maze.goal.tolist(),
            }

        df = df._append(data, ignore_index=True)

    df.to_pickle(args.output_path)
    logger.info(f"Saved to {args.output_path}")
    logger.info(f"df.head():\n{df.head()}")


if __name__ == "__main__":
    main()
