import argparse
import random
import subprocess
from time import perf_counter_ns, time
from typing import List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from stara_astar.astar import AStar as AstarNaive
from stara_astar_numba.astar_numba import AStarNumba
from stara_astar_stdlib.a_star_stdlib import AStarStdLib
from stara_cpp.stara_cpp import AStar as AstarCpp
from stara_cpp.stara_cpp import load_maze as load_cpp_maze
from stara_maze_generator.pathfinder.base import PathfinderBase
from stara_maze_generator.vmaze import VMaze
from stara_rs.stara_rs import MazeSolver
from tqdm import tqdm

from batch_mazes.generator import generate_maze


class AstarCxx(PathfinderBase):
    def __init__(self, maze: VMaze) -> None:
        super().__init__(maze)
        mmap = maze.maze_map.copy()

        # cast mmap elements to np.int32

        mmap = mmap.astype(np.int32)

        maze_ptr = load_cpp_maze(mmap)
        self.solver = AstarCpp(maze_ptr)

    def find_path(
        self,
        start: NDArray[np.int32] | Tuple[int, int],
        goal: NDArray[np.int32] | Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        return self.solver.find_path(start, goal)


class AStarRS(PathfinderBase):
    def __init__(self, maze):
        super().__init__(maze)
        self.solver = MazeSolver()
        self.solver.load(self.maze.maze_map)

    def find_path(
        self,
        start: NDArray[np.int32] | Tuple[int, int],
        goal: NDArray[np.int32] | Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        return self.solver.astar(start, goal)


def run_find_path(solver: PathfinderBase, maze: VMaze, N=1_000) -> float:
    start = (maze.start[0], maze.start[1])
    goal = (maze.goal[0], maze.goal[1])
    start_time = perf_counter_ns()
    for _ in range(N):
        solver.find_path(start, goal)
    end_time = perf_counter_ns()
    # https://docs.python.org/3/library/time.html#time.get_clock_info
    # ---------------
    # Macbook Pro, 16" 2021 M1 Max, 32 GB RAM
    # >>> time.get_clock_info('monotonic')
    # namespace(implementation='mach_absolute_time()', monotonic=True, adjustable=False, resolution=4.166666666666666e-08)
    # >>> time.get_clock_info('time')
    # namespace(implementation='clock_gettime(CLOCK_REALTIME)', monotonic=False, adjustable=True, resolution=1.0000000000000002e-06)
    # >>> time.get_clock_info('perf_counter')
    # namespace(implementation='mach_absolute_time()', monotonic=True, adjustable=False, resolution=4.166666666666666e-08)
    return (end_time - start_time) / N


def shuffe_row_benchmark(maze: VMaze, runs=1_000):
    data = {
        "naive": {"runs": runs, "instance": AstarNaive(maze), "times": []},
        "stdlib": {"runs": runs, "instance": AStarStdLib(maze), "times": []},
        "rust": {"runs": runs, "instance": AStarRS(maze), "times": []},
        "numba": {"runs": runs, "instance": AStarNumba(maze), "times": []},
        "cpp": {"runs": runs, "instance": AstarCxx(maze), "times": []},
    }
    has_runs_left = any(option["runs"] > 0 for option in data.values())
    while has_runs_left:
        random_impl = random.choice(
            list([key for key in data.keys() if data[key]["runs"] > 0])
        )
        option = data[random_impl]
        option["runs"] -= 1
        time = run_find_path(option["instance"], maze)
        option["times"].append(time)
        has_runs_left = any(option["runs"] > 0 for option in data.values())

    return data


def get_maze_from_dataframe(row, maze_size, maze_valid_path_count):
    if row["seed"] is None:
        raise ValueError("seed is required")
    return generate_maze(maze_size, row["seed"], maze_valid_path_count)


def process_row_shuffle(row, maze_size, maze_valid_path_count, nuitka_bin):
    # check if row has 'maze' column
    if "maze" not in row:
        row["maze"] = None
    maze: VMaze = row["maze"]
    if maze is None:
        maze = get_maze_from_dataframe(row, maze_size, maze_valid_path_count)
    results = shuffe_row_benchmark(maze, nuitka_bin)
    for key, value in results.items():
        row[key] = np.mean(value["times"])
    return row


def process_row(row, maze_size, maze_valid_path_count, nuitka_path, RUN_ID):
    # check if row has 'maze' column
    if "maze" not in row:
        row["maze"] = None
    maze: VMaze = row["maze"]
    if maze is None:
        maze = get_maze_from_dataframe(row, maze_size, maze_valid_path_count)

    # Naive A* search
    naive = AstarNaive(maze)
    row["naive"] = run_find_path(naive, maze)

    # StdLib A* search
    stdlib = AStarStdLib(maze)
    row["stdlib"] = run_find_path(stdlib, maze)

    # Rust A* search
    rs = AStarRS(maze)
    row["pyo3"] = run_find_path(rs, maze)

    # numba A* search
    numba = AStarNumba(maze)
    row["numba"] = run_find_path(numba, maze)

    # C++ A* search
    cxx = AstarCxx(maze)
    row["cpp"] = run_find_path(cxx, maze)

    # Nuitka
    process = subprocess.run(
        [
            str(e)
            for e in [
                nuitka_path,
                "--maze-size",
                maze_size,
                "--maze-seed",
                maze.seed,
                "--benchmark-id",
                RUN_ID,
            ]
        ]
    )
    if process.returncode != 0:
        logger.warning(f"Nuitka exit code: {process.returncode}")
    # else:
    #     df = pd.read_csv(f"maze_{RUN_ID}.csv")
    #     row["nuitka"] = df["times"]

    return row


def strip_df(df) -> pd.DataFrame:
    try:
        df = df.drop(columns=["maze"])
    except KeyError:
        logger.warning("maze column not found")
    try:
        df = df.drop(columns=["min_valid_paths"])
    except KeyError:
        logger.warning("min_valid_paths column not found")
    try:
        df = df.drop(columns=["start"])
    except KeyError:
        logger.warning("start column not found")
    try:
        df = df.drop(columns=["goal"])
    except KeyError:
        logger.warning("goal column not found")
    df = df.dropna()
    # 'size' column is a category
    df["size"] = pd.Categorical(df["size"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--nuitka-bin", type=str, default="astar_nuitka.bin")
    parser.add_argument("--maze-file", type=str, default="10x10_1000.pkl")
    args = parser.parse_args()
    RUN_ID = uuid4()

    mazes: pd.DataFrame | pd.Series = pd.read_pickle(args.maze_file)
    maze_size = mazes.iloc[0]["size"]
    logger.info(f"Maze size: {maze_size}")

    tqdm.pandas(desc="[Benchmark] Processing mazes")

    if args.shuffle:
        logger.info("Shuffler Method")
        logger.info("reading mazes into memory")
        mazes = mazes.progress_apply(
            lambda row: process_row_shuffle(
                row, row["size"], row["min_valid_paths"], args.nuitka_bin
            ),
            axis=1,
        )
    else:
        logger.info("Sequential Method")
        logger.info("reading mazes into memory")
        mazes = mazes.progress_apply(
            lambda row: process_row(
                row, row["size"], row["min_valid_paths"], args.nuitka_bin, RUN_ID
            ),
            axis=1,
        )

    #  merge the results
    mazes = strip_df(mazes)

    logger.info("Merging results on 'seed' column")
    # TODO: merge results from nuitka

    #  <maze-file>.pkl.json contains the results of the nuitka run
    #  so we can just read that file and merge the results
    #  on the seed column
    # nuitka_runs = pd.read_csv(f"maze_")
    # mazes = .merge(mazes, on="seed")

    logger.info("df.head():\n{}".format(mazes.head()))

    file_name = f"{args.maze_file}_{RUN_ID}_{maze_size}.pkl"

    logger.info(f"Saving results to {file_name}")
    mazes.to_pickle(file_name)
    logger.info("Done")

    return 0


if __name__ == "__main__":
    main()
