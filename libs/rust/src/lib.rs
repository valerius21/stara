use pyo3::prelude::*;

#[pyclass]
struct MazeSolver {
    width: usize,
    height: usize,
    data: Vec<u8>,            // Flat array: 0 = wall, 1 = path
    visited: Vec<u8>,         // Visit tracking: 0 = unvisited, 1 = open, 2 = closed
    parent: Vec<u32>,         // Changed to u32 to support larger mazes
    open: [(i32, u32); 1024], // Changed to u32 indices
    open_len: usize,
}

#[pymethods]
impl MazeSolver {
    #[new]
    fn new() -> Self {
        MazeSolver {
            width: 0,
            height: 0,
            data: Vec::new(),
            visited: Vec::new(),
            parent: Vec::new(),
            open: [(0, 0); 1024],
            open_len: 0,
        }
    }

    fn load(&mut self, maze: Vec<Vec<usize>>) -> PyResult<()> {
        self.height = maze.len();
        self.width = maze[0].len();
        let size = self.width * self.height;

        self.data = Vec::with_capacity(size);
        self.visited = vec![0; size];
        self.parent = vec![0; size];

        // Flatten maze into single vector, 0 = impassable
        for row in maze {
            self.data.extend(row.iter().map(|&x| x as u8));
        }
        Ok(())
    }

    fn astar(&mut self, start: (i32, i32), goal: (i32, i32)) -> PyResult<Vec<(i32, i32)>> {
        if self.data.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Maze not loaded",
            ));
        }

        let start_idx = self.get_index(start.0, start.1);
        let goal_idx = self.get_index(goal.0, goal.1);

        // Check if start or goal is impassable
        if self.data[start_idx] == 0 || self.data[goal_idx] == 0 {
            return Ok(vec![]);
        }

        self.visited.fill(0);
        self.open_len = 1;
        self.open[0] = (manhattan_distance(start, goal), start_idx as u32);
        self.visited[start_idx as usize] = 1;

        while self.open_len > 0 {
            let (_, current_idx) = self.pop_min();
            let current_idx = current_idx as usize;

            if current_idx == goal_idx as usize {
                return Ok(self.reconstruct_path(start, goal, start_idx, goal_idx));
            }

            self.visited[current_idx] = 2;

            let x = (current_idx / self.width) as i32;
            let y = (current_idx % self.width) as i32;

            // Inline neighbor checks
            static DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
            for &(dx, dy) in &DIRS {
                let nx = x + dx;
                let ny = y + dy;

                if nx >= 0 && (nx as usize) < self.height && ny >= 0 && (ny as usize) < self.width {
                    let idx = self.get_index(nx, ny);
                    if self.visited[idx] != 2 && self.data[idx] != 0 {
                        if self.visited[idx] == 0 {
                            let f = manhattan_distance((nx, ny), goal);
                            self.push_open(f, idx as u32);
                            self.visited[idx] = 1;
                            self.parent[idx] = current_idx as u32;
                        }
                    }
                }
            }
        }
        Ok(vec![])
    }
}

impl MazeSolver {
    #[inline(always)]
    fn get_index(&self, x: i32, y: i32) -> usize {
        (x as usize) * self.width + (y as usize)
    }

    #[inline(always)]
    fn pop_min(&mut self) -> (i32, u32) {
        let mut min_idx = 0;
        let mut min_score = self.open[0].0;

        for i in 1..self.open_len {
            if self.open[i].0 < min_score {
                min_idx = i;
                min_score = self.open[i].0;
            }
        }

        self.open_len -= 1;
        let result = self.open[min_idx];
        self.open[min_idx] = self.open[self.open_len];
        result
    }

    #[inline(always)]
    fn push_open(&mut self, f_score: i32, idx: u32) {
        if self.open_len < 1024 {
            self.open[self.open_len] = (f_score, idx);
            self.open_len += 1;
        }
    }

    fn reconstruct_path(
        &self,
        start: (i32, i32),
        goal: (i32, i32),
        start_idx: usize,
        goal_idx: usize,
    ) -> Vec<(i32, i32)> {
        let mut path = Vec::with_capacity(self.width + self.height);
        let mut current_idx = goal_idx;

        // Start with goal
        path.push(goal);
        current_idx = self.parent[current_idx] as usize;

        // Add intermediate nodes
        while current_idx != start_idx {
            let x = (current_idx / self.width) as i32;
            let y = (current_idx % self.width) as i32;
            path.push((x, y));
            current_idx = self.parent[current_idx] as usize;
        }

        // Add start
        path.push(start);
        path.reverse();
        path
    }
}

#[inline(always)]
fn manhattan_distance(pos1: (i32, i32), pos2: (i32, i32)) -> i32 {
    (pos1.0 - pos2.0).abs() + (pos1.1 - pos2.1).abs()
}

#[pymodule]
fn stara_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MazeSolver>()?;
    Ok(())
}
