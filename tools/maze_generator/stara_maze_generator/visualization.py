from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from stara_maze_generator.vmaze import VMaze


def export_html(maze: "VMaze", dest_path: Path, draw_solution: bool = False) -> None:
    """
    Export maze as HTML visualization.

    Args:
        maze: The maze to export
        dest_path: Path where to save the HTML file
        draw_solution: Whether to draw the solution path
    """
    HEADER = f"""
        <html>
        <head>
            <title>Maze #{maze.seed} {maze.rows}x{maze.cols}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body>
        <h1>Maze #{maze.seed} {maze.rows}x{maze.cols} {str(maze.pathfinding_algorithm).replace("Pathfinder.", "")}</h1>
        <div id="maze-root">
        """

    FOOTER = f"""
        </div>
        <div id="maze-legend">
            <span>
                P = Passage
            </span>
            <span>  
                W = Wall
            </span>
            <span>
                S = Start
            </span>
            <span>
                G = Goal
            </span>
        </div>
        </div>
        </body>
        <style>
       body {{
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            h1 {{
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                margin: 1rem 0;
            }}
            #maze-legend {{
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                margin: 1rem 0;
                display: flex;
                flex-direction: row;
                gap: 1rem;
            }}
            #maze-root {{
                display: grid;
                grid-template-columns: repeat({maze.cols}, minmax(0, 1fr));
                grid-template-rows: repeat(1, minmax(0, 1fr));
                max-width: calc({maze.cols} * 2rem);
                gap: 0;
                justify-items: center;
                align-items: center;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                border: 2rem solid #27272a;
                margin: 0;
                padding: 0;
            }}
            .cell-start {{
                background-color: #6ee7b7; /* emerald-300 */
                color: #09090b;
                text-align: center;
                height: 2rem;
                width: 2rem;
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;
            }}
            .cell-goal {{
                background-color: #fca5a5; /* red-300 */
                text-align: center;
                color: #09090b;
                height: 2rem;
                width: 2rem;
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;
            }}
            .cell-passage {{
                background-color: #fafafa; /* zinc-50 */
                text-align: center;
                color: #09090b;
                height: 2rem;
                width: 2rem;
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;
            }}
            .cell-wall {{
                background-color: #27272a; /* zinc-800 */
                text-align: center;
                color: #3f3f46;
                height: 2rem;
                width: 2rem;
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;
            }}
            .cell-path {{
                background-color: rgba(147, 197, 253, 1); /* blue-300 with 0.5 opacity */
                text-align: center;
                color: #09090b;
                height: 2rem;
                width: 2rem;
                display: flex;
                justify-content: center;
                align-items: center;
                text-align: center;
            }}
        </style>
        </html>
        """

    MERGED = HEADER

    for row_idx in range(maze.rows):
        MERGED += '<div class="row">'
        for col_idx in range(maze.cols):
            value = maze.maze_map[row_idx, col_idx]
            if maze.start[0] == row_idx and maze.start[1] == col_idx:
                MERGED += '<div class="cell-start">S</div>'
            elif maze.goal[0] == row_idx and maze.goal[1] == col_idx:
                MERGED += '<div class="cell-goal">G</div>'
            elif (row_idx, col_idx) in maze.path and draw_solution:
                MERGED += '<div class="cell-path">P</div>'
            elif value == 0:
                MERGED += '<div class="cell-wall">W</div>'
            elif value == 1:
                MERGED += '<div class="cell-passage">P</div>'
        MERGED += "</div>"

    MERGED += FOOTER

    # write html to file
    with open(dest_path, "w") as f:
        f.write(MERGED)
