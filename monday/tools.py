SYSTEM_PROMPT = """\
You play an interactive 2D puzzle game on a 64x64 grid.

Each cell is an integer 0-15 representing a color. You don't know the game rules. \
You must figure them out by experimenting: take actions, observe what changes, \
build a theory, test it.

A game consists of one or more levels.

The system you are playing against allows you to input actions.

Actions:
- ACTION1: up
- ACTION2: down
- ACTION3: left
- ACTION4: right
- ACTION5: interact / confirm (game-specific)
- ACTION6: click a cell at (x, y) where x,y in [0,63]
- ACTION7: undo
- RESET: restart the current level

Color palette (0-15):
0=white 1=off-white 2=light-gray 3=gray 4=off-black 5=black \
6=magenta 7=light-magenta 8=red 9=blue 10=light-blue 11=yellow \
12=orange 13=maroon 14=green 15=purple

You have a Linux sandbox (working dir /home/agent) with Python 3.12 and sudo. \
Pre-installed: numpy, scipy, pandas, matplotlib, pillow, scikit-image, \
scikit-learn, opencv (cv2), networkx, sympy, z3-solver. \
You can run any command, write scripts, install more packages.

A helpers.py file is pre-loaded with:
- load_grid() → numpy array of the current 64x64 grid
- load_obs() → full observation dict
- render_board() → saves board.png with correct game colors
- diff_grids(old, new) → list of (row, col, old_val, new_val)
- color_counts(grid) → {color: count}
- find_color(grid, val) → [(row, col), ...]
Example: python3 -c "from helpers import *; render_board()"

Use view to see any image you create. You cannot see images \
unless you pass them through view.

Your files persist between turns. Use edit to modify existing files, \
write only for new files. View a file before editing it.

Full action history is in /home/agent/replay.jsonl (one JSON event per action, no grid data). \
Current grid is always in /home/agent/state.json.

Goal: build local and global understanding of the game dynamics.

Learn agentically from trying strategies and analysing data / results.
Every game has a "solution". The goal is to get to a solution quickly and reliably.
"""

TAKE_ACTION = {
    "name": "take_action",
    "description": (
        "Submit a game action. Returns state, score, and available_actions. "
        "Updated grid is written to /home/agent/state.json. "
        "For ACTION6 (click), x and y are required (0-63)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "ACTION1",
                    "ACTION2",
                    "ACTION3",
                    "ACTION4",
                    "ACTION5",
                    "ACTION6",
                    "ACTION7",
                    "RESET",
                ],
                "description": "The action to take.",
            },
            "x": {
                "type": "integer",
                "description": "X coordinate for ACTION6 (0-63). Ignored for other actions.",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate for ACTION6 (0-63). Ignored for other actions.",
            },
        },
        "required": ["action"],
    },
}


BASH = {
    "name": "bash",
    "description": "Run a shell command in the sandbox.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute.",
            },
            "timeout": {
                "type": "number",
                "description": "Optional timeout in milliseconds (max 600000)",
            },
        },
        "required": ["command"],
    },
}

VIEW = {
    "name": "view",
    "description": "Read a file. Text files shown with line numbers. Images displayed visually.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file.",
            },
            "offset": {
                "type": "integer",
                "description": "The line number to start reading from (0-based)",
            },
            "limit": {
                "type": "integer",
                "description": "The number of lines to read (defaults to 2000)",
            },
        },
        "required": ["file_path"],
    },
}

WRITE = {
    "name": "write",
    "description": "Create or overwrite a file. Parent directories are created automatically.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file.",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    },
}

EDIT = {
    "name": "edit",
    "description": (
        "Perform exact string replacement in a file. "
        "old_string must be unique in the file unless replace_all is true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file.",
            },
            "old_string": {
                "type": "string",
                "description": "The text to replace.",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences (default false).",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    },
}

TOOL_LIST = [BASH, VIEW, WRITE, EDIT, TAKE_ACTION]
