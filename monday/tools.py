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

You have a Linux sandbox (working dir /home/agent) with Python 3.12, numpy, \
matplotlib, pillow, and sudo. You can run any command, write scripts, install packages.

A helpers.py file is pre-loaded with:
- load_grid() → numpy array of the current 64x64 grid
- load_obs() → full observation dict
- render_board() → saves board.png with correct game colors
- diff_grids(old, new) → list of (row, col, old_val, new_val)
- color_counts(grid) → {color: count}
- find_color(grid, val) → [(row, col), ...]
Example: python3 -c "from helpers import *; render_board()"

Use view_image to see any image you create. You cannot see files \
unless you pass them through view_image.

Goal: build local and global understanding of the game dynamics.

Learn agentically from trying strategies and analysing data / results.
Every game has a "solution". The goal is to get to a solution quickly and reliably.
"""

TAKE_ACTION = {
    "name": "take_action",
    "description": (
        "Submit a game action. Returns state, score, available_actions, a board image, "
        "a diff of changes, and memory of previously tried actions. "
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
                "description": "The command to execute",
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
    "description": "Read a text file with line numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path to the file to read",
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
                "description": "The path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    },
}

VIEW_IMAGE = {
    "name": "view_image",
    "description": (
        "View an image from the sandbox. The image is added to your visual context "
        "so you can see and analyze it. Use this after saving plots or renderings."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to an image file in the sandbox (png, jpg, gif, webp).",
            },
        },
        "required": ["file_path"],
    },
}

TOOL_LIST = [BASH, VIEW, WRITE, TAKE_ACTION, VIEW_IMAGE]
