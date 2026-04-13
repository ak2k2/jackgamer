"""Jack Gamer. A Google gemini harness with vision capabiltites"""

from arcengine import GameAction
from arc_agi import OperationMode
import arc_agi
from google.genai import types
from google import genai
from log import write_log, start_server, _render_grid
import argparse
import base64 as b64_mod
import hashlib
import json
import logging
import mimetypes
import subprocess
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)


client = genai.Client()

MODELS = ["gemini-3-flash-preview", "gemini-3.1-pro-preview"]
MODEL = MODELS[1]

SYSTEM_PROMPT = """\
You are playing an interactive puzzle game on a 64x64 grid.

Each cell is an integer 0-15 representing a color. You don't know the game rules. \
You must figure them out by experimenting: take actions, observe what changes, \
build a theory, test it.

After each action you receive:
- state: NOT_FINISHED, WIN, or GAME_OVER
- score: levels completed so far (a score increase means a level was solved)
- win_levels: total levels needed to win
- available_actions: which actions are legal right now (only use these)
- A board image showing the current grid state
- A diff of what changed since the last action
- A memory of actions already tried from this board state

The full 64x64 grid is also written to /home/agent/state.json after every action.

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
- diff_grids(old, new) → list of (row, col, old_val, new_val)
- color_counts(grid) → {color: count}
- find_color(grid, val) → [(row, col), ...]
Use: python3 -c "from helpers import *; grid = load_grid(); print(color_counts(grid))"

Goal: reach WIN state in as few actions as possible.\
"""

RUN_COMMAND = {
    "name": "run_command",
    "description": (
        "Run a shell command in the sandbox. Working directory is /home/agent. "
        "Python 3.12, numpy, matplotlib, pillow are available. "
        "Commands run as bash -c so pipes and chaining work."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute as a clean string.",
            },
        },
        "required": ["command"],
    },
}

VIEW_FILE = {
    "name": "view_file",
    "description": (
        "View an image or PDF from the sandbox. The file is added to your visual "
        "context so you can see and analyze it. Supports images (png, jpg, gif, webp) "
        'and PDFs. Do NOT use this for text files — read those with run_command("cat path") instead.'
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to an image or PDF in the sandbox.",
            },
        },
        "required": ["path"],
    },
}

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

RENDER_BOARD = {
    "name": "render_board",
    "description": "Render the current game board as a PNG image. No arguments needed.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

TOOLS = [RUN_COMMAND, VIEW_FILE, TAKE_ACTION, RENDER_BOARD]

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=TOOLS)],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    temperature=1.0,
    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    candidate_count=1,
    stop_sequences=[],
    thinking_config=types.ThinkingConfig(thinking_level="high"),
)


env = None
current_obs = None
state_action_memory = {}  # {grid_hash: {action: {"changed": bool, "diff": str}}}


def to_dict(obs):
    return {
        "state": obs.state.name,
        "score": obs.levels_completed,
        "win_levels": obs.win_levels,
        "grid": [layer.tolist() for layer in obs.frame],
        "available_actions": [
            GameAction.from_id(a).name for a in obs.available_actions
        ],
    }


def _hash_grid(grid):
    return hashlib.md5(str(grid).encode()).hexdigest()[:12]


def _diff_grids(old, new):
    """diff of changes with hints when stuck."""
    if not old or not new:
        return ""
    old_g = old[-1] if isinstance(old[0], list) and isinstance(old[0][0], list) else old
    new_g = new[-1] if isinstance(new[0], list) and isinstance(new[0][0], list) else new
    changes = []
    for r in range(min(len(old_g), len(new_g))):
        for c in range(min(len(old_g[r]), len(new_g[r]))):
            if old_g[r][c] != new_g[r][c]:
                changes.append(f"{old_g[r][c]}→{new_g[r][c]} at ({r},{c})")
    if not changes:
        return "no change"
    if len(changes) > 20:
        return f"{len(changes)} cells changed. first 20: " + "; ".join(changes[:20])
    return "; ".join(changes)


def _get_tried_actions(grid):
    h = _hash_grid(grid)
    tried = state_action_memory.get(h, {})
    if not tried:
        return ""
    lines = []
    for action, result in tried.items():
        status = "changed" if result["changed"] else "no change"
        line = f"  {action}: {status}"
        if result["changed"] and result["diff"]:
            line += f" ({result['diff'][:100]})"
        lines.append(line)
    return "Already tried from this board state:\n" + "\n".join(lines)


def _record_action(grid, action_name, changed, diff):
    h = _hash_grid(grid)
    state_action_memory.setdefault(h, {})[action_name] = {
        "changed": changed,
        "diff": diff,
    }


def _sync_state_to_sandbox(obs):
    state_json = json.dumps(obs)
    subprocess.run(
        [
            "docker",
            "exec",
            "-w",
            "/home/agent",
            "sandbox",
            "bash",
            "-c",
            f"cat > state.json << 'STATEEOF'\n{state_json}\nSTATEEOF",
        ],
        capture_output=True,
        timeout=10,
    )


def _seed_sandbox_helpers():
    helpers = '''\
import json
import numpy as np

def load_grid():
    """Load the current grid as a numpy array."""
    return np.array(json.load(open("state.json"))["grid"][-1])

def load_obs():
    """Load the full observation."""
    return json.load(open("state.json"))

def diff_grids(old, new):
    """Return list of (row, col, old_val, new_val) for changed cells."""
    return [(r, c, old[r][c], new[r][c])
            for r in range(len(old)) for c in range(len(old[r]))
            if old[r][c] != new[r][c]]

def color_counts(grid):
    """Count occurrences of each color value."""
    from collections import Counter
    return dict(Counter(int(v) for row in grid for v in row))

def find_color(grid, val):
    """Return list of (row, col) where grid == val."""
    return [(r, c) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c] == val]
'''
    subprocess.run(
        [
            "docker",
            "exec",
            "-w",
            "/home/agent",
            "sandbox",
            "bash",
            "-c",
            f"cat > helpers.py << 'HELPEOF'\n{helpers}\nHELPEOF",
        ],
        capture_output=True,
        timeout=10,
    )


def _render_board_bytes():
    """Try to render current observation grid as PNG bytes."""
    if not current_obs:
        return None
    b64 = _render_grid(current_obs["grid"])
    return b64_mod.b64decode(b64) if b64 else None


def step(args):
    global current_obs
    action_name = args["action"]

    if action_name == "RESET":
        obs = to_dict(env.reset())
        current_obs = obs
        _sync_state_to_sandbox(obs)
        return obs

    ga = GameAction.from_name(action_name)
    data = (
        {"x": int(args["x"]), "y": int(args["y"])} if action_name == "ACTION6" else None
    )
    obs = to_dict(env.step(ga, data=data))
    current_obs = obs
    _sync_state_to_sandbox(obs)
    return obs


VALID_ACTIONS = {
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
    "RESET",
}


def validate_action(args):
    action = args.get("action")
    if action not in VALID_ACTIONS:
        return f"invalid action: {action}"
    if (
        action != "RESET"
        and current_obs
        and action not in current_obs.get("available_actions", [])
    ):
        return f"action {action} not available. available: {current_obs['available_actions']}"
    if action == "ACTION6":
        x, y = args.get("x"), args.get("y")
        if x is None or y is None:
            return "ACTION6 requires x and y coordinates"
        if not (0 <= x <= 63 and 0 <= y <= 63):
            return f"coordinates out of range: x={x}, y={y} (must be 0-63)"
    return None


def execute_tool(name, args):
    try:
        return _execute_tool(name, args)
    except subprocess.TimeoutExpired:
        return {"result": "error: command timed out"}
    except Exception as e:
        return {"result": f"error: {type(e).__name__}: {e}"}


def _execute_tool(name, args):
    if name == "run_command":
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-w",
                "/home/agent",
                "sandbox",
                "bash",
                "-c",
                args["command"],
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {"result": (result.stdout + result.stderr).strip()}

    if name == "view_file":
        path = args["path"]
        result = subprocess.run(
            ["docker", "exec", "sandbox", "cat", path],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {"result": f"error: {result.stderr.decode().strip()}"}
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return {"result": f"Displaying {path}", "_bytes": result.stdout, "_mime": mime}

    if name == "take_action":
        error = validate_action(args)
        if error:
            return {"result": f"error: {error}"}
        before_grid = current_obs["grid"] if current_obs else None
        obs = step(args)
        summary = {k: v for k, v in obs.items() if k != "grid"}

        action_name = args["action"]
        changed = before_grid is not None and obs["grid"] != before_grid
        diff = _diff_grids(before_grid, obs["grid"]) if before_grid else ""
        if before_grid:
            _record_action(before_grid, action_name, changed, diff)
        if not changed and before_grid is not None:
            summary["note"] = "grid unchanged — no effect."
        elif diff and diff != "no change":
            summary["diff"] = diff

        tried = _get_tried_actions(obs["grid"])
        if tried:
            summary["memory"] = tried

        img = _render_board_bytes()
        if img:
            return {"result": summary, "_bytes": img, "_mime": "image/png"}
        return {"result": summary}

    if name == "render_board":
        img = _render_board_bytes()
        if not img:
            return {"result": "error: no game state yet"}
        return {
            "result": f"Board rendered (state={current_obs['state']} score={current_obs['score']}/{current_obs['win_levels']})",
            "_bytes": img,
            "_mime": "image/png",
        }

    return {"result": f"unknown tool: {name}"}


def play(game_id, max_actions=500, mode="normal"):
    global env, current_obs

    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", "sandbox"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or "true" not in r.stdout.lower():
        raise RuntimeError("sandbox container not running. run: docker compose up -d")

    arcade = arc_agi.Arcade(operation_mode=OperationMode(mode))
    card_id = arcade.open_scorecard(tags=["agent"])
    env = arcade.make(game_id, scorecard_id=card_id)

    start_server()

    current_obs = to_dict(env.reset())
    _sync_state_to_sandbox(current_obs)
    _seed_sandbox_helpers()
    state_action_memory.clear()
    logger.info(
        "game=%s  state=%s  score=%d/%d",
        game_id,
        current_obs["state"],
        current_obs["score"],
        current_obs["win_levels"],
    )

    obs_summary = {k: v for k, v in current_obs.items() if k != "grid"}
    prompt = (
        f"New game started. Initial state:\n{json.dumps(obs_summary, indent=2)}\n\n"
        f"The full grid is in /home/agent/state.json. "
        f"Start by calling render_board to see the board, then take actions to explore."
    )
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    write_log(contents, model=MODEL)

    action_count = 0
    max_turns = max_actions * 10
    turn_count = 0
    usage = {
        "prompt_tokens": 0,
        "prompt_tokens_total": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
    }

    while True:
        turn_count += 1
        if turn_count > max_turns:
            logger.info("reached max turns (%d)", max_turns)
            break

        logger.info("calling gemini (%d actions so far)...", action_count)
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=CONFIG,
        )

        m = response.usage_metadata
        if m:
            usage["prompt_tokens"] = m.prompt_token_count or 0
            usage["prompt_tokens_total"] += m.prompt_token_count or 0
            usage["output_tokens"] += m.candidates_token_count or 0
            usage["thinking_tokens"] += m.thoughts_token_count or 0

        model_content = response.candidates[0].content
        calls = [p for p in model_content.parts if p.function_call]

        if not calls:
            text = (response.text or "").strip()
            logger.info("model: %s", text[:200] if text else "(thinking)")
            contents.append(model_content)
            write_log(contents, usage=usage, model=MODEL)
            continue

        contents.append(model_content)
        result_parts = []

        for p in calls:
            fc = p.function_call
            output = execute_tool(fc.name, fc.args)

            if fc.name == "take_action":
                action_count += 1
                logger.info(
                    "[%d] %s  state=%s  score=%d/%d",
                    action_count,
                    fc.args.get("action"),
                    current_obs["state"],
                    current_obs["score"],
                    current_obs["win_levels"],
                )
            else:
                preview = (
                    output["result"]
                    if isinstance(output["result"], str)
                    else str(output["result"])
                )
                logger.info("%s: %s", fc.name, preview[:200])

            fr_kwargs = {
                "name": fc.name,
                "response": {"result": output["result"]},
                "id": fc.id,
            }
            if "_bytes" in output:
                fr_kwargs["parts"] = [
                    types.FunctionResponsePart(
                        inline_data=types.FunctionResponseBlob(
                            mime_type=output["_mime"],
                            display_name="board.png",
                            data=output["_bytes"],
                        )
                    )
                ]

            result_parts.append(
                types.Part(function_response=types.FunctionResponse(**fr_kwargs))
            )

        contents.append(types.Content(role="user", parts=result_parts))
        write_log(contents, usage=usage, model=MODEL)

        if current_obs["state"] == "WIN":
            logger.info("WIN in %d actions!", action_count)
            break

        if action_count >= max_actions:
            logger.info("reached max actions (%d)", max_actions)
            break

    scorecard = arcade.close_scorecard(card_id)
    if scorecard:
        logger.info("scorecard: %s", card_id)
        logger.info(json.dumps(scorecard.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ft09")
    p.add_argument("--max-actions", type=int, default=500)
    p.add_argument(
        "--mode", default="normal", choices=["normal", "online", "competition"]
    )
    args = p.parse_args()
    play(args.game, args.max_actions, args.mode)
