from typing import Any, Optional

from arcengine import FrameDataRaw, GameAction

from arc import MyArcSession
from arc_agi import OperationMode
from arc_agi import Arcade
import logging
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from sandbox import SandboxOrchestrator
import io
import json
from google.genai import types
from google import genai
import mimetypes
from tools import TOOL_LIST, SYSTEM_PROMPT, TAKE_ACTION

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)


def current_state_prompt(obs: FrameDataRaw) -> str:
    return f"""Currently playing game: {obs.game_id}. On level #{obs.levels_completed} out of #{obs.win_levels}.

"""


def starting_state_prompt(obs: FrameDataRaw) -> str:
    res = f"""Starting the following game: {obs.game_id}

"""

    return (
        f"New game started. Initial state:\n{json.dumps(obs, indent=2)}\n\n"
        f"The full grid is in /home/agent/state.json. "
        f"Start by calling render_board to see the board, then take actions to explore."
    )


class JackAgent:
    def __init__(self, sbx: SandboxOrchestrator, arc_session: MyArcSession):
        self.client = genai.Client()
        self.model = ["gemini-3.1-flash-lite-preview",
                      "gemini-3-flash-preview", "gemini-3.1-pro-preview"][2]
        self.contents: list[types.Content] = []
        self.sbx: SandboxOrchestrator = sbx
        self.arc_session: MyArcSession = arc_session
        self.replay_path = str(arc_session.env._recording_filename)
        self.usage = {"prompt_tokens": 0, "output_tokens": 0,
                      "thinking_tokens": 0, "total_prompt_tokens": 0,
                      "image_prompt_tokens": 0, "image_output_tokens": 0}
        self._seed_helpers()
        self._sync_state()
        self.clear()

    def _seed_helpers(self):
        self.sbx.write("/home/agent/helpers.py", '''\
import json
import numpy as np
from PIL import Image

PALETTE = np.array([
    [0xFF,0xFF,0xFF],[0xCC,0xCC,0xCC],[0x99,0x99,0x99],[0x66,0x66,0x66],
    [0x33,0x33,0x33],[0x00,0x00,0x00],[0xE5,0x3A,0xA3],[0xFF,0x7B,0xCC],
    [0xF9,0x3C,0x31],[0x1E,0x93,0xFF],[0x88,0xD8,0xF1],[0xFF,0xDC,0x00],
    [0xFF,0x85,0x1B],[0x92,0x12,0x31],[0x4F,0xCC,0x30],[0xA3,0x56,0xD6],
], dtype=np.uint8)

def load_grid():
    """Load the current grid as a numpy array."""
    return np.array(json.load(open("state.json"))["grid"])

def load_obs():
    """Load the full observation."""
    return json.load(open("state.json"))

def render_board(path="board.png"):
    """Render current board with correct game colors, save as PNG."""
    grid = load_grid()
    rgb = PALETTE[np.clip(grid.astype(np.uint8), 0, 15)]
    img = Image.fromarray(rgb).resize((512, 512), Image.NEAREST)
    img.save(path)
    return path

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
''')

    def _sync_state(self):
        obs = self.arc_session.obs
        state = obs.model_dump(mode="json")
        if obs.frame:
            state["grid"] = obs.frame[-1].tolist()
        self.sbx.write("/home/agent/state.json", json.dumps(state))

    def clear(self):
        self.contents = []

    # --- standalone image annotation (Nano Banana, separate from main context) ---
    PALETTE_RGB = np.array([
        [0xFF, 0xFF, 0xFF], [0xCC, 0xCC, 0xCC], [0x99, 0x99, 0x99], [0x66, 0x66, 0x66],
        [0x33, 0x33, 0x33], [0x00, 0x00, 0x00], [0xE5, 0x3A, 0xA3], [0xFF, 0x7B, 0xCC],
        [0xF9, 0x3C, 0x31], [0x1E, 0x93, 0xFF], [0x88, 0xD8, 0xF1], [0xFF, 0xDC, 0x00],
        [0xFF, 0x85, 0x1B], [0x92, 0x12, 0x31], [0x4F, 0xCC, 0x30], [0xA3, 0x56, 0xD6],
    ], dtype=np.uint8)

    def _frame_to_png(self, frame: np.ndarray, size: int = 512) -> bytes:
        rgb = self.PALETTE_RGB[np.clip(np.asarray(frame, dtype=np.uint8), 0, 15)]
        img = Image.fromarray(rgb).resize((size, size), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def annotate(self, prompt: str, frame: Optional[np.ndarray] = None) -> Optional[Image.Image]:
        """Call Nano Banana 2 to edit the current (or given) board frame per the prompt.
        Returns a PIL Image or None if the model refused. Tracks tokens in self.usage."""
        if frame is None:
            frame = self.arc_session.obs.frame[-1]
        png_bytes = self._frame_to_png(frame)
        res = self.client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[prompt, Image.open(io.BytesIO(png_bytes))],
        )
        m = res.usage_metadata
        if m:
            self.usage["image_prompt_tokens"] += m.prompt_token_count or 0
            self.usage["image_output_tokens"] += m.candidates_token_count or 0
        for part in res.parts:
            if part.inline_data and part.inline_data.data:
                return Image.open(io.BytesIO(part.inline_data.data))
        return None

    def execute_tool(self, name: str, args: Optional[dict[str, Any]] = None):
        try:
            args = args or {}
            if name == TAKE_ACTION["name"]:
                action_name: str = args.get("action")
                available = [GameAction.from_id(
                    a).name for a in self.arc_session.obs.available_actions]
                if action_name != "RESET" and action_name not in available:
                    return {"result": f"error: {action_name} not available. available: {available}"}
                data = None
                if action_name == "ACTION6":
                    x, y = args.get("x"), args.get("y")
                    if x is None or y is None:
                        return {"result": "error: ACTION6 requires x and y coordinates"}
                    if not (0 <= int(x) <= 63 and 0 <= int(y) <= 63):
                        return {"result": f"error: coordinates out of range: x={x}, y={y} (must be 0-63)"}
                    data = {"x": int(x), "y": int(y)}
                self.arc_session.do_action_from_name(
                    action_name=action_name, data=data)
                # sync state + replay to sandbox
                self._sync_state()
                self.sbx.write("/home/agent/replay.jsonl",
                               open(self.replay_path).read())
                return {"result": (
                    f"state={self.arc_session.obs.state.name} score={self.arc_session.obs.levels_completed}/{self.arc_session.obs.win_levels} "
                    f"available={[GameAction.from_id(a).name for a in self.arc_session.obs.available_actions]}\n"
                    f"Current grid and full obs written to /home/agent/state.json. "
                    f"Full action history in /home/agent/replay.jsonl."
                )}
            elif name == "view":
                file_path = args.get("file_path", "")
                mime = mimetypes.guess_type(file_path)[0] or ""
                if mime.startswith("image/"):
                    raw = self.sbx.read_file(file_path)
                    return {"result": f"Displaying {file_path}", "_bytes": raw, "_mime": mime}
                return {"result": self.sbx.view(**args)}
            else:
                return {"result": self.sbx.func_callable_map[name](**args)}
        except Exception as e:
            return {"result": f"error: {type(e).__name__}: {e}"}

    def generate_response(self):
        res: types.GenerateContentResponse = self.client.models.generate_content(
            model=self.model,
            contents=self.contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[types.Tool(function_declarations=TOOL_LIST)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="AUTO") # change to ANY to force function calls
                ),
                temperature=1.2,
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                candidate_count=1,
                stop_sequences=[],
                thinking_config=types.ThinkingConfig(thinking_level="high"),
            ),
        )
        m = res.usage_metadata
        if m:
            self.usage["prompt_tokens"] = m.prompt_token_count or 0
            self.usage["total_prompt_tokens"] += m.prompt_token_count or 0
            self.usage["output_tokens"] += m.candidates_token_count or 0
            self.usage["thinking_tokens"] += m.thoughts_token_count or 0

        model_content: types.Content = res.candidates[0].content
        calls: list[types.Part] = [
            p for p in model_content.parts if p.function_call]

        self.contents.append(model_content)

        if not calls:
            return []

        result_parts: list[types.Part] = []
        executed: list[dict] = []

        for p in calls:
            fc: types.FunctionCall = p.function_call

            output: dict = self.execute_tool(
                name=fc.name, args=fc.args)
            executed.append({"name": fc.name, "args": dict(
                fc.args), "result": output["result"]})

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
                            display_name=output.get("_name", "image.png"),
                            data=output["_bytes"],
                        )
                    )
                ]

            result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(**fr_kwargs))
            )

        self.contents.append(types.Content(role="user", parts=result_parts))
        return executed


def main():
    arcade = Arcade(operation_mode=OperationMode("normal"))
    scorecard_id: str = arcade.open_scorecard(tags=["jackagent"])

    arc_session = MyArcSession(
        game_id=["ls20", "ft09"][0], arcade=arcade, scorecard_id=scorecard_id
    )
    print(arc_session.obs)
    sbx = SandboxOrchestrator()

    print(sbx.bash("echo sandbox ready"))
    print(sbx.bash("python3 -c 'print(1+122)'"))

    agent = JackAgent(sbx=sbx, arc_session=arc_session)

    print(agent.contents)

    agent.contents = []


if __name__ == "__main__":
    main()
