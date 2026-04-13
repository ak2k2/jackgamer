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
import json
from google.genai import types
from google import genai
from tools import TOOL_LIST, SYSTEM_PROMPT, TAKE_ACTION

load_dotenv()


PALETTE = np.array(
    [
        [0xFF, 0xFF, 0xFF],
        [0xCC, 0xCC, 0xCC],
        [0x99, 0x99, 0x99],
        [0x66, 0x66, 0x66],
        [0x33, 0x33, 0x33],
        [0x00, 0x00, 0x00],
        [0xE5, 0x3A, 0xA3],
        [0xFF, 0x7B, 0xCC],
        [0xF9, 0x3C, 0x31],
        [0x1E, 0x93, 0xFF],
        [0x88, 0xD8, 0xF1],
        [0xFF, 0xDC, 0x00],
        [0xFF, 0x85, 0x1B],
        [0x92, 0x12, 0x31],
        [0x4F, 0xCC, 0x30],
        [0xA3, 0x56, 0xD6],
    ],
    dtype=np.uint8,
)


def render_frame(frame: np.ndarray) -> Image.Image:
    rgb = PALETTE[np.clip(np.asarray(frame, dtype=np.uint8), 0, 15)]
    return Image.fromarray(rgb).resize((512, 512), Image.NEAREST)


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
        self.model = ["gemini-3-flash-preview", "gemini-3.1-pro-preview"][1]
        self.contents: list[types.Content] = []
        self.sbx: SandboxOrchestrator = sbx
        self.arc_session: MyArcSession = arc_session
        self.replay_path = str(arc_session.env._recording_filename)
        self.clear()

    def clear(self):
        self.contents = []

    def execute_tool(self, name: str, args: Optional[dict[str, Any]] = None):
        try:
            args = args or {}
            if name == TAKE_ACTION["name"]:
                action_name: str = args.get("action")
                data = None
                if action_name == "ACTION6":
                    x, y = args.get("x"), args.get("y")
                    if x is None or y is None:
                        return {"result": "error: ACTION6 requires x and y coordinates"}
                    if not (0 <= int(x) <= 63 and 0 <= int(y) <= 63):
                        return {"result": f"error: coordinates out of range: x={x}, y={y} (must be 0-63)"}
                    data = {"x": int(x), "y": int(y)}
                self.arc_session.do_action_from_name(action_name=action_name, data=data)
                # sync state + replay to sandbox
                obs = self.arc_session.obs
                state = obs.model_dump(mode="json")
                state["grid"] = obs.frame[-1].tolist()
                self.sbx.write("/home/agent/state.json", json.dumps(state))
                self.sbx.write("/home/agent/replay.jsonl",
                               open(self.replay_path).read())
                return {"result": (
                    f"state={self.arc_session.obs.state.name} score={self.arc_session.obs.levels_completed}/{self.arc_session.obs.win_levels} "
                    f"available={[GameAction.from_id(a).name for a in self.arc_session.obs.available_actions]}\n"
                    f"Current grid and full obs written to /home/agent/state.json. "
                    f"Full action history in /home/agent/replay.jsonl."
                )}
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
                # force a tool call: https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#function_calling_modes
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY")  # AUTO to allow text
                ),
                temperature=1.0,
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                candidate_count=1,
                stop_sequences=[],
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
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
            executed.append({"name": fc.name, "args": dict(fc.args), "result": output["result"]})

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
