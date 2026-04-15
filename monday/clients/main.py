from arcengine import GameAction
from arc import MyArcSession
from arc_agi import OperationMode, Arcade
import io
import logging
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from sandbox import SandboxOrchestrator
from google.genai import types
from agent import JackAgent

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

PALETTE = np.array([
    [0xFF,0xFF,0xFF],[0xCC,0xCC,0xCC],[0x99,0x99,0x99],[0x66,0x66,0x66],
    [0x33,0x33,0x33],[0x00,0x00,0x00],[0xE5,0x3A,0xA3],[0xFF,0x7B,0xCC],
    [0xF9,0x3C,0x31],[0x1E,0x93,0xFF],[0x88,0xD8,0xF1],[0xFF,0xDC,0x00],
    [0xFF,0x85,0x1B],[0x92,0x12,0x31],[0x4F,0xCC,0x30],[0xA3,0x56,0xD6],
], dtype=np.uint8)

PRICING = {
    "gemini-3-flash-preview":  {"in": 0.50, "in_200k": 0.50, "out": 3.00, "out_200k": 3.00},
    "gemini-3.1-pro-preview":  {"in": 2.00, "in_200k": 4.00, "out": 12.00, "out_200k": 18.00},
}


def render_frame(frame: np.ndarray, size: int = 192) -> Image.Image:
    rgb = PALETTE[np.clip(np.asarray(frame, dtype=np.uint8), 0, 15)]
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)


def estimate_cost(usage, model):
    p = PRICING.get(model, PRICING["gemini-3-flash-preview"])
    ctx = usage["prompt_tokens"]
    rate_in = p["in_200k"] if ctx > 200_000 else p["in"]
    rate_out = p["out_200k"] if ctx > 200_000 else p["out"]
    cost_in = usage["total_prompt_tokens"] * rate_in / 1_000_000
    cost_out = (usage["output_tokens"] + usage["thinking_tokens"]) * rate_out / 1_000_000
    return cost_in + cost_out


# --- setup ---
arcade = Arcade(operation_mode=OperationMode("normal"))
scorecard_id: str = arcade.open_scorecard(tags=["jackagent"])
arc_session = MyArcSession(
    game_id=["ls20", "ft09", "vc33"][0], arcade=arcade, scorecard_id=scorecard_id
)
sbx = SandboxOrchestrator()
agent = JackAgent(sbx=sbx, arc_session=arc_session)

# --- kick off ---
agent.contents = [
    types.Content(role="user", parts=[types.Part(text=(
        f"Game: {arc_session.obs.game_id} | "
        f"Score: {arc_session.obs.levels_completed}/{arc_session.obs.win_levels} | "
        f"Actions: {[GameAction.from_id(a).name for a in arc_session.obs.available_actions]}\n\n"
        f"Render the board, look at it, figure out the rules, solve the game."
    ))])
]

# --- loop ---
for turn in range(100):
    calls = agent.generate_response()
    u = agent.usage
    cost = estimate_cost(u, agent.model)
    obs = agent.arc_session.obs
    score = f"{obs.levels_completed}/{obs.win_levels}"
    state = obs.state.name

    print(f"\n{'─' * 70}")
    print(f"  turn {turn+1}    {state} {score}    ctx {u['prompt_tokens']:,}  out {u['output_tokens']:,}  think {u['thinking_tokens']:,}  ${cost:.4f}")
    print(f"{'─' * 70}")

    if not calls:
        for p in agent.contents[-1].parts:
            if hasattr(p, "text") and p.text:
                print(p.text[:500])
        continue

    for c in calls:
        name = c["name"]
        result = str(c["result"])

        if name == "take_action":
            action = c["args"].get("action", "")
            xy = f" ({c['args'].get('x')},{c['args'].get('y')})" if action == "ACTION6" else ""
            print(f"\n  {action}{xy}    {state} {score}")
            if obs.frame:
                display(render_frame(obs.frame[-1]))

        elif name == "bash":
            cmd = c["args"].get("command", "")
            print(f"\n  $ {cmd}")
            for line in result[:300].splitlines():
                print(f"    {line}")

        elif name == "view":
            path = c["args"].get("file_path", "")
            if result.startswith("Displaying"):
                print(f"\n  view  {path}")
                try:
                    raw = agent.sbx.read_file(path)
                    display(Image.open(io.BytesIO(raw)).resize((192, 192), Image.NEAREST))
                except Exception:
                    pass
            else:
                lines = result.count("\n") + 1
                print(f"\n  view  {path}  ({lines} lines)")

        elif name == "write":
            print(f"\n  write  {c['args'].get('file_path', '')}")

        elif name == "edit":
            print(f"\n  edit  {c['args'].get('file_path', '')}")

        else:
            print(f"\n  {name}  {str(c['args'])[:100]}")

    if state == "WIN":
        print(f"\n{'=' * 70}")
        print(f"  WIN in {turn+1} turns    ${cost:.4f}")
        print(f"{'=' * 70}")
        break
