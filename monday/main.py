

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


agent.contents = [
    types.Content(role="user", parts=[types.Part(text=(
        f"Game: {arc_session.obs.game_id} | "
        f"Score: {arc_session.obs.levels_completed}/{arc_session.obs.win_levels} | "
        f"Actions: {[GameAction.from_id(a).name for a in arc_session.obs.available_actions]}\n\n"
        f"Current grid is in /home/agent/state.json. "
        f"Full action history is in /home/agent/replay.jsonl.\n\n"
        f"Workflow:\n"
        f"1. Render: python3 -c 'from helpers import *; render_board()' then view_image('board.png')\n"
        f"2. Analyze the grid — find objects, colors, structure, symmetries\n"
        f"3. Take an action, then render + view again to see what changed\n"
        f"4. Write your theory of the rules to notes.md — update it as you learn\n"
        f"5. Use matplotlib to annotate images — draw boxes, arrows, highlights to verify your understanding\n"
        f"6. Never guess randomly. Build a theory, test it, refine it.\n\n"
        f"You can create any visualization you want with matplotlib/pillow and see it with view_image. "
        f"Solve this game."
    ))])
]

agent.contents.append(types.Content(role="user", parts=[
    types.Part(text="solve quickly")]))

for _ in range(100):
    calls = agent.generate_response()
    if not calls:
        last = agent.contents[-1]
        for p in last.parts:
            if hasattr(p, "text") and p.text:
                print(f"[thinking] {p.text[:300]}\n")
    for c in calls:
        print(f"[{c['name']}] {c['args']}")
        print(f"  → {str(c['result'])[:300]}\n")
        if c["name"] == "take_action" and agent.arc_session.obs.frame:
            display(render_frame(agent.arc_session.obs.frame[-1]))
        if c["name"] == "view_image":
            try:
                raw = agent.sbx.read_file(
                    agent.sbx._resolve(c["args"]["file_path"]))
                display(Image.open(io.BytesIO(raw)))
            except:
                pass
