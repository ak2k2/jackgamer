"""Run JackAgent on a game and print results."""
import argparse
import logging
import json

from arcengine import GameAction
from arc import MyArcSession
from arc_agi import OperationMode, Arcade
from sandbox import SandboxOrchestrator
from google.genai import types
from agent import JackAgent
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)


def play(game_id: str, max_turns: int = 200, mode: str = "normal"):
    arcade = Arcade(operation_mode=OperationMode(mode))
    scorecard_id = arcade.open_scorecard(tags=["jackagent"])
    logger.info("scorecard: %s", scorecard_id)

    arc_session = MyArcSession(
        game_id=game_id, arcade=arcade, scorecard_id=scorecard_id
    )
    sbx = SandboxOrchestrator()
    logger.info("sandbox ready: %s", sbx.bash("echo ok"))

    agent = JackAgent(sbx=sbx, arc_session=arc_session)

    obs = arc_session.obs
    agent.contents = [
        types.Content(role="user", parts=[types.Part(text=(
            f"Game: {obs.game_id} | "
            f"Score: {obs.levels_completed}/{obs.win_levels} | "
            f"Actions: {[GameAction.from_id(a).name for a in obs.available_actions]}\n\n"
            f"Current grid is in /home/agent/state.json. "
            f"Full action history is in /home/agent/replay.jsonl. "
            f"Solve this game."
        ))])
    ]

    action_count = 0
    for turn in range(max_turns):
        calls = agent.generate_response()

        if not calls:
            last = agent.contents[-1]
            for p in last.parts:
                if hasattr(p, "text") and p.text:
                    logger.info("[thinking] %s", p.text[:200])
            continue

        for c in calls:
            if c["name"] == "take_action":
                action_count += 1
                logger.info("[%d] %s → %s", action_count, c["args"].get("action"), str(c["result"])[:200])
            else:
                logger.info("[%s] %s", c["name"], str(c["result"])[:200])

        obs = arc_session.obs
        if obs.state.name == "WIN":
            logger.info("WIN in %d actions!", action_count)
            break

        if obs.state.name == "GAME_OVER":
            logger.info("GAME_OVER at %d actions", action_count)
            break

    scorecard = arcade.close_scorecard(scorecard_id)
    if scorecard:
        logger.info("scorecard: %s", scorecard_id)
        logger.info(json.dumps(scorecard.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ft09")
    p.add_argument("--max-turns", type=int, default=500)
    p.add_argument("--mode", default="normal", choices=["normal", "online", "competition"])
    args = p.parse_args()
    play(args.game, args.max_turns, args.mode)
