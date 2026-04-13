from arcengine import GameAction, FrameDataRaw
from arc_agi import EnvironmentWrapper
from arc_agi import Arcade
from dotenv import load_dotenv

load_dotenv()


class MyArcSession:
    def __init__(self, game_id: str, arcade: Arcade, scorecard_id: str):
        self.env: EnvironmentWrapper = arcade.make(
            game_id=game_id, scorecard_id=scorecard_id
        )
        self.last_obs: FrameDataRaw | None = None
        self.obs: FrameDataRaw | None = self.env.reset()

    def reset(self):
        self.prev_obs = None
        self.obs = self.env.reset()

    def do_action(self, action: GameAction):
        assert action.name in {
            "ACTION1",
            "ACTION2",
            "ACTION3",
            "ACTION4",
            "ACTION5",
            "ACTION6",
            "ACTION7",
            "RESET",
        }
        if action == GameAction.RESET:
            self.reset()
            return
        self.last_obs = self.obs
        data = {}  # TODO: support click
        self.obs = self.env.step(action, data=data)

    def do_action_from_name(self, action_name: str):
        ga = GameAction.from_name(action_name)
        self.do_action(action=ga)
