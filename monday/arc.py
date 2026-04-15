from arcengine import GameAction, FrameDataRaw
from arc_agi import EnvironmentWrapper
from arc_agi import Arcade
from dotenv import load_dotenv
load_dotenv()


class MyArcSession:
    def __init__(self, game_id: str, arcade: Arcade, scorecard_id: str):
        self.env: EnvironmentWrapper = arcade.make(
            game_id=game_id, scorecard_id=scorecard_id, save_recording=True, include_frame_data=False)
        self.last_obs: FrameDataRaw | None = None
        self.obs: FrameDataRaw | None = self.env.reset()
        self.has_reset_this_level = False

    def reset(self):
        self.last_obs = None
        self.obs = self.env.reset()

    def do_action(self, action: GameAction, data: dict | None = None):
        assert action.name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4",
                               "ACTION5", "ACTION6", "ACTION7", "RESET"}
        if action == GameAction.RESET:
            self.reset()
            return
        self.last_obs = self.obs
        self.obs = self.env.step(action, data=data)

    def tick_state(self):
        if self.obs.state:
            pass

    def do_action_from_name(self, action_name: str, data: dict | None = None):
        ga = GameAction.from_name(action_name)
        self.do_action(action=ga, data=data)
