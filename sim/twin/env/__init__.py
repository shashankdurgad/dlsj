"""The twin's RL / evaluation environment.

    from sim.twin.env import TwinEnv, TwinEnvCfg, TwinGym

Importing this package does not start Isaac; TwinEnv(...) launches it (headless, cameras, GB10 render fix) if needed.
TwinGym (Physical AI Studio Gym) is imported lazily because physicalai pulls in its own heavy dependencies.
"""
from sim.twin.env.twin_env import CAMS, PICK_TASK, PLACE_TASK, TwinEnv, TwinEnvCfg, launch_app

__all__ = ["CAMS", "PICK_TASK", "PLACE_TASK", "TwinEnv", "TwinEnvCfg", "TwinGym", "launch_app"]


def __getattr__(name):
    if name == "TwinGym":
        from sim.twin.env.studio_gym import TwinGym

        return TwinGym
    raise AttributeError(name)
