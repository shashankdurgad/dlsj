"""python -m pytest sim/twin/rl/test_gate.py"""
import pytest

from sim.twin.rl.gate import Fader, is_pick_prompt

PICK = ["pick up the blue brick", "pick up the brown brick", "pick up the red brick", "pick up the white brick",
        "pick up the yellow brick", "pick up the brick and lift it up", "pick up a brick from the table",
        "pick up the brick and hold it in front of the inspection camera", "Pick up the brick", "grasp the block",
        "grab the red brick", "lift the duplo brick", "PICK UP THE BRICK AND LIFT IT UP"]
NOT_PICK = ["place the brick in the orange bowl", "place the brick in the purple bowl",
            "put the brick in the {colour} bowl", "place the brick in the {colour} bowl",
            "bring the brick to the inspection camera", "put the brick in the purple bowl",
            "place the brick in the yellow bowl", "", None, "go home", "hold the brick still",
            "release the brick"]


@pytest.mark.parametrize("p", PICK)
def test_pick(p):
    assert is_pick_prompt(p)


@pytest.mark.parametrize("p", NOT_PICK)
def test_not_pick(p):
    assert not is_pick_prompt(p)


def test_fader():
    f = Fader(4)
    assert f.update("pick up the brick and lift it up") == 1.0
    vals = [f.update("put the brick in the red bowl") for _ in range(5)]
    assert vals == [0.75, 0.5, 0.25, 0.0, 0.0]
    assert f.update("pick up a brick from the table") == 1.0
