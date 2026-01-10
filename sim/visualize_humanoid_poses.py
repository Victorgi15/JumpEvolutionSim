"""Visualize humanoid following a sequence of target poses."""

import sys

sys.path.insert(0, ".")
from physics.engine import World
from creature.humanoid import HumanoidCreature
from render.viewer import Viewer


def demo(duration=10.0, dt=1 / 240.0, force_scale: float = 1.0):
    world = World()
    world.dt = dt
    h = HumanoidCreature({}, world, base_x=0.0, force_scale=force_scale)
    # define simple poses: stand, crouch, reach (targets are relative to pelvis index)
    stand = {
        "duration": 1.5,
        "targets": {5: (0.0, 0.4), 6: (-0.25, 0.3), 7: (0.25, 0.3)},
    }
    crouch = {
        "duration": 1.5,
        "targets": {3: (-0.2, -0.05), 4: (0.2, -0.05), 5: (0.0, 0.2)},
    }
    reach = {"duration": 1.5, "targets": {8: (-0.6, 0.6), 9: (0.6, 0.6), 5: (0.0, 0.9)}}
    h.set_pose_sequence([stand, crouch, reach])

    viewer = Viewer(world, creature=h)

    t = 0.0

    def step():
        nonlocal t
        h.step_controller(t, dt)
        h.step_actuators(t, dt)
        world.step(dt)
        t += dt

    viewer.run_loop(step)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--force-scale", type=float, default=1.0, help="Scale multiplier for muscle forces")
    args = p.parse_args()
    demo(force_scale=args.force_scale)
