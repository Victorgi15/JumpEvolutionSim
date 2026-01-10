"""Visualize humanoid following a sequence of joint angle poses."""

import sys
import math

sys.path.insert(0, ".")
from physics.engine import World
from creature.humanoid import HumanoidCreature
from render.viewer import Viewer


def demo(
    duration=10.0,
    dt=1 / 240.0,
    force_scale: float = 1.0,
    cycle_freq: float = 1.5,
    stiffness: float = 6.0,
    damping: float = 1.2,
    fps: int = 60,
    random_pose: bool = False,
    seed: int | None = None,
):
    world = World()
    world.dt = dt
    genome = {"cycle_freq": cycle_freq, "stiffness": stiffness, "damping": damping}
    h = HumanoidCreature(
        genome,
        world,
        base_x=0.0,
        force_scale=force_scale,
        cycle_freq=cycle_freq,
        stiffness=stiffness,
        damping=damping,
    )

    joint_names = [j["name"] for j in h.joints]
    rest_angles = [p["rest_angle"] for p in h.joint_params]

    stand = {
        "duration": 1.5,
        "targets": {"hip_left": 0.0, "hip_right": 0.0, "shoulder_left": 0.0, "shoulder_right": 0.0},
    }
    crouch = {
        "duration": 1.5,
        "targets": {"hip_left": -25.0, "hip_right": -25.0},
    }
    reach = {
        "duration": 1.5,
        "targets": {"shoulder_left": 25.0, "shoulder_right": 25.0},
    }

    if random_pose:
        import random

        rng = random.Random(seed)
        pose = {}
        for name in joint_names:
            pose[name] = rng.uniform(-40.0, 40.0)
        pose_sequence = [{"duration": duration, "targets": pose}]
    else:
        pose_sequence = [stand, crouch, reach]

    total_duration = sum(p["duration"] for p in pose_sequence)

    viewer = Viewer(world, creature=h)
    t = 0.0

    def apply_pose(time_now: float):
        # find pose index and blend to next
        if total_duration <= 0:
            return
        tmod = time_now % total_duration
        acc = 0.0
        idx = 0
        for i, pose in enumerate(pose_sequence):
            acc += pose["duration"]
            if tmod <= acc:
                idx = i
                break
        next_idx = (idx + 1) % len(pose_sequence)
        pose = pose_sequence[idx]
        next_pose = pose_sequence[next_idx]
        dur = pose["duration"]
        prev_end = acc - dur
        alpha = 0.0 if dur <= 0 else (tmod - prev_end) / dur

        for j, name in enumerate(joint_names):
            a0 = pose["targets"].get(name, 0.0)
            a1 = next_pose["targets"].get(name, 0.0)
            a_deg = a0 * (1.0 - alpha) + a1 * alpha
            h.joint_params[j]["target_angle"] = rest_angles[j] + math.radians(a_deg)
            h.joint_params[j]["amp"] = 0.0

    def step():
        nonlocal t
        apply_pose(t)
        h.step_controller(t, dt)
        h.step_actuators(t, dt)
        world.step(dt)
        t += dt

    viewer.run_loop(step, fps=fps)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--force-scale", type=float, default=1.0, help="Scale multiplier")
    p.add_argument("--cycle-freq", type=float, default=1.5, help="Cycle frequency (Hz)")
    p.add_argument("--stiffness", type=float, default=6.0, help="Joint stiffness")
    p.add_argument("--damping", type=float, default=1.2, help="Joint damping")
    p.add_argument("--fps", type=int, default=60, help="Frames per second")
    p.add_argument("--random-pose", dest="random_pose", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=1 / 240.0)

    args = p.parse_args()
    demo(
        duration=args.duration,
        dt=args.dt,
        force_scale=args.force_scale,
        cycle_freq=args.cycle_freq,
        stiffness=args.stiffness,
        damping=args.damping,
        fps=args.fps,
        random_pose=args.random_pose,
        seed=args.seed,
    )
