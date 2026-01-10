"""Run many randomized humanoid simulations and visualize the top-k by max pelvis height."""

import sys

# ensure project root is on sys.path when running this script directly
sys.path.insert(0, ".")
import random
import argparse
import math
import json
from physics.engine import World
from creature.humanoid import HumanoidCreature

try:
    import pygame
except Exception:
    pygame = None


def random_joint_params(rng, num_joints, amp_min, amp_max, cycle_freq, stiffness, damping):
    params = []
    for _ in range(num_joints):
        params.append(
            {
                "amp": rng.uniform(amp_min, amp_max),
                "phase": rng.uniform(0.0, 2.0 * math.pi),
                "freq": cycle_freq,
                "stiffness": stiffness,
                "damping": damping,
            }
        )
    return params


def run_sim(duration, dt, force_scale, cycle_freq, joint_params):
    w = World()
    w.dt = dt
    genome = {"joint_params": joint_params, "cycle_freq": cycle_freq}
    h = HumanoidCreature(
        genome,
        w,
        base_x=0.0,
        force_scale=force_scale,
        cycle_freq=cycle_freq,
    )
    steps = int(duration / dt)
    max_pelvis = -1e9
    for i in range(steps):
        t = i * dt
        h.step_controller(t, dt)
        h.step_actuators(t, dt)
        w.step(dt)
        max_pelvis = max(max_pelvis, h.particles[0].y)
    return max_pelvis


def record_sim(duration, dt, params):
    force_scale = params["force_scale"]
    cycle_freq = params["cycle_freq"]
    joint_params = params["joint_params"]
    w = World()
    w.dt = dt
    genome = {"joint_params": joint_params, "cycle_freq": cycle_freq}
    h = HumanoidCreature(
        genome,
        w,
        base_x=0.0,
        force_scale=force_scale,
        cycle_freq=cycle_freq,
    )
    steps = int(duration / dt)
    frames = []
    for i in range(steps):
        t = i * dt
        h.step_controller(t, dt)
        h.step_actuators(t, dt)
        w.step(dt)
        pts = [(p.x, p.y, p.inv_mass) for p in w.particles]
        frames.append({"particles": pts})
    return frames


def visualize_three(frames_list, dt, scale=200.0):
    if pygame is None:
        print("Pygame not available: cannot visualize")
        return
    pygame.init()
    width = 1200
    height = 420
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    cols = len(frames_list)
    cell_w = width // cols
    cell_h = height
    running = True
    steps = min(len(f) for f in frames_list)
    step = 0
    finished = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    step = 0
                    finished = False
                elif event.key == pygame.K_q:
                    running = False
        screen.fill((30, 30, 30))
        if not finished and step < steps:
            for i, frames in enumerate(frames_list):
                frame = frames[step]
                vx = i * cell_w
                vy = 0
                gy = vy + cell_h - 40
                pygame.draw.rect(screen, (50, 120, 50), (vx, gy, cell_w, 40))
                pts = frame["particles"]
                edges = [
                    (0, 1),
                    (0, 2),
                    (1, 3),
                    (2, 4),
                    (0, 5),
                    (5, 6),
                    (5, 7),
                    (6, 8),
                    (7, 9),
                ]
                for a, b in edges:
                    x1 = vx + int(cell_w / 2 + pts[a][0] * scale)
                    y1 = vy + int(cell_h - 80 - pts[a][1] * scale)
                    x2 = vx + int(cell_w / 2 + pts[b][0] * scale)
                    y2 = vy + int(cell_h - 80 - pts[b][1] * scale)
                    pygame.draw.line(screen, (200, 200, 200), (x1, y1), (x2, y2), 3)
                for p in pts:
                    x = vx + int(cell_w / 2 + p[0] * scale)
                    y = vy + int(cell_h - 80 - p[1] * scale)
                    col = (200, 50, 50) if p[2] != 0 else (100, 100, 100)
                    pygame.draw.circle(screen, col, (x, y), 6)
            pygame.display.flip()
            clock.tick(1.0 / dt if dt > 0 else 60)
            step += 1
            if step >= steps:
                finished = True
        else:
            clock.tick(30)
    pygame.quit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=1 / 240.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--force-scale", type=float, default=1.0)
    p.add_argument("--cycle-freq", type=float, default=1.5)
    p.add_argument("--amp-min", type=float, default=0.1)
    p.add_argument("--amp-max", type=float, default=0.6)
    p.add_argument("--stiffness", type=float, default=6.0)
    p.add_argument("--damping", type=float, default=1.2)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--replay-file", type=str, default=None)
    p.add_argument("--replay-index", type=int, default=0)

    args = p.parse_args()

    if args.replay_file is not None:
        with open(args.replay_file, "r") as f:
            saved = json.load(f)
        if args.replay_index < 0 or args.replay_index >= len(saved):
            print("replay-index out of range")
            return
        r = saved[args.replay_index]
        frames = record_sim(args.duration, args.dt, r)
        visualize_three([frames], args.dt)
        return

    rng = random.Random(args.seed)
    results = []
    for i in range(args.n):
        joint_params = random_joint_params(
            rng,
            num_joints=4,
            amp_min=args.amp_min,
            amp_max=args.amp_max,
            cycle_freq=args.cycle_freq,
            stiffness=args.stiffness,
            damping=args.damping,
        )
        mx = run_sim(
            args.duration,
            args.dt,
            args.force_scale,
            args.cycle_freq,
            joint_params,
        )
        results.append(
            {
                "index": i,
                "max_pelvis": mx,
                "force_scale": args.force_scale,
                "cycle_freq": args.cycle_freq,
                "joint_params": joint_params,
            }
        )
        if not args.quiet and (i % max(1, args.n // 10) == 0):
            print(f"run {i}/{args.n} max_pelvis={mx:.3f}")

    results.sort(key=lambda r: r["max_pelvis"], reverse=True)
    topk = results[: args.topk]
    print("Top K results:")
    for j, r in enumerate(topk):
        print(j + 1, r["max_pelvis"], r["force_scale"], r["cycle_freq"])

    if args.save_json is not None:
        with open(args.save_json, "w") as f:
            json.dump(results, f)
        print(f"Saved results to {args.save_json}")

    frames_list = []
    for r in topk:
        frames = record_sim(args.duration, args.dt, r)
        frames_list.append(frames)
    visualize_three(frames_list, args.dt)


if __name__ == "__main__":
    main()
