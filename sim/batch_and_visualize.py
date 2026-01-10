"""Run many randomized humanoid simulations and visualize the top-k by max pelvis height.

Usage:
    python sim/batch_and_visualize.py --n 1000 --duration 1.0 --dt 1/240 --seed 0 --topk 3

This will run `n` simulations with random parameters (force_scale, cycle_freq, pose targets) and
select the top-k runs by maximum pelvis height. It then replays the top-k side-by-side in a single Pygame window.
"""

import sys
import random
import argparse
import math
import time
from physics.engine import World
from creature.humanoid import HumanoidCreature

try:
    import pygame
except Exception:
    pygame = None


def random_pose(rng, num_particles):
    # generate a single pose: dict of index->(x_rel, y_rel)
    pose = {}
    for i in range(num_particles):
        # with some probability leave it unspecified
        if rng.random() < 0.6:
            x = rng.uniform(-0.5, 0.5)
            y = rng.uniform(-0.2, 1.2)
            pose[i] = (x, y)
    return pose


def run_sim(seed, duration, dt, force_scale, cycle_freq, balance_assist, pose):
    w = World()
    w.dt = dt
    h = HumanoidCreature({}, w, base_x=0.0, force_scale=force_scale, cycle_freq=cycle_freq, balance_assist=balance_assist)
    # disable cyclic limb muscles to focus on pose pulls
    h.muscle_edges = []
    h.set_pose_sequence([{"duration": duration, "targets": pose}])
    steps = int(duration / dt)
    max_pelvis = -1e9
    # run
    for i in range(steps):
        t = i * dt
        h.step_controller(t, dt)
        h.step_actuators(t, dt)
        w.step(dt)
        max_pelvis = max(max_pelvis, h.particles[0].y)
    return max_pelvis


def record_sim(duration, dt, params):
    # re-run simulation and record frames (particle positions + activations)
    force_scale = params['force_scale']
    cycle_freq = params['cycle_freq']
    pose = params['pose']
    balance_assist = params.get('balance_assist', True)
    w = World(); w.dt = dt
    h = HumanoidCreature({}, w, base_x=0.0, force_scale=force_scale, cycle_freq=cycle_freq, balance_assist=balance_assist)
    h.muscle_edges = []
    h.set_pose_sequence([{"duration": duration, "targets": pose}])
    steps = int(duration / dt)
    frames = []
    for i in range(steps):
        t = i * dt
        h.step_controller(t, dt)
        h.step_actuators(t, dt)
        w.step(dt)
        # snapshot
        pts = [(p.x, p.y, p.inv_mass) for p in w.particles]
        acts = [ { 'p1': a['p1'], 'p2': a['p2'], 'activation': a['activation'], 'force': a['force']} for a in getattr(h, 'last_activations', []) ]
        targets = dict(getattr(h, 'current_targets', {}))
        frames.append({'particles': pts, 'activations': acts, 'targets': targets})
    return frames


def visualize_three(frames_list, dt, scale=200.0):
    if pygame is None:
        print('Pygame not available: cannot visualize')
        return
    pygame.init()
    width = 1200
    height = 420
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    n = len(frames_list)
    cols = n
    rows = 1
    cell_w = width // cols
    cell_h = height // rows
    running = True
    steps = min(len(f) for f in frames_list)
    step = 0
    while running and step < steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((30, 30, 30))
        for i, frames in enumerate(frames_list):
            frame = frames[step]
            # draw background and ground
            vx = i * cell_w
            vy = 0
            gy = vy + cell_h - 40
            pygame.draw.rect(screen, (50, 120, 50), (vx, gy, cell_w, 40))
            # draw constraints as lines between particle indices
            pts = frame['particles']
            # we know humanoid topology: edges are hardcoded ordering
            edges = [(0,1),(0,2),(1,3),(2,4),(0,5),(5,6),(5,7),(6,8),(7,9)]
            for a,b in edges:
                x1 = vx + int(cell_w/2 + pts[a][0]*scale)
                y1 = vy + int(cell_h - 80 - pts[a][1]*scale)
                x2 = vx + int(cell_w/2 + pts[b][0]*scale)
                y2 = vy + int(cell_h - 80 - pts[b][1]*scale)
                pygame.draw.line(screen, (200,200,200), (x1,y1),(x2,y2), 3)
            # draw particles
            for p in pts:
                x = vx + int(cell_w/2 + p[0]*scale)
                y = vy + int(cell_h - 80 - p[1]*scale)
                col = (200,50,50) if p[2] != 0 else (100,100,100)
                pygame.draw.circle(screen, col, (x,y), 6)
            # draw targets
            for idx,(tx,ty) in frame['targets'].items():
                sx = vx + int(cell_w/2 + tx*scale)
                sy = vy + int(cell_h - 80 - ty*scale)
                pygame.draw.circle(screen, (50,200,50), (sx,sy), 4)
        pygame.display.flip()
        clock.tick(1.0/dt if dt>0 else 60)
        step += 1
    pygame.quit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--duration', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=1/240.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    rng = random.Random(args.seed)
    results = []
    for i in range(args.n):
        fs = rng.uniform(0.5, 5.0)
        cf = rng.uniform(0.5, 3.0)
        pose = random_pose(rng, 10)
        mx = run_sim(args.seed + i, args.duration, args.dt, fs, cf, True, pose)
        results.append({'index': i, 'max_pelvis': mx, 'force_scale': fs, 'cycle_freq': cf, 'pose': pose})
        if not args.quiet and (i % max(1, args.n//10) == 0):
            print(f'run {i}/{args.n} max_pelvis={mx:.3f} fs={fs:.2f} cf={cf:.2f}')
    # pick topk
    results.sort(key=lambda r: r['max_pelvis'], reverse=True)
    topk = results[:args.topk]
    print('Top K results:')
    for j, r in enumerate(topk):
        print(j+1, r['max_pelvis'], r['force_scale'], r['cycle_freq'])
    # re-run topk and record frames
    frames_list = []
    for r in topk:
        frames = record_sim(args.duration, args.dt, r)
        frames_list.append(frames)
    # visualize side-by-side
    visualize_three(frames_list, args.dt)

if __name__ == '__main__':
    main()
