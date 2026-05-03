"""Visualize one controlled triad: three particles and two rigid links."""

import argparse
from collections.abc import Callable
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics.engine import DistanceConstraint, Particle, World
from render.viewer import Viewer

MoleculeBuilder = Callable[[World, float], tuple[Particle, ...]]
ClampFn = Callable[[float, float, float], float]
ExtraControls = Callable[[dict[str, float]], list[dict[str, object]]]
ExtraControlSetter = Callable[[dict[str, float], str, float, ClampFn], bool]
ExtraStep = Callable[
    [World, tuple[Particle, ...], dict[str, float], float], dict[str, float] | None
]


def build_triad(world: World, start_y: float) -> tuple[Particle, Particle, Particle]:
    left = world.add_particle(Particle(-0.25, start_y, 0.0, 0.0, 1.0))
    joint = world.add_particle(Particle(0.0, start_y + 0.4, 0.0, 0.0, 1.0))
    right = world.add_particle(Particle(0.25, start_y, 0.0, 0.0, 1.0))

    world.add_constraint(
        DistanceConstraint(left, joint, math.hypot(joint.x - left.x, joint.y - left.y))
    )
    world.add_constraint(
        DistanceConstraint(joint, right, math.hypot(right.x - joint.x, right.y - joint.y))
    )
    return left, joint, right


def run_controlled_molecule(
    build_molecule: MoleculeBuilder,
    base_angle_deg: float = 60.0,
    amp_deg: float = 25.0,
    cycle_freq: float = 0.6,
    stiffness: float = 360.0,
    damping: float = 0.7,
    max_torque: float = 220.0,
    gravity: float = 9.81,
    start_y: float = 0.6,
    fps: int = 60,
    time_scale: float = 1.0,
    max_time: float | None = None,
    extra_control_state: dict[str, float] | None = None,
    extra_controls: ExtraControls | None = None,
    extra_control_setter: ExtraControlSetter | None = None,
    extra_step: ExtraStep | None = None,
) -> None:
    world = World(gravity=(0.0, -abs(gravity)))
    world.dt = 1 / 240.0
    particles = build_molecule(world, start_y)
    left, joint, right = particles[0], particles[1], particles[2]

    control_state = {
        "cycle_freq": cycle_freq,
        "min_angle_deg": base_angle_deg - abs(amp_deg),
        "max_angle_deg": base_angle_deg + abs(amp_deg),
        "stiffness": stiffness,
        "damping": damping,
        "max_torque": max_torque,
    }
    if extra_control_state is not None:
        control_state.update(extra_control_state)
    t = 0.0
    phase = 0.0
    sim_accumulator = 0.0
    frame_dt = 1.0 / max(1, fps)
    clock_state = {
        "time": 0.0,
        "phase": 0.0,
        "target_angle_deg": base_angle_deg,
        "current_angle_deg": math.degrees(world.joint_angle(joint, left, right)),
        "error_deg": 0.0,
        "torque": 0.0,
        "cycle_freq": cycle_freq,
        "min_angle_deg": control_state["min_angle_deg"],
        "max_angle_deg": control_state["max_angle_deg"],
        "stiffness": control_state["stiffness"],
        "damping": control_state["damping"],
        "max_torque": control_state["max_torque"],
    }
    if extra_control_state is not None:
        clock_state.update(extra_control_state)

    def clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    def set_control(key: str, value: float) -> None:
        if key == "cycle_freq":
            control_state[key] = clamp(value, 0.05, 8.0)
        elif key == "min_angle_deg":
            max_angle = control_state["max_angle_deg"]
            control_state[key] = clamp(value, 10.0, max_angle - 1.0)
        elif key == "max_angle_deg":
            min_angle = control_state["min_angle_deg"]
            control_state[key] = clamp(value, min_angle + 1.0, 170.0)
        elif key == "stiffness":
            control_state[key] = clamp(value, 0.0, 400.0)
        elif key == "damping":
            control_state[key] = clamp(value, 0.0, 90.0)
        elif key == "max_torque":
            control_state[key] = clamp(value, 1.0, 1200.0)
        elif extra_control_setter is not None:
            extra_control_setter(control_state, key, value, clamp)

    def controls() -> list[dict[str, object]]:
        specs = [
            {
                "key": "cycle_freq",
                "label": "clock",
                "value": control_state["cycle_freq"],
                "min": 0.05,
                "max": 8.0,
                "unit": "Hz",
            },
            {
                "key": "min_angle_deg",
                "label": "min",
                "value": control_state["min_angle_deg"],
                "min": 10.0,
                "max": 170.0,
                "unit": "deg",
            },
            {
                "key": "max_angle_deg",
                "label": "max",
                "value": control_state["max_angle_deg"],
                "min": 10.0,
                "max": 170.0,
                "unit": "deg",
            },
            {
                "key": "stiffness",
                "label": "stiff",
                "value": control_state["stiffness"],
                "min": 0.0,
                "max": 400.0,
                "unit": "",
            },
            {
                "key": "damping",
                "label": "damp",
                "value": control_state["damping"],
                "min": 0.0,
                "max": 90.0,
                "unit": "",
            },
            {
                "key": "max_torque",
                "label": "torque",
                "value": control_state["max_torque"],
                "min": 1.0,
                "max": 1200.0,
                "unit": "",
            },
        ]
        if extra_controls is not None:
            specs.extend(extra_controls(control_state))
        return specs

    def advance_simulation(dt: float) -> None:
        nonlocal t, phase
        cycle = control_state["cycle_freq"]
        min_angle = control_state["min_angle_deg"]
        max_angle = control_state["max_angle_deg"]
        current_stiffness = control_state["stiffness"]
        current_damping = control_state["damping"]
        current_max_torque = control_state["max_torque"]
        target_center = math.radians((min_angle + max_angle) * 0.5)
        amplitude = math.radians((max_angle - min_angle) * 0.5)
        phase = (phase + cycle * dt) % 1.0
        target = target_center + amplitude * math.sin(2.0 * math.pi * phase)
        torque, error = world.apply_joint_angle_pd(
            joint,
            left,
            right,
            target,
            current_stiffness,
            current_damping,
            dt,
            current_max_torque,
        )
        extra_state = {}
        if extra_step is not None:
            extra_state = extra_step(world, particles, control_state, dt) or {}
        world.step(dt)
        t += dt
        state = {
            "time": t,
            "phase": phase,
            "target_angle_deg": math.degrees(target),
            "current_angle_deg": math.degrees(world.joint_angle(joint, left, right)),
            "error_deg": math.degrees(error),
            "torque": torque,
            "cycle_freq": cycle,
            "min_angle_deg": min_angle,
            "max_angle_deg": max_angle,
            "stiffness": current_stiffness,
            "damping": current_damping,
            "max_torque": current_max_torque,
        }
        state.update(extra_state)
        clock_state.update(state)

    def step() -> None:
        nonlocal sim_accumulator
        sim_accumulator += frame_dt * time_scale
        max_substeps = max(1, int(math.ceil((frame_dt * max(1.0, time_scale)) / world.dt)) + 2)
        substeps = 0

        while sim_accumulator >= world.dt and substeps < max_substeps:
            advance_simulation(world.dt)
            sim_accumulator -= world.dt
            substeps += 1

        if substeps == max_substeps:
            sim_accumulator = min(sim_accumulator, world.dt)

    Viewer(
        world,
        clock_provider=lambda: clock_state,
        controls_provider=controls,
        control_setter=set_control,
    ).run_loop(
        step, fps=fps, max_time=max_time
    )


def demo(
    base_angle_deg: float = 60.0,
    amp_deg: float = 25.0,
    cycle_freq: float = 0.6,
    stiffness: float = 360.0,
    damping: float = 0.7,
    max_torque: float = 220.0,
    gravity: float = 9.81,
    start_y: float = 0.6,
    fps: int = 60,
    time_scale: float = 1.0,
    max_time: float | None = None,
) -> None:
    run_controlled_molecule(
        build_triad,
        base_angle_deg=base_angle_deg,
        amp_deg=amp_deg,
        cycle_freq=cycle_freq,
        stiffness=stiffness,
        damping=damping,
        max_torque=max_torque,
        gravity=gravity,
        start_y=start_y,
        fps=fps,
        time_scale=time_scale,
        max_time=max_time,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-angle", dest="base_angle_deg", type=float, default=60.0)
    parser.add_argument("--amp", dest="amp_deg", type=float, default=25.0)
    parser.add_argument("--cycle-freq", type=float, default=0.6)
    parser.add_argument("--stiffness", type=float, default=360.0)
    parser.add_argument("--damping", type=float, default=0.7)
    parser.add_argument("--max-torque", type=float, default=220.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--start-y", type=float, default=0.6)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-time", type=float, default=None)
    args = parser.parse_args()
    demo(**vars(args))


if __name__ == "__main__":
    main()
