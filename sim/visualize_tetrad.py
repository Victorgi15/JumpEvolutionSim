"""Visualize one controlled tetrad: a triad plus one central branch."""

import argparse
from dataclasses import dataclass
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics.engine import DistanceConstraint, Particle, World
from render.viewer import Viewer
from sim.visualize_triad import build_triad


@dataclass
class Position:
    pivot_targets_deg: list[float]


@dataclass
class TetradGenome:
    positions: list[Position]
    clock_hz: float


def normalize_angle_deg(angle: float) -> float:
    return angle % 360.0


def signed_angle_deg(angle: float) -> float:
    normalized = normalize_angle_deg(angle)
    if normalized > 180.0:
        normalized -= 360.0
    return normalized


def shortest_delta_deg(start: float, end: float) -> float:
    return (end - start + 180.0) % 360.0 - 180.0


def build_tetrad(
    world: World, start_y: float, branch_length: float = 0.45
) -> tuple[Particle, Particle, Particle, Particle]:
    left, joint, right = build_triad(world, start_y)
    branch = world.add_particle(
        Particle(joint.x, joint.y + branch_length, 0.0, 0.0, 1.0)
    )
    world.add_constraint(
        DistanceConstraint(
            joint,
            branch,
            math.hypot(branch.x - joint.x, branch.y - joint.y),
        )
    )
    return left, joint, right, branch


def default_tetrad_genome(
    angle1_deg: float = 60.0,
    angle2_deg: float = 145.0,
    clock_hz: float = 0.5,
) -> TetradGenome:
    positions = [
        Position([angle1_deg, angle2_deg]),
        Position([angle1_deg + 35.0, angle2_deg - 35.0]),
        Position([angle1_deg - 25.0, angle2_deg + 45.0]),
    ]
    for position in positions:
        position.pivot_targets_deg[:] = [
            normalize_angle_deg(angle) for angle in position.pivot_targets_deg
        ]
    return TetradGenome(positions, clock_hz)


def evaluate_tetrad(
    genome: TetradGenome,
    duration: float = 10.0,
    stiffness: float = 360.0,
    damping: float = 0.7,
    max_torque: float = 220.0,
    gravity: float = 9.81,
    start_y: float = 0.6,
    branch_length: float = 0.45,
    score_points_per_meter: float = 100.0,
    score_energy_penalty: float = 0.05,
) -> dict[str, float]:
    world = World(gravity=(0.0, -abs(gravity)))
    world.dt = 1 / 240.0
    particles = build_tetrad(world, start_y, branch_length)
    left, joint, right, branch = particles
    positions = [
        Position([normalize_angle_deg(angle) for angle in position.pivot_targets_deg])
        for position in genome.positions
    ]
    command_targets_deg = positions[0].pivot_targets_deg.copy()
    max_target_rate_deg_s = 90.0
    playback_time = 0.0
    position_index = 0
    energy_pivot1_j = 0.0
    energy_pivot2_j = 0.0
    start_com_x = sum(p.x for p in particles) / len(particles)

    steps = max(1, int(duration / world.dt))
    for _ in range(steps):
        dt = world.dt
        playback_time += dt
        position_index = int(playback_time * genome.clock_hz) % len(positions)
        targets = positions[position_index].pivot_targets_deg

        max_step = max_target_rate_deg_s * dt
        for index, target in enumerate(targets):
            delta = shortest_delta_deg(command_targets_deg[index], target)
            if abs(delta) <= max_step:
                command_targets_deg[index] = normalize_angle_deg(target)
            else:
                command_targets_deg[index] = normalize_angle_deg(
                    command_targets_deg[index] + math.copysign(max_step, delta)
                )

        target1 = math.radians(signed_angle_deg(command_targets_deg[0]))
        target2 = math.radians(signed_angle_deg(command_targets_deg[1]))
        omega1_before = world.joint_angular_velocity(joint, left, right)
        omega2_before = world.joint_angular_velocity(joint, right, branch)
        torque1, _ = world.apply_joint_angle_pd(
            joint, left, right, target1, stiffness, damping, dt, max_torque
        )
        torque2, _ = world.apply_joint_angle_pd(
            joint, right, branch, target2, stiffness, damping, dt, max_torque
        )
        omega1_after = world.joint_angular_velocity(joint, left, right)
        omega2_after = world.joint_angular_velocity(joint, right, branch)
        power1 = abs(torque1 * 0.5 * (omega1_before + omega1_after))
        power2 = abs(torque2 * 0.5 * (omega2_before + omega2_after))
        energy_pivot1_j += power1 * dt
        energy_pivot2_j += power2 * dt
        world.step(dt)

    energy_total_j = energy_pivot1_j + energy_pivot2_j
    com_x = sum(p.x for p in particles) / len(particles)
    distance_x = com_x - start_com_x
    distance_points = distance_x * score_points_per_meter
    energy_penalty_points = energy_total_j * score_energy_penalty
    score = distance_points - energy_penalty_points
    return {
        "score": score,
        "distance_x_m": distance_x,
        "energy_total_j": energy_total_j,
        "energy_pivot1_j": energy_pivot1_j,
        "energy_pivot2_j": energy_pivot2_j,
        "score_distance_points": distance_points,
        "score_energy_penalty_points": energy_penalty_points,
        "clock_hz": genome.clock_hz,
    }


def demo(
    angle1_deg: float = 60.0,
    angle2_deg: float = 145.0,
    stiffness: float = 360.0,
    damping: float = 0.7,
    max_torque: float = 220.0,
    gravity: float = 9.81,
    start_y: float = 0.6,
    branch_length: float = 0.45,
    fps: int = 60,
    time_scale: float = 1.0,
    max_time: float | None = None,
    log: bool = False,
    log_interval: float = 0.25,
    score_points_per_meter: float = 100.0,
    score_energy_penalty: float = 0.05,
    genome: TetradGenome | None = None,
    autoplay: bool = False,
) -> None:
    world = World(gravity=(0.0, -abs(gravity)))
    world.dt = 1 / 240.0
    particles = build_tetrad(world, start_y, branch_length)
    left, joint, right, branch = particles
    start_com_x = sum(p.x for p in particles) / len(particles)

    if genome is None:
        genome = default_tetrad_genome(angle1_deg, angle2_deg)
    positions = [
        Position([normalize_angle_deg(angle) for angle in position.pivot_targets_deg])
        for position in genome.positions
    ]
    command_targets_deg = positions[0].pivot_targets_deg.copy()
    max_target_rate_deg_s = 90.0

    playback = {
        "running": autoplay,
        "time": 0.0,
        "position_index": 0,
        "clock_hz": genome.clock_hz,
    }
    energy = {
        "pivot1_j": 0.0,
        "pivot2_j": 0.0,
        "total_j": 0.0,
        "pivot1_w": 0.0,
        "pivot2_w": 0.0,
        "total_w": 0.0,
    }
    hud_state = {
        "mode": "pivot_angles",
        "time": 0.0,
        "position_index": 0,
        "position_count": len(positions),
        "playing": playback["running"],
        "clock_hz": playback["clock_hz"],
        "angle1_target_deg": positions[0].pivot_targets_deg[0],
        "angle1_applied_setpoint_deg": command_targets_deg[0],
        "angle1_applied_target_deg": signed_angle_deg(command_targets_deg[0]),
        "angle1_current_deg": math.degrees(world.joint_angle(joint, left, right)),
        "angle1_current_unsigned_deg": normalize_angle_deg(
            math.degrees(world.joint_angle(joint, left, right))
        ),
        "angle1_error_deg": 0.0,
        "angle1_torque": 0.0,
        "angle2_target_deg": positions[0].pivot_targets_deg[1],
        "angle2_applied_setpoint_deg": command_targets_deg[1],
        "angle2_applied_target_deg": signed_angle_deg(command_targets_deg[1]),
        "angle2_current_deg": math.degrees(world.joint_angle(joint, right, branch)),
        "angle2_current_unsigned_deg": normalize_angle_deg(
            math.degrees(world.joint_angle(joint, right, branch))
        ),
        "angle2_error_deg": 0.0,
        "angle2_torque": 0.0,
        "max_torque": max_torque,
        "energy_total_j": 0.0,
        "energy_pivot1_j": 0.0,
        "energy_pivot2_j": 0.0,
        "power_total_w": 0.0,
        "power_pivot1_w": 0.0,
        "power_pivot2_w": 0.0,
        "distance_x_m": 0.0,
        "score": 0.0,
        "score_distance_points": 0.0,
        "score_energy_penalty_points": 0.0,
    }
    sim_accumulator = 0.0
    frame_dt = 1.0 / max(1, fps)
    next_log_time = 0.0

    def clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    def active_position() -> Position:
        return positions[int(playback["position_index"])]

    def active_targets() -> list[float]:
        if playback["running"]:
            index = int(playback["time"] * playback["clock_hz"]) % len(positions)
            playback["position_index"] = index
        return active_position().pivot_targets_deg

    def set_control(key: str, value: float) -> None:
        if key == "clock_hz":
            playback["clock_hz"] = clamp(value, 0.1, 2.0)
            return

        if key == "play_positions":
            playback["running"] = not playback["running"]
            playback["time"] = 0.0
            playback["position_index"] = 0
            return

        parts = key.split("_")
        if (
            len(parts) != 2
            or not parts[0].startswith("p")
            or not parts[1].startswith("a")
        ):
            return

        position_index = int(parts[0][1:]) - 1
        pivot_index = int(parts[1][1:]) - 1
        if 0 <= position_index < len(positions) and 0 <= pivot_index < 2:
            target = normalize_angle_deg(clamp(value, 0.0, 360.0))
            positions[position_index].pivot_targets_deg[pivot_index] = target
            if not playback["running"] and position_index == int(
                playback["position_index"]
            ):
                command_targets_deg[pivot_index] = target

    def controls() -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        for position_index, position in enumerate(positions, start=1):
            for pivot_index, value in enumerate(position.pivot_targets_deg, start=1):
                specs.append(
                    {
                        "key": f"p{position_index}_a{pivot_index}",
                        "label": f"P{position_index} a{pivot_index}",
                        "value": value,
                        "min": 0.0,
                        "max": 360.0,
                        "unit": "deg",
                    }
                )
        specs.append(
            {
                "key": "clock_hz",
                "label": "clock",
                "value": playback["clock_hz"],
                "min": 0.1,
                "max": 2.0,
                "unit": "Hz",
            }
        )
        specs.append(
            {
                "type": "button",
                "key": "play_positions",
                "label": "stop" if playback["running"] else "start",
                "active": playback["running"],
            }
        )
        return specs

    def advance_simulation(dt: float) -> None:
        nonlocal next_log_time
        if playback["running"]:
            playback["time"] += dt

        targets = active_targets()
        if playback["running"]:
            max_step = max_target_rate_deg_s * dt
            for index, target in enumerate(targets):
                delta = shortest_delta_deg(command_targets_deg[index], target)
                if abs(delta) <= max_step:
                    command_targets_deg[index] = normalize_angle_deg(target)
                else:
                    command_targets_deg[index] = normalize_angle_deg(
                        command_targets_deg[index] + math.copysign(max_step, delta)
                    )

        applied_target1_deg = signed_angle_deg(command_targets_deg[0])
        applied_target2_deg = signed_angle_deg(command_targets_deg[1])
        target1 = math.radians(applied_target1_deg)
        target2 = math.radians(applied_target2_deg)
        omega1_before = world.joint_angular_velocity(joint, left, right)
        omega2_before = world.joint_angular_velocity(joint, right, branch)
        torque1, error1 = world.apply_joint_angle_pd(
            joint, left, right, target1, stiffness, damping, dt, max_torque
        )
        torque2, error2 = world.apply_joint_angle_pd(
            joint, right, branch, target2, stiffness, damping, dt, max_torque
        )
        omega1_after = world.joint_angular_velocity(joint, left, right)
        omega2_after = world.joint_angular_velocity(joint, right, branch)
        omega1 = 0.5 * (omega1_before + omega1_after)
        omega2 = 0.5 * (omega2_before + omega2_after)
        # Count absolute actuator work so aggressive oscillations are penalized
        # even when they alternately inject and brake mechanical energy.
        power1 = abs(torque1 * omega1)
        power2 = abs(torque2 * omega2)
        energy["pivot1_w"] = power1
        energy["pivot2_w"] = power2
        energy["total_w"] = power1 + power2
        energy["pivot1_j"] += power1 * dt
        energy["pivot2_j"] += power2 * dt
        energy["total_j"] = energy["pivot1_j"] + energy["pivot2_j"]
        world.step(dt)
        com_x = sum(p.x for p in particles) / len(particles)
        distance_x = com_x - start_com_x
        distance_points = distance_x * score_points_per_meter
        energy_penalty_points = energy["total_j"] * score_energy_penalty
        score = distance_points - energy_penalty_points
        hud_state.update(
            {
                "time": hud_state["time"] + dt,
                "position_index": int(playback["position_index"]),
                "position_count": len(positions),
                "playing": playback["running"],
                "clock_hz": playback["clock_hz"],
                "angle1_target_deg": targets[0],
                "angle1_applied_setpoint_deg": command_targets_deg[0],
                "angle1_applied_target_deg": applied_target1_deg,
                "angle1_current_deg": math.degrees(
                    world.joint_angle(joint, left, right)
                ),
                "angle1_current_unsigned_deg": normalize_angle_deg(
                    math.degrees(world.joint_angle(joint, left, right))
                ),
                "angle1_error_deg": math.degrees(error1),
                "angle1_torque": torque1,
                "angle2_target_deg": targets[1],
                "angle2_applied_setpoint_deg": command_targets_deg[1],
                "angle2_applied_target_deg": applied_target2_deg,
                "angle2_current_deg": math.degrees(
                    world.joint_angle(joint, right, branch)
                ),
                "angle2_current_unsigned_deg": normalize_angle_deg(
                    math.degrees(world.joint_angle(joint, right, branch))
                ),
                "angle2_error_deg": math.degrees(error2),
                "angle2_torque": torque2,
                "energy_total_j": energy["total_j"],
                "energy_pivot1_j": energy["pivot1_j"],
                "energy_pivot2_j": energy["pivot2_j"],
                "power_total_w": energy["total_w"],
                "power_pivot1_w": energy["pivot1_w"],
                "power_pivot2_w": energy["pivot2_w"],
                "distance_x_m": distance_x,
                "score": score,
                "score_distance_points": distance_points,
                "score_energy_penalty_points": energy_penalty_points,
            }
        )
        if log and hud_state["time"] >= next_log_time:
            particles_state = [left, joint, right, branch]
            com_y = sum(p.y for p in particles_state) / len(particles_state)
            max_speed = max(math.hypot(p.vx, p.vy) for p in particles_state)
            print(
                "[tetrad] "
                f"t={hud_state['time']:.2f}s "
                f"pos={int(hud_state['position_index']) + 1}/{len(positions)} "
                f"playing={hud_state['playing']} "
                f"clock={playback['clock_hz']:.2f}Hz "
                f"set=({targets[0]:.1f},{targets[1]:.1f}) "
                f"applied_set=({command_targets_deg[0]:.1f},"
                f"{command_targets_deg[1]:.1f}) "
                f"cmd=({applied_target1_deg:.1f},{applied_target2_deg:.1f}) "
                f"measured=({hud_state['angle1_current_unsigned_deg']:.1f},"
                f"{hud_state['angle2_current_unsigned_deg']:.1f}) "
                f"signed=({hud_state['angle1_current_deg']:.1f},"
                f"{hud_state['angle2_current_deg']:.1f}) "
                f"err=({hud_state['angle1_error_deg']:.1f},"
                f"{hud_state['angle2_error_deg']:.1f}) "
                f"tau=({torque1:.1f},{torque2:.1f}) "
                f"power=({power1:.2f},{power2:.2f})W "
                f"energy={energy['total_j']:.2f}J "
                f"dx={distance_x:.3f}m "
                f"score={score:.2f} "
                f"com=({com_x:.2f},{com_y:.2f}) "
                f"vmax={max_speed:.2f}"
            )
            next_log_time += log_interval

    def step() -> None:
        nonlocal sim_accumulator
        sim_accumulator += frame_dt * time_scale
        max_substeps = max(
            1, int(math.ceil((frame_dt * max(1.0, time_scale)) / world.dt)) + 2
        )
        substeps = 0

        while sim_accumulator >= world.dt and substeps < max_substeps:
            advance_simulation(world.dt)
            sim_accumulator -= world.dt
            substeps += 1

        if substeps == max_substeps:
            sim_accumulator = min(sim_accumulator, world.dt)

    Viewer(
        world,
        clock_provider=lambda: hud_state,
        controls_provider=controls,
        control_setter=set_control,
    ).run_loop(step, fps=fps, max_time=max_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle1", dest="angle1_deg", type=float, default=60.0)
    parser.add_argument("--angle2", dest="angle2_deg", type=float, default=145.0)
    parser.add_argument("--stiffness", type=float, default=360.0)
    parser.add_argument("--damping", type=float, default=0.7)
    parser.add_argument("--max-torque", type=float, default=220.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--start-y", type=float, default=0.6)
    parser.add_argument("--branch-length", type=float, default=0.45)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log-interval", type=float, default=0.25)
    parser.add_argument("--score-points-per-meter", type=float, default=100.0)
    parser.add_argument("--score-energy-penalty", type=float, default=0.05)
    args = parser.parse_args()
    demo(**vars(args))


if __name__ == "__main__":
    main()
