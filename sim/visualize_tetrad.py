"""Visualize one controlled tetrad: a triad plus one central branch."""

import argparse
from dataclasses import dataclass, field
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
    branch_attachments: list[int] = field(default_factory=list)


def normalize_angle_deg(angle: float) -> float:
    return angle % 360.0


def signed_angle_deg(angle: float) -> float:
    normalized = normalize_angle_deg(angle)
    if normalized > 180.0:
        normalized -= 360.0
    return normalized


def shortest_delta_deg(start: float, end: float) -> float:
    return (end - start + 180.0) % 360.0 - 180.0


def playback_position_index(playback_time: float, clock_hz: float, position_count: int) -> int:
    return int(playback_time * clock_hz * position_count) % position_count


def build_tetrad(
    world: World,
    start_y: float,
    genome: TetradGenome | None = None,
    branch_length: float = 0.45,
) -> list[Particle]:
    if genome is None:
        genome = default_tetrad_genome()
    num_angles = len(genome.positions[0].pivot_targets_deg)
    branch_attachments = normalized_branch_attachments(genome)
    left = world.add_particle(
        Particle(-branch_length, start_y, 0.0, 0.0, 1.0)
    )
    joint = world.add_particle(Particle(0.0, start_y, 0.0, 0.0, 1.0))
    right = world.add_particle(
        Particle(branch_length, start_y, 0.0, 0.0, 1.0)
    )
    particles = [left, joint, right]
    world.add_constraint(
        DistanceConstraint(left, joint, branch_length)
    )
    world.add_constraint(
        DistanceConstraint(joint, right, branch_length)
    )
    for attachment_index in branch_attachments:
        anchor = particles[attachment_index]
        new_p = world.add_particle(
            Particle(anchor.x, anchor.y + branch_length, 0.0, 0.0, 1.0)
        )
        world.add_constraint(
            DistanceConstraint(anchor, new_p, branch_length)
        )
        particles.append(new_p)
    return particles


def normalized_branch_attachments(genome: TetradGenome) -> list[int]:
    num_angles = len(genome.positions[0].pivot_targets_deg)
    attachments = []
    for branch_index in range(max(0, num_angles - 1)):
        particle_count = 3 + branch_index
        if branch_index < len(genome.branch_attachments):
            attachment = genome.branch_attachments[branch_index]
        elif branch_index == 0:
            attachment = 1
        else:
            attachment = particle_count - 1
        attachments.append(max(0, min(particle_count - 1, int(attachment))))
    return attachments


def reference_particle_for_attachment(
    attachment_index: int, parents: dict[int, int]
) -> int:
    if attachment_index == 1:
        return 2
    return parents.get(attachment_index, 1)


def build_joint_pairs(genome: TetradGenome) -> list[tuple[int, int, int]]:
    num_angles = len(genome.positions[0].pivot_targets_deg)
    pairs = [(1, 0, 2)]
    parents = {0: 1, 2: 1}
    for branch_index, attachment_index in enumerate(normalized_branch_attachments(genome)):
        new_particle_index = 3 + branch_index
        reference_index = reference_particle_for_attachment(attachment_index, parents)
        pairs.append((attachment_index, reference_index, new_particle_index))
        parents[new_particle_index] = attachment_index
    return pairs


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
    return TetradGenome(positions, clock_hz, [1])


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
    score_airborne_penalty: float = 80.0,
) -> dict[str, float]:
    world = World(gravity=(0.0, -abs(gravity)))
    world.dt = 1 / 240.0
    particles = build_tetrad(world, start_y, genome, branch_length)
    num_angles = len(genome.positions[0].pivot_targets_deg)
    positions = [
        Position([normalize_angle_deg(angle) for angle in position.pivot_targets_deg])
        for position in genome.positions
    ]
    command_targets_deg = positions[0].pivot_targets_deg.copy()
    joint_pairs = build_joint_pairs(genome)
    max_target_rate_deg_s = 90.0
    playback_time = 0.0
    position_index = 0
    energy_joints = [0.0] * num_angles
    airborne_time = 0.0
    ground_epsilon = 0.03

    start_com_x = sum(p.x for p in particles) / len(particles)

    steps = max(1, int(duration / world.dt))
    for _ in range(steps):
        dt = world.dt
        playback_time += dt
        position_index = playback_position_index(
            playback_time, genome.clock_hz, len(positions)
        )
        targets = positions[position_index].pivot_targets_deg

        max_step = max_target_rate_deg_s * dt
        for idx in range(num_angles):
            delta = shortest_delta_deg(command_targets_deg[idx], targets[idx])
            if abs(delta) <= max_step:
                command_targets_deg[idx] = normalize_angle_deg(targets[idx])
            else:
                command_targets_deg[idx] = normalize_angle_deg(
                    command_targets_deg[idx] + math.copysign(max_step, delta)
                )

        for idx, (joint_idx, left_idx, right_idx) in enumerate(joint_pairs):
            p_joint = particles[joint_idx]
            p_left = particles[left_idx]
            p_right = particles[right_idx]
            target = math.radians(signed_angle_deg(command_targets_deg[idx]))
            omega_before = world.joint_angular_velocity(p_joint, p_left, p_right)
            torque, _ = world.apply_joint_angle_pd(
                p_joint, p_left, p_right, target, stiffness, damping, dt, max_torque
            )
            omega_after = world.joint_angular_velocity(p_joint, p_left, p_right)
            power = abs(torque * 0.5 * (omega_before + omega_after))
            energy_joints[idx] += power * dt
        world.step(dt)
        if not any(p.y <= ground_epsilon for p in particles):
            airborne_time += dt

    energy_total_j = sum(energy_joints)
    com_x = sum(p.x for p in particles) / len(particles)
    distance_x = com_x - start_com_x
    distance_points = distance_x * score_points_per_meter
    energy_penalty_points = energy_total_j * score_energy_penalty
    airborne_penalty_points = airborne_time * score_airborne_penalty
    score = distance_points - energy_penalty_points - airborne_penalty_points
    return {
        "score": score,
        "distance_x_m": distance_x,
        "airborne_time_s": airborne_time,
        "energy_total_j": energy_total_j,
        "energy_pivot1_j": energy_joints[0] if len(energy_joints) > 0 else 0.0,
        "energy_pivot2_j": energy_joints[1] if len(energy_joints) > 1 else 0.0,
        "score_distance_points": distance_points,
        "score_energy_penalty_points": energy_penalty_points,
        "score_airborne_penalty_points": airborne_penalty_points,
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
    score_airborne_penalty: float = 80.0,
    genome: TetradGenome | None = None,
    autoplay: bool = False,
) -> None:
    world = World(gravity=(0.0, -abs(gravity)))
    world.dt = 1 / 240.0
    if genome is None:
        genome = default_tetrad_genome(angle1_deg, angle2_deg)
    particles = build_tetrad(world, start_y, genome, branch_length)
    num_angles = len(genome.positions[0].pivot_targets_deg)
    joint_pairs = build_joint_pairs(genome)
    start_com_x = sum(p.x for p in particles) / len(particles)

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
        "joints": [0.0] * num_angles,
        "total_j": 0.0,
        "powers": [0.0] * num_angles,
        "total_w": 0.0,
    }
    airborne_time = 0.0
    ground_epsilon = 0.03
    hud_state = {
        "mode": "pivot_angles",
        "time": 0.0,
        "position_index": 0,
        "position_count": len(positions),
        "playing": playback["running"],
        "clock_hz": playback["clock_hz"],
        "angle1_target_deg": positions[0].pivot_targets_deg[0] if num_angles > 0 else 0.0,
        "angle1_applied_setpoint_deg": command_targets_deg[0] if num_angles > 0 else 0.0,
        "angle1_applied_target_deg": signed_angle_deg(command_targets_deg[0]) if num_angles > 0 else 0.0,
        "angle1_current_deg": math.degrees(world.joint_angle(particles[1], particles[0], particles[2])) if len(particles) > 2 else 0.0,
        "angle1_current_unsigned_deg": normalize_angle_deg(
            math.degrees(world.joint_angle(particles[1], particles[0], particles[2]))
        ) if len(particles) > 2 else 0.0,
        "angle1_error_deg": 0.0,
        "angle1_torque": 0.0,
        "angle2_target_deg": positions[0].pivot_targets_deg[1] if num_angles > 1 else 0.0,
        "angle2_applied_setpoint_deg": command_targets_deg[1] if num_angles > 1 else 0.0,
        "angle2_applied_target_deg": signed_angle_deg(command_targets_deg[1]) if num_angles > 1 else 0.0,
        "angle2_current_deg": math.degrees(world.joint_angle(particles[1], particles[2], particles[3])) if len(particles) > 3 else 0.0,
        "angle2_current_unsigned_deg": normalize_angle_deg(
            math.degrees(world.joint_angle(particles[1], particles[2], particles[3]))
        ) if len(particles) > 3 else 0.0,
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
        "airborne_time_s": 0.0,
        "score_airborne_penalty_points": 0.0,
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
            index = playback_position_index(
                playback["time"], playback["clock_hz"], len(positions)
            )
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
        if 0 <= position_index < len(positions) and 0 <= pivot_index < num_angles:
            target = normalize_angle_deg(clamp(value, 0.0, 360.0))
            positions[position_index].pivot_targets_deg[pivot_index] = target
            if not playback["running"] and position_index == int(
                playback["position_index"]
            ):
                command_targets_deg[pivot_index] = target

    def controls() -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        for position_index, position in enumerate(positions, start=1):
            for pivot_index in range(num_angles):
                specs.append(
                    {
                        "key": f"p{position_index}_a{pivot_index + 1}",
                        "label": f"P{position_index} a{pivot_index + 1}",
                        "value": position.pivot_targets_deg[pivot_index],
                        "min": 0.0,
                        "max": 360.0,
                        "unit": "deg",
                    }
                )
        specs.append(
            {
                "key": "clock_hz",
                "label": "cycle",
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
        nonlocal next_log_time, airborne_time
        if playback["running"]:
            playback["time"] += dt

        targets = active_targets()
        if playback["running"]:
            max_step = max_target_rate_deg_s * dt
            for idx in range(num_angles):
                delta = shortest_delta_deg(command_targets_deg[idx], targets[idx])
                if abs(delta) <= max_step:
                    command_targets_deg[idx] = normalize_angle_deg(targets[idx])
                else:
                    command_targets_deg[idx] = normalize_angle_deg(
                        command_targets_deg[idx] + math.copysign(max_step, delta)
                    )

        errors = []
        torques = []
        for idx, (joint_idx, left_idx, right_idx) in enumerate(joint_pairs):
            p_joint = particles[joint_idx]
            p_left = particles[left_idx]
            p_right = particles[right_idx]
            target = math.radians(signed_angle_deg(command_targets_deg[idx]))
            omega_before = world.joint_angular_velocity(p_joint, p_left, p_right)
            torque, error = world.apply_joint_angle_pd(
                p_joint, p_left, p_right, target, stiffness, damping, dt, max_torque
            )
            omega_after = world.joint_angular_velocity(p_joint, p_left, p_right)
            omega = 0.5 * (omega_before + omega_after)
            power = abs(torque * omega)
            energy["powers"][idx] = power
            energy["joints"][idx] += power * dt
            errors.append(error)
            torques.append(torque)
        energy["total_j"] = sum(energy["joints"])
        energy["total_w"] = sum(energy["powers"])
        world.step(dt)
        if not any(p.y <= ground_epsilon for p in particles):
            airborne_time += dt
        com_x = sum(p.x for p in particles) / len(particles)
        distance_x = com_x - start_com_x
        distance_points = distance_x * score_points_per_meter
        energy_penalty_points = energy["total_j"] * score_energy_penalty
        airborne_penalty_points = airborne_time * score_airborne_penalty
        score = distance_points - energy_penalty_points - airborne_penalty_points
        hud_state.update(
            {
                "time": hud_state["time"] + dt,
                "position_index": int(playback["position_index"]),
                "position_count": len(positions),
                "playing": playback["running"],
                "clock_hz": playback["clock_hz"],
                "angle1_target_deg": targets[0] if num_angles > 0 else 0.0,
                "angle1_applied_setpoint_deg": command_targets_deg[0] if num_angles > 0 else 0.0,
                "angle1_applied_target_deg": signed_angle_deg(command_targets_deg[0]) if num_angles > 0 else 0.0,
                "angle1_current_deg": math.degrees(world.joint_angle(particles[1], particles[0], particles[2])) if len(particles) > 2 else 0.0,
                "angle1_current_unsigned_deg": normalize_angle_deg(
                    math.degrees(world.joint_angle(particles[1], particles[0], particles[2]))
                ) if len(particles) > 2 else 0.0,
                "angle1_error_deg": math.degrees(errors[0]) if len(errors) > 0 else 0.0,
                "angle1_torque": torques[0] if len(torques) > 0 else 0.0,
                "angle2_target_deg": targets[1] if num_angles > 1 else 0.0,
                "angle2_applied_setpoint_deg": command_targets_deg[1] if num_angles > 1 else 0.0,
                "angle2_applied_target_deg": signed_angle_deg(command_targets_deg[1]) if num_angles > 1 else 0.0,
                "angle2_current_deg": math.degrees(world.joint_angle(particles[1], particles[2], particles[3])) if len(particles) > 3 else 0.0,
                "angle2_current_unsigned_deg": normalize_angle_deg(
                    math.degrees(world.joint_angle(particles[1], particles[2], particles[3]))
                ) if len(particles) > 3 else 0.0,
                "angle2_error_deg": math.degrees(errors[1]) if len(errors) > 1 else 0.0,
                "angle2_torque": torques[1] if len(torques) > 1 else 0.0,
                "energy_total_j": energy["total_j"],
                "energy_pivot1_j": energy["joints"][0] if len(energy["joints"]) > 0 else 0.0,
                "energy_pivot2_j": energy["joints"][1] if len(energy["joints"]) > 1 else 0.0,
                "power_total_w": energy["total_w"],
                "power_pivot1_w": energy["powers"][0] if len(energy["powers"]) > 0 else 0.0,
                "power_pivot2_w": energy["powers"][1] if len(energy["powers"]) > 1 else 0.0,
                "distance_x_m": distance_x,
                "score": score,
                "score_distance_points": distance_points,
                "score_energy_penalty_points": energy_penalty_points,
                "airborne_time_s": airborne_time,
                "score_airborne_penalty_points": airborne_penalty_points,
            }
        )
        if log and hud_state["time"] >= next_log_time:
            particles_state = particles
            com_y = sum(p.y for p in particles_state) / len(particles_state)
            max_speed = max(math.hypot(p.vx, p.vy) for p in particles_state)
            print(
                "[tetrad] "
                f"t={hud_state['time']:.2f}s "
                f"pos={int(hud_state['position_index']) + 1}/{len(positions)} "
                f"playing={hud_state['playing']} "
                f"clock={playback['clock_hz']:.2f}Hz "
                f"set=({', '.join(f'{t:.1f}' for t in targets)}) "
                f"applied_set=({', '.join(f'{c:.1f}' for c in command_targets_deg)}) "
                f"measured=({', '.join(f'{normalize_angle_deg(math.degrees(world.joint_angle(particles[joint_idx], particles[left_idx], particles[right_idx]))):.1f}' for joint_idx, left_idx, right_idx in joint_pairs)}) "
                f"energy={energy['total_j']:.2f}J "
                f"air={airborne_time:.2f}s "
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
    parser.add_argument("--score-airborne-penalty", type=float, default=80.0)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--generation", type=int, default=None)
    args = parser.parse_args()

    genome = None
    if args.log_path and args.generation is not None:
        import json
        found = False
        with open(args.log_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data["generation"] == args.generation and data["rank"] == 1:
                    genome_dict = data["genome"]
                    positions = [Position(list(p)) for p in genome_dict["positions"]]
                    genome = TetradGenome(
                        positions,
                        genome_dict["clock_hz"],
                        list(genome_dict.get("branch_attachments", [])),
                    )
                    found = True
                    break
        if not found:
            raise ValueError(f"Generation {args.generation} not found in {args.log_path}")

    demo(
        angle1_deg=args.angle1_deg,
        angle2_deg=args.angle2_deg,
        stiffness=args.stiffness,
        damping=args.damping,
        max_torque=args.max_torque,
        gravity=args.gravity,
        start_y=args.start_y,
        branch_length=args.branch_length,
        fps=args.fps,
        time_scale=args.time_scale,
        max_time=args.max_time,
        log=args.log,
        log_interval=args.log_interval,
        score_points_per_meter=args.score_points_per_meter,
        score_energy_penalty=args.score_energy_penalty,
        score_airborne_penalty=args.score_airborne_penalty,
        genome=genome,
        autoplay=genome is not None,
    )


if __name__ == "__main__":
    main()
