import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics.engine import DistanceConstraint, Particle, World
from render import viewer
from sim import evolve_tetrad
from sim import visualize_tetrad
from sim import visualize_triad
from sim.visualize_triad import build_triad
from sim.visualize_tetrad import (
    build_joint_pairs,
    build_tetrad,
    normalized_branch_attachments,
    playback_position_index,
    shortest_delta_deg,
    signed_angle_deg,
)


def test_link_preserves_length():
    world = World()
    p1 = world.add_particle(Particle(0.0, 1.0, 0.0, -5.0, 1.0))
    p2 = world.add_particle(Particle(0.0, 2.0, 0.0, 5.0, 1.0))
    world.add_constraint(DistanceConstraint(p1, p2, 1.0))

    for _ in range(60):
        world.step(world.dt)

    length = math.hypot(p2.x - p1.x, p2.y - p1.y)
    assert length == pytest.approx(1.0, abs=1e-2)


def test_joint_angle_controller_reduces_error():
    world = World(gravity=(0.0, 0.0))
    world.dt = 1 / 240.0
    left, joint, right = build_triad(world, start_y=0.0)

    start_angle = world.joint_angle(joint, left, right)
    target = start_angle - 0.6
    err0 = abs(world._wrap_angle(target - start_angle))

    for _ in range(240):
        world.apply_joint_angle_pd(joint, left, right, target, 12.0, 4.0, world.dt)
        world.step(world.dt)

    end_angle = world.joint_angle(joint, left, right)
    err1 = abs(world._wrap_angle(target - end_angle))
    assert err1 < err0


def test_link_angle_controller_reduces_error():
    world = World(gravity=(0.0, 0.0))
    world.dt = 1 / 240.0
    anchor = world.add_particle(Particle(0.0, 0.7, 0.0, 0.0, 1.0))
    tip = world.add_particle(Particle(0.0, 1.2, 0.0, 0.0, 1.0))
    world.add_constraint(DistanceConstraint(anchor, tip, 0.5))

    start_angle = world.link_angle(anchor, tip)
    target = math.radians(30.0)
    err0 = abs(world._wrap_angle(target - start_angle))

    for _ in range(240):
        world.apply_link_angle_pd(anchor, tip, target, 60.0, 4.0, world.dt, 100.0)
        world.step(world.dt)

    end_angle = world.link_angle(anchor, tip)
    err1 = abs(world._wrap_angle(target - end_angle))
    assert err1 < err0


def test_visualize_triad_cli_arguments_match_demo_signature(monkeypatch):
    captured = {}

    def fake_demo(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(visualize_triad, "demo", fake_demo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_triad",
            "--base-angle",
            "70",
            "--amp",
            "10",
            "--cycle-freq",
            "1.2",
            "--max-torque",
            "42",
        ],
    )

    visualize_triad.main()

    assert captured["base_angle_deg"] == 70
    assert captured["amp_deg"] == 10
    assert captured["cycle_freq"] == 1.2
    assert captured["max_torque"] == 42


def test_visualize_triad_advances_one_render_frame_in_real_time(monkeypatch):
    captured = {}

    class FakeViewer:
        def __init__(self, world, clock_provider, controls_provider, control_setter):
            captured["clock_provider"] = clock_provider

        def run_loop(self, step_callback, fps, max_time):
            step_callback()
            captured["time"] = captured["clock_provider"]()["time"]

    monkeypatch.setattr(visualize_triad, "Viewer", FakeViewer)

    visualize_triad.demo(fps=60)

    assert captured["time"] == pytest.approx(1 / 60)


def test_tetrad_adds_one_branch_to_the_center_joint():
    world = World(gravity=(0.0, 0.0))
    left, joint, right, branch = build_tetrad(world, start_y=0.0, branch_length=0.5)

    assert len(world.particles) == 4
    assert len(world.constraints) == 3
    assert world.constraints[0].p1 is left
    assert world.constraints[0].p2 is joint
    assert world.constraints[1].p1 is joint
    assert world.constraints[1].p2 is right
    assert world.constraints[2].p1 is joint
    assert world.constraints[2].p2 is branch
    assert math.hypot(branch.x - joint.x, branch.y - joint.y) == pytest.approx(0.5)


def test_tetrad_branch_attachments_define_added_segment_joints():
    genome = visualize_tetrad.TetradGenome(
        [
            visualize_tetrad.Position([10.0, 20.0, 30.0]),
            visualize_tetrad.Position([40.0, 50.0, 60.0]),
            visualize_tetrad.Position([70.0, 80.0, 90.0]),
        ],
        0.5,
        [1, 2],
    )
    world = World(gravity=(0.0, 0.0))
    particles = build_tetrad(world, start_y=0.0, genome=genome, branch_length=0.5)

    assert len(particles) == 5
    assert world.constraints[2].p1 is particles[1]
    assert world.constraints[2].p2 is particles[3]
    assert world.constraints[3].p1 is particles[2]
    assert world.constraints[3].p2 is particles[4]
    assert build_joint_pairs(genome) == [(1, 0, 2), (1, 2, 3), (2, 1, 4)]


def test_visualize_tetrad_cli_arguments_match_demo_signature(monkeypatch):
    captured = {}

    def fake_demo(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(visualize_tetrad, "demo", fake_demo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_tetrad",
            "--angle1",
            "75",
            "--angle2",
            "135",
            "--branch-length",
            "0.7",
            "--log",
            "--log-interval",
            "0.5",
            "--score-points-per-meter",
            "150",
            "--score-energy-penalty",
            "0.2",
            "--score-airborne-penalty",
            "70",
        ],
    )

    visualize_tetrad.main()

    assert captured["angle1_deg"] == 75
    assert captured["angle2_deg"] == 135
    assert captured["branch_length"] == 0.7
    assert captured["log"] is True
    assert captured["log_interval"] == 0.5
    assert captured["score_points_per_meter"] == 150
    assert captured["score_energy_penalty"] == 0.2
    assert captured["score_airborne_penalty"] == 70


def test_visualize_tetrad_angle_controls_update_targets(monkeypatch):
    captured = {}

    class FakeViewer:
        def __init__(self, world, clock_provider, controls_provider, control_setter):
            captured["clock_provider"] = clock_provider
            captured["control_setter"] = control_setter
            captured["controls_provider"] = controls_provider

        def run_loop(self, step_callback, fps, max_time):
            keys = [spec["key"] for spec in captured["controls_provider"]()]
            assert keys == [
                "p1_a1",
                "p1_a2",
                "p2_a1",
                "p2_a2",
                "p3_a1",
                "p3_a2",
                "clock_hz",
                "play_positions",
            ]
            captured["control_setter"]("p1_a1", 272.0)
            captured["control_setter"]("p1_a2", 338.0)
            step_callback()
            captured.update(captured["clock_provider"]())

    monkeypatch.setattr(visualize_tetrad, "Viewer", FakeViewer)

    visualize_tetrad.demo(fps=60)

    assert captured["mode"] == "pivot_angles"
    assert captured["angle1_target_deg"] == 272.0
    assert captured["angle2_target_deg"] == 338.0
    assert captured["angle1_applied_target_deg"] == -88.0
    assert captured["angle2_applied_target_deg"] == -22.0


def test_visualize_tetrad_displays_setpoint_and_signed_command(monkeypatch):
    captured = {}

    class FakeViewer:
        def __init__(self, world, clock_provider, controls_provider, control_setter):
            captured["clock_provider"] = clock_provider
            captured["control_setter"] = control_setter

        def run_loop(self, step_callback, fps, max_time):
            captured["control_setter"]("p1_a1", 338.0)
            step_callback()
            captured.update(captured["clock_provider"]())

    monkeypatch.setattr(visualize_tetrad, "Viewer", FakeViewer)

    visualize_tetrad.demo(fps=60)

    assert captured["angle1_target_deg"] == 338.0
    assert captured["angle1_applied_target_deg"] == signed_angle_deg(338.0)
    assert captured["angle1_applied_target_deg"] == -22.0


def test_visualize_tetrad_play_button_loops_positions(monkeypatch):
    captured = {}

    class FakeViewer:
        def __init__(self, world, clock_provider, controls_provider, control_setter):
            captured["clock_provider"] = clock_provider
            captured["control_setter"] = control_setter

        def run_loop(self, step_callback, fps, max_time):
            captured["control_setter"]("p1_a1", 10.0)
            captured["control_setter"]("p1_a2", 20.0)
            captured["control_setter"]("p2_a1", 30.0)
            captured["control_setter"]("p2_a2", 40.0)
            captured["control_setter"]("play_positions", 1.0)
            for _ in range(41):
                step_callback()
            captured.update(captured["clock_provider"]())

    monkeypatch.setattr(visualize_tetrad, "Viewer", FakeViewer)

    visualize_tetrad.demo(fps=60)

    assert captured["playing"] is True
    assert captured["position_index"] == 1
    assert captured["clock_hz"] == pytest.approx(0.5)
    assert captured["angle1_target_deg"] == 30.0
    assert captured["angle2_target_deg"] == 40.0


def test_visualize_tetrad_clock_knob_changes_position_speed(monkeypatch):
    captured = {}

    class FakeViewer:
        def __init__(self, world, clock_provider, controls_provider, control_setter):
            captured["clock_provider"] = clock_provider
            captured["control_setter"] = control_setter

        def run_loop(self, step_callback, fps, max_time):
            captured["control_setter"]("p2_a1", 30.0)
            captured["control_setter"]("p2_a2", 40.0)
            captured["control_setter"]("clock_hz", 1.0)
            captured["control_setter"]("play_positions", 1.0)
            for _ in range(21):
                step_callback()
            captured.update(captured["clock_provider"]())

    monkeypatch.setattr(visualize_tetrad, "Viewer", FakeViewer)

    visualize_tetrad.demo(fps=60)

    assert captured["clock_hz"] == pytest.approx(1.0)
    assert captured["position_index"] == 1
    assert captured["angle1_target_deg"] == 30.0
    assert captured["angle2_target_deg"] == 40.0


def test_visualize_tetrad_playback_smooths_applied_targets(monkeypatch):
    captured = {}

    class FakeViewer:
        def __init__(self, world, clock_provider, controls_provider, control_setter):
            captured["clock_provider"] = clock_provider
            captured["control_setter"] = control_setter

        def run_loop(self, step_callback, fps, max_time):
            captured["control_setter"]("p1_a1", 60.0)
            captured["control_setter"]("p1_a2", 145.0)
            captured["control_setter"]("p2_a1", 180.0)
            captured["control_setter"]("p2_a2", 300.0)
            captured["control_setter"]("play_positions", 1.0)
            for _ in range(41):
                step_callback()
            captured.update(captured["clock_provider"]())

    monkeypatch.setattr(visualize_tetrad, "Viewer", FakeViewer)

    visualize_tetrad.demo(fps=60)

    assert captured["angle1_target_deg"] == 180.0
    assert captured["angle2_target_deg"] == 300.0
    assert captured["angle1_applied_setpoint_deg"] != 180.0
    assert captured["angle2_applied_setpoint_deg"] != 300.0
    assert abs(shortest_delta_deg(60.0, captured["angle1_applied_setpoint_deg"])) < 5.0
    assert abs(shortest_delta_deg(145.0, captured["angle2_applied_setpoint_deg"])) < 5.0


def test_playback_clock_is_full_cycle_frequency():
    assert playback_position_index(0.0, 0.5, 3) == 0
    assert playback_position_index(0.67, 0.5, 3) == 1
    assert playback_position_index(1.34, 0.5, 3) == 2
    assert playback_position_index(2.01, 0.5, 3) == 0
    assert playback_position_index(2.01, 0.5, 6) == 0


def test_visualize_tetrad_tracks_actuator_energy(monkeypatch):
    captured = {}

    class FakeViewer:
        def __init__(self, world, clock_provider, controls_provider, control_setter):
            captured["clock_provider"] = clock_provider
            captured["control_setter"] = control_setter

        def run_loop(self, step_callback, fps, max_time):
            captured["control_setter"]("p1_a1", 90.0)
            captured["control_setter"]("p1_a2", 220.0)
            for _ in range(60):
                step_callback()
            captured.update(captured["clock_provider"]())

    monkeypatch.setattr(visualize_tetrad, "Viewer", FakeViewer)

    visualize_tetrad.demo(fps=60)

    assert captured["energy_total_j"] > 0.0
    assert captured["energy_total_j"] == pytest.approx(
        captured["energy_pivot1_j"] + captured["energy_pivot2_j"]
    )
    assert captured["power_total_w"] >= 0.0


def test_visualize_tetrad_scores_distance_minus_energy(monkeypatch):
    captured = {}

    class FakeViewer:
        def __init__(self, world, clock_provider, controls_provider, control_setter):
            captured["clock_provider"] = clock_provider

        def run_loop(self, step_callback, fps, max_time):
            for _ in range(60):
                step_callback()
            captured.update(captured["clock_provider"]())

    monkeypatch.setattr(visualize_tetrad, "Viewer", FakeViewer)

    visualize_tetrad.demo(
        fps=60,
        score_points_per_meter=10.0,
        score_energy_penalty=0.5,
        score_airborne_penalty=2.0,
    )

    assert captured["score"] == pytest.approx(
        captured["score_distance_points"]
        - captured["score_energy_penalty_points"]
        - captured["score_airborne_penalty_points"]
    )
    assert captured["score_distance_points"] == pytest.approx(
        captured["distance_x_m"] * 10.0
    )
    assert captured["score_energy_penalty_points"] == pytest.approx(
        captured["energy_total_j"] * 0.5
    )
    assert captured["score_airborne_penalty_points"] == pytest.approx(
        captured["airborne_time_s"] * 2.0
    )


def test_evolve_tetrad_evaluates_population():
    results = evolve_tetrad.evaluate_population(
        population_size=3,
        duration=0.1,
        seed=123,
        min_positions=3,
        max_positions=3,
        min_angles=2,
        max_angles=2,
    )
    best = evolve_tetrad.best_candidate(results)

    assert len(results) == 3
    assert all(len(result.genome.positions) == 3 for result in results)
    assert all(0.1 <= result.genome.clock_hz <= 2.0 for result in results)
    assert all("score" in result.metrics for result in results)
    assert all("airborne_time_s" in result.metrics for result in results)
    assert best.metrics["score"] == max(result.metrics["score"] for result in results)


def test_random_tetrad_genome_randomizes_positions_angles_and_attachments():
    rng = random.Random(10)
    genomes = [
        evolve_tetrad.random_tetrad_genome(
            rng,
            min_positions=2,
            max_positions=5,
            min_angles=2,
            max_angles=5,
        )
        for _ in range(20)
    ]
    position_counts = {len(genome.positions) for genome in genomes}
    angle_counts = {len(genome.positions[0].pivot_targets_deg) for genome in genomes}

    assert len(position_counts) > 1
    assert len(angle_counts) > 1
    for genome in genomes:
        angle_count = len(genome.positions[0].pivot_targets_deg)
        assert len(normalized_branch_attachments(genome)) == angle_count - 1
        assert all(
            len(position.pivot_targets_deg) == angle_count
            for position in genome.positions
        )


def test_evolve_tetrad_selects_and_mutates_population():
    rng = random.Random(1)
    results = evolve_tetrad.evaluate_population(
        population_size=10,
        duration=0.05,
        seed=123,
    )
    selected = evolve_tetrad.select_best(results, selection_ratio=0.2)
    next_gen = evolve_tetrad.next_generation(
        selected,
        population_size=10,
        rng=rng,
        mutation_rate=1.0,
        angle_sigma=5.0,
        clock_sigma=0.1,
        immigrant_ratio=0.2,
    )

    assert len(selected) == 2
    assert len(next_gen) == 10
    assert 0.1 <= next_gen[-1].clock_hz <= 2.0
    assert any(
        next_gen[-1].positions[index].pivot_targets_deg
        != selected[0].genome.positions[index].pivot_targets_deg
        for index in range(3)
    )


def test_evolve_tetrad_add_member_can_attach_to_existing_articulations():
    attachments = set()
    base = visualize_tetrad.TetradGenome(
        [
            visualize_tetrad.Position([10.0, 20.0]),
            visualize_tetrad.Position([30.0, 40.0]),
            visualize_tetrad.Position([50.0, 60.0]),
        ],
        0.5,
        [1],
    )

    for seed in range(30):
        mutated = evolve_tetrad.mutate_genome(
            base,
            random.Random(seed),
            mutation_rate=0.0,
            add_member_prob=1.0,
            remove_member_prob=0.0,
        )
        attachments.add(normalized_branch_attachments(mutated)[-1])

    assert len(attachments) > 1
    assert attachments.issubset({0, 1, 2, 3})


def test_evolve_tetrad_can_add_and_remove_positions():
    base = visualize_tetrad.TetradGenome(
        [
            visualize_tetrad.Position([10.0, 20.0]),
            visualize_tetrad.Position([30.0, 40.0]),
            visualize_tetrad.Position([50.0, 60.0]),
        ],
        0.5,
        [1],
    )

    added = evolve_tetrad.mutate_genome(
        base,
        random.Random(4),
        mutation_rate=0.0,
        add_member_prob=0.0,
        remove_member_prob=0.0,
        add_position_prob=1.0,
        remove_position_prob=0.0,
        max_positions=8,
    )
    removed = evolve_tetrad.mutate_genome(
        added,
        random.Random(5),
        mutation_rate=0.0,
        add_member_prob=0.0,
        remove_member_prob=0.0,
        add_position_prob=0.0,
        remove_position_prob=1.0,
        min_positions=2,
    )

    assert len(added.positions) == 4
    assert all(len(position.pivot_targets_deg) == 2 for position in added.positions)
    assert len(removed.positions) == 3


def test_evolve_tetrad_logs_top_three_per_generation():
    log_path = ROOT / "logs" / "test_evolve_tetrad.jsonl"
    log_path.unlink(missing_ok=True)
    results = evolve_tetrad.evolve_population(
        population_size=5,
        generations=2,
        duration=0.05,
        seed=123,
        selection_ratio=0.4,
        log_path=log_path,
    )
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(results) == 5
    assert len(lines) == 6
    assert '"generation": 1' in lines[0]
    assert '"rank": 1' in lines[0]
    assert '"genome"' in lines[0]
    log_path.unlink(missing_ok=True)


def test_evolve_tetrad_injects_random_immigrants():
    rng = random.Random(2)
    parent = evolve_tetrad.CandidateResult(
        1,
        visualize_tetrad.TetradGenome(
            [
                visualize_tetrad.Position([10.0, 20.0]),
                visualize_tetrad.Position([30.0, 40.0]),
                visualize_tetrad.Position([50.0, 60.0]),
            ],
            0.5,
        ),
        {"score": 1.0},
    )

    next_gen = evolve_tetrad.next_generation(
        [parent],
        population_size=5,
        rng=rng,
        mutation_rate=0.0,
        immigrant_ratio=0.4,
        min_positions=4,
        max_positions=4,
        min_angles=4,
        max_angles=4,
    )

    assert next_gen[0].clock_hz == pytest.approx(0.5)
    assert next_gen[1].clock_hz == pytest.approx(0.5)
    assert next_gen[2].clock_hz == pytest.approx(0.5)
    assert any(genome.clock_hz != pytest.approx(0.5) for genome in next_gen[3:])
    assert all(len(genome.positions) == 4 for genome in next_gen[3:])
    assert all(len(genome.positions[0].pivot_targets_deg) == 4 for genome in next_gen[3:])


def test_visualize_tetrad_uses_supplied_genome(monkeypatch):
    captured = {}
    genome = visualize_tetrad.TetradGenome(
        [
            visualize_tetrad.Position([10.0, 20.0]),
            visualize_tetrad.Position([30.0, 40.0]),
            visualize_tetrad.Position([50.0, 60.0]),
        ],
        1.25,
    )

    class FakeViewer:
        def __init__(self, world, clock_provider, controls_provider, control_setter):
            captured["clock_provider"] = clock_provider
            captured["controls_provider"] = controls_provider

        def run_loop(self, step_callback, fps, max_time):
            step_callback()
            captured.update(captured["clock_provider"]())
            captured["controls"] = captured["controls_provider"]()

    monkeypatch.setattr(visualize_tetrad, "Viewer", FakeViewer)

    visualize_tetrad.demo(genome=genome, autoplay=True, fps=60)

    assert captured["playing"] is True
    assert captured["clock_hz"] == pytest.approx(1.25)
    assert captured["controls"][0]["value"] == 10.0
    assert captured["controls"][1]["value"] == 20.0


def test_viewer_camera_follows_creature_center(monkeypatch):
    monkeypatch.setattr(viewer, "PYGAME_AVAILABLE", False)
    world = World(gravity=(0.0, 0.0))
    world.add_particle(Particle(2.0, 0.0, 0.0, 0.0, 1.0))
    world.add_particle(Particle(4.0, 0.0, 0.0, 0.0, 1.0))
    sim_viewer = viewer.Viewer(world, width=800, height=600, scale=100.0)

    sim_viewer._update_camera()

    assert sim_viewer.camera_x > 0.0
    assert sim_viewer.offset_x < sim_viewer.default_offset_x
