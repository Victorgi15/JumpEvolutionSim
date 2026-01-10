"""Genome representation and random generator for joint-angle creatures."""

import random


def default_genome():
    return {
        "type": "chain",
        "segments": [0.4, 0.5],
        "masses": [1.0, 0.8],
        "joint_params": [
            {"amp": 0.25, "freq": 1.5, "phase": 0.0, "stiffness": 6.0, "damping": 1.2}
        ],
    }


def humanoid_genome():
    return {
        "type": "humanoid",
        "joint_params": [
            {"amp": 0.35, "phase": 0.0},
            {"amp": 0.35, "phase": 3.14},
            {"amp": 0.25, "phase": 3.14},
            {"amp": 0.25, "phase": 0.0},
        ],
        "cycle_freq": 1.5,
        "stiffness": 6.0,
        "damping": 1.2,
    }


def random_genome(rng=None):
    if rng is None:
        rng = random.Random()
    segments = [rng.uniform(0.25, 0.6), rng.uniform(0.25, 0.7)]
    masses = [rng.uniform(0.5, 2.0), rng.uniform(0.4, 1.5)]
    joint_params = [
        {
            "amp": rng.uniform(0.05, 0.6),
            "freq": rng.uniform(0.5, 3.0),
            "phase": rng.uniform(0, 6.28),
            "stiffness": rng.uniform(3.0, 10.0),
            "damping": rng.uniform(0.5, 3.0),
        }
    ]
    return {"segments": segments, "masses": masses, "joint_params": joint_params}
