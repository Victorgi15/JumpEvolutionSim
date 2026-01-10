"""Genome representation and random generator for a two-segment creature."""

import random


def default_genome():
    # Default chain genome (kept for backward compatibility)
    return {
        "type": "chain",
        "segments": [0.4, 0.5],
        "masses": [1.0, 0.8],
        "muscles": [
            {"force_max": 150.0, "stiffness": 1.0},
            {"force_max": 120.0, "stiffness": 1.0}
        ],
        # clocks per muscle: freq (Hz), phase (rad), amp (relative length change)
        "clocks": [
            {"freq": 1.5, "phase": 0.0, "amp": 0.2},
            {"freq": 1.5, "phase": 0.5, "amp": 0.2}
        ],
        # target relative length factor (1.0 = nominal)
        "rest_factors": [1.0, 1.0]
    }


def humanoid_genome():
    # simple humanoid genome: placeholder muscle params on limbs
    g = {
        "type": "humanoid",
        "muscle_params": [
            {"force_max": 150.0},
            {"force_max": 150.0},
            {"force_max": 100.0},
            {"force_max": 100.0}
        ],
        "clocks": {
            "freq": 1.5,
            "phase_offset": 0.5,
            "amp": 1.0
        }
    }
    return g


def random_genome(rng=None):
    if rng is None:
        rng = random.Random()
    g = {
        "segments": [rng.uniform(0.25, 0.6), rng.uniform(0.25, 0.7)],
        "masses": [rng.uniform(0.5, 2.0), rng.uniform(0.4, 1.5)],
        "muscles": [
            {"force_max": rng.uniform(50, 300), "stiffness": rng.uniform(0.5, 2.0)},
            {"force_max": rng.uniform(50, 300), "stiffness": rng.uniform(0.5, 2.0)},
        ],
        "clocks": [
            {
                "freq": rng.uniform(0.5, 3.0),
                "phase": rng.uniform(0, 6.28),
                "amp": rng.uniform(0.05, 0.4),
            },
            {
                "freq": rng.uniform(0.5, 3.0),
                "phase": rng.uniform(0, 6.28),
                "amp": rng.uniform(0.05, 0.4),
            },
        ],
        "rest_factors": [1.0, 1.0],
    }
    return g
