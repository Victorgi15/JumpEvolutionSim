"""Try a first random population of tetrad control genomes."""

import argparse
from dataclasses import dataclass
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.visualize_tetrad import Position, TetradGenome, demo, evaluate_tetrad


@dataclass
class CandidateResult:
    index: int
    genome: TetradGenome
    metrics: dict[str, float]


def random_tetrad_genome(rng: random.Random) -> TetradGenome:
    positions = [
        Position([rng.uniform(0.0, 360.0), rng.uniform(0.0, 360.0)])
        for _ in range(3)
    ]
    return TetradGenome(positions, rng.uniform(0.1, 2.0))


def evaluate_population(
    population_size: int = 10,
    duration: float = 10.0,
    seed: int | None = None,
    score_points_per_meter: float = 100.0,
    score_energy_penalty: float = 0.05,
) -> list[CandidateResult]:
    rng = random.Random(seed)
    results = []
    for index in range(1, population_size + 1):
        genome = random_tetrad_genome(rng)
        metrics = evaluate_tetrad(
            genome,
            duration=duration,
            score_points_per_meter=score_points_per_meter,
            score_energy_penalty=score_energy_penalty,
        )
        results.append(CandidateResult(index, genome, metrics))
    return results


def format_genome(genome: TetradGenome) -> str:
    positions = []
    for position in genome.positions:
        a1, a2 = position.pivot_targets_deg
        positions.append(f"({a1:6.1f}, {a2:6.1f})")
    return f"clock={genome.clock_hz:.2f}Hz positions=[{', '.join(positions)}]"


def print_results(results: list[CandidateResult]) -> None:
    for result in results:
        metrics = result.metrics
        print(
            f"#{result.index:02d} "
            f"score={metrics['score']:8.2f} "
            f"dx={metrics['distance_x_m']:+7.3f}m "
            f"energy={metrics['energy_total_j']:8.2f}J "
            f"{format_genome(result.genome)}"
        )


def best_candidate(results: list[CandidateResult]) -> CandidateResult:
    return max(results, key=lambda result: result.metrics["score"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--view-time", type=float, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--score-points-per-meter", type=float, default=100.0)
    parser.add_argument("--score-energy-penalty", type=float, default=0.05)
    args = parser.parse_args()

    results = evaluate_population(
        population_size=args.population,
        duration=args.duration,
        seed=args.seed,
        score_points_per_meter=args.score_points_per_meter,
        score_energy_penalty=args.score_energy_penalty,
    )
    print_results(results)
    best = best_candidate(results)
    print()
    print(
        "Best "
        f"#{best.index:02d} "
        f"score={best.metrics['score']:.2f} "
        f"dx={best.metrics['distance_x_m']:+.3f}m "
        f"energy={best.metrics['energy_total_j']:.2f}J"
    )
    print(format_genome(best.genome))

    if not args.no_view:
        demo(
            genome=best.genome,
            autoplay=True,
            fps=args.fps,
            max_time=args.view_time,
            score_points_per_meter=args.score_points_per_meter,
            score_energy_penalty=args.score_energy_penalty,
        )


if __name__ == "__main__":
    main()
