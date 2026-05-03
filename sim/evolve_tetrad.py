"""Evolve tetrad control genomes with a simple genetic loop."""

import argparse
import json
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


def clone_genome(genome: TetradGenome) -> TetradGenome:
    return TetradGenome(
        [Position(position.pivot_targets_deg.copy()) for position in genome.positions],
        genome.clock_hz,
    )


def mutate_genome(
    genome: TetradGenome,
    rng: random.Random,
    mutation_rate: float = 0.35,
    angle_sigma: float = 12.0,
    clock_sigma: float = 0.12,
) -> TetradGenome:
    mutated = clone_genome(genome)
    for position in mutated.positions:
        for index, angle in enumerate(position.pivot_targets_deg):
            if rng.random() <= mutation_rate:
                position.pivot_targets_deg[index] = (
                    angle + rng.gauss(0.0, angle_sigma)
                ) % 360.0
    if rng.random() <= mutation_rate:
        mutated.clock_hz = max(
            0.1, min(2.0, mutated.clock_hz + rng.gauss(0.0, clock_sigma))
        )
    return mutated


def evaluate_population(
    population_size: int = 10,
    duration: float = 10.0,
    seed: int | None = None,
    genomes: list[TetradGenome] | None = None,
    score_points_per_meter: float = 100.0,
    score_energy_penalty: float = 0.05,
) -> list[CandidateResult]:
    rng = random.Random(seed)
    if genomes is None:
        genomes = [random_tetrad_genome(rng) for _ in range(population_size)]

    results = []
    for index, genome in enumerate(genomes, start=1):
        metrics = evaluate_tetrad(
            genome,
            duration=duration,
            score_points_per_meter=score_points_per_meter,
            score_energy_penalty=score_energy_penalty,
        )
        results.append(CandidateResult(index, genome, metrics))
    return results


def select_best(
    results: list[CandidateResult], selection_ratio: float = 0.10
) -> list[CandidateResult]:
    selected_count = max(1, int(round(len(results) * selection_ratio)))
    return sorted(results, key=lambda result: result.metrics["score"], reverse=True)[
        :selected_count
    ]


def next_generation(
    selected: list[CandidateResult],
    population_size: int,
    rng: random.Random,
    mutation_rate: float = 0.35,
    angle_sigma: float = 12.0,
    clock_sigma: float = 0.12,
    immigrant_ratio: float = 0.10,
) -> list[TetradGenome]:
    genomes = [clone_genome(result.genome) for result in selected]
    immigrant_count = max(0, int(round(population_size * immigrant_ratio)))
    mutation_slots = max(0, population_size - len(genomes) - immigrant_count)

    for _ in range(mutation_slots):
        parent = rng.choice(selected).genome
        genomes.append(
            mutate_genome(parent, rng, mutation_rate, angle_sigma, clock_sigma)
        )

    while len(genomes) < population_size:
        genomes.append(random_tetrad_genome(rng))

    return genomes[:population_size]


def genome_to_dict(genome: TetradGenome) -> dict[str, object]:
    return {
        "clock_hz": genome.clock_hz,
        "positions": [position.pivot_targets_deg for position in genome.positions],
    }


def log_generation_top3(
    log_path: Path,
    generation: int,
    results: list[CandidateResult],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    top3 = sorted(results, key=lambda result: result.metrics["score"], reverse=True)[:3]
    with log_path.open("a", encoding="utf-8") as handle:
        for rank, result in enumerate(top3, start=1):
            handle.write(
                json.dumps(
                    {
                        "generation": generation,
                        "rank": rank,
                        "candidate": result.index,
                        "metrics": result.metrics,
                        "genome": genome_to_dict(result.genome),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def evolve_population(
    population_size: int = 1000,
    generations: int = 1,
    duration: float = 10.0,
    seed: int | None = None,
    selection_ratio: float = 0.10,
    mutation_rate: float = 0.35,
    angle_sigma: float = 12.0,
    clock_sigma: float = 0.12,
    immigrant_ratio: float = 0.10,
    score_points_per_meter: float = 100.0,
    score_energy_penalty: float = 0.05,
    log_path: Path | None = None,
) -> list[CandidateResult]:
    rng = random.Random(seed)
    genomes = [random_tetrad_genome(rng) for _ in range(population_size)]
    final_results: list[CandidateResult] = []

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    for generation in range(1, generations + 1):
        final_results = evaluate_population(
            population_size=population_size,
            duration=duration,
            genomes=genomes,
            score_points_per_meter=score_points_per_meter,
            score_energy_penalty=score_energy_penalty,
        )
        selected = select_best(final_results, selection_ratio)
        best = selected[0]
        print(
            f"generation {generation:03d} "
            f"best_score={best.metrics['score']:.2f} "
            f"dx={best.metrics['distance_x_m']:+.3f}m "
            f"energy={best.metrics['energy_total_j']:.2f}J "
            f"selected={len(selected)}/{population_size} "
            f"immigrants={int(round(population_size * immigrant_ratio))}"
        )
        if log_path is not None:
            log_generation_top3(log_path, generation, final_results)
        if generation < generations:
            genomes = next_generation(
                selected,
                population_size,
                rng,
                mutation_rate,
                angle_sigma,
                clock_sigma,
                immigrant_ratio,
            )

    return final_results


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
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--selection-ratio", type=float, default=0.10)
    parser.add_argument("--mutation-rate", type=float, default=0.35)
    parser.add_argument("--angle-sigma", type=float, default=12.0)
    parser.add_argument("--clock-sigma", type=float, default=0.12)
    parser.add_argument("--immigrant-ratio", type=float, default=0.10)
    parser.add_argument("--log-path", type=Path, default=Path("logs/evolve_tetrad.jsonl"))
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--view-time", type=float, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--score-points-per-meter", type=float, default=100.0)
    parser.add_argument("--score-energy-penalty", type=float, default=0.05)
    args = parser.parse_args()

    results = evolve_population(
        population_size=args.population,
        generations=args.generations,
        duration=args.duration,
        seed=args.seed,
        selection_ratio=args.selection_ratio,
        mutation_rate=args.mutation_rate,
        angle_sigma=args.angle_sigma,
        clock_sigma=args.clock_sigma,
        immigrant_ratio=args.immigrant_ratio,
        score_points_per_meter=args.score_points_per_meter,
        score_energy_penalty=args.score_energy_penalty,
        log_path=args.log_path,
    )
    if args.generations == 1:
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
    print(f"log={args.log_path}")

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
