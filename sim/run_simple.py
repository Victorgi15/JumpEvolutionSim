"""Run a simple simulation for 5 seconds with a default genome creature and report max COM height."""

import time
from physics.engine import World, Particle
from creature.genome import default_genome
from creature.creature import Creature


def evaluate(
    genome,
    duration=5.0,
    dt=1 / 240.0,
    alpha=1e-3,
    log_csv: str = None,
    seed: int = None,
):
    """Simulate genome for duration and return metrics: height, energy, fitness.

    Optional: append metrics to CSV `log_csv`.
    """
    world = World()
    world.dt = dt
    c = Creature(genome, world, base_x=0.0)

    t = 0.0
    steps = int(duration / dt)
    max_y = -1e9
    energy = 0.0
    for i in range(steps):
        # controller update
        c.step_controller(t, dt)
        # actuators: apply muscle forces and accumulate energy
        energy += c.step_actuators(t, dt)
        # step world
        world.step(dt)
        com = world.center_of_mass()
        max_y = max(max_y, com[1])
        t += dt

    fitness = max_y - alpha * energy
    metrics = {"height": max_y, "energy": energy, "fitness": fitness, "alpha": alpha}
    if seed is not None:
        metrics["seed"] = seed
    if log_csv is not None:
        try:
            from experiments.logger import append_metrics_csv

            append_metrics_csv(log_csv, metrics)
        except Exception as e:
            print("Failed to write CSV:", e)
    return metrics


def run(duration=5.0, dt=1 / 240.0):
    genome = default_genome()
    metrics = evaluate(genome, duration=duration, dt=dt)
    print(
        f"Max COM height over {duration}s: {metrics['height']:.4f} m, energy={metrics['energy']:.4f}, fitness={metrics['fitness']:.6f}"
    )
    return metrics


if __name__ == "__main__":
    run()
