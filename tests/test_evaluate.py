from creature.genome import default_genome
from sim.run_simple import evaluate


def test_evaluate_basic():
    g = default_genome()
    metrics = evaluate(g, duration=0.5, dt=1 / 120.0)
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"height", "energy", "fitness"}
    assert isinstance(metrics["height"], float)
    assert isinstance(metrics["energy"], float)
    assert isinstance(metrics["fitness"], float)
    # height should be >= 0 since COM above ground
    assert metrics["height"] >= 0.0
    assert metrics["energy"] >= 0.0


if __name__ == "__main__":
    print("Running evaluate test...")
    test_evaluate_basic()
    print("OK")
