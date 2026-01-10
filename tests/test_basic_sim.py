"""Minimal test to ensure simulation runs and returns a numeric max height."""

from sim.run_simple import run


def test_run_short():
    h = run(duration=1.0, dt=1 / 120.0)
    assert isinstance(h, float)
    assert h >= 0.0


if __name__ == "__main__":
    print("Running short test...")
    test_run_short()
    print("OK")
