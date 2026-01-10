import sys

sys.path.insert(0, ".")
from sim.run_simple import run

if __name__ == "__main__":
    h = run(duration=1.0, dt=1 / 120.0)
    print("height=", h)
