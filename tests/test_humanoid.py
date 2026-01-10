from creature.humanoid import HumanoidCreature
from physics.engine import World


def test_humanoid_lengths_stable():
    w = World()
    w.dt = 1 / 240.0
    h = HumanoidCreature({}, w, base_x=0.0)
    # record nominal lengths
    nominal = [c.target_length for c in h.constraints]
    # simulate for 2 seconds
    steps = int(2.0 / w.dt)
    for _ in range(steps):
        h.step_actuators(0.0, w.dt)
        w.step(w.dt)
    # check lengths do not drift beyond tolerance
    tol = 5e-3
    for c, L0 in zip(h.constraints, nominal):
        dx = c.p2.x - c.p1.x
        dy = c.p2.y - c.p1.y
        dist = (dx * dx + dy * dy) ** 0.5
        assert abs(dist - L0) < tol


if __name__ == "__main__":
    test_humanoid_lengths_stable()
    print("OK")
