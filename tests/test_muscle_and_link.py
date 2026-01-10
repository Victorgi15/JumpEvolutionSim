import math
from physics.engine import World, Particle, DistanceConstraint


def test_joint_angle_controller_reduces_error():
    w = World()
    a = Particle(-0.2, 0.0, 0.0, 0.0, 1.0)
    b = Particle(0.2, 0.0, 0.0, 0.0, 1.0)
    c = Particle(0.2, 0.4, 0.0, 0.0, 1.0)
    w.add_particle(a)
    w.add_particle(b)
    w.add_particle(c)
    L_ab = math.hypot(b.x - a.x, b.y - a.y)
    L_bc = math.hypot(c.x - b.x, c.y - b.y)
    w.add_constraint(DistanceConstraint(a, b, L_ab))
    w.add_constraint(DistanceConstraint(b, c, L_bc))

    start_angle = w.joint_angle(b, a, c)
    target = start_angle - 0.6
    err0 = abs(w._wrap_angle(target - start_angle))
    for _ in range(120):
        w.apply_joint_angle_pd(b, a, c, target, stiffness=6.0, damping=1.2, dt=w.dt)
        w.step(w.dt)
    end_angle = w.joint_angle(b, a, c)
    err1 = abs(w._wrap_angle(target - end_angle))
    assert err1 < err0


def test_link_preserves_length():
    w = World()
    p1 = Particle(0, 1.0, 0.0, -5.0, 1.0)
    p2 = Particle(0, 2.0, 0.0, 5.0, 1.0)
    w.add_particle(p1)
    w.add_particle(p2)
    c = DistanceConstraint(p1, p2, 1.0)
    w.add_constraint(c)
    # step several frames and ensure length stays close to 1.0
    for _ in range(60):
        w.step(w.dt)
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dist = (dx**2 + dy**2) ** 0.5
    assert abs(dist - 1.0) < 1e-2


if __name__ == "__main__":
    test_joint_angle_controller_reduces_error()
    test_link_preserves_length()
    print("OK")
