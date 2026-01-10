from physics.engine import World, Particle, DistanceConstraint


def test_muscle_energy_nonnegative():
    w = World()
    p1 = Particle(0, 0, 0.0, 0.0, 1.0)
    p2 = Particle(0, 0.5, 0.0, -0.5, 1.0)
    w.add_particle(p1)
    w.add_particle(p2)
    energy = w.muscle_pair(p1, p2, force=100.0, dt=1/120.0)
    assert isinstance(energy, float)
    assert energy >= 0.0


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
    dist = (dx ** 2 + dy ** 2) ** 0.5
    assert abs(dist - 1.0) < 1e-2


if __name__ == '__main__':
    test_muscle_energy_nonnegative()
    test_link_preserves_length()
    print('OK')
