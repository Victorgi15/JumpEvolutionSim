"""Build a two-segment creature in the physics World using the genome.

We model each segment as a Particle pair (start/end) connected by a distance constraint.
For simplicity a segment is represented by a single particle at its end; the 'root' is fixed at y=0 initially.
Muscles are modelled as distance constraints whose target_length varies over time according to clocks.
"""

from typing import Dict, List
import math
from physics.engine import Particle, DistanceConstraint, World


class Creature:
    def __init__(self, genome: Dict, world: World, base_x=0.0):
        self.genome = genome
        self.world = world
        self.base_x = base_x
        self.particles: List[Particle] = []
        self.constraints: List[DistanceConstraint] = []
        self._build()

    def _build(self):
        segs = self.genome["segments"]
        masses = self.genome["masses"]
        # root particle anchored to ground (inv_mass=0)
        root = Particle(self.base_x, 0.0, 0.0, 0.0, 0.0)
        self.world.add_particle(root)
        self.particles.append(root)

        # create chain of end particles
        x = self.base_x
        y = 0.0
        for i, L in enumerate(segs):
            # place next particle vertically upwards
            y += L
            m = masses[i]
            p = Particle(x, y, 0.0, 0.0, 1.0 / m if m > 0 else 0.0)
            self.world.add_particle(p)
            self.particles.append(p)
            # distance constraint to previous particle (rigid segment)
            c = DistanceConstraint(self.particles[-2], p, L, stiffness=1.0)
            self.world.add_constraint(c)
            self.constraints.append(c)

        # muscles: for simplicity each muscle controls the segment's current length target
        # we keep references to the same DistanceConstraint objects and will update target_length
        self.muscle_constraints = self.constraints[:]  # one per segment

    def step_controller(self, t: float, dt: float):
        # update target lengths based on clocks
        clocks = self.genome.get("clocks", [])
        rest_factors = self.genome.get("rest_factors", [])
        for i, c in enumerate(self.muscle_constraints):
            L0 = self.genome["segments"][i]
            clock = clocks[i]
            amp = clock["amp"]
            freq = clock["freq"]
            phase = clock["phase"]
            # sine wave modifies rest length factor
            factor = 1.0 + amp * math.sin(2 * math.pi * freq * t + phase)
            target = L0 * factor * rest_factors[i]
            # interpolate to new target to simulate stiffness
            c.target_length = target

    def center_of_mass(self):
        return self.world.center_of_mass()
