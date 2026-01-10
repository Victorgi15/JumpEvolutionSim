"""Humanoid creature builder matching the requested structure.

Creates a central spine with shoulders and hips; simple symmetric 2-DoF arms/legs.
"""

from typing import Dict, List
from physics.engine import Particle, DistanceConstraint, World


class HumanoidCreature:
    def __init__(self, genome: Dict, world: World, base_x=0.0):
        self.genome = genome
        self.world = world
        self.base_x = base_x
        self.particles: List[Particle] = []
        self.constraints: List[DistanceConstraint] = []
        self.muscle_edges = []
        self._build()

    def _build(self):
        # default geometry (meters)
        # nodes defined relative to base_x, base y = 0
        # order: root (pelvis center), hip left, hip right, knee left, knee right, torso top, shoulder left, shoulder right, elbow left, elbow right, head
        nodes = [
            (0.0, 0.6),  # pelvis
            (-0.2, 0.4),  # hip left
            (0.2, 0.4),  # hip right
            (-0.2, 0.1),  # knee left
            (0.2, 0.1),  # knee right
            (0.0, 1.0),  # torso top
            (-0.3, 0.9),  # shoulder left
            (0.3, 0.9),  # shoulder right
            (-0.45, 0.7),  # elbow left
            (0.45, 0.7),  # elbow right
            (0.0, 1.25),  # head
        ]

        # create particles
        for x, y in nodes:
            p = Particle(self.base_x + x, y, 0.0, 0.0, 1.0)
            self.world.add_particle(p)
            self.particles.append(p)

        # edges: (i,j,length,is_muscle,compliance)
        edges = [
            (0, 1, 0.24, False, 0.0),  # pelvis to hip left
            (0, 2, 0.24, False, 0.0),  # pelvis to hip right
            (1, 3, 0.35, True, 0.0),  # hip left to knee
            (2, 4, 0.35, True, 0.0),  # hip right to knee
            (0, 5, 0.5, False, 0.0),  # pelvis to torso top
            (5, 6, 0.32, True, 0.0),  # torso top to shoulder left
            (5, 7, 0.32, True, 0.0),  # torso top to shoulder right
            (6, 8, 0.3, True, 0.0),  # shoulder left -> elbow
            (7, 9, 0.3, True, 0.0),  # shoulder right -> elbow
            (5, 10, 0.28, False, 0.0),  # torso top -> head
        ]

        for i, j, L, is_muscle, comp in edges:
            p1 = self.particles[i]
            p2 = self.particles[j]
            c = DistanceConstraint(p1, p2, L, stiffness=1.0, compliance=comp)
            self.world.add_constraint(c)
            self.constraints.append(c)
            if is_muscle:
                # muscles will use genome params (if provided) mapped by index
                default_muscle = {"force_max": 120.0, "stiffness": 1.0}
                self.muscle_edges.append({"constraint": c, "params": default_muscle})

    def step_controller(self, t: float, dt: float):
        # torso-level simple sway or no change; muscles actuated elsewhere
        pass

    def step_actuators(self, t: float, dt: float) -> float:
        # simple activation pattern for muscles (sine-based)
        total_energy = 0.0
        for i, m in enumerate(self.muscle_edges):
            c = m["constraint"]
            params = m["params"]
            # simple phase per muscle
            phase = (i % 2) * 0.5
            act = 0.5 * (
                1.0
                + __import__("math").sin(2 * __import__("math").pi * 1.5 * t + phase)
            )
            force = params["force_max"] * act
            # apply along constraint endpoints
            energy = self.world.muscle_pair(c.p1, c.p2, force, dt)
            total_energy += energy
            # store for visualization
            if not hasattr(self, "last_activations"):
                self.last_activations = []
            self.last_activations.append(
                {"p1": c.p1, "p2": c.p2, "activation": act, "force": force}
            )
        return total_energy

    def center_of_mass(self):
        return self.world.center_of_mass()
