"""Humanoid creature builder matching the requested structure.

Creates a central spine with shoulders and hips; simple symmetric 2-DoF arms/legs.
"""

from typing import Dict, List
from physics.engine import Particle, DistanceConstraint, World


class HumanoidCreature:
    def __init__(self, genome: Dict, world: World, base_x=0.0, force_scale: float = 1.0):
        self.genome = genome
        self.world = world
        self.base_x = base_x
        self.force_scale = force_scale
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
        """Advance the internal pose clock and set current_targets (world positions) for particles.

        Expects `self.pose_sequence` to be a list of dicts: { 'duration': float, 'targets': {idx:(x_rel,y_rel)} }
        where targets are positions relative to pelvis.
        """
        if not hasattr(self, "pose_sequence") or len(self.pose_sequence) == 0:
            return
        if not hasattr(self, "pose_index"):
            self.pose_index = 0
            self.pose_elapsed = 0.0
        self.pose_elapsed += dt
        duration = self.pose_sequence[self.pose_index].get("duration", 1.0)
        if self.pose_elapsed >= duration:
            self.pose_elapsed = 0.0
            self.pose_index = (self.pose_index + 1) % len(self.pose_sequence)
        current_pose = self.pose_sequence[self.pose_index].get("targets", {})
        pelvis = self.particles[0]
        px, py = pelvis.x, pelvis.y
        # compute world targets only when entering a new pose so targets stay fixed in world frame
        if (
            not hasattr(self, "_last_pose_index")
            or self._last_pose_index != self.pose_index
        ):
            self.current_targets = {}
            for idx, rel in current_pose.items():
                self.current_targets[int(idx)] = (px + rel[0], py + rel[1])
            self._last_pose_index = self.pose_index

    def step_actuators(self, t: float, dt: float) -> float:
        """Actuate cyclic muscles and PD-track pose targets if present.

        Returns energy consumed during this timestep.
        """
        total_energy = 0.0
        # cyclic limb muscles
        self.last_activations = []
        for i, m in enumerate(self.muscle_edges):
            c = m["constraint"]
            params = m["params"]
            phase = (i % 2) * 0.5
            act = 0.5 * (
                1.0
                + __import__("math").sin(2 * __import__("math").pi * 1.5 * t + phase)
            )
            force = params["force_max"] * self.force_scale * act
            energy = self.world.muscle_pair(c.p1, c.p2, force, dt)
            total_energy += energy
            self.last_activations.append(
                {"p1": c.p1, "p2": c.p2, "activation": act, "force": force}
            )
        # PD controllers for pose targets
        if hasattr(self, "current_targets") and len(self.current_targets) > 0:
            pelvis = self.particles[0]
            kp = 80.0
            kd = 40.0
            max_force = 200.0 * self.force_scale
            for idx, (tx, ty) in self.current_targets.items():
                if idx < 0 or idx >= len(self.particles):
                    continue
                p = self.particles[idx]
                # update or create anchor particle at target position
                if not hasattr(self, "anchors"):
                    self.anchors = {}
                if idx not in self.anchors:
                    # create a fixed anchor (inv_mass=0) so it doesn't move and acts as target
                    anchor = Particle(tx, ty, 0.0, 0.0, 0.0)
                    self.world.add_particle(anchor)
                    self.anchors[idx] = anchor
                else:
                    anchor = self.anchors[idx]
                    anchor.x = tx
                    anchor.y = ty
                # compute direction toward anchor
                dx = tx - p.x
                dy = ty - p.y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist == 0:
                    continue
                ux = dx / dist
                uy = dy / dist
                v_along = p.vx * ux + p.vy * uy
                # apply deadzone to avoid chatter when very close
                if dist < 0.02:
                    force_mag = 0.0
                else:
                    force_mag = kp * dist - kd * v_along
                    force_mag = max(-max_force, min(max_force, force_mag))
                energy = self.world.muscle_pair(p, anchor, force_mag, dt)
                total_energy += energy
                self.last_activations.append(
                    {
                        "p1": p,
                        "p2": anchor,
                        "activation": abs(force_mag) / max_force,
                        "force": abs(force_mag),
                    }
                )
        return total_energy

    def set_pose_sequence(self, sequence):
        """Set pose sequence: list of dicts { 'duration': float, 'targets': {idx:(x_rel,y_rel)} }.
        targets are positions relative to pelvis.
        """
        self.pose_sequence = sequence
        self.pose_index = 0
        self.pose_elapsed = 0.0

    def center_of_mass(self):
        return self.world.center_of_mass()
