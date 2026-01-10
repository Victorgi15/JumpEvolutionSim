"""Humanoid creature builder matching the requested structure.

Creates a central spine with shoulders and hips; simple symmetric 2-DoF arms/legs.
"""

from typing import Dict, List
from physics.engine import Particle, DistanceConstraint, World


class HumanoidCreature:
    def __init__(
        self,
        genome: Dict,
        world: World,
        base_x=0.0,
        force_scale: float = 1.0,
        balance_assist: bool = True,
        balance_kp: float = 80.0,
        balance_kd: float = 10.0,
        max_up_g_factor: float = 2.5,
    ):
        self.genome = genome
        self.world = world
        self.base_x = base_x
        self.force_scale = force_scale
        # balance assist params
        self.balance_assist = balance_assist
        self.balance_kp = balance_kp
        self.balance_kd = balance_kd
        # maximum upward force allowed in multiples of weight (mass * g)
        self.max_up_g_factor = max_up_g_factor
        self.particles: List[Particle] = []
        self.constraints: List[DistanceConstraint] = []
        self.muscle_edges = []
        # controller diagnostics
        self.controller_time = 0.0
        self.pose_progress = 0.0
        self._last_pose_index = None
        # anchor storage for targets and pelvis support
        self.anchors = {}
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
        ]

        for i, j, L, is_muscle, comp in edges:
            p1 = self.particles[i]
            p2 = self.particles[j]
            c = DistanceConstraint(p1, p2, L, stiffness=1.0, compliance=comp)
            self.world.add_constraint(c)
            self.constraints.append(c)
            if is_muscle:
                # muscles will use genome params (if provided) mapped by index
                # stronger default for leg muscles (hip->knee)
                leg_pairs = {(1, 3), (2, 4)}
                shoulder_pairs = {(5, 6), (5, 7), (6, 8), (7, 9)}
                if (i, j) in leg_pairs:
                    default_muscle = {"force_max": 380.0, "stiffness": 1.2}
                elif (i, j) in shoulder_pairs:
                    default_muscle = {"force_max": 160.0, "stiffness": 1.0}
                else:
                    default_muscle = {"force_max": 120.0, "stiffness": 1.0}
                self.muscle_edges.append({"constraint": c, "params": default_muscle})

    def step_controller(self, t: float, dt: float):
        """Advance the internal pose clock and set current_targets (world positions) for particles.

        Expects `self.pose_sequence` to be a list of dicts: { 'duration': float, 'targets': {idx:(x_rel,y_rel)} }
        where targets are positions relative to pelvis.
        """
        if not hasattr(self, "pose_sequence") or len(self.pose_sequence) == 0:
            return
        # expose controller time
        self.controller_time = t
        if not hasattr(self, "pose_index"):
            self.pose_index = 0
            self.pose_elapsed = 0.0
        self.pose_elapsed += dt
        duration = self.pose_sequence[self.pose_index].get("duration", 1.0)
        # pose progress (clamped to 0..1)
        self.pose_progress = min(
            1.0, self.pose_elapsed / duration if duration > 0 else 0.0
        )
        if self.pose_elapsed >= duration:
            self.pose_elapsed = 0.0
            self.pose_index = (self.pose_index + 1) % len(self.pose_sequence)
        current_pose = self.pose_sequence[self.pose_index].get("targets", {})
        pelvis = self.particles[0]
        px, py = pelvis.x, pelvis.y
        # compute world targets only when entering a new pose so targets stay fixed in world frame
        if self._last_pose_index != self.pose_index:
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
            # clamp upward component to avoid "flying" when muscles are too strong
            dx = c.p2.x - c.p1.x
            dy = c.p2.y - c.p1.y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist != 0:
                ux = dx / dist
                uy = dy / dist
                if uy > 0:
                    # approximate mass of p1 (if p1 is the actuator) else use both
                    m1 = 0.0 if c.p1.inv_mass == 0 else 1.0 / c.p1.inv_mass
                    # max upward force allowed based on weight
                    max_up_force = m1 * abs(self.world.gravity[1]) * self.max_up_g_factor
                    # vertical component is force * uy
                    if force * uy > max_up_force and uy > 0:
                        force = max_up_force / uy
            energy = self.world.muscle_pair(c.p1, c.p2, force, dt)
            total_energy += energy
            self.last_activations.append(
                {"p1": c.p1, "p2": c.p2, "activation": act, "force": force}
            )
        # PD controllers for pose targets
        if hasattr(self, "current_targets") and len(self.current_targets) > 0:
            pelvis = self.particles[0]
            kp = 80.0 * self.force_scale
            kd = 40.0 * self.force_scale
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
                # clamp upward component to prevent launching
                ux = ux
                uy = uy
                if uy > 0:
                    m_p = 0.0 if p.inv_mass == 0 else 1.0 / p.inv_mass
                    max_up_force = m_p * abs(self.world.gravity[1]) * self.max_up_g_factor * self.force_scale
                    if force_mag * uy > max_up_force:
                        force_mag = max_up_force / uy
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
        # Balance assist: keep COM over support (feet) by applying small horizontal force to pelvis
        if self.balance_assist:
            try:
                pelvis = self.particles[0]
                # choose support points (knees act as feet in this simple model)
                left_foot = self.particles[3]
                right_foot = self.particles[4]
                left_x = left_foot.x
                right_x = right_foot.x
                min_x = min(left_x, right_x)
                max_x = max(left_x, right_x)
                com_x, _ = self.world.center_of_mass()
                # only correct if COM is outside support region (small deadzone)
                tol = 0.02
                target_x = None
                if com_x < (min_x - tol):
                    target_x = min_x
                elif com_x > (max_x + tol):
                    target_x = max_x
                if target_x is not None:
                    error_x = target_x - com_x
                    # compute PD force and clamp
                    max_bal = 200.0 * self.force_scale
                    bal_force = self.balance_kp * error_x - self.balance_kd * pelvis.vx
                    bal_force = max(-max_bal, min(max_bal, bal_force))
                    # create or update pelvis support anchor (at target_x)
                    if "pelvis_support" not in self.anchors:
                        anchor = Particle(target_x, pelvis.y, 0.0, 0.0, 0.0)
                        self.world.add_particle(anchor)
                        self.anchors["pelvis_support"] = anchor
                    else:
                        anchor = self.anchors["pelvis_support"]
                        anchor.x = target_x
                        anchor.y = pelvis.y
                    energy = self.world.muscle_pair(pelvis, anchor, bal_force, dt)
                    total_energy += energy
                    self.last_activations.append(
                        {
                            "p1": pelvis,
                            "p2": anchor,
                            "activation": abs(bal_force) / max_bal,
                            "force": abs(bal_force),
                        }
                    )
            except Exception:
                pass
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
