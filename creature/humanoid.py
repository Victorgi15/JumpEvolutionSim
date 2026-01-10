"""Humanoid creature builder with joint angle actuators."""

from typing import Dict, List
import math
from physics.engine import Particle, DistanceConstraint, World


class HumanoidCreature:
    def __init__(
        self,
        genome: Dict,
        world: World,
        base_x=0.0,
        force_scale: float = 1.0,
        cycle_freq: float = 1.5,
        stiffness: float = 6.0,
        damping: float = 1.2,
        default_amp: float = 0.35,
    ):
        self.genome = genome
        self.world = world
        self.base_x = base_x
        self.force_scale = force_scale
        self.cycle_freq = cycle_freq
        self.stiffness = stiffness
        self.damping = damping
        self.default_amp = default_amp
        self.particles: List[Particle] = []
        self.constraints: List[DistanceConstraint] = []
        self.joints: List[dict] = []
        self.joint_params: List[dict] = []
        self.current_joint_targets: List[float] = []
        self.controller_time = 0.0
        self._build()

    def _build(self):
        # default geometry (meters)
        # order: pelvis, hip L, hip R, knee L, knee R, torso top, shoulder L, shoulder R, elbow L, elbow R
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

        for x, y in nodes:
            p = Particle(self.base_x + x, y, 0.0, 0.0, 1.0)
            self.world.add_particle(p)
            self.particles.append(p)

        edges = [
            (0, 1, 0.24),
            (0, 2, 0.24),
            (1, 3, 0.35),
            (2, 4, 0.35),
            (0, 5, 0.5),
            (5, 6, 0.32),
            (5, 7, 0.32),
            (6, 8, 0.3),
            (7, 9, 0.3),
        ]

        for i, j, L in edges:
            p1 = self.particles[i]
            p2 = self.particles[j]
            c = DistanceConstraint(p1, p2, L, stiffness=1.0, compliance=0.0)
            self.world.add_constraint(c)
            self.constraints.append(c)

        # joint definitions (joint index, left index, right index, default phase)
        self.joints = [
            {"name": "hip_left", "joint": 1, "left": 0, "right": 3, "phase": 0.0},
            {"name": "hip_right", "joint": 2, "left": 0, "right": 4, "phase": math.pi},
            {"name": "shoulder_left", "joint": 6, "left": 5, "right": 8, "phase": math.pi},
            {"name": "shoulder_right", "joint": 7, "left": 5, "right": 9, "phase": 0.0},
        ]

        self._init_joint_params()

    def _init_joint_params(self):
        defaults = self.genome.get("joint_params", [])
        base_freq = self.genome.get("cycle_freq", self.cycle_freq)
        base_k = self.genome.get("stiffness", self.stiffness)
        base_d = self.genome.get("damping", self.damping)

        self.joint_params = []
        for i, jdef in enumerate(self.joints):
            joint = self.particles[jdef["joint"]]
            left = self.particles[jdef["left"]]
            right = self.particles[jdef["right"]]
            rest_angle = self.world.joint_angle(joint, left, right)
            params = defaults[i] if i < len(defaults) else {}
            params = {
                "rest_angle": params.get("rest_angle", rest_angle),
                "target_angle": params.get("target_angle", None),
                "amp": params.get("amp", self.default_amp),
                "freq": params.get("freq", base_freq),
                "phase": params.get("phase", jdef["phase"]),
                "stiffness": params.get("stiffness", base_k),
                "damping": params.get("damping", base_d),
            }
            self.joint_params.append(params)

    def step_controller(self, t: float, dt: float):
        self.controller_time = t
        self.current_joint_targets = []
        for params in self.joint_params:
            base = params["rest_angle"]
            if params["target_angle"] is not None:
                base = params["target_angle"]
            amp = params["amp"]
            freq = params["freq"]
            phase = params["phase"]
            target = base + amp * math.sin(2.0 * math.pi * freq * t + phase)
            self.current_joint_targets.append(target)

    def step_actuators(self, t: float, dt: float) -> float:
        """Apply joint spring-damper torques.

        Returns energy consumed during this timestep.
        """
        total_energy = 0.0
        self.last_activations = []
        if not self.current_joint_targets:
            self.step_controller(t, dt)
        for i, jdef in enumerate(self.joints):
            joint = self.particles[jdef["joint"]]
            left = self.particles[jdef["left"]]
            right = self.particles[jdef["right"]]
            params = self.joint_params[i]
            target = self.current_joint_targets[i]
            stiffness = params["stiffness"] * self.force_scale
            damping = params["damping"] * self.force_scale
            torque, err, energy = self.world.apply_joint_angle_pd(
                joint, left, right, target, stiffness, damping, dt
            )
            total_energy += energy
            scale = max(1e-6, abs(stiffness * (abs(params["amp"]) + 1e-3)))
            activation = min(1.0, abs(torque) / scale)
            self.last_activations.append(
                {"p1": left, "p2": joint, "activation": activation, "force": abs(torque)}
            )
            self.last_activations.append(
                {"p1": right, "p2": joint, "activation": activation, "force": abs(torque)}
            )
        return total_energy

    def center_of_mass(self):
        return self.world.center_of_mass()
