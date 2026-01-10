"""Build a chain creature with rigid bones and joint angle actuators."""

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
        self.joints: List[tuple[int, int, int]] = []
        self.joint_params: List[dict] = []
        self.joint_rest_angles: List[float] = []
        self.current_joint_targets: List[float] = []
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

        # joints: internal nodes only
        for i in range(1, len(self.particles) - 1):
            self.joints.append((i, i - 1, i + 1))

        # initialize joint params
        self._init_joint_params()

    def _init_joint_params(self):
        defaults = self.genome.get("joint_params", [])
        self.joint_params = []
        self.joint_rest_angles = []
        for i, (j_idx, l_idx, r_idx) in enumerate(self.joints):
            joint = self.particles[j_idx]
            left = self.particles[l_idx]
            right = self.particles[r_idx]
            rest_angle = self.world.joint_angle(joint, left, right)
            params = defaults[i] if i < len(defaults) else {}
            # set base values if missing
            params = {
                "rest_angle": params.get("rest_angle", rest_angle),
                "amp": params.get("amp", 0.0),
                "freq": params.get("freq", 1.5),
                "phase": params.get("phase", 0.0),
                "stiffness": params.get("stiffness", 6.0),
                "damping": params.get("damping", 1.2),
                "target_angle": params.get("target_angle", None),
            }
            self.joint_params.append(params)
            self.joint_rest_angles.append(rest_angle)
        self.cycle_freq = self.joint_params[0]["freq"] if self.joint_params else 1.0

    def step_controller(self, t: float, dt: float):
        # compute target angles for each joint
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
        for i, (j_idx, l_idx, r_idx) in enumerate(self.joints):
            joint = self.particles[j_idx]
            left = self.particles[l_idx]
            right = self.particles[r_idx]
            params = self.joint_params[i]
            target = self.current_joint_targets[i]
            stiffness = params["stiffness"]
            damping = params["damping"]
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
