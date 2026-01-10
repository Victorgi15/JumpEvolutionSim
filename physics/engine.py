"""Minimal physics engine: particles + distance constraints + simple ground collision.

This is a simple Position-Based Dynamics-like solver (iterative constraint projection)
with semi-implicit Euler integration for velocities.
"""

from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    inv_mass: float

    def apply_impulse(self, ix: float, iy: float):
        if self.inv_mass == 0:
            return
        self.vx += ix * self.inv_mass
        self.vy += iy * self.inv_mass


class DistanceConstraint:
    def __init__(
        self,
        p1: Particle,
        p2: Particle,
        target_length: float,
        stiffness: float = 1.0,
        compliance: float = 0.0,
    ):
        self.p1 = p1
        self.p2 = p2
        self.target_length = target_length
        self.stiffness = stiffness
        # compliance between 0 (rigid) and 1 (fully compliant)
        self.compliance = compliance

    def solve(self):
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return
        error = dist - self.target_length
        # positional correction distributed by inverse mass
        w1 = self.p1.inv_mass
        w2 = self.p2.inv_mass
        wsum = w1 + w2
        if wsum == 0:
            return
        correction = (error / dist) * self.stiffness
        corrx = dx * correction
        corry = dy * correction
        if w1 > 0:
            self.p1.x += corrx * (w1 / wsum)
            self.p1.y += corry * (w1 / wsum)
        if w2 > 0:
            self.p2.x -= corrx * (w2 / wsum)
            self.p2.y -= corry * (w2 / wsum)


class World:
    def __init__(self, gravity: Tuple[float, float] = (0.0, -9.81)):
        self.particles: List[Particle] = []
        self.constraints: List[DistanceConstraint] = []
        self.gravity = gravity
        self.dt = 1 / 120.0
        self.substeps = 1
        self.iterations = 8
        self.restitution = 0.0
        self.friction = 0.8

    def add_particle(self, p: Particle):
        self.particles.append(p)
        return p

    def add_constraint(self, c: DistanceConstraint):
        self.constraints.append(c)
        return c

    def step(self, dt=None):
        if dt is None:
            dt = self.dt
        # high-level step decomposition
        self.Gravity(dt)

        # velocity-level link constraint should be applied BEFORE movement to prevent
        # velocities from causing immediate stretch during the position integration.
        for _ in range(self.iterations):
            self.Link()

        # integrate positions
        self.Movement(dt)

        # positional constraint solver (to correct residuals)
        for _ in range(self.iterations):
            for c in self.constraints:
                c.solve()

        # simple ground collision and friction (y=0 ground)
        for p in self.particles:
            if p.y < 0:
                p.y = 0
                if p.vy < 0:
                    p.vy = -p.vy * self.restitution
                p.vx *= self.friction

        # small global damping to reduce numerical oscillations (keeps system stable)
        damping = 0.998
        for p in self.particles:
            if p.inv_mass == 0:
                continue
            p.vx *= damping
            p.vy *= damping

    def center_of_mass(self):
        sx = sy = sm = 0.0
        for p in self.particles:
            m = 0.0 if p.inv_mass == 0 else 1.0 / p.inv_mass
            sx += p.x * m
            sy += p.y * m
            sm += m
        if sm == 0:
            return (0.0, 0.0)
        return (sx / sm, sy / sm)

    # --- New helper primitives requested ---
    def Movement(self, dt: float):
        """Integrate particle positions from velocities (semi-implicit handled in Gravity)."""
        for p in self.particles:
            # fixed particles (inv_mass == 0) do not move
            if p.inv_mass == 0:
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt

    def Gravity(self, dt: float):
        """Apply gravity to particle velocities."""
        g_x, g_y = self.gravity
        for p in self.particles:
            if p.inv_mass == 0:
                continue
            p.vx += g_x * dt
            p.vy += g_y * dt

    def Link(self):
        """Velocity-level constraint: remove relative velocity along link direction to avoid length change.

        Applies per-constraint compliance (c.compliance) and runs for `self.iterations`.
        """
        # iterate to improve enforcement
        for c in self.constraints:
            p1 = c.p1
            p2 = c.p2
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue
            nx = dx / dist
            ny = dy / dist
            # relative velocity along the link
            rvx = p2.vx - p1.vx
            rvy = p2.vy - p1.vy
            v_rel = rvx * nx + rvy * ny
            # compute impulse (velocity correction) to reduce that component
            w1 = p1.inv_mass
            w2 = p2.inv_mass
            wsum = w1 + w2
            if wsum == 0:
                continue
            # desired change depends on compliance of this constraint
            dv = -v_rel * (1.0 - c.compliance)
            # convert dv to impulses for each particle: J = dv / (w1+w2)
            j = dv / wsum
            jx = j * nx
            jy = j * ny
            # apply velocity changes Δv = ± j * inv_mass_contrib
            if p1.inv_mass != 0:
                p1.vx -= jx * w1
                p1.vy -= jy * w1
            if p2.inv_mass != 0:
                p2.vx += jx * w2
                p2.vy += jy * w2

    @staticmethod
    def _wrap_angle(rad: float) -> float:
        while rad > math.pi:
            rad -= 2.0 * math.pi
        while rad < -math.pi:
            rad += 2.0 * math.pi
        return rad

    def joint_angle(self, joint: Particle, left: Particle, right: Particle) -> float:
        """Return signed joint angle at `joint` between vectors (left-joint) and (right-joint)."""
        lx = left.x - joint.x
        ly = left.y - joint.y
        rx = right.x - joint.x
        ry = right.y - joint.y
        Ll = math.hypot(lx, ly)
        Lr = math.hypot(rx, ry)
        if Ll == 0 or Lr == 0:
            return 0.0
        dot = lx * rx + ly * ry
        cross = lx * ry - ly * rx
        return math.atan2(cross, dot)

    def joint_angular_velocity(
        self, joint: Particle, left: Particle, right: Particle
    ) -> float:
        """Estimate relative angular velocity between the two segments around `joint`."""
        lx = left.x - joint.x
        ly = left.y - joint.y
        rx = right.x - joint.x
        ry = right.y - joint.y
        Ll = math.hypot(lx, ly)
        Lr = math.hypot(rx, ry)
        if Ll == 0 or Lr == 0:
            return 0.0
        tlx = -ly / Ll
        tly = lx / Ll
        trx = -ry / Lr
        try_ = rx / Lr
        vla_x = left.vx - joint.vx
        vla_y = left.vy - joint.vy
        vrc_x = right.vx - joint.vx
        vrc_y = right.vy - joint.vy
        w_left = (vla_x * tlx + vla_y * tly) / Ll
        w_right = (vrc_x * trx + vrc_y * try_) / Lr
        return w_right - w_left

    def apply_joint_torque(
        self, joint: Particle, left: Particle, right: Particle, torque: float, dt: float
    ) -> float:
        """Apply equal and opposite torques on the two segments around a joint."""
        self._apply_pair_couple(joint, left, torque, dt)
        self._apply_pair_couple(joint, right, -torque, dt)
        return 0.0

    def apply_joint_angle_pd(
        self,
        joint: Particle,
        left: Particle,
        right: Particle,
        target_angle: float,
        stiffness: float,
        damping: float,
        dt: float,
    ):
        """Apply a spring-damper torque driving the joint toward `target_angle`."""
        angle = self.joint_angle(joint, left, right)
        w_rel = self.joint_angular_velocity(joint, left, right)
        err = self._wrap_angle(target_angle - angle)
        torque = stiffness * err - damping * w_rel
        self.apply_joint_torque(joint, left, right, torque, dt)
        energy = max(0.0, torque * w_rel) * dt
        return torque, err, energy

    def _apply_pair_couple(
        self, p_left: Particle, p_right: Particle, torque: float, dt: float
    ) -> None:
        dx = p_right.x - p_left.x
        dy = p_right.y - p_left.y
        s = math.hypot(dx, dy)
        if s == 0:
            return
        nx = -dy / s
        ny = dx / s
        F = torque / s
        J = F * dt
        p_left.apply_impulse(J * nx, J * ny)
        p_right.apply_impulse(-J * nx, -J * ny)
