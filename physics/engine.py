"""Minimal 2D physics for a particle triad."""

from dataclasses import dataclass
import math
from typing import List, Tuple


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    inv_mass: float

    def apply_impulse(self, ix: float, iy: float) -> None:
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
    ):
        self.p1 = p1
        self.p2 = p2
        self.target_length = target_length
        self.stiffness = stiffness

    def solve(self) -> None:
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return

        w1 = self.p1.inv_mass
        w2 = self.p2.inv_mass
        wsum = w1 + w2
        if wsum == 0:
            return

        correction = ((dist - self.target_length) / dist) * self.stiffness
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
        self.iterations = 8
        self.restitution = 0.0
        self.friction = 0.8

    def add_particle(self, p: Particle) -> Particle:
        self.particles.append(p)
        return p

    def add_constraint(self, c: DistanceConstraint) -> DistanceConstraint:
        self.constraints.append(c)
        return c

    def step(self, dt: float | None = None) -> None:
        if dt is None:
            dt = self.dt

        self._apply_gravity(dt)
        for _ in range(self.iterations):
            self._solve_link_velocities()

        self._integrate(dt)
        for _ in range(self.iterations):
            for c in self.constraints:
                c.solve()

        self._collide_with_ground()
        self._damp_velocities()

    def _apply_gravity(self, dt: float) -> None:
        gx, gy = self.gravity
        for p in self.particles:
            if p.inv_mass == 0:
                continue
            p.vx += gx * dt
            p.vy += gy * dt

    def _solve_link_velocities(self) -> None:
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
            relative_vx = p2.vx - p1.vx
            relative_vy = p2.vy - p1.vy
            relative_speed = relative_vx * nx + relative_vy * ny

            w1 = p1.inv_mass
            w2 = p2.inv_mass
            wsum = w1 + w2
            if wsum == 0:
                continue

            impulse = -relative_speed / wsum
            ix = impulse * nx
            iy = impulse * ny
            if w1 > 0:
                p1.vx -= ix * w1
                p1.vy -= iy * w1
            if w2 > 0:
                p2.vx += ix * w2
                p2.vy += iy * w2

    def _integrate(self, dt: float) -> None:
        for p in self.particles:
            if p.inv_mass == 0:
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt

    def _collide_with_ground(self) -> None:
        for p in self.particles:
            if p.y >= 0:
                continue
            p.y = 0
            if p.vy < 0:
                p.vy = -p.vy * self.restitution
            p.vx *= self.friction

    def _damp_velocities(self) -> None:
        for p in self.particles:
            if p.inv_mass == 0:
                continue
            p.vx *= 0.9995
            p.vy *= 0.9995

    @staticmethod
    def _wrap_angle(rad: float) -> float:
        while rad > math.pi:
            rad -= 2.0 * math.pi
        while rad < -math.pi:
            rad += 2.0 * math.pi
        return rad

    @staticmethod
    def joint_angle(joint: Particle, left: Particle, right: Particle) -> float:
        lx = left.x - joint.x
        ly = left.y - joint.y
        rx = right.x - joint.x
        ry = right.y - joint.y
        if math.hypot(lx, ly) == 0 or math.hypot(rx, ry) == 0:
            return 0.0

        dot = lx * rx + ly * ry
        cross = lx * ry - ly * rx
        return math.atan2(cross, dot)

    @staticmethod
    def joint_angular_velocity(joint: Particle, left: Particle, right: Particle) -> float:
        lx = left.x - joint.x
        ly = left.y - joint.y
        rx = right.x - joint.x
        ry = right.y - joint.y
        left_len = math.hypot(lx, ly)
        right_len = math.hypot(rx, ry)
        if left_len == 0 or right_len == 0:
            return 0.0

        left_tx = -ly / left_len
        left_ty = lx / left_len
        right_tx = -ry / right_len
        right_ty = rx / right_len

        left_vx = left.vx - joint.vx
        left_vy = left.vy - joint.vy
        right_vx = right.vx - joint.vx
        right_vy = right.vy - joint.vy

        left_w = (left_vx * left_tx + left_vy * left_ty) / left_len
        right_w = (right_vx * right_tx + right_vy * right_ty) / right_len
        return right_w - left_w

    @staticmethod
    def link_angle(anchor: Particle, tip: Particle) -> float:
        return math.atan2(tip.y - anchor.y, tip.x - anchor.x)

    @staticmethod
    def link_angular_velocity(anchor: Particle, tip: Particle) -> float:
        dx = tip.x - anchor.x
        dy = tip.y - anchor.y
        length = math.hypot(dx, dy)
        if length == 0:
            return 0.0

        tx = -dy / length
        ty = dx / length
        relative_vx = tip.vx - anchor.vx
        relative_vy = tip.vy - anchor.vy
        return (relative_vx * tx + relative_vy * ty) / length

    def apply_link_angle_pd(
        self,
        anchor: Particle,
        tip: Particle,
        target_angle: float,
        stiffness: float,
        damping: float,
        dt: float,
        max_torque: float | None = None,
    ) -> tuple[float, float]:
        angle = self.link_angle(anchor, tip)
        angular_velocity = self.link_angular_velocity(anchor, tip)
        error = self._wrap_angle(target_angle - angle)
        torque = stiffness * error - damping * angular_velocity
        if max_torque is not None:
            torque = max(-max_torque, min(max_torque, torque))

        self._apply_pair_couple(anchor, tip, -torque, dt)
        return torque, error

    def apply_joint_angle_pd(
        self,
        joint: Particle,
        left: Particle,
        right: Particle,
        target_angle: float,
        stiffness: float,
        damping: float,
        dt: float,
        max_torque: float | None = None,
    ) -> tuple[float, float]:
        angle = self.joint_angle(joint, left, right)
        angular_velocity = self.joint_angular_velocity(joint, left, right)
        error = self._wrap_angle(target_angle - angle)
        torque = stiffness * error - damping * angular_velocity
        if max_torque is not None:
            torque = max(-max_torque, min(max_torque, torque))

        self._apply_pair_couple(joint, left, torque, dt)
        self._apply_pair_couple(joint, right, -torque, dt)
        return torque, error

    @staticmethod
    def _apply_pair_couple(
        p_left: Particle, p_right: Particle, torque: float, dt: float
    ) -> None:
        dx = p_right.x - p_left.x
        dy = p_right.y - p_left.y
        length = math.hypot(dx, dy)
        if length == 0:
            return

        nx = -dy / length
        ny = dx / length
        impulse = (torque / length) * dt
        p_left.apply_impulse(impulse * nx, impulse * ny)
        p_right.apply_impulse(-impulse * nx, -impulse * ny)
