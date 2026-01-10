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
        self, p1: Particle, p2: Particle, target_length: float, stiffness: float = 1.0
    ):
        self.p1 = p1
        self.p2 = p2
        self.target_length = target_length
        self.stiffness = stiffness

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
        # density used to compute mass from link lengths
        self.linear_density = 1.0

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
        self.Movement(dt)

        # velocity-level link constraint to avoid stretching during integration
        self.Link()

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

    def Link(self, compliance: float = 0.0):
        """Velocity-level constraint: remove relative velocity along link direction to avoid length change.

        This projects out the component of relative velocity along the constraint direction
        so that the constraint length does not change due to velocities. The optional
        compliance allows partial correction (0.0 = full correction).
        """
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
            # compute impulse (velocity correction) to cancel that component
            w1 = p1.inv_mass
            w2 = p2.inv_mass
            wsum = w1 + w2
            if wsum == 0:
                continue
            # desired change in relative velocity to enforce zero along normal
            dv = -v_rel * (1.0 - compliance)
            # convert dv to impulses for each particle: J = dv / (w1+w2)
            j = dv / wsum
            jx = j * nx
            jy = j * ny
            # apply opposite impulses
            if p1.inv_mass != 0:
                p1.vx -= jx * (1.0 / p1.inv_mass)
                p1.vy -= jy * (1.0 / p1.inv_mass)
            if p2.inv_mass != 0:
                p2.vx += jx * (1.0 / p2.inv_mass)
                p2.vy += jy * (1.0 / p2.inv_mass)

    def Contract(
        self, top: Particle, left: Particle, right: Particle, force: float, dt: float
    ):
        """Apply a muscle contraction that acts on three points (two lower endpoints and a top).

        The muscle applies equal impulses to `left` and `right` directing them towards their
        centroid and slightly downward; the top receives the opposite impulse (conserving momentum).
        `force` is peak force (N); impulses applied are force * dt.
        """
        # centroid of left and right
        cx = 0.5 * (left.x + right.x)
        cy = 0.5 * (left.y + right.y)
        impulses = []
        total_ix = 0.0
        total_iy = 0.0
        for p in (left, right):
            # direction from point to centroid
            dx = cx - p.x
            dy = cy - p.y
            # encourage downward motion too
            dy -= abs(0.2 * (p.y - top.y))
            norm = math.hypot(dx, dy)
            if norm == 0:
                ux, uy = 0.0, -1.0
            else:
                ux = dx / norm
                uy = dy / norm
            imp_mag = force * dt * 0.5  # split half to each side
            ix = ux * imp_mag
            iy = uy * imp_mag
            impulses.append((ix, iy))
            total_ix += ix
            total_iy += iy

        # apply impulses to left and right
        for p, (ix, iy) in zip((left, right), impulses):
            p.apply_impulse(ix, iy)

        # apply opposite impulse to top to conserve momentum
        top.apply_impulse(-total_ix, -total_iy)

    def SetLength(self, constraint: DistanceConstraint, new_length: float):
        """Set target length for a constraint and update masses of attached particles based on
        adjacent link lengths (simple linear density model)."""
        constraint.target_length = new_length
        # recompute masses for particles: sum of adjacent constraint lengths * density
        # build adjacency
        adj = {p: [] for p in self.particles}
        for c in self.constraints:
            adj[c.p1].append(c)
            adj[c.p2].append(c)
        for p, neigh in adj.items():
            if p.inv_mass == 0:
                continue
            total_length = 0.0
            for c in neigh:
                # use the constraint target length as segment length
                total_length += c.target_length
            mass = max(1e-6, total_length * self.linear_density)
            p.inv_mass = 1.0 / mass
