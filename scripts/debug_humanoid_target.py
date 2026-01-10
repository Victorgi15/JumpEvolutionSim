import sys, math

sys.path.insert(0, ".")
from physics.engine import World
from creature.humanoid import HumanoidCreature

w = World()
w.dt = 1 / 240.0
h = HumanoidCreature({}, w, base_x=0.0)
h.muscle_edges = []
tr = (0.8, 0.6)
h.set_pose_sequence([{"duration": 1.0, "targets": {9: tr}}])
# compute target world
pel = h.particles[0]
tx, ty = pel.x + tr[0], pel.y + tr[1]
print("target", tx, ty)
for i in range(1, 121):
    t = (i - 1) * w.dt
    h.step_controller(t, w.dt)
    h.step_actuators(t, w.dt)
    w.step(w.dt)
    p = h.particles[9]
    d = math.hypot(p.x - tx, p.y - ty)
    if i % 5 == 0:
        print(
            f"step {i}, t={i*w.dt:.3f}, dist={d:.4f}, pos=({p.x:.3f},{p.y:.3f}), vel=({p.vx:.3f},{p.vy:.3f})"
        )
