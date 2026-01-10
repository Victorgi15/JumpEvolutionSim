from creature.humanoid import HumanoidCreature
from physics.engine import World


def test_humanoid_joint_moves_toward_target():
    w = World()
    w.dt = 1 / 240.0
    h = HumanoidCreature({}, w, base_x=0.0)
    # pick left hip joint
    idx = next(i for i, j in enumerate(h.joints) if j["name"] == "hip_left")
    jdef = h.joints[idx]
    rest = h.joint_params[idx]["rest_angle"]
    target = rest - 0.5
    h.joint_params[idx]["target_angle"] = target
    h.joint_params[idx]["amp"] = 0.0
    # initial error
    joint = h.particles[jdef["joint"]]
    left = h.particles[jdef["left"]]
    right = h.particles[jdef["right"]]
    err0 = abs(w._wrap_angle(target - w.joint_angle(joint, left, right)))
    steps = int(0.5 / w.dt)
    t = 0.0
    for _ in range(steps):
        h.step_controller(t, w.dt)
        h.step_actuators(t, w.dt)
        w.step(w.dt)
        t += w.dt
    err1 = abs(w._wrap_angle(target - w.joint_angle(joint, left, right)))
    assert err1 < err0, f"Angle error did not decrease: {err0} -> {err1}"


if __name__ == "__main__":
    test_humanoid_joint_moves_toward_target()
    print("OK")
