"""Verified 18-point XHand kinematic anchors shared with sidecar export."""
import numpy as np
from openpi.spatial.kinematics import RobotKinematics

FINGERS = (
    ("thumb", ("right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2"), "right_hand_thumb_rota_tip"),
    ("index", ("right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2"), "right_hand_index_rota_tip"),
    ("middle", ("right_hand_mid_joint1", "right_hand_mid_joint2"), "right_hand_mid_tip"),
    ("ring", ("right_hand_ring_joint1", "right_hand_ring_joint2"), "right_hand_ring_tip"),
    ("pinky", ("right_hand_pinky_joint1", "right_hand_pinky_joint2"), "right_hand_pinky_tip"),
)
PALM_LINK = "right_hand_link"
EXTRA_PALM_REFERENCE_LINKS = ("right_hand_ee_link", "right_hand_back_link")


def select_keypoints(kin: RobotKinematics, tf: dict[str, np.ndarray]):
    """Return reproducible ordered 18-point skeleton, based on verified URDF."""
    joints = kin.joint_by_name
    hand_revolutes = {j.name for j in kin.joints if j.joint_type != "fixed" and j.name.startswith("right_hand_")}
    listed = {j for _, chain, _ in FINGERS for j in chain}
    if hand_revolutes != listed:
        raise ValueError(f"URDF hand movable topology mismatch: missing={hand_revolutes-listed}, extra={listed-hand_revolutes}")
    if PALM_LINK not in tf:
        raise KeyError(PALM_LINK)

    names = ["palm/frame_origin"]
    groups = ["palm"]
    kinds = ["reference"]
    xyz = [np.asarray(tf[PALM_LINK][:3, 3], dtype=float)]
    edges = []
    for finger, chain, tip in FINGERS:
        prev = 0
        for jname in chain:
            joint = joints[jname]
            if joint.joint_type not in ("revolute", "continuous"):
                raise ValueError(f"Expected revolute joint: {jname}")
            child = joint.child_link
            if child not in tf:
                raise KeyError(child)
            pos = np.asarray(tf[child][:3, 3], dtype=float)
            idx = len(xyz)
            names.append(jname)
            groups.append(finger)
            kinds.append("axis")
            xyz.append(pos)
            edges.append((prev, idx))
            prev = idx
        if tip not in tf or tip not in kin.links:
            raise KeyError(f"Missing tip link {tip}")
        idx = len(xyz)
        names.append(tip)
        groups.append(finger)
        kinds.append("tip_link_origin")
        xyz.append(np.asarray(tf[tip][:3, 3], dtype=float))
        edges.append((prev, idx))
    out = np.stack(xyz).astype(np.float32)
    assert len(names) == 18 and out.shape == (18, 3)
    if not np.isfinite(out).all():
        raise ValueError("Invalid skeleton coordinates")
    return names, groups, kinds, out, edges


