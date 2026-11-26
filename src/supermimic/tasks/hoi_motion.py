"""Prepare a single OmniContact Carry Box clip for the robot and the free box.

Example with the Sim2Sim carrybox_motion_4_with_contact.npz clip::

  python -m supermimic.tasks.hoi_motion raw.npz carry.npz --box-size .30 .30 .30

Lengths are metres. Sim2Sim table1/2_pos_w are centres of horizontal slabs
with 0.01 m half-thickness and 0.20 m half-widths (omnicontact_carry_box.xml).
Explicit table overrides specify top centre XYZ, yaw (rad) and half-widths XY.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _joint_names():
  legs = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
  arms = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
  )
  names = [f"{s}_{j}_joint" for s in ("left", "right") for j in legs]
  names += [f"waist_{j}_joint" for j in ("yaw", "roll", "pitch")]
  names += [f"{s}_{j}_joint" for s in ("left", "right") for j in arms]
  return names


# OmniContact Sim2Sim, policy/omnicontact/config/OmniContact.yaml: lab2mj.
_LAB_TO_MJ = [
  0,
  3,
  6,
  9,
  13,
  17,
  1,
  4,
  7,
  10,
  14,
  18,
  2,
  5,
  8,
  11,
  15,
  19,
  21,
  23,
  25,
  27,
  12,
  16,
  20,
  22,
  24,
  26,
  28,
]


def _array(data, key, shape):
  a = np.asarray(data[key], dtype=np.float64)
  if a.shape != shape or not np.isfinite(a).all():
    raise ValueError(f"{key}: expected finite array of shape {shape}, got {a.shape}")
  return a


def _support(data, key, override):
  if override is None and key not in data:
    # No table: infinite level ground; its XY origin carries no task target.
    return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  if override is None:
    centers = _array(data, key, (len(data["joint_pos"]), 3))
    if not np.allclose(centers, centers[:1], atol=1e-3, rtol=0):
      raise ValueError(f"{key} moves; a single fixed support surface is required")
    x, y, z = centers[0]
    z += 0.01
    if z <= 0.001:
      return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    yaw, hx, hy = 0.0, 0.20, 0.20
  else:
    x, y, z, yaw, hx, hy = override
    if not np.isfinite(override).all() or min(hx, hy, z) <= 0:
      raise ValueError("A finite table needs a positive top height and half-widths")
  return [x, y, z, np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2), hx, hy, 1.0]


def prepare(
  source, output, box_size=None, mass_kg=2.0, start_table=None, goal_table=None
):
  """Write one synchronized 50 Hz NPZ. A named joint array takes precedence."""
  with np.load(source, allow_pickle=False) as data:
    length = len(data["joint_pos"])
    fps = float(data["fps"].item())
    if length < 2 or not np.isfinite(fps) or fps <= 0:
      raise ValueError("Expected at least two samples at a positive frame rate")
    t = np.arange(length) / fps
    target_t = np.arange(int(np.floor(t[-1] * 50 + 1e-8)) + 1) / 50
    target_t = np.minimum(target_t, t[-1])
    if len(target_t) < 2:
      raise ValueError("Clip is too short for the 50 Hz control rate")

    def linear(values):
      return np.stack([np.interp(target_t, t, col) for col in values.T], axis=-1)

    def root_field(suffix, width):
      key = "base_" + suffix
      if key in data:
        return _array(data, key, (length, width))
      # Some released augmentation clips only carry the 39-body arrays.
      return _array(data, "body_" + suffix, (length, 39, width))[:, 0]

    def rotations(key, quat):
      if np.any(np.linalg.norm(quat, axis=-1) < 1e-8):
        raise ValueError(f"{key} contains a zero quaternion")
      r = Slerp(t, Rotation.from_quat(quat[:, [1, 2, 3, 0]]))(target_t)
      return r, r.as_quat()[:, [3, 0, 1, 2]]

    def velocity(key, position):
      if key in data:
        return linear(_array(data, key, (length, position.shape[1])))
      return np.gradient(position, target_t, axis=0)

    def angular_velocity(key, orientations):
      if key in data:
        return linear(_array(data, key, (length, 3)))
      omega = (orientations[1:] * orientations[:-1].inv()).as_rotvec() * 50
      return np.concatenate((omega[:1], 0.5 * (omega[1:] + omega[:-1]), omega[-1:]))

    joint_names = _joint_names()
    if "joint_names" in data:
      names = data["joint_names"].astype(str).tolist()
      if sorted(names) != sorted(joint_names):
        raise ValueError("Expected the 29 unique named G1 joints")
      indices = [names.index(name) for name in joint_names]
    else:
      indices = _LAB_TO_MJ
    joint_lab = linear(_array(data, "joint_pos", (length, 29)))
    joint_vel = velocity("joint_vel", joint_lab)[:, indices]
    root_pos = linear(root_field("pos_w", 3))
    root_rot, root_quat = rotations("base_quat_w", root_field("quat_w", 4))
    item_pos = linear(_array(data, "object_pos_w", (length, 3)))
    item_rot, item_quat = rotations(
      "object_quat_w", _array(data, "object_quat_w", (length, 4))
    )
    item = np.concatenate(
      (
        item_pos,
        item_quat,
        velocity("object_lin_vel_w", item_pos),
        angular_velocity("object_ang_vel_w", item_rot),
      ),
      axis=-1,
    )
    labels = _array(data, "contact_info", (length, 4, 1))
    if not np.isin(labels, [0, 1]).all():
      raise ValueError("contact_info must contain binary ankle/wrist targets")
    contacts = labels[np.rint(target_t * fps).astype(int), 2:4, 0]
    if box_size is None:
      if "carry_box_selected_size_cm" not in data:
        raise ValueError("Provide --box-size X Y Z in metres for this capture")
      box_size = np.asarray(data["carry_box_selected_size_cm"]).reshape(-1) * 0.01
    half_size = np.asarray(box_size) / 2
    if (
      half_size.shape != (3,)
      or not np.isfinite(half_size).all()
      or np.any(half_size <= 0)
    ):
      raise ValueError("Box size must be three positive lengths")
    if not np.isfinite(mass_kg) or mass_kg <= 0:
      raise ValueError("Box mass must be positive")
    supports = [
      _support(data, "table1_pos_w", start_table),
      _support(data, "table2_pos_w", goal_table),
    ]
    for frame, role, surface in ((0, "start", supports[0]), (-1, "goal", supports[1])):
      if contacts[frame].any() or np.linalg.norm(item[frame, 7:10]) > 0.1:
        continue
      bottom = item_pos[frame, 2] - np.abs(item_rot[frame].as_matrix()[2]) @ half_size
      if not surface[-1] and abs(bottom) > 0.035:
        raise ValueError(
          f"Unheld {role} box bottom is {bottom:.3f} m above scene zero; "
          f"check box axes/size and provide --{role}-table for elevated support"
        )
    values = dict(
      joint_pos=joint_lab[:, indices],
      joint_vel=joint_vel,
      root_pos_w=root_pos,
      root_quat_w=root_quat,
      root_lin_vel_w=(
        linear(root_field("lin_vel_w", 3))
        if "body_lin_vel_w" in data or "base_lin_vel_w" in data
        else velocity("base_lin_vel_w", root_pos)
      ),
      root_ang_vel_w=(
        linear(root_field("ang_vel_w", 3))
        if "body_ang_vel_w" in data or "base_ang_vel_w" in data
        else angular_velocity("base_ang_vel_w", root_rot)
      ),
      item_state_w=item,
      wrist_contacts=contacts,
      half_size=half_size,
      mass_kg=mass_kg,
      supports_w=supports,
      fps=50.0,
    )
  # Exclusive creation also protects the source when the two paths coincide.
  with Path(output).open("xb") as stream:
    np.savez_compressed(
      stream,
      joint_names=np.asarray(joint_names),
      **{k: np.asarray(v, dtype=np.float32) for k, v in values.items()},
    )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--box-size", nargs=3, type=float, metavar=("X", "Y", "Z"))
  parser.add_argument("--mass-kg", type=float, default=2.0)
  for arg in ("--start-table", "--goal-table"):
    parser.add_argument(
      arg, nargs=6, type=float, metavar=("X", "Y", "TOP_Z", "YAW", "HX", "HY")
    )
  args = parser.parse_args()
  prepare(**vars(args))


if __name__ == "__main__":
  main()
