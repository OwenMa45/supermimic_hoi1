"""OmniContact conversion, including data variants used in the Carry Box task."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from supermimic.tasks.hoi_motion import prepare


def _clip():
  # 0.2 s at 90 Hz maps to 11 control samples including both endpoints.
  t = np.arange(19) / 90
  pose = np.stack((0.5 * t, -t, np.full_like(t, 0.15)), axis=-1)
  quat = Rotation.from_euler("z", 2 * t[:, None]).as_quat()[:, [3, 0, 1, 2]]
  labels = np.zeros((len(t), 4, 1), dtype=int)
  labels[:, :2] = 1
  labels[3:9, 2] = 1
  labels[9:15, 3] = 1
  return {
    "joint_pos": np.arange(29)[None] + t[:, None],
    "body_pos_w": np.repeat((pose + [0, 0, 0.65])[:, None], 39, axis=1),
    "body_quat_w": np.repeat(quat[:, None], 39, axis=1),
    "fps": np.array([90]),
    "object_pos_w": pose,
    "object_quat_w": quat,
    "contact_info": labels,
    "carry_box_selected_size_cm": [30, 30, 30],
  }


def _convert(tmp_path, clip, **options):
  source, output = tmp_path / "raw.npz", tmp_path / "prepared.npz"
  np.savez(source, **clip)
  prepare(source, output, **options)
  return np.load(output, allow_pickle=False)


def test_shared_time_grid_and_contact_order(tmp_path):
  with _convert(tmp_path, _clip()) as prepared:
    t = np.arange(11) / 50
    assert prepared["fps"].item() == 50
    item = prepared["item_state_w"]
    assert item.shape == (11, 13)
    np.testing.assert_allclose(item[:, 0], 0.5 * t, atol=1e-6)
    np.testing.assert_allclose(item[:, 1], -t, atol=1e-6)
    np.testing.assert_allclose(prepared["root_pos_w"][:, 0], item[:, 0])
    np.testing.assert_allclose(item[:, 12], 2.0, atol=1e-5)
    orientation = Rotation.from_quat(item[:, [4, 5, 6, 3]])
    np.testing.assert_allclose(orientation.as_rotvec()[:, 2], 2 * t, atol=1e-6)
    np.testing.assert_array_equal(
      prepared["wrist_contacts"][:, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
      prepared["wrist_contacts"][:, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    )
    np.testing.assert_allclose(prepared["half_size"], [0.15, 0.15, 0.15])
    np.testing.assert_array_equal(prepared["supports_w"][:, 9], [0, 0])
    names = prepared["joint_names"].tolist()
    for name, lab_idx in (
      ("left_knee_joint", 9),
      ("right_hip_pitch_joint", 1),
      ("waist_pitch_joint", 8),
      ("right_wrist_yaw_joint", 28),
    ):
      i = names.index(name)
      np.testing.assert_allclose(prepared["joint_pos"][:, i], lab_idx + t, atol=2e-6)
      np.testing.assert_allclose(prepared["joint_vel"][:, i], 1.0)


def test_named_source_joints_override_default_permutation(tmp_path):
  clip = _clip()
  canonical = (
    [
      f"{s}_{j}_joint"
      for s in ("left", "right")
      for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [
      f"{s}_{j}_joint"
      for s in ("left", "right")
      for j in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
      )
    ]
  )
  clip["joint_names"] = canonical[::-1]
  clip["joint_vel"] = np.broadcast_to(np.arange(29), (19, 29))
  with _convert(tmp_path, clip) as prepared:
    assert prepared["joint_names"].tolist() == canonical
    np.testing.assert_array_equal(prepared["joint_pos"][0], np.arange(28, -1, -1))
    np.testing.assert_array_equal(prepared["joint_vel"][-1], np.arange(28, -1, -1))


@pytest.mark.parametrize("role", ["start", "goal"])
def test_missing_stationary_box_support(tmp_path, role):
  clip = _clip()
  clip["object_pos_w"][:] = [0.0, 0.0, 0.45]
  clip["contact_info"][:] = 0
  # Provide the opposite support so we test both endpoints independently.
  other_role = "goal" if role == "start" else "start"
  with pytest.raises(ValueError, match=f"--{role}-table"):
    _convert(tmp_path, clip, **{other_role + "_table": (0, 0, 0.3, 0, 0.2, 0.2)})


def test_unsupported_moving_table_and_explicit_top(tmp_path):
  clip = _clip()
  centers = np.zeros((19, 3))
  centers[:, 0] = np.arange(19) / 100
  clip["table1_pos_w"] = centers
  with pytest.raises(ValueError, match="moves"):
    _convert(tmp_path, clip)
  with pytest.raises(ValueError, match="positive"):
    _convert(tmp_path, clip, start_table=(0, 0, 0.20, 0, -0.1, 0.2))
  with _convert(tmp_path, clip, start_table=(0.1, 0.3, 0.20, 0, 0.16, 0.2)) as result:
    np.testing.assert_allclose(
      result["supports_w"][0], [0.1, 0.3, 0.2, 1, 0, 0, 0, 0.16, 0.2, 1]
    )


def test_recorded_table_is_slab_center(tmp_path):
  clip = _clip()
  clip["table1_pos_w"] = np.tile([0.2, -0.1, 0.26], (19, 1))
  clip["table2_pos_w"] = np.tile([0.4, 0.3, -0.01], (19, 1))
  with _convert(tmp_path, clip) as result:
    np.testing.assert_allclose(result["supports_w"][0, :3], [0.2, -0.1, 0.27])
    assert result["supports_w"][0, 9] == 1
    assert result["supports_w"][1, 9] == 0
