import math

import pytest
import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from supermimic.hoi_item import (
  decode_item,
  encode_item,
  encode_supports,
  integrate_item,
)


def _pose(xyz, angle=0.0):
  return torch.tensor(
    [*xyz, math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)], dtype=torch.float64
  )


def test_item_fields_and_roundtrip():
  root = _pose((10.0, 2.0, 1.0), math.pi / 2)
  item = torch.cat((_pose((10.0, 3.0, 0.5)), root.new_tensor([1, 0, 0, 0, 1, 0])))
  half = root.new_tensor([0.2, 0.3, 0.4])
  wrists = item[:3] + root.new_tensor([[0.0, 0.3, 0.1], [0.0, -0.3, 0.0]])
  rep = encode_item(item, root, half, wrists)
  expected = root.new_tensor(
    [
      1,
      0,
      -0.5,
      0,
      1,
      -1,
      0,
      0,
      0,
      0,
      -1,
      0,
      1,
      0,
      0,
      0.2,
      0.3,
      0.4,
      0,
      0.3,
      0.1,
      0,
      -0.3,
      0,
    ]
  )
  torch.testing.assert_close(rep, expected)
  torch.testing.assert_close(decode_item(rep, root), item)
  flipped = torch.cat((item[:3], -item[3:7], item[7:]))
  torch.testing.assert_close(encode_item(flipped, root, half, wrists), rep)
  torch.testing.assert_close(
    encode_item(item.expand(2, 3, 13), root, half, wrists), rep.expand(2, 3, 24)
  )


def test_integration_uses_old_root_and_updates_orientation():
  root = _pose((0, 0, 0), math.pi / 2)
  item = torch.cat((_pose((1, 0, 0)), root.new_zeros(6)))
  delta = root.new_tensor([1.0, 0, 0, 0, 0, 1.0]).requires_grad_()
  result = integrate_item(item, root, delta, 0.2, 0.75)
  torch.testing.assert_close(result[:3], root.new_tensor([1, 0.15, 0]))
  torch.testing.assert_close(result[7:], root.new_tensor([0, 1, 0, 0, 0, 1]))
  torch.testing.assert_close(
    matrix_from_quat(result[3:7]), matrix_from_quat(_pose((0, 0, 0), 0.15)[3:])
  )
  result.sum().backward()
  assert torch.isfinite(delta.grad).all() and torch.all(delta.grad != 0)
  zero = torch.zeros_like(delta, requires_grad=True)
  integrate_item(item, root, zero, 0.2, 0.75).sum().backward()
  assert torch.isfinite(zero.grad).all()


def test_new_root_and_wrists_do_not_move_static_box():
  root = _pose((0, 0, 1))
  new_root = _pose((1, 0, 0.9), math.pi / 2)
  item = torch.cat((_pose((0.1, 0.1, 0.4)), root.new_zeros(6)))
  wrists = root.new_tensor([[0, 0.2, 0.5], [0, -0.2, 0.5]])
  half = root.new_tensor([0.2, 0.2, 0.4])
  predicted = integrate_item(item, root, root.new_zeros(6), 0.02, 0.6)
  rep = encode_item(predicted, new_root, half, wrists)
  torch.testing.assert_close(decode_item(rep, new_root), item)
  moved = encode_item(predicted, new_root, half, wrists + root.new_tensor([0, 0, 0.2]))
  torch.testing.assert_close(moved[:18], rep[:18])
  torch.testing.assert_close(moved[18:] - rep[18:], root.new_tensor([0, 0, 0.2] * 2))


def test_shared_translation_and_yaw_preserve_all_box_features():
  from mjlab.utils.lab_api.math import quat_apply, quat_mul

  root = _pose((1.0, 0.5, 0.9), 0.2)
  item = torch.cat((_pose((1.4, 0.2, 0.4), -0.6), root.new_tensor([1, 2, 3, 4, 5, 6])))
  wrists = root.new_tensor([[1.3, 0.1, 0.4], [1.4, 0.4, 0.5]])
  half = root.new_tensor([0.1, 0.15, 0.2])
  yaw = _pose((0, 0, 0), 1.1)[3:]
  shift = root.new_tensor([5.0, -3.0, 0.0])

  def transform_pose(pose):
    return torch.cat((quat_apply(yaw, pose[:3]) + shift, quat_mul(yaw, pose[3:7])))

  moved_item = torch.cat(
    (
      transform_pose(item),
      quat_apply(yaw, item[7:10]),
      quat_apply(yaw, item[10:]),
    )
  )
  moved_wrists = quat_apply(yaw.expand(2, 4), wrists) + shift
  torch.testing.assert_close(
    encode_item(moved_item, transform_pose(root), half, moved_wrists),
    encode_item(item, root, half, wrists),
  )


@pytest.mark.parametrize("angle", [0.0, 1.2, -1.6])
def test_support_frames_for_ground_and_table(angle):
  root = _pose((2, 5, 1), angle)
  surface = torch.stack(
    (
      torch.cat((_pose((99, -99, 0), 0.75), root.new_tensor([100, 100, 0]))),
      torch.cat((_pose((2, 5, 0.5), angle), root.new_tensor([0.3, 0.4, 1]))),
    )
  )
  actual = encode_supports(root.expand(3, 7), surface).reshape(3, 2, 12)
  expected = root.new_tensor(
    [
      [0, 0, -1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, -0.5, 1, 0, 0, 1, 0, 0, 0.3, 0.4, 1],
    ]
  )
  torch.testing.assert_close(actual, expected.expand_as(actual))


@pytest.mark.parametrize("is_table", [False, True])
def test_evaluation_counts_lift_and_only_supported_release(is_table):
  from supermimic.scripts.evaluate_hoi import Thresholds, episode_metrics

  z = 0.4 if is_table else 0.0
  floor = torch.cat((_pose((0, 0, 0)), torch.zeros(3, dtype=torch.float64)))
  target = torch.cat((_pose((1, 0, z)), floor.new_tensor([0.5, 0.5, float(is_table)])))
  state = torch.cat((_pose((1, 0, z + 0.15)), floor.new_zeros(6)))
  items = state.expand(6, 3, 13).clone()
  items[:2, :, :3] = floor.new_tensor([0, 0, 0.3])
  hands = torch.zeros(6, 3, 2, dtype=torch.bool)
  hands[:2] = True
  hands[:, 1] = True  # Episode 1 keeps its hands on the box.
  on_target = torch.ones(6, 3, dtype=torch.bool)
  on_target[:, 2] = False  # Episode 2 hovers near target without support.
  result = episode_metrics(
    items,
    hands,
    on_target,
    torch.ones_like(on_target),
    floor.new_tensor([0.15] * 3),
    torch.stack((floor, target)),
    state,
    0.1,
    Thresholds(),
  )
  assert result["lifted"] == 3
  assert result["end_to_end_rate"] == pytest.approx(1 / 3)


def test_reference_contact_boundaries_are_not_physical_success():
  from supermimic.scripts.evaluate_hoi import _phases

  contacts = torch.zeros(30, 2)
  contacts[8:20] = 1
  phases = _phases(contacts)
  assert phases[[0, 8, 14, 20, 29]].tolist() == [0, 1, 2, 3, 4]
  with pytest.raises(ValueError, match="release"):
    _phases(contacts[:16])
