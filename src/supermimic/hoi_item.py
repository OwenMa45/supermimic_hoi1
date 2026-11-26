"""Carry Box rigid-body state and support geometry for SuperMimic.

Coordinates are in meters, seconds and radians. Quaternions use wxyz. The
box frame is centered on its geometric center and aligned with its edges.
Linear velocity is measured at that center. Pelvis orientation includes
roll and pitch. All functions preserve leading batch or trajectory axes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from mjlab.utils.lab_api.math import matrix_from_quat

from supermimic.utils.helpers import (
  matrix_to_quat,
  quat_integrate,
  rot6d_to_matrix,
  small_matmul,
)


def _rotate(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
  return (rotation * vector.unsqueeze(-2)).sum(-1)


def encode_item(
  item_w: torch.Tensor,
  root_pose_w: torch.Tensor,
  half_size: torch.Tensor,
  wrists_w: torch.Tensor,
) -> torch.Tensor:
  """Encode a world rigid body (13), pelvis pose (7), box half sizes (3)
  and left/right wrist positions (..., 2, 3) as one flat 24-vector.

  Fields are pelvis-relative center and rotation, absolute linear/angular
  velocities expressed in the pelvis frame, box half sizes, and left then
  right wrist coordinates in the box frame. A constant half_size broadcasts.
  """
  if item_w.shape[-1] != 13 or root_pose_w.shape[-1] != 7:
    raise ValueError("Expected world box state (..., 13) and pelvis pose (..., 7)")
  if half_size.shape[-1] != 3 or wrists_w.shape[-2:] != (2, 3):
    raise ValueError("Expected half sizes (..., 3) and left/right wrists (..., 2, 3)")
  batch = torch.broadcast_shapes(
    item_w.shape[:-1],
    root_pose_w.shape[:-1],
    half_size.shape[:-1],
    wrists_w.shape[:-2],
  )
  item_w = item_w.expand(*batch, 13)
  world_to_root = matrix_from_quat(root_pose_w[..., 3:7]).transpose(-2, -1)
  box_to_world = matrix_from_quat(item_w[..., 3:7])
  box_to_root = small_matmul(world_to_root, box_to_world)
  wrists_box = _rotate(
    box_to_world.transpose(-2, -1).unsqueeze(-3), wrists_w - item_w[..., None, :3]
  )
  return torch.cat(
    (
      _rotate(world_to_root, item_w[..., :3] - root_pose_w[..., :3]),
      box_to_root[..., :2].flatten(-2),
      _rotate(world_to_root, item_w[..., 7:10]),
      _rotate(world_to_root, item_w[..., 10:13]),
      half_size.expand(*batch, 3),
      wrists_box.flatten(-2),
    ),
    dim=-1,
  )


def decode_item(item_rep: torch.Tensor, root_pose_w: torch.Tensor) -> torch.Tensor:
  """Recover [world center, wxyz, world linear velocity, world angular velocity]."""
  if item_rep.shape[-1] != 24 or root_pose_w.shape[-1] != 7:
    raise ValueError("Expected encoded item (..., 24) and pelvis pose (..., 7)")
  root_to_world = matrix_from_quat(root_pose_w[..., 3:7])
  box_to_world = small_matmul(root_to_world, rot6d_to_matrix(item_rep[..., 3:9]))
  return torch.cat(
    (
      root_pose_w[..., :3] + _rotate(root_to_world, item_rep[..., :3]),
      matrix_to_quat(box_to_world),
      _rotate(root_to_world, item_rep[..., 9:12]),
      _rotate(root_to_world, item_rep[..., 12:15]),
    ),
    dim=-1,
  )


def integrate_item(
  item_w: torch.Tensor,
  root_pose_w: torch.Tensor,
  delta_root: torch.Tensor,
  dt: float,
  substep_coeff: float,
) -> torch.Tensor:
  """Advance the world box state with one predicted [Δv, Δω] in the old pelvis frame.

  Deltas cover the entire control step. The pose update uses the same
  substep coefficient as the human representation, (decimation + 1)/(2*decimation).
  Re-encode the result with the new pelvis and new wrists for the next step.
  """
  if (item_w.shape[-1], root_pose_w.shape[-1], delta_root.shape[-1]) != (13, 7, 6):
    raise ValueError("Expected box (..., 13), pelvis (..., 7), velocity delta (..., 6)")
  root_to_world = matrix_from_quat(root_pose_w[..., 3:7])
  dv = _rotate(root_to_world, delta_root[..., :3])
  dw = _rotate(root_to_world, delta_root[..., 3:])
  v, w = item_w[..., 7:10], item_w[..., 10:13]
  next_pos = item_w[..., :3] + dt * (v + substep_coeff * dv)
  quat = item_w[..., 3:7].expand(*next_pos.shape[:-1], 4)
  next_quat = quat_integrate(quat, w + substep_coeff * dw, dt)
  return torch.cat((next_pos, next_quat, v + dv, w + dw), dim=-1)


def _horizontal_frame(root_to_world: torch.Tensor) -> torch.Tensor:
  """Level frame following pelvis heading, with a fallback when pelvis x is vertical."""
  forward = root_to_world[..., :2, 0]
  left = root_to_world[..., :2, 1]
  from_left = torch.stack((left[..., 1], -left[..., 0]), dim=-1)
  heading = F.normalize(
    torch.where(forward.square().sum(-1, keepdim=True) > 1e-8, forward, from_left),
    dim=-1,
  )
  x, y = heading.unbind(-1)
  zero, one = torch.zeros_like(x), torch.ones_like(x)
  return torch.stack((x, -y, zero, y, x, zero, zero, zero, one), dim=-1).unflatten(
    -1, (3, 3)
  )


def encode_supports(
  root_pose_w: torch.Tensor, supports_w: torch.Tensor
) -> torch.Tensor:
  """Encode source and target support top faces relative to the current pelvis.

  supports_w has shape (..., 2, 10), each row [center(3), wxyz(4),
  half extents along face x/y(2), finite(1)]. A finite face uses its given
  axes; its local z points out of the top. An infinite horizontal floor uses
  only center.z and is represented directly below the pelvis at its heading.
  Output rows [center(3), rotation6d(6), half extents(2), finite(1)] are flattened.
  """
  if root_pose_w.shape[-1] != 7 or supports_w.shape[-2:] != (2, 10):
    raise ValueError("Expected pelvis (..., 7) and source/target supports (..., 2, 10)")
  batch = torch.broadcast_shapes(root_pose_w.shape[:-1], supports_w.shape[:-2])
  supports_w = supports_w.expand(*batch, 2, 10)
  root_pos = root_pose_w[..., None, :3]
  root_to_world = matrix_from_quat(root_pose_w[..., 3:7])
  world_to_root = root_to_world.transpose(-2, -1).unsqueeze(-3)
  finite = supports_w[..., 9:10].bool()
  offset = supports_w[..., :3] - root_pos
  floor_offset = torch.cat((torch.zeros_like(offset[..., :2]), offset[..., 2:3]), -1)
  center = _rotate(world_to_root, torch.where(finite, offset, floor_offset))
  # The floor row may have unused zero quaternion placeholders.
  safe_quat = torch.where(finite, supports_w[..., 3:7], root_pose_w[..., None, 3:7])
  axes_w = torch.where(
    finite.unsqueeze(-1),
    matrix_from_quat(safe_quat),
    _horizontal_frame(root_to_world).unsqueeze(-3),
  )
  rotation6d = small_matmul(world_to_root, axes_w)[..., :2].flatten(-2)
  extents = torch.where(finite, supports_w[..., 7:9], 0.0)
  return torch.cat((center, rotation6d, extents, finite.to(center.dtype)), -1).flatten(
    -2
  )
