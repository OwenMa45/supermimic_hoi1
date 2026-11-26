"""Actor, dynamics and reference-window contracts of single-box GenCoord."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_apply

import supermimic.representations as robot_repr
from supermimic.hoi_representation import (
  HoiGenCoordRepresentation,
  item_goal_window,
  item_tracking_reward,
  merge_item_goal,
)


def two_wrist_fk(root, quat, joints, root_vel, root_omega, joint_vel):
  """Two moving wrist links, keeping both the joint and root gradient paths."""
  zero = torch.zeros_like(joints[..., :2])
  offsets = torch.stack(
    (joints[..., :2], zero + root.new_tensor([0.2, -0.2]), zero), -1
  )
  rotations = quat.unsqueeze(-2).expand(*offsets.shape[:-1], 4)
  positions = root.unsqueeze(-2) + quat_apply(rotations, offsets)
  return positions, rotations, torch.zeros_like(positions), torch.zeros_like(positions)


def make_rep(use_wrists):
  rep = HoiGenCoordRepresentation(29, torch.zeros(29), use_wrists)
  rep.dt = 0.02
  rep.substep_coeff = 0.75
  rep.fk = two_wrist_fk
  rep.wrist_indices = [0, 1]
  rep.half_size = torch.tensor([0.1, 0.2, 0.3])
  rep.supports_w = torch.zeros(2, 10)
  rep.supports_w[:, 3] = 1.0
  return rep


def full_state(*batch):
  state = torch.zeros(*batch, 84)
  state[..., 2] = 0.8
  state[..., 3] = 1.0
  state[..., 71:74] = torch.tensor([0.2, 0, 0.4])
  state[..., 74] = 1.0
  return state


@pytest.mark.parametrize("use_wrists,item_dim", [(False, 18), (True, 24)])
def test_replay_rollout_loss_and_proprio(use_wrists, item_dim):
  rep = make_rep(use_wrists)
  full = full_state(2, 3)
  aux = rep.extract_aux_state(full)
  obs = rep.represent(full)
  assert obs.shape == (2, 3, 68 + item_dim + 24)
  assert aux.shape == (2, 3, 20)
  assert (rep.full_state_dim, rep.prediction_dim) == (84, 41)
  torch.testing.assert_close(rep.reconstruct_full_state(aux, obs), full)

  delta = torch.zeros(2, 3, rep.prediction_dim)
  delta[..., 6] = 1.0  # Left arm joint velocity increment.
  delta[..., 35] = 0.5  # Box gains world x velocity under the identity root.
  delta[..., 40] = 0.1
  delta.requires_grad_()
  next_aux, next_rep = rep.apply_prediction(aux, obs, delta, rep.dt)
  replayed = rep.represent(rep.reconstruct_full_state(next_aux, next_rep))
  torch.testing.assert_close(replayed, next_rep)
  if use_wrists:
    assert not torch.allclose(next_rep[..., 86:92], obs[..., 86:92])
    wrist_grad = torch.autograd.grad(next_rep[..., 86].sum(), delta, retain_graph=True)[
      0
    ]
    assert (wrist_grad[..., 6].abs() > 0).all()
  blocks = rep.loss_blocks(next_rep, next_aux)
  assert list(blocks) == list(rep.LOSS_WEIGHT_CONFIG)
  assert blocks["item_rot"].shape == (2, 3, 9)
  target = full[..., 71:].clone()
  target[..., 0] += 0.1
  reward = item_tracking_reward(next_aux[..., 7:], target).sum()
  grad = torch.autograd.grad(reward, delta)[0]
  assert torch.isfinite(grad).all()
  assert (grad[..., 35] > 0).all()
  assert (grad[..., 40] < 0).all()

  rep.actor_obs_mode = "proprio"
  joint_state = rep.joint_state(next_rep)
  action = torch.zeros_like(joint_state[..., :29])
  ctx = action[..., :0]
  sensor = rep.compute_actor_sensor(next_rep, ctx, next_aux, joint_state, action)
  assert sensor.shape[-1] == rep.proprio_obs_dim(29)
  torch.testing.assert_close(sensor[..., 93:], next_rep[..., 68:])
  torch.testing.assert_close(
    rep.compute_actor_sensor(next_rep, ctx, sim_actor_obs=sensor), sensor
  )
  assert not rep.proprio_noise_bounds(29, "cpu")[93:].any()


def test_reference_window_layout_clip_boundaries_and_actor_parity(monkeypatch):
  monkeypatch.setattr(robot_repr, "_compiled_assemble_goal", robot_repr._assemble_goal)
  rep = make_rep(True)
  # In each clip, the final frame is repeated as soon as the stride reaches it.
  frames = 8
  item = full_state(frames)[..., 71:].clone()
  item[:, 0] = torch.arange(frames)
  flags = torch.stack((torch.arange(frames) % 2, torch.arange(frames) // 4), -1).float()
  motion = SimpleNamespace(
    motion_ends=torch.tensor([4, 8]),
    item_state_w=item,
    wrist_contacts=flags,
    root_pos_w=torch.zeros(frames, 3),
    root_quat_w=full_state(frames)[:, 3:7],
    joint_pos=torch.zeros(frames, 29),
    joint_vel=torch.zeros(frames, 29),
  )
  state = full_state(2)
  state[:, 0] = 1.5
  aux = rep.extract_aux_state(state)
  times = torch.tensor([2, 4])
  reference = item_goal_window(motion, times, aux[..., :7])
  per_frame = reference.reshape(2, 10, 11)
  indices = torch.tensor([[2, *([3] * 9)], [4, *([7] * 9)]])
  torch.testing.assert_close(per_frame[..., 0], item[indices, 0] - 1.5)
  torch.testing.assert_close(per_frame[..., 9:], flags[indices])
  robot_goal, _ = robot_repr.GenCoordRepresentation.compute_wm_goal(
    rep, motion, times, aux
  )
  real_goal = merge_item_goal(robot_goal, reference)
  wm_goal, actor_goal = rep.compute_wm_goal(motion, times, aux)
  torch.testing.assert_close(real_goal, wm_goal)
  torch.testing.assert_close(robot_repr.extract_actor_goal(real_goal), actor_goal)
  assert (wm_goal.shape[-1], actor_goal.shape[-1]) == (780, 750)
