"""G1 generalized coordinates with one free rigid box and two fixed supports."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from supermimic.hoi_item import encode_item, encode_supports, integrate_item
from supermimic.representations import (
  GOAL_REL_ORI_DIM,
  GOAL_REL_POS_DIM,
  PROPRIO_NOISE_HALF_WIDTHS,
  GenCoordRepresentation,
  assemble_anchor_window,
  extract_actor_goal,
  goal_window_indices,
)
from supermimic.utils.gencoord_helpers import compute_gencoord_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp.commands import MotionLoader

  from supermimic.sim_context import SimContext


def item_goal_window(
  motion: MotionLoader, motion_times: torch.Tensor, root_pose_w: torch.Tensor
) -> torch.Tensor:
  """Pack each reference frame as [box center(3), orientation(6), wrist flags(2)]."""
  indices = goal_window_indices(motion, motion_times)
  target = motion.item_state_w[indices]
  pos, ori = assemble_anchor_window(
    root_pose_w[..., :3], root_pose_w[..., 3:7], target[..., :3], target[..., 3:7]
  )
  shape = indices.shape
  frames = torch.cat(
    (pos.reshape(*shape, 3), ori.reshape(*shape, 6), motion.wrist_contacts[indices]),
    dim=-1,
  )
  return frames.flatten(-2)


def merge_item_goal(
  robot_wm_goal: torch.Tensor, item_goal: torch.Tensor
) -> torch.Tensor:
  """Keep the robot's relative anchor fields last for the existing actor extractor."""
  tail = GOAL_REL_POS_DIM + GOAL_REL_ORI_DIM
  return torch.cat(
    (robot_wm_goal[..., :-tail], item_goal, robot_wm_goal[..., -tail:]), -1
  )


def item_tracking_reward(item_w: torch.Tensor, target_w: torch.Tensor) -> torch.Tensor:
  """Unit-scale box pose reward. The caller applies the environment time step.

  For unit quaternions the squared chordal rotation error 4 sin²(theta/2)
  agrees locally with squared radians and is smooth at a perfect match.
  """
  pos_sq = (item_w[..., :3] - target_w[..., :3]).square().sum(-1)
  alignment = (item_w[..., 3:7] * target_w[..., 3:7]).sum(-1)
  rot_sq = 4.0 * (1.0 - alignment.square()).clamp_min(0.0)
  return torch.exp(-pos_sq / 0.15**2 - rot_sq / 0.5**2)


class HoiGenCoordRepresentation(GenCoordRepresentation):
  """Physical replay [human, item13], local observation [human, item, supports].

  The auxiliary integrator state carries the world root pose and physical box
  state. Kinematic and scene observations are regenerated at every rollout step.
  """

  LOSS_WEIGHT_CONFIG = {
    **GenCoordRepresentation.LOSS_WEIGHT_CONFIG,
    "item_pos": "nominal_error_body_pos",
    "item_rot": "nominal_error_body_rot",
    "item_vel": "nominal_error_body_vel",
    "item_avel": "nominal_error_body_avel",
  }

  def __init__(
    self,
    num_joints: int,
    default_joint_pos: torch.Tensor,
    use_wrist_features: bool = True,
  ):
    super().__init__(num_joints, default_joint_pos)
    self.robot_full_dim = self.full_state_dim
    self.robot_rep_dim = self.rep_state_dim
    self.robot_prediction_dim = self.prediction_dim
    self.item_dim = 24 if use_wrist_features else 18
    self.full_state_dim += 13
    self.rep_state_dim += self.item_dim + 24
    self.prediction_dim += 6

  def bind(self, env: ManagerBasedRlEnv, sim_context: SimContext) -> None:
    super().bind(env, sim_context)
    command = env.command_manager.get_term("motion")
    self.register_buffer("half_size", command.motion.half_size, persistent=False)
    self.register_buffer("supports_w", command.motion.supports_w, persistent=False)
    self.wrist_indices = [
      command.cfg.body_names.index(f"{side}_wrist_yaw_link")
      for side in ("left", "right")
    ]

  def _encode_state(
    self, robot_rep: torch.Tensor, root: torch.Tensor, item: torch.Tensor
  ) -> torch.Tensor:
    pos, _, _, _ = super().world_frame_bodies(root, robot_rep)
    wrists = pos[..., self.wrist_indices, :]
    item_rep = encode_item(item, root, self.half_size, wrists)
    return torch.cat(
      (
        robot_rep,
        item_rep[..., : self.item_dim],
        encode_supports(root, self.supports_w),
      ),
      dim=-1,
    )

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    robot, item = full_state.split((self.robot_full_dim, 13), dim=-1)
    robot_rep = compute_gencoord_state(robot.reshape(-1, self.robot_full_dim))
    return self._encode_state(
      robot_rep.reshape(*robot.shape[:-1], self.robot_rep_dim), robot[..., :7], item
    )

  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    return torch.cat((full_state[..., :7], full_state[..., self.robot_full_dim :]), -1)

  def apply_prediction(
    self,
    aux_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    root, item = aux_state.split((7, 13), dim=-1)
    robot_delta, item_delta = prediction.split((self.robot_prediction_dim, 6), dim=-1)
    next_root, next_robot_rep = super().apply_prediction(
      root, curr_rep_state[..., : self.robot_rep_dim], robot_delta, dt
    )
    next_item = integrate_item(item, root, item_delta, dt, self.substep_coeff)
    return (
      torch.cat((next_root, next_item), -1),
      self._encode_state(next_robot_rep, next_root, next_item),
    )

  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    robot = super().reconstruct_full_state(
      aux_state[..., :7], rep_state[..., : self.robot_rep_dim]
    )
    return torch.cat((robot, aux_state[..., 7:]), -1)

  def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
    return super().joint_state(rep_state[..., : self.robot_rep_dim])

  def world_frame_bodies(self, aux_state: torch.Tensor, rep_state: torch.Tensor):
    return super().world_frame_bodies(
      aux_state[..., :7], rep_state[..., : self.robot_rep_dim]
    )

  def compute_wm_obs(self, rep_state, aux_state, joint_state, sim_ctx):
    return super().compute_wm_obs(
      rep_state[..., : self.robot_rep_dim], aux_state, joint_state, sim_ctx
    )

  def proprio_term_widths(self, num_actions: int) -> list[int]:
    return [*super().proprio_term_widths(num_actions), self.item_dim, 24]

  def proprio_noise_bounds(
    self, num_actions: int, device: torch.device | str
  ) -> torch.Tensor:
    widths = self.proprio_term_widths(num_actions)
    halves = (*PROPRIO_NOISE_HALF_WIDTHS.values(), 0.0, 0.0)
    return torch.cat(
      [torch.full((w,), h, device=device) for w, h in zip(widths, halves, strict=True)]
    )

  def compute_actor_sensor(
    self,
    rep_state,
    sim_ctx,
    aux_state=None,
    joint_state=None,
    last_action=None,
    sim_actor_obs=None,
    noisy=False,
  ):
    if self.actor_obs_mode == "rep_state":
      return rep_state
    if sim_actor_obs is None:
      sensor = super().compute_actor_sensor(
        rep_state, sim_ctx, aux_state, joint_state, last_action, noisy=False
      )
      sensor = torch.cat((sensor, rep_state[..., self.robot_rep_dim :]), dim=-1)
    else:
      sensor = sim_actor_obs
    return self.add_proprio_noise(sensor) if noisy else sensor

  def compute_wm_goal(self, motion, motion_times, aux_state):
    robot_wm_goal, _ = super().compute_wm_goal(motion, motion_times, aux_state)
    item_goal = item_goal_window(motion, motion_times, aux_state[..., :7])
    goal = merge_item_goal(robot_wm_goal, item_goal)
    return goal, extract_actor_goal(goal)

  def loss_blocks(self, rep_state, aux_state):
    blocks = super().loss_blocks(
      rep_state[..., : self.robot_rep_dim], aux_state[..., :7]
    )
    item = aux_state[..., 7:]
    rot_matrix = matrix_from_quat(item[..., 3:7])
    return {
      **blocks,
      "item_pos": item[..., :3],
      "item_rot": rot_matrix.flatten(-2) * math.sqrt(0.5),
      "item_vel": item[..., 7:10],
      "item_avel": item[..., 10:13],
    }
