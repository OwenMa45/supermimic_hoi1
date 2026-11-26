"""Single-scene G1 Carry Box task with a free, contact-driven rigid body."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg

from supermimic.hoi_item import encode_item, encode_supports
from supermimic.hoi_representation import item_goal_window, item_tracking_reward
from supermimic.tasks.env_cfgs import gencoord_env_cfg


def _scene_data(path):
  if isinstance(path, list) and len(path) == 1:
    path = path[0]
  if not isinstance(path, str) or not Path(path).expanduser().is_file():
    raise ValueError("Carry Box needs one prepared NPZ file; pass --motion-file")
  with np.load(Path(path).expanduser(), allow_pickle=False) as data:
    keys = {"item_state_w", "wrist_contacts", "half_size", "supports_w", "mass_kg"}
    if missing := keys.difference(data.files):
      raise ValueError(
        f"Missing {sorted(missing)}; prepare the clip with supermimic.tasks.hoi_motion"
      )
    result = {k: np.asarray(data[k], dtype=np.float32) for k in keys}
    length = data["joint_pos"].shape[0]
    if length < 2 or not np.isclose(data["fps"].item(), 50.0):
      raise ValueError("Expected a prepared 50 Hz clip with at least two frames")
  shapes = {
    "item_state_w": (length, 13),
    "wrist_contacts": (length, 2),
    "half_size": (3,),
    "supports_w": (2, 10),
    "mass_kg": (),
  }
  for key, value in result.items():
    if value.shape != shapes[key] or not np.isfinite(value).all():
      raise ValueError(f"{key} must be finite and have shape {shapes[key]}")
  if np.any(result["half_size"] <= 0) or result["mass_kg"] <= 0:
    raise ValueError("Box dimensions and mass must be positive")
  if not np.isin(result["wrist_contacts"], [0, 1]).all():
    raise ValueError("wrist_contacts must be binary")
  if not np.allclose(
    np.linalg.norm(result["item_state_w"][:, 3:7], axis=-1), 1.0, atol=1e-4
  ):
    raise ValueError("Prepared box orientations must be unit wxyz quaternions")
  for support in result["supports_w"]:
    if not np.isclose(np.linalg.norm(support[3:7]), 1.0, atol=1e-4):
      raise ValueError("Support orientation must be a unit quaternion")
    if abs(support[4]) > 1e-5 or abs(support[5]) > 1e-5 or support[9] not in (0, 1):
      raise ValueError("Expected a horizontal finite table or ground")
    if support[9] and (np.any(support[7:9] <= 0) or support[2] <= 0):
      raise ValueError("Finite support needs positive half-widths and top height")
    if not support[9] and abs(support[2]) > 1e-5:
      raise ValueError("Ground must be at scene height zero")
  return result


def _cuboid_spec(size, mass, floating):
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(
    name="box" if floating else "table", mocap=not floating
  )
  if floating:
    body.add_freejoint()
  body.add_geom(
    name="box_collision" if floating else "top_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=size,
    mass=mass,
    condim=3,
    priority=1,
    friction=(0.8, 0.005, 0.0001),
    rgba=(0.70, 0.46, 0.22, 1.0) if floating else (0.30, 0.48, 0.58, 1.0),
  )
  return spec


def configure_hoi_scene(cfg, *, with_wrists=True):
  """Finalize geometry after --motion-file is resolved and before scene creation."""
  command = cfg.commands.get("motion")
  if not isinstance(command, CarryBoxMotionCommandCfg):
    return
  data = _scene_data(command.motion_file)
  first = data["item_state_w"][0]
  cfg.scene.entities["box"] = EntityCfg(
    spec_fn=partial(
      _cuboid_spec, tuple(data["half_size"]), float(data["mass_kg"]), True
    ),
    init_state=EntityCfg.InitialStateCfg(pos=tuple(first[:3]), rot=tuple(first[3:7])),
  )
  for i, name in enumerate(("start_table", "goal_table")):
    cfg.scene.entities.pop(name, None)
    row = data["supports_w"][i]
    if not row[-1] or (i == 1 and np.array_equal(row, data["supports_w"][0])):
      continue
    position = row[:3].copy()
    position[2] -= 0.01
    cfg.scene.entities[name] = EntityCfg(
      spec_fn=partial(_cuboid_spec, (*row[7:9], 0.01), 1.0, False),
      init_state=EntityCfg.InitialStateCfg(pos=tuple(position), rot=tuple(row[3:7])),
    )
  cfg.observations["actor"].terms["item_obs"].params["with_wrists"] = with_wrists


class CarryBoxMotionCommand(MotionCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    if not np.isclose(self.motion.fps * env.step_dt, 1.0):
      raise ValueError("One motion frame must equal one control step for Carry Box")
    data = _scene_data(cfg.motion_file)
    for key in ("item_state_w", "wrist_contacts", "half_size", "supports_w"):
      setattr(self.motion, key, torch.as_tensor(data[key], device=self.device))
    self.box = env.scene["box"]
    self.wrist_ids = [
      cfg.body_names.index(f"{side}_wrist_yaw_link") for side in ("left", "right")
    ]
    self.tables = []
    for i, name in enumerate(("start_table", "goal_table")):
      if name in env.cfg.scene.entities:
        pose = self.motion.supports_w[i, :7].clone()
        pose[2] -= 0.01
        self.tables.append((env.scene[name], pose))

  def _write_reference_state_to_sim(self, env_ids, *robot_state):
    # Both reset_to_frame and sampled resets call this hook. Ordinary steps
    # only advance the reference; the physical box responds to contacts.
    super()._write_reference_state_to_sim(env_ids, *robot_state)
    origins = self._env.scene.env_origins[env_ids]
    state = self.motion.item_state_w[self.time_steps[env_ids]].clone()
    state[:, :3] += origins
    self.box.write_root_state_to_sim(state, env_ids=env_ids)
    for table, local_pose in self.tables:
      pose = local_pose.expand(len(origins), -1).clone()
      pose[:, :3] += origins
      table.write_mocap_pose_to_sim(pose, env_ids=env_ids)


@dataclass(kw_only=True)
class CarryBoxMotionCommandCfg(MotionCommandCfg):
  def build(self, env):
    return CarryBoxMotionCommand(self, env)


def box_state_w(env):
  """Environment-local world axes: centre, quaternion, linear and angular velocity."""
  box = env.scene["box"].data
  return torch.cat(
    (
      box.root_link_pos_w - env.scene.env_origins,
      box.root_link_quat_w,
      box.root_link_lin_vel_w,
      box.root_link_ang_vel_w,
    ),
    dim=-1,
  )


def wrist_positions_w(env):
  command = env.command_manager.get_term("motion")
  return command.robot_body_pos_w[:, command.wrist_ids] - env.scene.env_origins[:, None]


def _root_pose(env):
  robot = env.scene["robot"].data
  return torch.cat(
    (robot.root_link_pos_w - env.scene.env_origins, robot.root_link_quat_w), dim=-1
  )


def _current_item(env, with_wrists=True):
  motion = env.command_manager.get_term("motion").motion
  values = encode_item(
    box_state_w(env), _root_pose(env), motion.half_size, wrist_positions_w(env)
  )
  return values if with_wrists else values[..., :18]


def _current_supports(env):
  return encode_supports(
    _root_pose(env), env.command_manager.get_term("motion").motion.supports_w
  )


def _item_goal(env):
  cmd = env.command_manager.get_term("motion")
  return item_goal_window(cmd.motion, cmd.time_steps, _root_pose(env))


def _item_reward(env):
  cmd = env.command_manager.get_term("motion")
  return item_tracking_reward(box_state_w(env), cmd.motion.item_state_w[cmd.time_steps])


def carry_box_env_cfg(play=False):
  cfg = gencoord_env_cfg(play)
  base = cfg.commands["motion"]
  command = CarryBoxMotionCommandCfg(
    **{f.name: getattr(base, f.name) for f in fields(base)}
  )
  command.pose_range = {}
  command.velocity_range = {}
  command.joint_position_range = (0.0, 0.0)
  command.viz.mode = "frames"
  cfg.commands["motion"] = command
  cfg.observations["full_state"].terms["item_state_w"] = ObservationTermCfg(
    func=box_state_w
  )
  actor = cfg.observations["actor"].terms
  actor["item_obs"] = ObservationTermCfg(
    func=_current_item, params={"with_wrists": True}
  )
  actor["supports_obs"] = ObservationTermCfg(func=_current_supports)
  group = cfg.observations["goal"]
  group.terms = {
    "command": group.terms["command"],
    "item_goal": ObservationTermCfg(func=_item_goal),
    "motion_anchor_pos_b": group.terms["motion_anchor_pos_b"],
    "motion_anchor_ori_b": group.terms["motion_anchor_ori_b"],
  }
  cfg.rewards["item_tracking"] = RewardTermCfg(func=_item_reward, weight=1.0)
  return cfg
