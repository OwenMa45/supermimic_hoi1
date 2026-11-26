"""Evaluate Carry Box policies in physics and their models on the same actions.

Example: python -m supermimic.scripts.evaluate_hoi --motion-file carry.npz
  --run entity/project/ablation18 entity/project/full24 --seed 42 --num-envs 16

A run may also be a local directory with actor.pt, world_model.pt and
runner_cfg.json (a serialized runner config, optionally under "runner_cfg").
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import matrix_from_quat


@dataclass
class Thresholds:
  lift_m: float = 0.05
  goal_xy_m: float = 0.12
  surface_m: float = 0.025
  speed_m_s: float = 0.08
  spin_rad_s: float = 0.3
  hold_s: float = 0.20


def surface_geometry(item, half_size, support):
  """Box center, lowest height and support footprint from oriented box extents."""
  axes = (
    matrix_from_quat(support[3:7])
    if bool(support[9])
    else torch.eye(3, device=item.device, dtype=item.dtype)
  )
  center = (item[..., :3] - support[:3]) @ axes
  radii = (axes.T @ matrix_from_quat(item[..., 3:7])).abs() @ half_size
  clearance = center[..., 2] - radii[..., 2]
  inside = ((center[..., :2].abs() + radii[..., :2]) <= support[7:9] + 0.005).all(-1)
  return center, clearance, inside | ~support[9].bool()


def episode_metrics(
  items, hands, on_target, valid, half_size, supports, goal, dt, limits
):
  """Score actual post-step states (time, episode, ...), excluding auto-resets."""
  _, clearance, _ = surface_geometry(items, half_size, supports[0])
  center, gap, inside = surface_geometry(items, half_size, supports[1])
  goal_center, _, _ = surface_geometry(goal, half_size, supports[1])
  lift = valid & hands.all(-1) & (clearance >= limits.lift_m)
  arrived = inside & (
    (center[..., :2] - goal_center[..., :2]).norm(dim=-1) <= limits.goal_xy_m
  )
  slow = (items[..., 7:10].norm(dim=-1) <= limits.speed_m_s) & (
    items[..., 10:13].norm(dim=-1) <= limits.spin_rad_s
  )
  lift_seen = lift.cummax(dim=0).values
  arrival = valid & lift_seen & arrived & (gap.abs() <= limits.surface_m)
  settled = arrival & on_target & ~hands.any(-1) & slow
  hold = max(1, math.ceil(limits.hold_s / dt))
  completed = torch.zeros_like(lift[0])
  if settled.shape[0] >= hold:
    completed = settled.unfold(0, hold, 1).all(-1).any(0)
  n, lifts, arrivals, ends = (
    valid.shape[1],
    lift.any(0).sum().item(),
    arrival.any(0).sum().item(),
    completed.sum().item(),
  )
  return dict(
    episodes=n,
    lifted=lifts,
    arrived_after_lift=arrivals,
    released_stably=ends,
    lift_rate=lifts / n,
    placement_given_arrival=ends / arrivals if arrivals else None,
    end_to_end_rate=ends / n,
  )


def _phases(contacts):
  """Approach, contact acquisition, carry, release, post-release from the reference."""
  held = contacts.bool().any(-1)
  active = held.nonzero().flatten()
  if not active.numel() or active[0] == 0 or active[-1] == len(held) - 1:
    raise ValueError(
      "Evaluation needs a clip covering pre-contact, carrying and release"
    )
  phases = torch.where(held, 2, 0)
  phases[active[-1] + 1 :] = 4
  # Label transition neighborhoods separately, including the first free frame.
  for boundary in (held[1:] != held[:-1]).nonzero().flatten().tolist():
    t = boundary + 1
    phases[max(t - 2, 0) : t + 3] = 1 if held[t] else 3
  return phases


def _add_sensors(cfg, target_finite):
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  box = ContactMatch(mode="geom", pattern="box_collision", entity="box")
  sensors = [
    ContactSensorCfg(
      name=f"eval_{side}",
      fields=("found",),
      primary=ContactMatch(
        mode="geom", pattern=f"{side}_(hand|wrist)_collision", entity="robot"
      ),
      secondary=box,
    )
    for side in ("left", "right")
  ]
  if target_finite:
    table = "goal_table" if "goal_table" in cfg.scene.entities else "start_table"
    target = ContactMatch(mode="geom", pattern="top_collision", entity=table)
  else:
    target = ContactMatch(mode="geom", pattern="terrain")
  sensors.append(
    ContactSensorCfg(
      name="eval_target", primary=box, secondary=target, fields=("found",)
    )
  )
  cfg.scene.sensors = (*cfg.scene.sensors, *sensors)


def _load(run_id, clip, num_envs, seed, device):
  import numpy as np
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg

  import supermimic.tasks  # noqa: F401
  from supermimic.runners.run_config import (
    check_representation_matches,
    load_runner_cfg,
  )
  from supermimic.runners.supermimic_runner import build_actor, build_world_model
  from supermimic.tasks.hoi_env import configure_hoi_scene

  path = Path(run_id)
  if path.is_dir():
    raw = json.loads((path / "runner_cfg.json").read_text(encoding="utf-8"))
    cfg = load_runner_cfg(raw if "runner_cfg" in raw else {"runner_cfg": raw})
  else:
    import wandb

    from supermimic.utils.wandb_artifacts import download_model_dir

    run = wandb.Api().run(run_id)
    path = download_model_dir(run)
    cfg = load_runner_cfg(run.config)
  task = "SuperMimic-GenCoord-HOI-G1"
  check_representation_matches(task, cfg.representation)
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs, env_cfg.seed = num_envs, seed
  env_cfg.commands["motion"].motion_file = str(clip)
  env_cfg.commands["motion"].sampling_mode = "start"
  configure_hoi_scene(env_cfg, with_wrists=cfg.use_wrist_features)
  with np.load(clip, allow_pickle=False) as data:
    target_finite = bool(data["supports_w"][1, 9])
  _add_sensors(env_cfg, target_finite)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
  try:
    wm = build_world_model(cfg, env, device)
    wm.load_bundle(torch.load(path / "world_model.pt", map_location=device))
    wm.dynamics.eval()
    actor = build_actor(cfg, wm, device)
    actor.load_state_dict(torch.load(path / "actor.pt", map_location=device))
    actor.set_slot_mix({"motion": 1.0})
    actor.eval()
    return env, wm, actor
  except BaseException:
    env.close()
    raise


def _simulate(env, wm, actor, seed, limits):
  from supermimic.tasks.task_spec import build_actor_obs_td

  env.unwrapped.seed(seed)
  obs, _ = env.reset()
  cmd = env.unwrapped.command_manager.get_term("motion")
  motion = cmd.motion
  stages = _phases(motion.wrist_contacts)
  rep = wm.representation
  # mjlab restores physical frame 0, then advances the command to target 1.
  if torch.any(cmd.time_steps != 1):
    raise ValueError("Expected a start reset with frame 1 as the first action target")
  scene = env.unwrapped.scene
  goal_task = next(t for t in actor.task_specs if t.name == "motion")
  states = [obs["full_state"].clone()]
  sim_contexts, actions, valids, phases, hands, surface = [], [], [], [], [], []
  alive = torch.ones_like(cmd.time_steps, dtype=torch.bool)
  history = None
  for _ in range(len(stages) - 1):
    current = states[-1]
    aux, x = rep.extract_aux_state(current), rep.represent(current)
    ctx = wm.sim_context.read(env.unwrapped)
    sensor = rep.compute_actor_sensor(x, ctx, aux_state=aux, sim_actor_obs=obs["actor"])
    if history is None:
      history = sensor.unsqueeze(1).repeat(1, actor.history_length, 1)
    else:
      history = torch.cat((history[:, 1:], sensor.unsqueeze(1)), dim=1)
    frames = cmd.time_steps.clone()
    goal = goal_task.extract_training_goal(motion, frames, aux)
    action = actor(build_actor_obs_td({"motion": goal}, history))
    obs, _, done, _ = env.step(action)
    alive = alive & ~done.bool().reshape_as(alive) & (cmd.time_steps == frames + 1)
    phases.append(stages[frames])
    valids.append(alive)
    states.append(obs["full_state"].clone())
    actions.append(action)
    # A checkpoint's training randomization ranges set its normalization.
    # Keep physical values so the other model can apply its own scales.
    sim_contexts.append(ctx * wm.sim_context.scale + wm.sim_context.mean)
    hands.append(
      torch.stack(
        [(scene[f"eval_{side}"].data.found > 0).any(-1) for side in ("left", "right")],
        dim=-1,
      )
    )
    surface.append((scene["eval_target"].data.found > 0).any(-1))
    if not alive.any():
      break
  state, valid = torch.stack(states), torch.stack(valids)
  outcome = episode_metrics(
    state[1:, :, -13:],
    torch.stack(hands),
    torch.stack(surface),
    valid,
    motion.half_size,
    motion.supports_w,
    motion.item_state_w[-1],
    env.unwrapped.step_dt,
    limits,
  )
  return outcome, (
    state,
    torch.stack(actions),
    torch.stack(sim_contexts),
    valid,
    torch.stack(phases),
  )


def _model_errors(wm, trace, horizon):
  state, action, raw_context, valid, phase = trace
  ctx = (raw_context - wm.sim_context.mean) / wm.sim_context.scale
  names = ("approach", "contact_on", "carry", "contact_off", "released")
  values, labels = [], []
  # Non-overlapping windows use the same physical states/actions for every model.
  for start in range(0, len(action) - horizon + 1, horizon):
    stop = start + horizon
    x = wm.representation.represent(state[start])
    aux = wm.representation.extract_aux_state(state[start])
    for t in range(start, stop):
      aux, x = wm.dynamics(x, action[t], aux, ctx[t])
    pred, true = aux[..., 7:], state[stop, :, -13:]
    dot = (pred[..., 3:7] * true[..., 3:7]).sum(-1).abs().clamp(0.0, 1.0)
    error = torch.stack(
      (
        (pred[..., :3] - true[..., :3]).norm(dim=-1),
        2 * dot.acos(),
        (pred[..., 7:10] - true[..., 7:10]).norm(dim=-1),
        (pred[..., 10:13] - true[..., 10:13]).norm(dim=-1),
      ),
      dim=-1,
    )
    keep = valid[stop - 1]
    values.append(error[keep])
    labels.append(phase[stop - 1, keep])
  result = {}
  for i, name in enumerate(names):
    rows = (
      torch.cat(values)[torch.cat(labels) == i] if values else state.new_empty((0, 4))
    )
    result[name] = {"samples": len(rows)}
    if len(rows):
      result[name].update(
        zip(
          ("center_m", "angle_rad", "linear_m_s", "angular_rad_s"),
          rows.mean(0).tolist(),
          strict=True,
        )
      )
  return result


@torch.inference_mode()
def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument(
    "--run", nargs="+", required=True, help="One run or an 18D/24D pair"
  )
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--horizon", type=int, default=8)
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  parser.add_argument("--output", type=Path)
  for name, default in asdict(Thresholds()).items():
    parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
  args = parser.parse_args()
  if not 1 <= len(args.run) <= 2 or len(set(args.run)) != len(args.run):
    parser.error("Supply one model or two distinct models")
  if args.num_envs < 1 or args.horizon < 1:
    parser.error("num-envs and horizon must be positive")
  limits = Thresholds(**{n: getattr(args, n) for n in asdict(Thresholds())})
  if any(not math.isfinite(v) or v <= 0 for v in asdict(limits).values()):
    parser.error("All scoring thresholds must be positive and finite")
  opened = {}
  report = dict(
    motion_file=str(args.motion_file.resolve()),
    seed=args.seed,
    thresholds=asdict(limits),
    results={},
  )
  try:
    for run in args.run:
      opened[run] = _load(run, args.motion_file, args.num_envs, args.seed, args.device)
    for behavior, (env, wm, actor) in opened.items():
      outcomes, trace = _simulate(env, wm, actor, args.seed, limits)
      report["results"][behavior] = {
        "policy_item_dim": wm.representation.item_dim,
        "real_sim": outcomes,
        "world_model_errors_on_this_policy_trace": {
          name: {
            str(h): _model_errors(other, trace, h) for h in sorted({1, args.horizon})
          }
          for name, (_, other, _) in opened.items()
        },
      }
  finally:
    for env, _, _ in opened.values():
      env.close()
  output = json.dumps(report, indent=2, allow_nan=False)
  print(output)
  if args.output:
    args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
