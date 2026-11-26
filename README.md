# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd supermimic
```

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation
- `SuperMimic-GenCoord-HOI-G1` — G1 and one physical box tracked from an OmniContact clip

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

### Carry Box

Prepare a single clip before training. For the public Sim2Sim
`carrybox_motion_4_with_contact.npz`, the box is a 0.30 m cube.

```bash
python -m supermimic.tasks.hoi_motion carrybox_motion_4_with_contact.npz carry4.npz --box-size .30 .30 .30
python -m supermimic.scripts.train SuperMimic-GenCoord-HOI-G1 --motion-file carry4.npz
```

The converter reads the selected dimensions (cm) from OmniContact size-augmented
clips, resamples at the G1 control rate of 50 Hz, and maps joint names to G1
order. Use `--start-table` and `--goal-table` for finite surfaces whose geometry
is absent from the source. Each takes top-centre XYZ, yaw in radians, and two
half-extents in metres. Set `--mass-kg` to the simulated box mass (default 2).

The actor and world model share 24 box features per frame. For the 18-feature
ablation, repeat training with `--runner.use-wrist-features False`. Evaluate the
two checkpoints on the same prepared clip, initial seed and action traces:

```bash
python -m supermimic.scripts.evaluate_hoi --motion-file carry4.npz --run path/to/18d-run path/to/24d-run --seed 42 --output metrics.json
```

Local run directories contain `runner_cfg.json`, `actor.pt`, and `world_model.pt`.
The evaluator also accepts W&B run paths. It measures lift and sustained
release in physics and reports model prediction errors by reference contact phase.
Its CLI exposes the physical thresholds in metres, seconds and radians.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
