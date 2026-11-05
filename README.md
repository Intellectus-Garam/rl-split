# rl-split

PPO training for IsaacLab Go2 rough terrain (`Isaac-Velocity-Rough-UnitreeGo2`, rsl_rl 5.5.1), split into two processes connected by int2DDS:

- **Worker** (DGX Spark): runs the simulation. There can be several.
- **Learner** (Jetson Orin or DGX Spark): runs PPO. No IsaacLab needed.

Both sides read the same `configs/go2_rough.yaml`.

Workers are numbered `0 .. num_workers-1` and each has its own `central_obs_<id>`, `central_action_<id>` and `local_transition_<id>` topics; `ctrl` and `local_weights` are shared. The Learner stacks the Workers' envs in `worker_id` order into one batch.

## Modes

### `central`: policy only on the Learner

Each Worker sends obs every step and gets its own slice of the actions back. The Learner waits for every Worker on every step.

```
worker ── central_obs ────▶ learner      obs, rewards, dones (~3.9MB/step per Worker)
worker ◀── central_action ── learner      actions (~0.2MB/step per Worker)
worker ◀── ctrl ──────────── learner      start / stop
```

### `local`: policy copy on the Worker

Each Worker collects 24 steps with its copy and streams them. The Learner updates on all Workers' steps and sends new weights to every Worker.

```
worker ── local_transition ─▶ learner    rsl_rl Transition (~4.5MB × 24/iter per Worker)
worker ◀── local_weights ──── learner    actor + critic state_dict (latest only)
worker ◀── ctrl ───────────── learner    start / stop
```

All topics are RELIABLE over UDP. `local_weights` is TRANSIENT_LOCAL / KEEP_LAST 1; the rest are KEEP_ALL (30 samples).

## Setup

int2DDS is built from source:

```bash
(cd <path-to-int2DDS> && cargo build --release -p int2dds-ffi -p int2dds-idl)
export PYTHONPATH=<path-to-int2DDS>/python
```

int2dds finds `<path-to-int2DDS>/target/release/libint2dds_ffi.so` on its own (no `INT2DDS_FFI_PATH` needed).

- Worker (Spark): see `requirements/worker.txt` (IsaacLab 3.0.0 with Newton physics; Isaac Sim is not needed).
- Learner: see `requirements/learner.txt` (Jetson AGX Orin: `requirements/learner-orin.txt`).

## Run

Start in any order (the Learner waits up to 10 min for the Workers). All use the same `--mode`, each in its own venv.

```bash
python3 scripts/worker.py --mode central     # or local
python3 scripts/learner.py --mode central
```

Several Workers: number them `0 .. N-1` with `--worker_id` and tell the Learner `N`.

```bash
python3 scripts/worker.py --mode central --worker_id 0
python3 scripts/worker.py --mode central --worker_id 1
python3 scripts/learner.py --mode central --num_workers 2
```

- Every Worker runs `env.num_envs` envs from the config.
- A Worker's seed is random (printed at startup) unless `--seed` is given. Give each Worker a different `--seed`.
- The Learner multiplies `num_mini_batches` by `--num_workers`, so the mini-batch size stays as configured and the extra data becomes extra updates. `learner.py --num_mini_batches` sets it directly instead. The values used are printed at startup and saved to `params/run.yaml` in the log directory.
- A Worker exits when the Learner stops. With `--keep_alive` it keeps the simulation up and waits for the next Learner.
- A `--worker_id` used twice stops the Learner with an error. A missing one makes it time out naming the unmatched topic.

Logs and checkpoints go to `logs/rsl_rl/<experiment>/<time>_<mode>/`.

## Messages

Defined in `rl_split/comm/rl_split.idl`. After editing it, regenerate:

```bash
int2dds-idl -p rl_split/comm/types.py rl_split/comm/rl_split.idl
```
