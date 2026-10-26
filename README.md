# rl-split

PPO training for IsaacLab Go2 rough terrain (`Isaac-Velocity-Rough-UnitreeGo2`, rsl_rl 5.5.1), split into two processes connected by int2DDS:

- **Worker** (DGX Spark): runs the simulation.
- **Learner** (Jetson Orin): runs PPO. No IsaacLab needed.

Both sides read the same `configs/go2_rough.yaml`.

## Modes

### `central`: policy only on the Learner

The Worker sends obs every step and gets actions back.

```
worker ── central_obs ────▶ learner      obs, rewards, dones (~3.9MB/step)
worker ◀── central_action ── learner      actions (~0.2MB/step)
worker ◀── ctrl ──────────── learner      start / stop
```

### `local`: policy copy on the Worker

The Worker collects 24 steps with its copy and streams them. The Learner updates and sends new weights.

```
worker ── local_transition ─▶ learner    rsl_rl Transition (~4.5MB × 24/iter)
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
- Learner (Orin): see `requirements/learner.txt`.

## Run

Start in any order (the Learner waits up to 10 min for the Worker). Both use the same `--mode`, each in its own venv.

```bash
python3 scripts/worker.py --mode central     # or local
python3 scripts/learner.py --mode central
```

Logs and checkpoints go to `logs/rsl_rl/<experiment>/<time>_<mode>/`.

## Messages

Defined in `rl_split/comm/rl_split.idl`. After editing it, regenerate:

```bash
int2dds-idl -p rl_split/comm/types.py rl_split/comm/rl_split.idl
```
