"""Startup handshake shared by both modes.

Workers are numbered 0 .. num_workers-1 (worker.py --worker_id) and all use env.num_envs.
Worker: create its topics -> wait for ctrl "start" -> run -> stop on ctrl "stop".
Learner: create every Worker's topics -> wait until all are matched -> send ctrl "start" -> ... -> send "stop".
"""

from __future__ import annotations

import json
import time

from rl_split.comm import types as msg
from rl_split.comm.convert import fmt_bytes, payload_bytes

# A Worker can take minutes to bring up the simulation.
WORKER_TIMEOUT_S = 600.0


def worker_rows(worker_id: int, num_envs: int) -> slice:
    """Worker worker_id's rows in the Learner's batch, where the Workers' envs are stacked in worker_id order."""
    return slice(worker_id * num_envs, (worker_id + 1) * num_envs)


def wait_workers(node, num_workers: int) -> None:
    """Learner: block until every Worker has readers for every Learner writer, so "start" reaches all of them."""
    print(f"[learner] waiting for {num_workers} worker(s)")
    node.wait_matched(timeout=WORKER_TIMEOUT_S)
    node.check_single_worker()
    print(f"[learner] {num_workers} worker(s) connected")


class CtrlSender:
    def __init__(self, node) -> None:
        self._node = node
        self._seq = 0

    def send(self, command: str, **args) -> None:
        self._seq += 1
        sample = msg.Ctrl(seq=self._seq, command=command, args_json=json.dumps(args))
        self._node.write("ctrl", sample)
        print(f"[learner] pub ctrl {command!r} seq {self._seq} ({fmt_bytes(payload_bytes(sample))})")

    def stop(self, linger: float = 2.0) -> None:
        """Send stop and give the reliable protocol time to deliver it before the process exits."""
        self.send("stop")
        time.sleep(linger)


def wait_start(inbox) -> dict:
    """Block until ctrl "start" and return its arguments."""
    print("[worker] waiting for ctrl start from the learner")
    while True:
        ctrl = inbox.get()
        _log_ctrl(ctrl)
        if ctrl.command == "start":
            return json.loads(ctrl.args_json)
        print(f"[worker] ignoring ctrl {ctrl.command!r} before start")


def stop_requested(inbox) -> bool:
    ctrl = inbox.poll()
    if ctrl is None:
        return False
    _log_ctrl(ctrl)
    if ctrl.command != "stop":
        print(f"[worker] ignoring ctrl {ctrl.command!r}")
    return ctrl.command == "stop"


def _log_ctrl(ctrl) -> None:
    print(f"[worker] sub ctrl {ctrl.command!r} seq {ctrl.seq} ({fmt_bytes(payload_bytes(ctrl))}) {ctrl.args_json}")
