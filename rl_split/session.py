"""Startup handshake and shutdown shared by both modes.

Workers are numbered 0 .. num_workers-1 (worker.py --worker_id) and all use env.num_envs.
Worker: create its topics -> wait for ctrl "start" -> run. ctrl "stop" ends it at any point,
including before "start" (check_stop / wait_start raise StopRequested). With worker.py --keep_alive
it then waits for the next "start" instead, so a new Learner can connect to the same Worker.
Learner: create every Worker's topics -> wait until all are matched -> send ctrl "start" -> ...
The learner scripts send "stop" in a finally around everything after the Node is created, so
it goes out on any exit Python sees (normal end, exception, Ctrl+C, SIGTERM, SIGHUP).
"""

from __future__ import annotations

import json
import signal
import sys
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


def exit_on_signals() -> None:
    """Make SIGINT, SIGTERM and SIGHUP raise, so finally blocks (and the ctrl "stop" in them) run.

    SIGINT is set explicitly because a process started in the background by a non-interactive
    shell (a script, nohup) starts with SIGINT ignored, and Python then keeps it ignored.
    """

    def _exit(signum, _frame):
        print(f"[learner] got {signal.Signals(signum).name}, exiting")
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGHUP, _exit)


class CtrlSender:
    """Learner's ctrl writer. Every Worker reads it, so wait_matched waits for num_workers readers."""

    def __init__(self, node, num_workers: int) -> None:
        self._node = node
        self._seq = 0
        node.writer("ctrl", num_readers=num_workers)

    def send(self, command: str, **args) -> None:
        self._seq += 1
        sample = msg.Ctrl(seq=self._seq, command=command, args_json=json.dumps(args))
        self._node.write("ctrl", sample)
        print(f"[learner] pub ctrl {command!r} seq {self._seq} ({fmt_bytes(payload_bytes(sample))})")

    def stop(self, linger: float = 2.0) -> None:
        """Send stop and give the reliable protocol time to deliver it before the process exits."""
        self.send("stop")
        time.sleep(linger)


class StopRequested(Exception):
    """ctrl "stop" arrived. Raised from any Worker wait so the Worker ends at once."""


def wait_start(inbox) -> dict:
    """Block until ctrl "start" and return its arguments. Raises StopRequested on "stop"."""
    print("[worker] waiting for ctrl start from the learner")
    while True:
        ctrl = inbox.get()
        _log_ctrl(ctrl)
        if ctrl.command == "start":
            return json.loads(ctrl.args_json)
        if ctrl.command == "stop":
            raise StopRequested
        print(f"[worker] ignoring ctrl {ctrl.command!r} before start")


def check_stop(inbox) -> None:
    """Raise StopRequested if ctrl "stop" has arrived. Other commands are ignored."""
    while (ctrl := inbox.poll()) is not None:
        _log_ctrl(ctrl)
        if ctrl.command == "stop":
            raise StopRequested
        print(f"[worker] ignoring ctrl {ctrl.command!r}")


def _log_ctrl(ctrl) -> None:
    print(f"[worker] sub ctrl {ctrl.command!r} seq {ctrl.seq} ({fmt_bytes(payload_bytes(ctrl))}) {ctrl.args_json}")
