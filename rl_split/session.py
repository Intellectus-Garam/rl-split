"""Startup handshake shared by both modes.

Worker: wait for ctrl "start" -> run -> stop on ctrl "stop".
Learner: wait for the Worker's readers -> send ctrl "start" -> ... -> send "stop".
"""

from __future__ import annotations

import json
import time

from rl_split.comm import types as msg
from rl_split.comm.convert import fmt_bytes, payload_bytes

# The Worker can take minutes to bring up the simulation.
WORKER_TIMEOUT_S = 600.0


def wait_worker(node) -> None:
    """Block until the Worker has readers for every Learner writer, so "start" is not lost."""
    print("[learner] waiting for the worker")
    node.wait_matched(timeout=WORKER_TIMEOUT_S)


class CtrlSender:
    def __init__(self, node) -> None:
        self._node = node
        self._seq = 0

    def send(self, command: str, **args) -> None:
        self._seq += 1
        sample = msg.Ctrl(seq=self._seq, command=command, args_json=json.dumps(args))
        self._node.write("ctrl", sample)
        print(f"[learner] sent ctrl {command!r} (seq {self._seq}, {fmt_bytes(payload_bytes(sample))})")

    def stop(self, linger: float = 2.0) -> None:
        """Send stop and give the reliable protocol time to deliver it before the process exits."""
        self.send("stop")
        time.sleep(linger)


def wait_start(inbox) -> dict:
    """Block until ctrl "start" and return its arguments."""
    print("[worker] waiting for ctrl start from the learner")
    while True:
        ctrl = inbox.get()
        if ctrl.command == "start":
            print(f"[worker] received ctrl 'start' (seq {ctrl.seq}, {fmt_bytes(payload_bytes(ctrl))}): {ctrl.args_json}")
            return json.loads(ctrl.args_json)
        print(f"[worker] received ctrl {ctrl.command!r} (seq {ctrl.seq}), ignoring before start")


def stop_requested(inbox) -> bool:
    ctrl = inbox.poll()
    if ctrl is not None and ctrl.command != "stop":
        print(f"[worker] received ctrl {ctrl.command!r} (seq {ctrl.seq}), ignoring")
    return ctrl is not None and ctrl.command == "stop"
