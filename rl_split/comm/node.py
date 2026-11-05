"""DDS participant, topics and QoS shared by both roles."""

from __future__ import annotations

import os
import time
from collections import deque

from int2dds import (
    DataReaderQos,
    DataWriterQos,
    DdsTimeout,
    DomainParticipant,
    Durability,
    History,
    Reliability,
    ResourceLimits,
    WaitSet,
)

from rl_split.comm import types as msg

# KEEP_ALL queue depth: one iteration's 24 transitions fit without blocking the writer.
STREAM_MAX_SAMPLES = 30
# max_blocking_time of a reliable write; Node.write retries after it.
WRITE_TIMEOUT_S = 10.0

# name -> (type, QoS kind). "state": latest value, delivered to late joiners.
# "stream": every sample in order, bounded queue.
TOPICS = {
    "ctrl": (msg.Ctrl, "stream"),
    "central_obs": (msg.CentralObs, "stream"),
    "central_action": (msg.CentralAction, "stream"),
    "local_transition": (msg.LocalTransition, "stream"),
    "local_weights": (msg.LocalWeights, "state"),
}
# Topics with one instance per Worker, named <name>_<worker_id> (see per_worker).
# The rest are shared: every Worker reads the same ctrl and local_weights.
PER_WORKER = {"central_obs", "central_action", "local_transition"}


def per_worker(name: str, worker_id: int) -> str:
    """Topic name of Worker worker_id's copy of a per-Worker topic."""
    if name not in PER_WORKER:
        raise ValueError(f"{name} is not a per-worker topic")
    return f"{name}_{worker_id}"


def _base(name: str) -> str:
    """central_obs_1 -> central_obs; shared names are returned as they are."""
    base, _, suffix = name.rpartition("_")
    return base if base in PER_WORKER and suffix.isdigit() else name


def _qos(kind: str, qos_class):
    reliability = Reliability("RELIABLE", max_blocking_time=WRITE_TIMEOUT_S)
    if kind == "state":
        return qos_class(
            reliability=reliability,
            durability=Durability("TRANSIENT_LOCAL"),
            history=History("KEEP_LAST", depth=1),
        )
    return qos_class(
        reliability=reliability,
        history=History("KEEP_ALL"),
        # max_samples_per_instance defaults to unlimited, which DDS rejects as larger than max_samples.
        resource_limits=ResourceLimits(max_samples=STREAM_MAX_SAMPLES, max_samples_per_instance=STREAM_MAX_SAMPLES),
    )


class Inbox:
    """Reader wrapper that hands out valid samples one at a time."""

    def __init__(self, reader) -> None:
        self.reader = reader
        self._waitset = WaitSet()
        self._waitset.attach(reader)
        self._pending: deque = deque()

    def poll(self):
        """Next sample if one has arrived, otherwise None."""
        if not self._pending:
            self._pending.extend(s.data for s in self.reader.take() if s.valid_data)
        return self._pending.popleft() if self._pending else None

    def drain(self) -> int:
        """Discard every sample that has arrived and return how many there were."""
        count = 0
        while self.poll() is not None:
            count += 1
        return count

    def get(self, timeout: float | None = None):
        """Block until a sample arrives. Returns None on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            sample = self.poll()
            if sample is not None:
                return sample
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return None
            try:
                self._waitset.wait(timeout=1.0 if remaining is None else min(remaining, 1.0))
            except DdsTimeout:
                pass


class Node:
    """One participant per process with lazily created writers and readers.

    Topic names are either a TOPICS key (shared) or per_worker(key, worker_id).
    """

    def __init__(self, dds_cfg: dict, role: str) -> None:
        self.cfg = dds_cfg
        # The core reads these when the first participant is created.
        os.environ.setdefault("INT2DDS_TRANSPORT", dds_cfg["transport"])
        os.environ.setdefault("INT2DDS_UDP_SOCKET_BUFFER", str(dds_cfg["udp_socket_buffer"]))
        self.dp = DomainParticipant(domain_id=dds_cfg["domain_id"], name=role)
        self._pub = self.dp.create_publisher()
        self._sub = self.dp.create_subscriber()
        self._topics: dict = {}
        self.writers: dict = {}
        self.inboxes: dict = {}
        self._num_readers: dict[str, int] = {}

    def _topic(self, name: str):
        if name not in self._topics:
            self._topics[name] = self.dp.create_topic(name, TOPICS[_base(name)][0])
        return self._topics[name]

    def writer(self, name: str, num_readers: int = 1):
        """num_readers: readers wait_matched waits for (the Worker count for shared Learner topics).
        It is set when the writer is created."""
        if name not in self.writers:
            qos = _qos(TOPICS[_base(name)][1], DataWriterQos)
            self.writers[name] = self._pub.create_datawriter(self._topic(name), qos)
            self._num_readers[name] = num_readers
        return self.writers[name]

    def inbox(self, name: str) -> Inbox:
        qos = _qos(TOPICS[_base(name)][1], DataReaderQos)
        self.inboxes[name] = Inbox(self._sub.create_datareader(self._topic(name), qos))
        return self.inboxes[name]

    def wait_matched(self, timeout: float | None = None) -> None:
        """Block until every writer has its num_readers readers. Samples on volatile topics are lost otherwise.

        Also checks for a duplicated Worker while waiting: with a --worker_id used twice another id is
        missing, so the wait itself would only end in a timeout.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while missing := {n: w.matched_readers for n, w in self.writers.items() if w.matched_readers < self._num_readers[n]}:
            self.check_single_worker()
            if deadline is not None and time.monotonic() > deadline:
                counts = ", ".join(f"{n}: {c}/{self._num_readers[n]}" for n, c in missing.items())
                raise TimeoutError(f"readers missing (topic: matched/expected): {counts}")
            time.sleep(0.05)

    def check_single_worker(self) -> None:
        """Learner: raise if a per-Worker topic has more than one Worker on it (a --worker_id used twice)."""
        counts = {n: w.matched_readers for n, w in self.writers.items() if _base(n) != n}
        counts |= {n: i.reader.matched_writers for n, i in self.inboxes.items() if _base(n) != n}
        if crowded := [n for n, c in counts.items() if c > 1]:
            raise RuntimeError(f"more than one Worker on {crowded}: is a --worker_id used twice?")

    def write(self, name: str, sample) -> None:
        """Write and retry while a KEEP_ALL queue is full."""
        writer = self.writer(name)
        while True:
            try:
                writer.write(sample)
                return
            except DdsTimeout:
                print(f"[dds] write to {name} blocked for {WRITE_TIMEOUT_S}s, retrying")

    def close(self) -> None:
        self.dp.close()
