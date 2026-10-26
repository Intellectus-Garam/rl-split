"""Background thread that serializes and writes samples so the caller does not block."""

from __future__ import annotations

import queue
import threading


class Sender:
    """Writes (topic, make_sample) jobs in order on its own thread.

    make_sample is called on the sender thread, so building the message (the
    expensive part for 4MB samples) also leaves the caller's thread.
    """

    def __init__(self, node, max_pending: int = 30) -> None:
        self._node = node
        self._jobs: queue.Queue = queue.Queue(maxsize=max_pending)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="dds-sender", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                topic, make_sample, log = job
                sample = make_sample()
                self._node.write(topic, sample)
                if log is not None:
                    print(log(sample))
            except BaseException as e:  # surfaced to the caller on the next put/flush
                self._error = e
            finally:
                self._jobs.task_done()

    def _check(self) -> None:
        if self._error is not None:
            raise RuntimeError("sender thread failed") from self._error

    def put(self, topic: str, make_sample, log=None) -> None:
        """log(sample) -> str is printed after the sample has been written."""
        self._check()
        self._jobs.put((topic, make_sample, log))

    def flush(self) -> None:
        """Wait until every queued sample has been handed to DDS."""
        self._jobs.join()
        self._check()

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join()
