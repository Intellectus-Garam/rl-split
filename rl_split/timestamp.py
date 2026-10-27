"""Prefix every line written to stdout/stderr with a timestamp."""

from __future__ import annotations

import atexit
import sys
import threading
from datetime import datetime


class TimestampedStream:
    """Wraps a text stream. Lines are buffered per thread and written whole, so
    prints from the sender thread do not get spliced into the main thread's lines."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._local = threading.local()

    def write(self, text: str) -> int:
        buf = getattr(self._local, "buf", "") + text
        *lines, self._local.buf = buf.split("\n")
        if lines:
            stamp = datetime.now().strftime("[%m-%d %H:%M:%S.%f")[:-3] + "] "
            with self._lock:
                self._stream.write("".join(f"{stamp}{line}\n" for line in lines))
                self._stream.flush()
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def close_line(self) -> None:
        """Write out a trailing partial line (at exit)."""
        if getattr(self._local, "buf", ""):
            self.write("\n")

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install() -> None:
    """Covers Python-level output only; native libraries writing to fd 1/2 directly are not stamped."""
    for name in ("stdout", "stderr"):
        stream = TimestampedStream(getattr(sys, name))
        setattr(sys, name, stream)
        atexit.register(stream.close_line)
