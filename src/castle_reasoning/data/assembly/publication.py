"""Data assembly publication."""

import copy
import threading
from pathlib import Path

import castle_reasoning.runtime.diagnostics.timing as timing
from castle_reasoning.runtime.diagnostics import io_timing


class SummaryPublisher:
    """One writer, one pending snapshot, and a checked final drain on context exit."""

    def __init__(self, path):
        self.path = Path(path)
        self._condition = threading.Condition()
        self._thread = None
        self._pending = None
        self._closing = False
        self._error = None

    def __enter__(self):
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("Summary publisher already started")
            self._thread = threading.Thread(target=self._write, name="summary-writer")
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as error:
            if exc is None:
                raise
            exc.add_note(f"Summary publication also failed: {error}")
        return False

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError(
                f"Summary publication failed: {self.path}"
            ) from self._error

    def check(self):
        """Report an asynchronous failure at a collector chunk boundary."""
        with self._condition:
            self._check_error()

    @timing.measured("publication.submit")
    def submit(self, value):
        """Copy under the caller's records lock; replace any older pending snapshot."""
        with self._condition:
            self._check_error()
            if self._thread is None or self._closing:
                raise RuntimeError("Summary publisher is not running")
            self._pending = copy.deepcopy(value)
            self._condition.notify()

    @timing.measured("publication.drain")
    def close(self):
        """Stop submissions and wait for the last accepted snapshot to be written."""
        with self._condition:
            self._closing = True
            thread = self._thread
            self._condition.notify()
        if thread is not None:
            thread.join()
        self.check()

    def _write(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._pending is not None or self._closing
                    )
                    if self._pending is None:
                        return
                    snapshot, self._pending = self._pending, None
                # No producer/records lock is held across encoding or filesystem I/O.
                with timing.span("publication.write", path=str(self.path)):
                    io_timing.write_json(self.path, snapshot)
        except Exception as error:
            with self._condition:
                self._error = error
                self._pending = None
                self._closing = True
                self._condition.notify()
