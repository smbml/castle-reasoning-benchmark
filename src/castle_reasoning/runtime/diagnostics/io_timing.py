"""Runtime diagnostics io_timing."""

import contextlib
import json
from pathlib import Path

import numpy as np

import castle_reasoning.runtime.diagnostics.timing as timing


class File:
    """Forward to the original buffered file; times are application-visible I/O."""

    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, data):
        with timing.span("io.write", bytes=len(data)):
            return self.stream.write(data)

    def read(self, size=-1):
        with timing.span("io.read", requested_bytes=size):
            return self.stream.read(size)

    def seek(self, *args):
        with timing.span("io.seek"):
            return self.stream.seek(*args)

    def flush(self):
        with timing.span("io.flush"):
            return self.stream.flush()

    def close(self):
        with timing.span("io.close"):
            return self.stream.close()


def savez_compressed(path, **arrays):
    if timing.child_config() is None:
        return np.savez_compressed(path, **arrays)
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    with timing.span("io.npz_save", path=path):
        with timing.span("io.open"):
            stream = File(open(path, "w+b"))
        try:
            np.savez_compressed(stream, **arrays)
        finally:
            stream.close()


@contextlib.contextmanager
def load(path):
    if timing.child_config() is None:
        with np.load(path) as saved:
            yield saved
        return
    with timing.span("io.npz_load", path=str(path)):
        with timing.span("io.open"):
            stream = File(Path(path).open("rb"))
        try:
            with np.load(stream) as saved:
                yield saved
        finally:
            stream.close()


def write_json(path: Path, value):
    """Atomically publishes strict JSON so partial writes are never read."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with timing.span("io.json_encode"):
        encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with timing.span("io.json_write", path=str(path), bytes=len(encoded.encode())):
        temporary.write_text(encoded)
        temporary.replace(path)
