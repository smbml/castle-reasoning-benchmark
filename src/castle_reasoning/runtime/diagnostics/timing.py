"""Runtime diagnostics timing."""

import contextlib
import contextvars
import functools
import inspect
import json
import os
import threading
import time
from pathlib import Path

_recorder = None


_context = contextvars.ContextVar("timing_context", default={})


_depth = contextvars.ContextVar("timing_depth", default=0)


class Recorder:
    """Buffer complete spans and mark incomplete process logs explicitly."""

    def __init__(self, directory, flush_events, context):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{os.getpid()}-{time.perf_counter_ns()}.jsonl"
        self.stream = self.path.open("x")
        self.lock = threading.Lock()
        self.events = []
        self.limit = flush_events
        self.last_flush = time.monotonic()
        self.context = context
        self.stream.write(
            json.dumps(dict(type="start", pid=os.getpid(), **context)) + "\n"
        )
        self.stream.flush()

    def add(self, record):
        with self.lock:
            self.events.append(record)
            if (
                len(self.events) >= self.limit
                or time.monotonic() - self.last_flush > 10
            ):
                self._flush()

    def _flush(self):
        start, cpu = time.perf_counter_ns(), time.thread_time_ns()
        for record in self.events:
            self.stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.stream.flush()
        self.events.clear()
        self.last_flush = time.monotonic()
        # Keep the flush itself visible without recursively triggering a flush.
        self.events.append(
            dict(
                name="trace.flush",
                start_ns=start,
                end_ns=time.perf_counter_ns(),
                cpu_ns=time.thread_time_ns() - cpu,
                pid=os.getpid(),
                tid=threading.get_native_id(),
                depth=_depth.get(),
                context=current(),
            )
        )

    def close(self):
        with self.lock:
            self._flush()
            for record in self.events:
                self.stream.write(json.dumps(record) + "\n")
            self.events.clear()
            self.stream.write('{"type":"end","dropped_events":0}\n')
            self.stream.close()


def start(directory, *, flush_events=4096, **context):
    global _recorder
    if _recorder is not None:
        raise RuntimeError("Tracing is already active")
    if flush_events < 1:
        raise ValueError("Positive trace buffer size required")
    _recorder = Recorder(directory, flush_events, context)


def stop():
    global _recorder
    if _recorder is not None:
        _recorder.close()
        _recorder = None


def current():
    return {} if _recorder is None else _recorder.context | _context.get()


def child_config(**context):
    if _recorder is None:
        return None
    return dict(directory=str(_recorder.directory), context=current() | context)


@contextlib.contextmanager
def span(name, /, **context):
    if _recorder is None:
        yield
        return
    merged = current() | context
    context_token = _context.set(merged)
    level = _depth.get()
    depth_token = _depth.set(level + 1)
    start_ns, cpu = time.perf_counter_ns(), time.thread_time_ns()
    try:
        yield
    finally:
        end_ns, cpu_ns = time.perf_counter_ns(), time.thread_time_ns() - cpu
        _depth.reset(depth_token)
        _context.reset(context_token)
        _recorder.add(
            dict(
                name=name,
                start_ns=start_ns,
                end_ns=end_ns,
                cpu_ns=cpu_ns,
                pid=os.getpid(),
                tid=threading.get_native_id(),
                depth=level,
                context=merged,
            )
        )


def measured(name, *fields):
    """Decorate only named phase boundaries, never per-frame inner loops."""

    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def call(*args, **kwargs):
            if _recorder is None:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs).arguments if fields else {}
            context = {key: bound[key] for key in fields if key in bound}
            if "world" in context:
                context["slot"] = context.pop("world")
            with span(name, **context):
                return function(*args, **kwargs)

        return call

    return decorate


def mark(name, /, **context):
    if _recorder is not None:
        with span(name, **context):
            pass


def checkpoint():
    if _recorder is not None:
        with _recorder.lock:
            _recorder._flush()
