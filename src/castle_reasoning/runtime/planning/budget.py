"""Runtime planning budget."""

import contextlib
import math
import time

import castle_reasoning.runtime.diagnostics.timing as timing
import castle_reasoning.runtime.planning.arm as arm


class Budget:
    """One calling thread's query scopes; never hold a scope during simulation.

    CPU limits assume single-threaded numerical libraries. The wall watchdog is
    cooperative: it detects overruns at checks, not inside a stuck native call.
    """

    def __init__(self, permits=None, watchdog_seconds=120):
        if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
            raise ValueError("Positive finite planner watchdog required")
        self.permits = permits
        self.watchdog_seconds = watchdog_seconds
        self._limits = []
        self.last_query = {}

    @property
    def active(self):
        """Whether this thread currently owns a planning scope."""
        return bool(self._limits)

    def check(self):
        """Check global wall, watchdog wall, and local CPU limits in that order."""
        if not self._limits:
            return
        deadline, watchdog, cpu_deadline = self._limits[-1]
        now = time.perf_counter()
        if now >= deadline:
            raise arm.PlanningBudgetError("experiment_wall_budget")
        if now >= watchdog:
            raise arm.PlanningBudgetError("planning_watchdog")
        if time.thread_time() >= cpu_deadline:
            raise arm.PlanningBudgetError("planning_cpu_budget")

    @contextlib.contextmanager
    def scope(self, cpu_seconds, deadline=float("inf")):
        """Admit a query, start its local clocks, and release on every exit."""
        if not math.isfinite(cpu_seconds) or cpu_seconds <= 0 or math.isnan(deadline):
            raise ValueError("Positive finite CPU budget and valid wall deadline required")
        acquired = False
        outer = not self._limits
        started = time.perf_counter()
        if outer:
            self.last_query = {"admission_seconds": 0.0}
        parent = self._limits[-1] if self._limits else (math.inf,) * 3
        deadline = min(deadline, parent[0])
        if time.perf_counter() >= deadline:
            raise arm.PlanningBudgetError("experiment_wall_budget")
        try:
            if not self._limits and self.permits is not None:
                remaining = deadline - time.perf_counter()
                with timing.span("planning.admission"):
                    acquired = self.permits.acquire(
                        timeout=None if math.isinf(remaining) else max(0, remaining)
                    )
                if not acquired:
                    raise arm.PlanningBudgetError("experiment_wall_budget")
            if outer:
                admitted = time.perf_counter()
                self.last_query.update(admitted_seconds=admitted,
                                       admission_seconds=admitted - started)
            self._limits.append((
                deadline,
                min(parent[1], time.perf_counter() + self.watchdog_seconds),
                min(parent[2], time.thread_time() + cpu_seconds),
            ))
            try:
                self.check()
                yield
                self.check()
            finally:
                self._limits.pop()
        finally:
            if outer:
                self.last_query["released_seconds"] = time.perf_counter()
            if acquired:
                self.permits.release()
