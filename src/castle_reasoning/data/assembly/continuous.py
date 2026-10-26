"""Data assembly continuous."""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

# Set limits before numerical libraries initialize their CPU thread pools.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import castle_reasoning.data.assembly.continuous_collect as continuous_collect  # noqa: E402
import castle_reasoning.runtime.diagnostics.telemetry as telemetry  # noqa: E402
import castle_reasoning.runtime.diagnostics.timing as timing  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=20)
    parser.add_argument("--attempts-per-target", type=int, default=1)
    parser.add_argument("--planning-seconds", type=float, default=8,
                        help="Calling-thread CPU seconds per local planning query.")
    parser.add_argument("--planning-workers", type=int,
                        help="Concurrent planning queries; defaults to allocated CPUs minus two.")
    parser.add_argument("--planning-watchdog-seconds", type=float, default=120,
                        help="Cooperative wall watchdog after planner admission.")
    parser.add_argument("--max-rollbacks", type=int, default=0,
                        help="Per stage, failures that may resume from the last "
                        "passed boundary instead of ending the attempt.")
    parser.add_argument("--planner-start-method", choices=("spawn", "forkserver"),
                        default="spawn",
                        help="forkserver shares preloaded imports between planner "
                        "processes, cutting their memory by about two thirds.")
    parser.add_argument("--wall-seconds", type=float, default=3600)
    parser.add_argument("--physics-timestep", type=float, default=0.002)
    parser.add_argument("--solver", choices=("cg", "newton"))
    parser.add_argument("--solver-iterations", type=int)
    parser.add_argument("--line-search-threads", type=int, choices=(32, 128))
    parser.add_argument(
        "--friction-cone",
        choices=("elliptic", "pyramidal"),
        help="MuJoCo friction approximation; omitted preserves the model default.",
    )
    parser.add_argument(
        "--replay-settle-seconds",
        type=int,
        default=10,
        help="Bounded verification-only quiet hold; zero disables it.",
    )
    parser.add_argument(
        "--contact-impedance",
        type=float,
        nargs=2,
        metavar=("D0", "DWIDTH"),
        help="Object/table contact impedance; omitted preserves original physics.",
    )
    parser.add_argument("--pause-planning-worlds", action="store_true",
                        help="Advance ready worlds while planners wait; requires patched runtime.")
    parser.add_argument("--skip-paused-preparation", action="store_true",
                        help="Use inactive placeholders for frozen worlds; requires paused planning.")
    parser.add_argument("--preparation-mode", choices=("owner", "producer"), default="owner")
    parser.add_argument("--queue-order", choices=("submitted", "longest_first"),
                        default="submitted",
                        help="Start targets with the most parts first within each "
                        "repeat round; attempt ids and seeds are unchanged.")
    parser.add_argument("--preparation-workers", type=int,
                        help="Concurrent preparation permits; no additional processes are created.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timing-output", type=Path,
                        help="Opt-in timing directory on node-local scratch.")
    args = vars(parser.parse_args())
    trace = args.pop("timing_output")
    args["plans"] = [json.loads(path.read_text()) for path in args["plans"]]
    if trace is not None:
        trace.mkdir(parents=True, exist_ok=False)
        (trace / "metadata.json").write_text(json.dumps({
            "argv": sys.argv, "pid": os.getpid(), "cwd": os.getcwd(),
            "start_unix_time": time.time(), "start_monotonic_ns": time.perf_counter_ns(),
        }, indent=2) + "\n")
        timing.start(trace)
    try:
        with telemetry.Monitor(trace) if trace is not None else contextlib.nullcontext():
            result = continuous_collect.execute(**args)
    finally:
        timing.stop()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
