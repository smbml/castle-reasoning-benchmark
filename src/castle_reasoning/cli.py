"""Cli."""

import argparse
import contextlib
import json
import sys

import castle_reasoning.data.download as data_download
import castle_reasoning.runtime.diagnostics.checks as checks


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-runtime")
    download = commands.add_parser("download")
    download.add_argument(
        "--tasks", required=True, help="Comma-separated task IDs"
    )
    download.add_argument("--root")
    download.add_argument("--revision")
    download.add_argument("--max-download-mb", type=float, default=256)
    args = parser.parse_args()
    try:
        if args.command == "check-runtime":
            with contextlib.redirect_stdout(sys.stderr):
                result = checks.check_runtime()
        else:
            receipt = data_download.download_tasks(
                [
                    value.strip()
                    for value in args.tasks.split(",")
                    if value.strip()
                ],
                args.root,
                revision=args.revision,
                max_bytes=int(args.max_download_mb * 1024 * 1024),
            )
            result = {
                "repo_id": receipt["repo_id"],
                "revision": receipt["revision"],
                "tasks": receipt["tasks"],
                "verified_files": len(receipt["files"]),
                "selected_bytes": receipt["selected_bytes"],
            }
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
