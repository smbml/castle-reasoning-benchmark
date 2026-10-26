"""Runtime rendering replay."""

import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import castle_reasoning.runtime.rendering.cameras as rendering_cameras


def render_review(
    model, qpos, qvel, directory: Path, video=True, goal_qpos=None, goal_qvel=None
):
    """Saves raw endpoints and a real-time, two-view 25 Hz H.264 review video."""
    start = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    cameras = rendering_cameras.Cameras(model)
    process = None
    try:
        for name, index in [("initial", 0), ("achieved", -1)]:
            frames = cameras.frames(qpos[index], qvel[index])
            Image.fromarray(np.concatenate(frames, axis=1)).save(
                directory / f"{name}.png"
            )
        target_frame = cameras.frames(
            goal_qpos if goal_qpos is not None else qpos[-1],
            goal_qvel if goal_qvel is not None else qvel[-1],
        )
        Image.fromarray(np.concatenate(target_frame, axis=1)).save(
            directory / "target.png"
        )
        if video:
            temporary = directory / "solve.tmp.mp4"
            process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "1024x384",
                    "-r",
                    "25",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                stdin=subprocess.PIPE,
            )
            for index in range(0, len(qpos), 2):
                frames = cameras.frames(qpos[index], qvel[index])
                process.stdin.write(np.concatenate(frames, axis=1).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("Video encoding failed.")
            temporary.replace(directory / "solve.mp4")
    finally:
        cameras.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    return time.perf_counter() - start


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--world", type=int, default=0)
    args = p.parse_args()
    trace = np.load(args.path, allow_pickle=False)
    qpos = trace["qpos"][:, args.world]
    qvel = trace["qvel"][:, args.world]
    model = mujoco.MjModel.from_binary_path(str(args.path.parent / "model.mjb"))
    if "initial_qpos" in trace:
        qpos = np.concatenate(
            [trace["initial_qpos"][args.world : args.world + 1], qpos]
        )
        qvel = np.concatenate(
            [trace["initial_qvel"][args.world : args.world + 1], qvel]
        )
    print(
        render_review(
            model,
            qpos,
            qvel,
            args.path.parent / f"{args.path.stem}-review-{args.world}",
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
