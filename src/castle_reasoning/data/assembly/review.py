"""Data assembly review."""

import html
import json
from pathlib import Path


def write_index(output: Path):
    """Publishes current metrics and one card per saved instance."""
    summary = json.loads((output / "summary.json").read_text())
    cards = []
    for directory in sorted(output.glob("instance[0-9][0-9][0-9]")):
        metadata = json.loads((directory / "metadata.json").read_text())
        number = metadata["instance"]
        prefix = directory.name
        image = (
            f'<img src="{prefix}/target.png" alt="Target assembly {number}">'
            if (directory / "target.png").exists()
            else "<p>Rendering pending</p>"
        )
        video = (
            f'<video controls preload="none" poster="{prefix}/target.png" src="{prefix}/solve.mp4"></video>'
            if (directory / "solve.mp4").exists()
            else ""
        )
        cards.append(f'''<article id="instance{number}"><h2>Instance {number:02d}</h2>
        <p>Candidate {metadata["candidate"]} · column heights {metadata["heights"]} · {metadata["action_count"] * 0.02:.1f} s demonstration</p>
        {image}{video}<p><a href="{prefix}/initial.png">Initial scene</a> · <a href="{prefix}/target.png">Sampled goal</a> · <a href="{prefix}/achieved.png">Achieved scene</a> ·
        <a href="{prefix}/metadata.json">Verification</a> · <a href="{prefix}/trajectory.npz">Actions and states</a> ·
        <a href="{prefix}/observations.npz">RGB and proprioception</a></p></article>''')
    metrics = [
        (
            "Saved / requested",
            f"{summary['saved_targets']} / {summary.get('requested_targets', 20)}",
        ),
        (
            "Verified / sampled targets",
            f"{summary['verified_targets']} / {summary['attempted_targets']} ({100 * summary['target_yield']:.1f}%)",
        ),
        (
            "Data generation, including setup and failed candidates",
            f"{summary['data_generation_seconds']:.1f} s",
        ),
        (
            "Saved action targets / generation second",
            f"{summary['saved_action_targets_per_second']:.1f}",
        ),
        (
            "End-to-end time including RGB and review artifacts",
            f"{summary.get('end_to_end_seconds', 0):.1f} s",
        ),
    ]
    rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>"
        for name, value in metrics
    )
    refresh = (
        '<meta http-equiv="refresh" content="30">'
        if summary["status"] != "complete"
        else ""
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">{refresh}
    <title>Verified block-building data</title><style>
    body{{font:17px/1.5 system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#17202b;background:#f5f7fa}}
    h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;border-bottom:1px solid #d7dee8;padding:10px}}
    article{{background:white;padding:20px;margin:24px 0;border:1px solid #d7dee8;border-radius:8px}}img,video{{width:100%;display:block;margin:12px 0}}a{{color:#135bb4}}
    .note{{max-width:950px}}code{{font-size:14px}}</style>
    <h1>Verified block-building data</h1><p>Status: <strong>{html.escape(summary["status"])}</strong> · MJ-Warp physics · OGBench UR5e / Robotiq</p>
    <p class="note">Qualification dataset: eight rigid cubes in randomized supported column assemblies. This is the first generator test; overhangs, temporary supports and general castle construction are not yet covered.</p>
    <table>{rows}</table><p>Video selection is fixed: one-based instances 1, 3, …, 19. Each video is real-time, two-camera reconstruction from the generated parked state, followed by return-home and settling.</p>
    <p class="note">Targets are instantiated and physically disassembled. Construction reverses the object order using newly computed forward controls. Admission requires target settling, disassembly, construction, independent recorded-action replay, <=5 mm center error, <=5° orientation error, a one-second stable hold, all required support contacts and an open gripper at home. The sampled goal is fixed before disassembly and shown separately from the achieved endpoint. The training trajectory contains T actions and T+1 states; RGB comes from those recorded simulator states.</p>
    <p><a href="targets-overview.png">All 20 sampled goals</a> · <a href="audit.json">Independent artifact audit</a> · <a href="README.md">Implemented methodology</a> · <a href="summary.json">Complete timing and yield report</a> · <a href="attempts.json">Every attempt and rejection</a> · <a href="config.json">Resolved configuration</a> · <a href="provenance.json">Source provenance</a></p>
    {"".join(cards)}</html>"""
    temporary = output / "index.tmp.html"
    temporary.write_text(page)
    temporary.replace(output / "index.html")


if __name__ == "__main__":
    import sys

    write_index(Path(sys.argv[1]))
