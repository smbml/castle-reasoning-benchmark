"""Data generation gallery."""

import argparse
import collections
import hashlib
import html
import importlib.metadata
import json
import shutil
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import castle_reasoning.data.generation.generator as generator
import castle_reasoning.data.generation.geometry as geometry
import castle_reasoning.data.generation.render as render


def target_id(candidate):
    return f"T{candidate + 1:04d}"


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _page(title, body, prefix=""):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Construction targets</title>
<link rel="stylesheet" href="{prefix}style.css"></head>
<body><main>{body}</main><footer>Reversible construction targets · Independent-load generation · Static geometry review</footer></body></html>'''


def _card(target):
    ident = target_id(target["candidate"])
    d = target["descriptors"]
    return f'''<a class="card" href="{ident}/index.html">
<img src="{ident}/front.png" width="1000" height="760" loading="lazy" alt="Target {ident}: {d["part_count"]} parts and {d["multi_support_parts"]} parts resting on multiple supports">
<div class="card-copy"><strong>{ident}</strong><span>{d["part_count"]} parts · {d["shape_types"]} shapes</span>
<small>{d["depth"]} support levels · {d["multi_support_parts"]} shared supports · {len(target["voids"])} reserved openings</small></div></a>'''


def _detail(target, directory, navigation):
    ident = target_id(target["candidate"])
    d = target["descriptors"]
    parts = [
        geometry.Part(tuple(p["origin"]), tuple(p["size"])) for p in target["parts"]
    ]
    rows = []
    incoming = collections.defaultdict(list)
    for contact in target["supports"]:
        incoming[contact["upper"]].append(str(contact["lower"] + 1))
    for i, p in enumerate(parts):
        kind, _ = render.material(p)
        dims = " × ".join(f"{s / 2:g}" for s in p.size)
        origin = ", ".join(f"{v / 2:g}" for v in p.origin)
        rows.append(
            f"<tr><td>{i + 1}</td><td>{kind}</td><td>{dims}</td><td>{origin}</td><td>{', '.join(incoming[i]) or 'ground'}</td></tr>"
        )
    operation_rows = "".join(
        f"<li><b>{op['operation'].replace('_', ' ')}</b> · parts {op['first'] + 1}–{op['first'] + op['count']}</li>"
        for op in target["operations"]
    )
    body = f'''<nav><a href="../index.html">← All targets</a><span>{navigation}</span></nav>
<header><p class="eyebrow">TARGET GEOMETRY / {ident}</p><h1>{ident}</h1>
<p>{d["part_count"]} parts · {d["shape_types"]} shape types · {d["depth"]} support levels · {d["multi_support_parts"]} multi-support parts</p>
<p class="status">Every construction prefix certified · Reverse disassembly witnessed · Robot feasibility unverified</p></header>
<div class="views"><figure><a href="front.png"><img src="front.png" width="1000" height="760" alt="{ident} perspective A"></a><figcaption>Perspective A · camera azimuth 225°, elevation 28°</figcaption></figure>
<figure><a href="back.png"><img src="back.png" width="1000" height="760" alt="{ident} opposite perspective"></a><figcaption>Perspective B · opposite side, same elevation</figcaption></figure>
<figure><a href="top.png"><img src="top.png" width="1000" height="760" alt="{ident} overhead view"></a><figcaption>Plan view · upper surfaces occlude parts below</figcaption></figure>
<figure><a href="elevation.png"><img src="elevation.png" width="1000" height="760" alt="{ident} front elevation"></a><figcaption>Front elevation · exposes vertical gaps and spans</figcaption></figure></div>
<div class="legend"><span class="pillar">Pillars</span><span class="beam">Beams</span><span class="slab">Slabs / plates</span><span class="block">Other blocks</span></div>
<section><h2>What is connected?</h2><p>Arrows run from the supporting part to the part above it. IDs match the part table. This graph shows horizontal load-bearing contacts; side contacts and the ground node are omitted.</p>
<a href="supports.svg"><img class="graph" src="supports.svg" alt="Numbered support graph for {ident}"></a></section>
<section class="columns"><div><h2>Target measurements</h2><dl>
<dt>Envelope, x × y × z</dt><dd>{" × ".join(f"{v / 2:g}" for v in d["dimensions"])} u</dd>
<dt>Solid fraction of envelope</dt><dd>{100 * d["fill_fraction"]:.1f}%</dd>
<dt>Branching supports</dt><dd>{d["branches"]}</dd><dt>Connected components</dt><dd>{d["components"]}</dd>
<dt>Reserved openings</dt><dd>{len(target["voids"])}</dd><dt>Seed / candidate index</dt><dd>{target["seed"]} / {target["candidate"]}</dd></dl>
<p><a class="button" href="target.json" download>Download target JSON</a></p>
<p class="muted">u is a nominal block unit, not a physical scale. The JSON uses half-unit integer coordinates. Displayed dimensions and corners are in u. Cuboids are axis-aligned, with identity orientation in world coordinates.</p></div>
<div><h2>Generation trace</h2><ol class="trace">{operation_rows}</ol><p>Construction: {" → ".join(str(i + 1) for i in target["construction_order"])}.<br>Disassembly: reverse this order. Every primitive placement passes independent-load balance and vertical object access.</p><p class="muted">These are object placements, not robot controls.</p></div></section>
<section><h2>Parts and dimensions</h2><div class="table-scroll"><table><thead><tr><th>ID</th><th>Shape class</th><th>XYZ dimensions, u</th><th>Minimum corner, u</th><th>Supported by</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>
<aside>For feedback, reference this gallery version, <b>{ident}</b>, and part IDs. Each added weight can reach ground without counterweights from other parts. The resulting object sequence is quasi-statically stable and vertically accessible. Robot grasp/collision feasibility and dynamics remain unverified.</aside>'''
    (directory / "index.html").write_text(_page(ident, body, "../"))


def _overview(targets, output):
    columns, cell_width, cell_height = 4, 360, 305
    canvas = Image.new("RGB", (columns * cell_width, 80 + 5 * cell_height), "#f4f5ef")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 17)
    heading = ImageFont.truetype(font_path, 25)
    draw.text(
        (22, 20),
        "20 procedural construction targets · geometry review",
        font=heading,
        fill="#18342d",
    )
    for i, target in enumerate(targets):
        ident = target_id(target["candidate"])
        image = Image.open(output / ident / "front.png").convert("RGB")
        image.thumbnail((cell_width - 12, cell_height - 40))
        x, y = (i % columns) * cell_width, 80 + (i // columns) * cell_height
        canvas.paste(image, (x + (cell_width - image.width) // 2, y))
        draw.text(
            (x + 16, y + cell_height - 32),
            f"{ident} · {target['descriptors']['part_count']} parts",
            font=font,
            fill="#18342d",
        )
    canvas.save(output / "overview.png")


def build(output: Path, seed=20260915, count=240):
    """Generate, validate, select, render, and export one review bundle."""
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    pool = generator.generate_pool(seed, count)
    generation_seconds = time.perf_counter() - started
    targets = pool["targets"]
    selected_ids = generator.select_coverage(targets, 20)
    by_id = {t["candidate"]: t for t in targets}
    selected = [by_id[i] for i in selected_ids]
    uncurated = targets[:12]
    displayed_ids = list(
        dict.fromkeys(selected_ids + [t["candidate"] for t in uncurated])
    )
    print(
        f"Generated {len(targets)}/{count} retained targets in {generation_seconds:.2f}s; rendering {len(displayed_ids)}",
        flush=True,
    )
    rejected = collections.Counter(
        t["rejection"] for t in pool["attempts"] if not t["accepted"]
    )
    render_started = time.perf_counter()
    for i, candidate in enumerate(displayed_ids):
        target = by_id[candidate]
        ident = target_id(candidate)
        directory = output / ident
        directory.mkdir()
        parts = [
            geometry.Part(tuple(p["origin"]), tuple(p["size"])) for p in target["parts"]
        ]
        for name, az, el in [
            ("front", 225, 28),
            ("back", 45, 28),
            ("top", 270, 90),
            ("elevation", 270, 0),
        ]:
            render.render(parts, directory / f"{name}.png", az, el)
        render.support_graph(parts, directory / "supports.svg")
        _write_json(directory / "target.json", target)
        previous = target_id(displayed_ids[(i - 1) % len(displayed_ids)])
        following = target_id(displayed_ids[(i + 1) % len(displayed_ids)])
        _detail(
            target,
            directory,
            f'<a href="../{previous}/index.html">Previous</a> · <a href="../{following}/index.html">Next →</a>',
        )
    _overview(selected, output)
    _write_json(output / "targets.json", targets)
    _write_json(output / "attempts.json", pool["attempts"])
    _write_json(
        output / "selection.json",
        {
            "coverage": selected_ids,
            "first_accepted": [t["candidate"] for t in uncurated],
        },
    )
    summary = {
        "generator_version": "independent-load-construction-v1",
        "seed": seed,
        "attempted": count,
        "retained": len(targets),
        "rejected": dict(rejected),
        "retained_fraction": len(targets) / count,
        "generation_seconds": generation_seconds,
        "render_export_seconds": time.perf_counter() - render_started,
        "selected": 20,
        "uncurated": len(uncurated),
        "rendered": len(displayed_ids),
        "unique_geometry": len({t["fingerprint"] for t in targets}),
        "topology_signatures": len({t["topology_fingerprint"] for t in targets}),
        "parts_range": [
            min(t["descriptors"]["part_count"] for t in targets),
            max(t["descriptors"]["part_count"] for t in targets),
        ],
        "versions": {
            p: importlib.metadata.version(p) for p in ("numpy", "scipy", "Pillow")
        },
        "geometry_unit": "half a nominal block unit; no physical scale",
        "physical_simulation_steps": 0,
        "static_contact_inset": 0.05,
        "timing_scope": "CPU generation/checks and render/export; excludes ZIP compression",
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.iterdir())
            if path.suffix in (".py", ".css")
        },
    }
    rejection_rows = "".join(
        f"<tr><td>{reason.replace('_', ' ')}</td><td>{number}</td></tr>"
        for reason, number in rejected.items()
    )
    body = f"""<nav><a href="#coverage">Coverage gallery</a><a href="#uncurated">Uncurated sample</a><a href="#distribution">Distribution & method</a></nav>
<header><p class="eyebrow">RESEARCH REVIEW / INDEPENDENT-LOAD GENERATION</p><h1>Reversible construction targets</h1>
<p class="lede">A generator of mixed cuboids, shared supports, open frames, and layered structures. Inspect the shapes before we teach a robot to build them.</p>
<p class="status">Every construction prefix certified · No dynamics or robot simulation</p>
<p>Fresh generator version: IDs are seed indices and may denote different geometry from the historical gallery. Each target includes a stable construction order and its reverse disassembly order.</p></header>
<div class="stats"><div><b>{count}</b><span>seeded candidates</span></div><div><b>{len(targets)}</b><span>retained after checks</span></div><div><b>{summary["parts_range"][0]}–{summary["parts_range"][1]}</b><span>parts per retained target</span></div><div><b>{generation_seconds:.1f}s</b><span>CPU generation + checks</span></div></div>
<div class="legend"><span class="pillar">Pillars</span><span class="beam">Beams</span><span class="slab">Slabs / plates</span><span class="block">Other blocks</span></div>
<section id="coverage"><div class="section-title"><h2>20 targets selected for coverage</h2><a href="overview.png">Download overview ↗</a></div>
<p>Click a target for four views, support graph, dimensions, and its generation trace. This selection spreads across measured structural features; it is not a uniform sample.</p><div class="cards">{"".join(_card(t) for t in selected)}</div></section>
<section id="uncurated"><h2>The first 12 accepted targets</h2><p>In seed order, before coverage selection. Rejected candidates are counted below and recorded in the full attempt log. Some targets also appear above.</p><div class="cards">{"".join(_card(t) for t in uncurated)}</div></section>
<section id="distribution"><h2>What distribution did we generate?</h2><div class="columns"><div><h3>Sampling</h3><p>Each candidate starts from a portal, courtyard frame, or slab. A seeded operation mixture composes extensions, branching frames, spans, perpendicular beams/slabs, and enclosures. Independent size and height strata cover compact, broad, and tall targets.</p>
<p>{count} seed indices are fixed in advance. Growth proposals that collide, enter a reserved opening, exceed the envelope, float, obstruct vertical placement, or need balancing counterweights are rejected without mutating the current target. Every part within a multi-part operation is checked. Whole targets that fail the size or static-load gate remain in the attempt log.</p>
<h3>Coverage selection</h3><p>Farthest-point sampling over normalized part count, height, footprint, solid fraction, support depth, branching, multi-support parts, shape count, and maximum horizontal aspect ratio. Seed: <code>{seed}</code>. Colors encode shape class and never count toward novelty.</p></div>
<div><h3>Acceptance accounting</h3><table><tbody><tr><td>Retained unique geometry</td><td>{len(targets)} / {count} ({100 * len(targets) / count:.1f}%)</td></tr>{rejection_rows}<tr><td>Distinct support-graph signatures</td><td>{summary["topology_signatures"]}</td></tr></tbody></table>
<p>Geometry is deduplicated under XY translation, part reordering, and global quarter-turn rotation. Graph signatures use directed color refinement, not an exact graph-isomorphism proof.</p>
<h3>Checks and limits</h3><p>Every retained structure has one object-contact component, nonintersecting solids, and positive-area vertical supports. For each placement, a unit weight on the new part must balance through all older supports with their weights set to zero. Each load is independently supported; positive sums certify every construction prefix. Contact corners are inset 5%. The completed structure also passes the ordinary volume-weighted equilibrium check.</p>
<p>Vertical insertion is checked for every part, so reversing the sequence gives clear object extraction. The guarantee assumes quasi-static gravity and horizontal nonadhesive contacts. Gripper/arm access, staging capacity, and dynamic stability still require qualification.</p></div></div></section>
<section><h2>Artifacts</h2><p class="downloads"><a class="button" href="targets.json" download>All accepted targets</a><a class="button" href="attempts.json" download>Every attempt</a><a class="button" href="summary.json">Timing & summary</a><a class="button" href="proposal.md">Generation method</a><a class="button" href="gallery.zip" download>Offline gallery ZIP</a></p>
<p>Reference a target ID and part IDs when giving feedback. Dimensions use nominal units; physical scale is deferred.</p></section>"""
    (output / "index.html").write_text(_page("Gallery", body))
    shutil.copyfile(Path(__file__).with_name("style.css"), output / "style.css")
    proposal = Path(__file__).with_name("resources") / "generation.md"
    shutil.copyfile(proposal, output / "proposal.md")
    summary["render_export_seconds"] = time.perf_counter() - render_started
    summary["generation_and_render_seconds"] = time.perf_counter() - started
    _write_json(output / "summary.json", summary)
    with zipfile.ZipFile(output / "gallery.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "gallery.zip":
                archive.write(path, path.relative_to(output))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--count", type=int, default=240)
    args = parser.parse_args()
    build(args.output, args.seed, args.count)
