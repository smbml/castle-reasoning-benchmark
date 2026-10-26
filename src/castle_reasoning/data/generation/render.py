"""Data generation render."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import castle_reasoning.data.generation.geometry as geometry


def material(part: geometry.Part) -> tuple[str, tuple[int, int, int]]:
    width, length, height = part.size
    if height > max(width, length):
        return "pillar", (204, 146, 92)
    if min(width, length) >= 4 and height <= 2:
        return "slab", (94, 139, 166)
    if max(width, length) >= 2 * min(width, length) and max(width, length) > height:
        return "beam", (84, 146, 132)
    return "block", (197, 114, 93)


def faces(part):
    low, high = np.asarray(part.origin), np.asarray(part.end)
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for side in (0, 1):
            vertices = []
            for a, b in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                point = low.astype(float).copy()
                point[axis] = high[axis] if side else low[axis]
                point[other[0]] = high[other[0]] if a else low[other[0]]
                point[other[1]] = high[other[1]] if b else low[other[1]]
                vertices.append(point)
            normal = np.zeros(3)
            normal[axis] = 1 if side else -1
            yield np.asarray(vertices), normal


def render(
    parts,
    path,
    azimuth=225,
    elevation=28,
    width=1000,
    height=760,
    colors=None,
    framing_parts=None,
):
    """Draw exact target solids with hidden surfaces removed by per-pixel depth."""
    az, el = np.deg2rad([azimuth, elevation])
    camera = np.asarray([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    right = np.asarray([-np.sin(az), np.cos(az), 0])
    up = np.cross(camera, right)
    basis = np.asarray([right, -up, camera]).T
    vertices = np.concatenate(
        [
            v
            for p in (parts if framing_parts is None else framing_parts)
            for v, _ in faces(p)
        ]
    )
    projected = vertices @ basis
    lo, hi = projected[:, :2].min(0), projected[:, :2].max(0)
    scale = min((width - 110) / (hi[0] - lo[0]), (height - 100) / (hi[1] - lo[1]))
    center = (lo + hi) / 2

    def project(points):
        result = np.asarray(points) @ basis
        result[:, :2] = (result[:, :2] - center) * scale + [width / 2, height / 2 - 8]
        return result

    image = Image.new("RGB", (width, height), "#fafaf7")
    floor = ImageDraw.Draw(image)
    bounds = (
        np.asarray([p.origin for p in parts]).min(0),
        np.asarray([p.end for p in parts]).max(0),
    )
    x0, y0 = bounds[0][:2] - 2
    x1, y1 = bounds[1][:2] + 2
    for axis in (0, 1):
        for value in range(int((x0, y0)[axis]), int((x1, y1)[axis]) + 1, 2):
            line = (
                [[value, y0, 0], [value, y1, 0]]
                if axis == 0
                else [[x0, value, 0], [x1, value, 0]]
            )
            floor.line([tuple(p[:2]) for p in project(line)], fill="#e8e9e2", width=1)
    shadow = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(shadow)
    for part in parts:
        x, y, _ = part.origin
        X, Y, _ = part.end
        points = project([[x, y, 0], [X, y, 0], [X, Y, 0], [x, Y, 0]])
        draw.polygon([tuple(p[:2] + [3, 5]) for p in points], fill=(42, 53, 50, 30))
    image = Image.alpha_composite(
        image.convert("RGBA"), shadow.filter(ImageFilter.GaussianBlur(5))
    ).convert("RGB")
    pixels = np.asarray(image).copy()
    depth = np.full((height, width), -np.inf)
    edges = []
    light = np.asarray([-0.4, -0.6, 1.0])
    light /= np.linalg.norm(light)
    for index, part in enumerate(parts):
        color = material(part)[1] if colors is None else colors[index]
        for vertices, normal in faces(part):
            if np.dot(normal, camera) <= 1e-8:
                continue
            points = project(vertices)
            shade = 0.76 + 0.23 * max(0, np.dot(normal, light))
            fill = np.asarray(color) * shade
            for indices in ((0, 1, 2), (0, 2, 3)):
                triangle = points[list(indices)]
                _triangle(triangle, fill, pixels, depth)
            edges.extend(zip(points, np.roll(points, -1, axis=0), strict=True))
    # Edges are depth-tested after all faces; internal face diagonals stay hidden.
    for a, b in edges:
        count = max(2, int(np.linalg.norm(a[:2] - b[:2]) * 2))
        points = np.linspace(a, b, count)
        xy = np.rint(points[:, :2]).astype(int)
        valid = (
            (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        )
        xy, points = xy[valid], points[valid]
        visible = points[:, 2] >= depth[xy[:, 1], xy[:, 0]] - 1 / scale
        xy = xy[visible]
        pixels[xy[:, 1], xy[:, 0]] = pixels[xy[:, 1], xy[:, 0]] * 0.7
    Image.fromarray(pixels).save(path)


def _triangle(triangle, color, pixels, depth):
    height, width = depth.shape
    lo = np.maximum(np.floor(triangle[:, :2].min(0)).astype(int), 0)
    hi = np.minimum(
        np.ceil(triangle[:, :2].max(0)).astype(int), [width - 1, height - 1]
    )
    if np.any(hi < lo):
        return
    x, y = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    a, b, c = triangle
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denominator) < 1e-8:
        return
    u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
    v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
    w = 1 - u - v
    z = u * a[2] + v * b[2] + w * c[2]
    region = np.s_[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
    mask = (u >= -1e-8) & (v >= -1e-8) & (w >= -1e-8) & (z > depth[region])
    depth[region][mask] = z[mask]
    pixels[region][mask] = color


def support_graph(parts, path):
    """Draw actual vertical support edges with per-part IDs and material colors."""
    contacts = geometry.supports(parts)
    levels = sorted({p.origin[2] for p in parts})
    width, height = 960, max(280, len(levels) * 72 + 60)
    positions = {}
    for row, z in enumerate(levels):
        nodes = sorted(
            (i for i, p in enumerate(parts) if p.origin[2] == z),
            key=lambda i: parts[i].center[:2],
        )
        for col, i in enumerate(nodes):
            positions[i] = (70 + (col + 0.5) * 840 / len(nodes), height - 45 - row * 72)
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Part support graph">',
        '<rect width="100%" height="100%" fill="#fafaf7"/>',
        '<defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#8eaaa0"/></marker></defs>',
    ]
    for z in levels:
        y = height - 45 - levels.index(z) * 72
        chunks.append(
            f'<text x="12" y="{y + 4}" font-family="sans-serif" font-size="12" fill="#66746e">z {z / 2:g}</text>'
        )
    for contact in contacts:
        x0, y0 = positions[contact.lower]
        x1, y1 = positions[contact.upper]
        chunks.append(
            f'<path d="M{x0} {y0 - 15} C{x0} {(y0 + y1) / 2}, {x1} {(y0 + y1) / 2}, {x1} {y1 + 18}" stroke="#9baca2" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>'
        )
    for i, (x, y) in positions.items():
        _, rgb = material(parts[i])
        color = "#" + "".join(f"{v:02x}" for v in rgb)
        chunks.append(
            f'<circle cx="{x}" cy="{y}" r="15" fill="{color}"/><text x="{x}" y="{y + 4}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="white">{i + 1}</text>'
        )
    chunks.append("</svg>")
    path.write_text("\n".join(chunks))
