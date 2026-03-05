"""
paste_stencil.py — Generate a solder-paste stencil SVG and a 3-D printable
alignment-jig STL from Gerber files.

Outputs (auto-named from the paste gerber):
  <base>_stencil.svg  — Laser-cut stencil with 4 alignment holes
  <base>_jig.stl      — 3-D jig: PCB pocket + 4 alignment pegs

Depends on gerber_to_svg.py for SVG rendering; does NOT modify it.
"""

import argparse
import math
import os
import re
import sys
import tempfile

import numpy as np
import trimesh
import trimesh.creation
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Reuse gerber_to_svg internals without calling convert()
# ---------------------------------------------------------------------------
from gerber_to_svg import GerberToSvg
from pygerber.gerberx3.tokenizer.tokenizer import Tokenizer
from pygerber.gerberx3.parser2.parser2 import Parser2


# ---------------------------------------------------------------------------
# Phase 3 – Outline parsing
# ---------------------------------------------------------------------------

def parse_outline(outline_file):
    """Parse a board-outline Gerber and return (min_x, max_x, min_y, max_y, svg_path_data).

    svg_path_data is a multi-subpath SVG path string (one M…Z per closed contour).
    """
    with open(outline_file, 'r') as f:
        source = f.read()

    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(source)
    parser = Parser2()
    command_buffer = parser.parse(tokens)

    bbox = command_buffer.get_bounding_box()
    if bbox is None:
        raise ValueError(f"Could not determine bounding box from outline file: {outline_file}")

    min_x = bbox.min_x.as_millimeters()
    max_x = bbox.max_x.as_millimeters()
    min_y = bbox.min_y.as_millimeters()
    max_y = bbox.max_y.as_millimeters()

    # Use GerberToSvg's element-collection machinery without converting
    helper = GerberToSvg(
        input_file=outline_file,
        output_file="/dev/null",  # never written
        outline_file=None,
    )
    helper._target_svg_list = helper.outline_elements
    for command in command_buffer.commands:
        helper.process_draw_command(command)
    helper._target_svg_list = None

    path_data = helper.build_outline_path_data()
    return float(min_x), float(max_x), float(min_y), float(max_y), path_data


# ---------------------------------------------------------------------------
# Phase 4 – SVG path → shapely polygons
# ---------------------------------------------------------------------------

def _arc_endpoint_to_center(x1, y1, rx, ry, large_arc, sweep, x2, y2):
    """Convert SVG arc endpoint parameters to center/angles (F.6.5 spec).

    Returns (cx, cy, theta1_deg, dtheta_deg).
    Only valid for circular arcs (rx == ry treated as circle).
    """
    # Degenerate or zero-length arc
    if abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9:
        return None

    # Use average radius for rx/ry and guard against zero
    r = (rx + ry) / 2.0
    if r < 1e-9:
        return None

    # Midpoint and chord vector
    mx_orig = (x1 + x2) / 2.0
    my_orig = (y1 + y2) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)
    if d < 1e-12:
        return None

    # Ensure radius is at least half the chord length
    if r < d / 2.0:
        r = d / 2.0

    # Distance from midpoint to center along perpendicular
    h_offset = math.sqrt(max(0.0, r * r - (d / 2.0) ** 2))

    # Unit perpendicular vector (rotated 90 degrees)
    ux = -dy / d
    uy = dx / d

    # Choose sign based on SVG large-arc and sweep flags
    sign = 1.0 if (large_arc != sweep) else -1.0
    cx = mx_orig + sign * h_offset * ux
    cy = my_orig + sign * h_offset * uy

    # Angles
    theta1 = math.atan2(y1 - cy, x1 - cx)
    theta2 = math.atan2(y2 - cy, x2 - cx)

    if sweep == 0:
        # counter-clockwise in standard math, but SVG sweep=0 is CCW
        dtheta = theta2 - theta1
        if dtheta > 0:
            dtheta -= 2 * math.pi
    else:
        # clockwise in standard math, sweep=1 is CW
        dtheta = theta2 - theta1
        if dtheta < 0:
            dtheta += 2 * math.pi

    return cx, cy, theta1, dtheta, r


def _sample_arc(x1, y1, rx, ry, large_arc, sweep, x2, y2, step_deg=1.0):
    """Return a list of (x, y) points approximating the arc (not including start point)."""
    result = _arc_endpoint_to_center(x1, y1, rx, ry, large_arc, sweep, x2, y2)
    if result is None:
        return [(x2, y2)]
    cx, cy, theta1, dtheta, r = result

    n_steps = max(2, int(abs(math.degrees(dtheta)) / step_deg))
    pts = []
    for i in range(1, n_steps + 1):
        t = theta1 + dtheta * i / n_steps
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    # Snap last point to exact endpoint
    if pts:
        pts[-1] = (x2, y2)
    return pts


def svg_path_to_polygons(path_data):
    """Parse an SVG path string into a list of shapely Polygons (one per M…Z subpath)."""
    NUM = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
    cmd_re = re.compile(r'([MLAZmlaz])([^MLAZmlaz]*)')

    polygons = []
    current_x, current_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0
    current_subpath = []

    for cmd, args in cmd_re.findall(path_data):
        args = args.strip()
        nums = [float(v) for v in re.findall(NUM, args)]

        if cmd == 'M':
            if len(nums) >= 2:
                current_x, current_y = nums[0], nums[1]
                start_x, start_y = current_x, current_y
                current_subpath = [(current_x, current_y)]

        elif cmd == 'L':
            if len(nums) >= 2:
                current_x, current_y = nums[0], nums[1]
                current_subpath.append((current_x, current_y))

        elif cmd == 'A':
            if len(nums) >= 7:
                rx, ry = nums[0], nums[1]
                large_arc, sweep = int(nums[3]), int(nums[4])
                x2, y2 = nums[5], nums[6]
                pts = _sample_arc(current_x, current_y, rx, ry, large_arc, sweep, x2, y2)
                current_subpath.extend(pts)
                current_x, current_y = x2, y2

        elif cmd == 'Z':
            if len(current_subpath) >= 3:
                try:
                    poly = ShapelyPolygon(current_subpath)
                    if poly.is_valid and not poly.is_empty:
                        polygons.append(poly)
                    else:
                        poly = poly.buffer(0)
                        if not poly.is_empty:
                            polygons.append(poly)
                except Exception:
                    pass
            current_subpath = []
            current_x, current_y = start_x, start_y

    return polygons


def get_outer_polygon(polygons):
    """Return the shapely polygon with the largest area (board outer perimeter)."""
    if not polygons:
        return None
    return max(polygons, key=lambda p: p.area)


# ---------------------------------------------------------------------------
# Phase 5 – SVG generation + post-processing
# ---------------------------------------------------------------------------

def alignment_corners(min_x, max_x, min_y, max_y, margin=15.0):
    """Return 4 alignment peg centre positions (mm, Gerber coordinate space)."""
    return [
        (min_x - margin, min_y - margin),
        (max_x + margin, min_y - margin),
        (max_x + margin, max_y + margin),
        (min_x - margin, max_y + margin),
    ]


def _expand_viewbox(svg_text, corners, circle_r=5.0, extra_margin=10.0):
    """Rewrite the <svg> width/height/viewBox to encompass alignment circles."""
    # Parse existing viewBox
    vb_match = re.search(r'viewBox="([^"]+)"', svg_text)
    if not vb_match:
        return svg_text
    vb_vals = [float(v) for v in vb_match.group(1).split()]
    if len(vb_vals) < 4:
        return svg_text
    vb_x, vb_y, vb_w, vb_h = vb_vals

    current_min_x = vb_x
    current_min_y = vb_y
    current_max_x = vb_x + vb_w
    current_max_y = vb_y + vb_h

    # Expand to cover circles + margin
    for cx, cy in corners:
        current_min_x = min(current_min_x, cx - circle_r - extra_margin)
        current_min_y = min(current_min_y, cy - circle_r - extra_margin)
        current_max_x = max(current_max_x, cx + circle_r + extra_margin)
        current_max_y = max(current_max_y, cy + circle_r + extra_margin)

    new_w = current_max_x - current_min_x
    new_h = current_max_y - current_min_y

    svg_text = re.sub(
        r'width="[^"]+"',
        f'width="{new_w:.4f}mm"',
        svg_text,
        count=1,
    )
    svg_text = re.sub(
        r'height="[^"]+"',
        f'height="{new_h:.4f}mm"',
        svg_text,
        count=1,
    )
    svg_text = re.sub(
        r'viewBox="[^"]+"',
        f'viewBox="{current_min_x:.4f} {current_min_y:.4f} {new_w:.4f} {new_h:.4f}"',
        svg_text,
        count=1,
    )
    return svg_text


def _inject_alignment_circles(svg_text, corners, jig_rect, circle_r=5.0):
    """Inject 4 alignment circles and the jig outer perimeter rect into the SVG.

    All coordinates are in Gerber space; the existing Y-inversion transform
    already handles the flip.

    jig_rect is (x, y, width, height) of the jig outer perimeter.
    """
    rx, ry, rw, rh = jig_rect
    rect_element = (
        f'  <rect x="{rx:.4f}" y="{ry:.4f}" width="{rw:.4f}" height="{rh:.4f}" '
        f'fill="none" stroke="blue" stroke-width="0.5" />'
    )
    circle_elements = '\n'.join(
        f'  <circle cx="{cx:.4f}" cy="{cy:.4f}" r="{circle_r}" '
        f'fill="none" stroke="blue" stroke-width="0.5" />'
        for cx, cy in corners
    )
    new_elements = rect_element + '\n' + circle_elements

    # Insert before the closing tag of the outermost transform group
    # The SVG structure ends with </g>\n</svg>, so insert before the last </g>
    insert_marker = '</g>\n</svg>'
    replacement = f'{new_elements}\n</g>\n</svg>'
    if insert_marker in svg_text:
        # Replace only the last occurrence
        idx = svg_text.rfind(insert_marker)
        svg_text = svg_text[:idx] + replacement + svg_text[idx + len(insert_marker):]
    return svg_text


def _scale_pads(svg_text, scale):
    """Scale each pad path in the copper group about its bounding-box centre.

    Wraps each <path> inside <g id="copper"> with an SVG transform of the form
    translate(cx,cy) scale(s,s) translate(-cx,-cy), which scales the pad in
    place. Use values below 1.0 to compensate for thicker stencils that would
    otherwise deposit too much paste.
    """
    if abs(scale - 1.0) < 1e-9:
        return svg_text

    NUM = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
    cmd_re = re.compile(r'([MLAZmlaz])([^MLAZmlaz]*)')

    def path_bbox_center(d):
        """Return the bounding-box centre of all endpoint coordinates in path d."""
        xs, ys = [], []
        for cmd, args in cmd_re.findall(d):
            nums = [float(v) for v in re.findall(NUM, args)]
            if cmd in ('M', 'L') and len(nums) >= 2:
                xs.append(nums[0])
                ys.append(nums[1])
            elif cmd == 'A' and len(nums) >= 7:
                xs.append(nums[5])
                ys.append(nums[6])
        if not xs:
            return None, None
        return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0

    def transform_path(m):
        tag = m.group(0)
        d_m = re.search(r'd="([^"]*)"', tag)
        if not d_m:
            return tag
        cx, cy = path_bbox_center(d_m.group(1))
        if cx is None:
            return tag
        t = (f'translate({cx:.5f},{cy:.5f}) scale({scale},{scale})'
             f' translate({-cx:.5f},{-cy:.5f})')
        if 'transform=' in tag:
            return re.sub(r'transform="[^"]*"', f'transform="{t}"', tag)
        return tag.replace('<path ', f'<path transform="{t}" ', 1)

    def scale_copper_group(m):
        return re.sub(r'<path\b[^/]*/>', transform_path, m.group(0))

    return re.sub(r'<g id="copper">.*?</g>', scale_copper_group, svg_text, flags=re.DOTALL)


def generate_stencil_svg(paste_file, svg_output, corner_radius, outline_corners, jig_rect, pad_scale=1.0):
    """Run GerberToSvg on the paste layer then post-process the output SVG."""
    converter = GerberToSvg(
        input_file=paste_file,
        output_file=svg_output,
        output_format='svg',
        corner_radius=corner_radius,
    )
    converter.convert()

    # Read the rendered SVG
    with open(svg_output, 'r', encoding='utf-8') as f:
        svg_text = f.read()

    # Expand canvas, scale pads, and inject alignment circles + jig perimeter rect
    svg_text = _expand_viewbox(svg_text, outline_corners)
    svg_text = _scale_pads(svg_text, pad_scale)
    svg_text = _inject_alignment_circles(svg_text, outline_corners, jig_rect)

    with open(svg_output, 'w', encoding='utf-8') as f:
        f.write(svg_text)

    print(f"Stencil SVG written to: {svg_output}")


# ---------------------------------------------------------------------------
# Phase 6 – STL jig generation
# ---------------------------------------------------------------------------

def generate_jig_stl(
    outer_polygon,
    min_x, max_x, min_y, max_y,
    board_thickness,
    peg_tolerance,
    jig_tolerance,
    corners,
    stl_output,
):
    """Build and export the 3-D alignment jig as an STL file.

    Jig structure:
      - A rectangular frame (outer_rect minus PCB outline) at board_thickness tall.
        The PCB cutout is expanded outward by jig_tolerance mm so the board
        drops in without force.
      - 4 cylindrical pegs rising board_thickness + 3 mm above z=0, centred at
        the same (x, y) positions used for the SVG alignment holes.
    """
    outer_r = dict(
        min_x=min_x - 30.0,
        max_x=max_x + 30.0,
        min_y=min_y - 30.0,
        max_y=max_y + 30.0,
    )
    outer_rect = shapely_box(
        outer_r['min_x'], outer_r['min_y'],
        outer_r['max_x'], outer_r['max_y'],
    )

    # Frame = outer rectangle minus exact PCB outline (expanded by jig_tolerance)
    if outer_polygon is not None and not outer_polygon.is_empty:
        try:
            cutout = outer_polygon.buffer(jig_tolerance) if jig_tolerance > 0 else outer_polygon
            frame_2d = outer_rect.difference(cutout)
        except Exception as e:
            print(f"Warning: shapely difference failed ({e}), using bounding box cutout.")
            pcb_box = shapely_box(min_x - jig_tolerance, min_y - jig_tolerance,
                                  max_x + jig_tolerance, max_y + jig_tolerance)
            frame_2d = outer_rect.difference(pcb_box)
    else:
        pcb_box = shapely_box(min_x - jig_tolerance, min_y - jig_tolerance,
                              max_x + jig_tolerance, max_y + jig_tolerance)
        frame_2d = outer_rect.difference(pcb_box)

    frame_mesh = trimesh.creation.extrude_polygon(frame_2d, height=board_thickness)

    # Alignment pegs
    peg_radius = (10.0 - peg_tolerance) / 2.0
    peg_height = board_thickness + 3.0
    peg_meshes = []
    for (px, py) in corners:
        cyl = trimesh.creation.cylinder(
            radius=peg_radius,
            height=peg_height,
            sections=64,
        )
        # trimesh.creation.cylinder centres the cylinder at z=0 (z ∈ [-h/2, h/2])
        # Translate so bottom of peg sits at z=0
        cyl.apply_translation([px, py, peg_height / 2.0])
        peg_meshes.append(cyl)

    all_meshes = [frame_mesh] + peg_meshes
    jig = trimesh.boolean.union(all_meshes, engine="manifold")

    jig.export(stl_output)
    print(f"Jig STL written to: {stl_output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a solder-paste stencil SVG and an alignment-jig STL "
            "from a paste Gerber and a board-outline Gerber."
        )
    )
    parser.add_argument("paste_file", help="Path to the paste Gerber file (F.Paste / B.Paste).")
    parser.add_argument("outline_file", help="Path to the board-outline Gerber file (Edge.Cuts).")
    parser.add_argument(
        "--corner-radius",
        type=float,
        default=0.0,
        metavar="MM",
        help="Radius in mm to apply to pad corners in the stencil SVG (default: 0).",
    )
    parser.add_argument(
        "--board-thickness",
        type=float,
        default=1.6,
        metavar="MM",
        help="PCB thickness in mm; sets the jig frame height (default: 1.6).",
    )
    parser.add_argument(
        "--peg-tolerance",
        type=float,
        default=0.2,
        metavar="MM",
        help=(
            "Clearance in mm subtracted from the 10 mm alignment-hole diameter "
            "to give the peg diameter for the STL (default: 0.2)."
        ),
    )
    parser.add_argument(
        "--jig-tolerance",
        type=float,
        default=0.1,
        metavar="MM",
        help=(
            "Amount in mm to expand the PCB outline cutout in the jig so the "
            "board drops in without force (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--pad-scale",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Scale each pad about its centre before cutting (default: 1.0, no change). "
            "Use values below 1.0 to reduce paste deposition for thicker stencils, "
            "e.g. 0.9 for a 10%% area reduction."
        ),
    )
    args = parser.parse_args()

    base = os.path.splitext(args.paste_file)[0]
    svg_output = base + "_stencil.svg"
    stl_output = base + "_jig.stl"

    # --- Parse outline ---
    print(f"Parsing outline: {args.outline_file}")
    min_x, max_x, min_y, max_y, path_data = parse_outline(args.outline_file)
    print(f"  Outline bbox: ({min_x:.3f}, {min_y:.3f}) – ({max_x:.3f}, {max_y:.3f}) mm")

    # --- Build outer polygon for STL ---
    outer_polygon = None
    if path_data:
        polygons = svg_path_to_polygons(path_data)
        outer_polygon = get_outer_polygon(polygons)
        if outer_polygon:
            print(f"  Outer polygon area: {outer_polygon.area:.2f} mm²")
        else:
            print("  Warning: could not extract outer polygon; STL will use bbox cutout.")
    else:
        print("  Warning: no outline path data extracted; STL will use bbox cutout.")

    # --- Compute alignment corner positions ---
    corners = alignment_corners(min_x, max_x, min_y, max_y, margin=15.0)
    print(f"  Alignment corners: {[(f'{x:.2f}', f'{y:.2f}') for x, y in corners]}")

    # --- Generate stencil SVG ---
    jig_rect = (min_x - 30.0, min_y - 30.0, (max_x - min_x) + 60.0, (max_y - min_y) + 60.0)
    print(f"\nRendering stencil SVG: {svg_output}")
    generate_stencil_svg(args.paste_file, svg_output, args.corner_radius, corners, jig_rect, args.pad_scale)

    # --- Generate jig STL ---
    print(f"\nBuilding jig STL: {stl_output}")
    generate_jig_stl(
        outer_polygon=outer_polygon,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        board_thickness=args.board_thickness,
        peg_tolerance=args.peg_tolerance,
        jig_tolerance=args.jig_tolerance,
        corners=corners,
        stl_output=stl_output,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
