**Gerber to SVG/PNG**

- **Purpose**: Convert Gerber files to clean SVG paths suitable for laser cutters, or export raster PNG renderings.

**Quick Start**
- **Install dependencies** (recommended in a virtual environment):

```
python -m pip install -r requirements.txt
```

- **Convert to SVG (default)** — auto-generates the output filename by appending `.svg` to the input filename:

```
python gerber_to_svg.py "input-file.gbr"
```

This produces `input-file.gbr.svg` containing traced vector paths with proper holes (uses `fill-rule="evenodd"`).

- **Export PNG** (raster rendering only):

```
python gerber_to_svg.py "input-file.gbr" --png
```

This produces `input-file.gbr.png` (pixel raster of the rendered Gerber at the configured DPI).

- **Include drill holes** — add an Excellon drill file with `--drill`:

```
python gerber_to_svg.py "input-file.gbr" --drill "drill-file.txt"
```

Drill holes will be rendered as red circles in the SVG output, with diameter matching the drill bit size. The drill file parser supports standard Excellon format with tool definitions (e.g., `T01C0.3000` for 0.3mm diameter) and metric/inch coordinate modes.

- **Drill diameter offset (`--drill-offset`)** — apply an offset to the parsed drill diameters when emitting drill circles:

```
python gerber_to_svg.py "input-file.gbr" --drill "drill-file.txt" --drill-offset 0.1
```

Use a positive value to enlarge holes (for example to compensate for masking/kerf when plating or soldermask), or a negative value to reduce holes. The offset is specified in millimeters and defaults to `0.0`.

Notes:
- If a negative offset results in a negative diameter for a given tool, the diameter will be clamped to `0.0` (no negative sizes are emitted).
- The offset is applied to the tool diameter parsed from the drill file before any unit conversion (the implementation ensures units are handled correctly and the final diameter is in millimeters).

- **Mirror final output for backside work** — useful when preparing artwork for the PCB underside. Add the `--mirror-x` flag to flip the final SVG (and PNG) horizontally:

```
python gerber_to_svg.py "input-file.gbr" --drill "drill-file.txt" --mirror-x
```

When used, the traced SVG output will be mirrored across the X axis (around the board center) so it can be used directly for backside processing.

- **Include outline Gerber** — supply a Gerber outline file (for the board profile) with `--outline`. The outline will be added to the final SVG as a separate blue `outline` group so you can see the board edge clearly:

```
python gerber_to_svg.py "input-file.gbr" --outline "outline.gbr"
```

The outline is emitted as stroked paths (blue, thin) in the `outline` group on top of the copper and drill layers. When an outline is provided, the copper layer is automatically clipped to the outline boundary (using SVG clipping paths), ensuring copper traces don't extend beyond the board edge.

- **Invert copper layer for CNC/laser etching** — use `--invert` to flip the copper layer so dark areas represent copper to be removed (useful for milling or laser etching where you want to mark the material to remove):

```
python gerber_to_svg.py "input-file.gbr" --invert
```

When inverted, the output shows a filled rectangle (bounded by the outline if provided, or the bounding box) with the copper traces subtracted using SVG's `evenodd` fill rule. This makes the dark areas represent the material that should be removed, leaving the light (white) areas as the copper traces.

**Behavior Notes**
- The script parses Gerber with `pygerber` and renders shapes directly into a raster image using OpenCV, then traces contours to produce clean SVG paths.
- Polarity-aware: clear regions are rendered as cutouts (holes) and dark regions as filled copper.
- The vector output uses `fill-rule="evenodd"`, so nested shapes (black inside white inside black) are handled correctly.
- Drill holes are parsed from Excellon format drill files and rendered in red in the SVG output (or as gray in PNG output).

**Files**
- `gerber_to_svg.py` — main script
- `requirements.txt` — Python deps

**Troubleshooting**
- If PNG export fails, ensure `opencv-python` and `Pillow` are installed. The script no longer depends on native Cairo.
- Large boards rendered at high DPI may produce large PNGs. If memory or time is a problem, consider decreasing the DPI inside the script (search for the `dpi` variable inside `render_and_trace`).

**Next steps / Tips**
- If you want a different output filename, rename the generated file after conversion or modify the script to accept an explicit output path.
- To tune contour simplification, modify the polygon approximation epsilon in `contour_to_path()`.