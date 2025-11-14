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