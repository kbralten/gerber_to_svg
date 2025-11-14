import argparse
import sys
import tempfile
import os
import numpy as np
import cv2
from PIL import Image
import io
from pygerber.gerberx3.tokenizer.tokenizer import Tokenizer
from pygerber.gerberx3.parser2.parser2 import Parser2
from pygerber.gerberx3.parser2.commands2.line2 import Line2
from pygerber.gerberx3.parser2.commands2.arc2 import Arc2
from pygerber.gerberx3.parser2.commands2.flash2 import Flash2
from pygerber.gerberx3.parser2.commands2.region2 import Region2
from pygerber.gerberx3.parser2.apertures2.circle2 import Circle2
from pygerber.gerberx3.parser2.apertures2.rectangle2 import Rectangle2
from pygerber.gerberx3.parser2.apertures2.obround2 import Obround2
from pygerber.gerberx3.state_enums import Polarity


class GerberToSvg:
    def __init__(self, input_file, output_file, output_format='svg'):
        self.input_file = input_file
        self.output_file = output_file
        self.output_format = output_format
        self.svg_elements = []
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

    def get_fill_color(self, command):
        """Determine fill color based on polarity. Clear = white, Dark = black."""
        if hasattr(command, 'transform') and command.transform:
            polarity = command.transform.polarity
            if polarity in (Polarity.Clear, Polarity.ClearRegion):
                return "white"
        return "black"
    
    def get_stroke_color(self, command):
        """Determine stroke color based on polarity. Clear = white, Dark = black."""
        if hasattr(command, 'transform') and command.transform:
            polarity = command.transform.polarity
            if polarity in (Polarity.Clear, Polarity.ClearRegion):
                return "white"
        return "black"

    def convert(self):
        try:
            # Read the Gerber file
            with open(self.input_file, 'r') as f:
                source_code = f.read()
            
            # Tokenize
            tokenizer = Tokenizer()
            tokens = tokenizer.tokenize(source_code)
            
            # Parse
            parser = Parser2()
            command_buffer = parser.parse(tokens)
            
            # Get bounding box
            bbox = command_buffer.get_bounding_box()
            if bbox:
                self.min_x = bbox.min_x.as_millimeters()
                self.max_x = bbox.max_x.as_millimeters()
                self.min_y = bbox.min_y.as_millimeters()
                self.max_y = bbox.max_y.as_millimeters()
            
            # Process each command
            for command in command_buffer.commands:
                self.process_draw_command(command)

            # Write intermediate SVG with overlapping shapes
            temp_svg = tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False)
            temp_svg_path = temp_svg.name
            temp_svg.close()
            
            self.write_svg_to_file(temp_svg_path)
            
            # Render SVG to PNG using Cairo
            print(f"Rendering intermediate SVG to raster image...")
            self.render_and_trace(temp_svg_path, self.output_file)
            
            # Clean up temp file
            os.unlink(temp_svg_path)
            
            print(f"Successfully converted '{self.input_file}' to '{self.output_file}'")

        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def process_draw_command(self, draw_command):
        if isinstance(draw_command, Line2):
            self.handle_line(draw_command)
        elif isinstance(draw_command, Arc2):
            self.handle_arc(draw_command)
        elif isinstance(draw_command, Flash2):
            self.handle_flash(draw_command)
        elif isinstance(draw_command, Region2):
            self.handle_region(draw_command)

    def handle_line(self, line):
        # For lines, we need to know the aperture to determine the stroke width
        aperture = line.aperture
        if isinstance(aperture, Circle2):
            stroke_width = aperture.diameter.as_millimeters()
            x1, y1 = line.start_point.x.as_millimeters(), line.start_point.y.as_millimeters()
            x2, y2 = line.end_point.x.as_millimeters(), line.end_point.y.as_millimeters()
            stroke_color = self.get_stroke_color(line)
            self.svg_elements.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{stroke_color}" stroke-width="{stroke_width}" stroke-linecap="round" />'
            )

    def handle_arc(self, arc):
        # Arcs are more complex and will be represented as a path
        aperture = arc.aperture
        if isinstance(aperture, Circle2):
            stroke_width = aperture.diameter.as_millimeters()
            x1, y1 = arc.start_point.x.as_millimeters(), arc.start_point.y.as_millimeters()
            x2, y2 = arc.end_point.x.as_millimeters(), arc.end_point.y.as_millimeters()
            center_x, center_y = arc.center_point.x.as_millimeters(), arc.center_point.y.as_millimeters()
            radius = ((x1 - center_x)**2 + (y1 - center_y)**2)**0.5
            
            # Calculate if it's a large arc (> 180 degrees)
            import math
            angle1 = math.atan2(y1 - center_y, x1 - center_x)
            angle2 = math.atan2(y2 - center_y, x2 - center_x)
            angle_diff = angle2 - angle1
            
            # Normalize angle difference
            if arc.is_clockwise:
                if angle_diff > 0:
                    angle_diff -= 2 * math.pi
                large_arc_flag = 1 if abs(angle_diff) > math.pi else 0
                sweep_flag = 0
            else:
                if angle_diff < 0:
                    angle_diff += 2 * math.pi
                large_arc_flag = 1 if abs(angle_diff) > math.pi else 0
                sweep_flag = 1

            stroke_color = self.get_stroke_color(arc)
            self.svg_elements.append(
                f'<path d="M {x1},{y1} A {radius},{radius} 0 {large_arc_flag} '
                f'{sweep_flag} {x2},{y2}" '
                f'stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none" stroke-linecap="round" />'
            )
    
    def handle_region(self, region):
        # Regions are filled polygons
        if not region.command_buffer or not region.command_buffer.commands:
            return
        
        # Start path at first segment start point
        path_parts = []
        first_command = region.command_buffer.commands[0]
        
        if hasattr(first_command, 'start_point'):
            path_parts.append(f"M {first_command.start_point.x.as_millimeters()},{first_command.start_point.y.as_millimeters()}")
        
        for segment in region.command_buffer.commands:
            if isinstance(segment, Line2):
                path_parts.append(f"L {segment.end_point.x.as_millimeters()},{segment.end_point.y.as_millimeters()}")
            elif isinstance(segment, Arc2):
                x2, y2 = segment.end_point.x.as_millimeters(), segment.end_point.y.as_millimeters()
                center_x, center_y = segment.center_point.x.as_millimeters(), segment.center_point.y.as_millimeters()
                x1, y1 = segment.start_point.x.as_millimeters(), segment.start_point.y.as_millimeters()
                radius = ((x1 - center_x)**2 + (y1 - center_y)**2)**0.5
                
                import math
                angle1 = math.atan2(y1 - center_y, x1 - center_x)
                angle2 = math.atan2(y2 - center_y, x2 - center_x)
                angle_diff = angle2 - angle1
                
                if segment.is_clockwise:
                    if angle_diff > 0:
                        angle_diff -= 2 * math.pi
                    large_arc_flag = 1 if abs(angle_diff) > math.pi else 0
                    sweep_flag = 0
                else:
                    if angle_diff < 0:
                        angle_diff += 2 * math.pi
                    large_arc_flag = 1 if abs(angle_diff) > math.pi else 0
                    sweep_flag = 1
                
                path_parts.append(f"A {radius},{radius} 0 {large_arc_flag} {sweep_flag} {x2},{y2}")
        
        path_parts.append("Z")  # Close the path
        path_d = " ".join(path_parts)
        fill_color = self.get_fill_color(region)
        self.svg_elements.append(f'<path d="{path_d}" fill="{fill_color}" />')

    def handle_flash(self, flash):
        aperture = flash.aperture
        cx, cy = flash.flash_point.x.as_millimeters(), flash.flash_point.y.as_millimeters()
        fill_color = self.get_fill_color(flash)

        if isinstance(aperture, Circle2):
            r = aperture.diameter.as_millimeters() / 2
            self.svg_elements.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill_color}" />'
            )
        elif isinstance(aperture, Rectangle2):
            width = aperture.x_size.as_millimeters()
            height = aperture.y_size.as_millimeters()
            x = cx - width / 2
            y = cy - height / 2
            self.svg_elements.append(
                f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill_color}" />'
            )
        elif isinstance(aperture, Obround2):
            width = aperture.x_size.as_millimeters()
            height = aperture.y_size.as_millimeters()
            # Obround is a rectangle with rounded ends
            if width > height:
                # Horizontal obround
                r = height / 2
                x1 = cx - width / 2 + r
                x2 = cx + width / 2 - r
                path_d = f"M {x1},{cy - r} L {x2},{cy - r} "
                path_d += f"A {r},{r} 0 0 1 {x2},{cy + r} "
                path_d += f"L {x1},{cy + r} "
                path_d += f"A {r},{r} 0 0 1 {x1},{cy - r} Z"
                self.svg_elements.append(f'<path d="{path_d}" fill="{fill_color}" />')
            else:
                # Vertical obround
                r = width / 2
                y1 = cy - height / 2 + r
                y2 = cy + height / 2 - r
                path_d = f"M {cx - r},{y1} L {cx - r},{y2} "
                path_d += f"A {r},{r} 0 0 0 {cx + r},{y2} "
                path_d += f"L {cx + r},{y1} "
                path_d += f"A {r},{r} 0 0 0 {cx - r},{y1} Z"
                self.svg_elements.append(f'<path d="{path_d}" fill="{fill_color}" />')


    def write_svg_to_file(self, filepath):
        """Write the intermediate SVG with overlapping shapes to a file."""
        # Use default bounds if none were found
        if self.min_x is None or self.max_x is None or self.min_y is None or self.max_y is None:
            self.min_x = 0
            self.max_x = 100
            self.min_y = 0
            self.max_y = 100

        # Add a small margin
        margin = 10
        min_x = self.min_x - margin
        min_y = self.min_y - margin
        width = (self.max_x - self.min_x) + 2 * margin
        height = (self.max_y - self.min_y) + 2 * margin

        with open(filepath, "w") as f:
            f.write(
                f'<svg width="{width}mm" height="{height}mm" '
                f'viewBox="{min_x} {min_y} {width} {height}" '
                'xmlns="http://www.w3.org/2000/svg">\n'
            )
            # Invert Y axis for SVG coordinate system (Gerber Y increases upward, SVG increases downward)
            center_y = (self.max_y + self.min_y) / 2
            f.write(f'<g transform="translate(0, {2 * center_y}) scale(1, -1)">\n')
            for element in self.svg_elements:
                f.write(f"  {element}\n")
            f.write("</g>\n")
            f.write("</svg>\n")
    
    def render_and_trace(self, svg_path, output_path):
        """Render SVG to image and trace contours to create clean SVG."""
        # Use default bounds if none were found
        if self.min_x is None or self.max_x is None or self.min_y is None or self.max_y is None:
            self.min_x = 0
            self.max_x = 100
            self.min_y = 0
            self.max_y = 100

        margin = 10
        min_x = float(self.min_x) - margin
        min_y = float(self.min_y) - margin
        width = float(self.max_x - self.min_x) + 2 * margin
        height = float(self.max_y - self.min_y) + 2 * margin
        
        # Render SVG to PNG using Pillow with svglib or direct rendering
        # We'll use a simple approach: render directly with Pillow if possible
        # Otherwise fall back to using the intermediate SVG drawing approach
        
        # For now, let's use a simpler approach: directly render shapes to image
        dpi = 3600  # High DPI for better resolution
        scale = dpi / 25.4  # Convert mm to pixels at given DPI
        pixel_width = int(width * scale)
        pixel_height = int(height * scale)
        
        print(f"Rendering at {pixel_width}x{pixel_height} pixels ({dpi} DPI)...")
        
        # Create a white image
        img = np.ones((pixel_height, pixel_width), dtype=np.uint8) * 255
        
        # Render each SVG element to the image
        self.render_svg_elements_to_image(img, scale, min_x, min_y, width, height)
        
        # If PNG output requested, save and return
        if self.output_format == 'png':
            cv2.imwrite(output_path, img)
            print(f"PNG saved to: {output_path}")
            return
        
        # Threshold to binary (black = copper, white = empty)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        print(f"Tracing contours...")
        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create new SVG with traced contours
        with open(output_path, 'w') as f:
            f.write(
                f'<svg width="{width}mm" height="{height}mm" '
                f'viewBox="{min_x} {min_y} {width} {height}" '
                'xmlns="http://www.w3.org/2000/svg">\n'
            )
            
            # Invert Y axis for SVG coordinate system
            center_y = (self.max_y + self.min_y) / 2
            f.write(f'<g transform="translate(0, {2 * center_y}) scale(1, -1)">\n')
            
            # Process contours with hierarchy
            if hierarchy is not None and len(contours) > 0:
                # hierarchy format: [Next, Previous, First_Child, Parent]
                hierarchy = hierarchy[0]
                
                # Process all contours recursively
                processed = set()
                
                def process_contour_tree(idx, depth=0):
                    """Recursively process contours and their children."""
                    if idx == -1 or idx in processed:
                        return []
                    
                    processed.add(idx)
                    path_parts = []
                    
                    # Add current contour
                    path_parts.append(self.contour_to_path(contours[idx], scale, min_x, min_y, height))
                    
                    # Process all children (holes at depth+1, filled at depth+2, etc.)
                    child_idx = hierarchy[idx][2]
                    while child_idx != -1:
                        # Recursively process child and its descendants
                        child_parts = process_contour_tree(child_idx, depth + 1)
                        path_parts.extend(child_parts)
                        child_idx = hierarchy[child_idx][0]  # Next sibling
                    
                    return path_parts
                
                # Process all top-level contours (parent = -1)
                for i in range(len(contours)):
                    if hierarchy[i][3] == -1 and i not in processed:
                        path_parts = process_contour_tree(i, 0)
                        if path_parts:
                            path_d = " ".join(path_parts)
                            f.write(f'  <path d="{path_d}" fill="black" fill-rule="evenodd" />\n')
            
            f.write("</g>\n")
            f.write("</svg>\n")
        
        print(f"Traced {len(contours)} contours")
    
    def render_svg_elements_to_image(self, img, scale, min_x, min_y, width, height):
        """Render SVG elements directly to a numpy image using OpenCV."""
        import xml.etree.ElementTree as ET
        import re
        
        for element in self.svg_elements:
            # Parse the element
            try:
                elem = ET.fromstring(element)
                tag = elem.tag
                fill = elem.get('fill', 'black')
                color = 0 if fill == 'black' else 255  # black = 0, white = 255
                
                if tag == 'circle':
                    cx = float(elem.get('cx'))
                    cy = float(elem.get('cy'))
                    r = float(elem.get('r'))
                    
                    # Convert to pixel coordinates
                    px = int((cx - min_x) * scale)
                    py = int((cy - min_y) * scale)
                    pr = int(r * scale)
                    
                    cv2.circle(img, (px, py), pr, color, -1)
                
                elif tag == 'rect':
                    x = float(elem.get('x'))
                    y = float(elem.get('y'))
                    w = float(elem.get('width'))
                    h = float(elem.get('height'))
                    
                    # Convert to pixel coordinates
                    px1 = int((x - min_x) * scale)
                    py1 = int((y - min_y) * scale)
                    px2 = int((x + w - min_x) * scale)
                    py2 = int((y + h - min_y) * scale)
                    
                    cv2.rectangle(img, (px1, py1), (px2, py2), color, -1)
                
                elif tag == 'line':
                    x1 = float(elem.get('x1'))
                    y1 = float(elem.get('y1'))
                    x2 = float(elem.get('x2'))
                    y2 = float(elem.get('y2'))
                    stroke_width = float(elem.get('stroke-width', 1))
                    stroke = elem.get('stroke', 'black')
                    color = 0 if stroke == 'black' else 255
                    
                    # Convert to pixel coordinates
                    px1 = int((x1 - min_x) * scale)
                    py1 = int((y1 - min_y) * scale)
                    px2 = int((x2 - min_x) * scale)
                    py2 = int((y2 - min_y) * scale)
                    thickness = max(1, int(stroke_width * scale))
                    
                    cv2.line(img, (px1, py1), (px2, py2), color, thickness, cv2.LINE_AA)
                
                elif tag == 'path':
                    d = elem.get('d')
                    points = self.parse_svg_path(d, scale, min_x, min_y)
                    if points and len(points) > 0:
                        pts = np.array(points, dtype=np.int32)
                        cv2.fillPoly(img, [pts], color)
                
            except Exception as e:
                print(f"Warning: Could not render element: {e}")
                continue
    
    def parse_svg_path(self, d, scale, min_x, min_y):
        """Parse SVG path data and convert to pixel coordinates."""
        import re
        
        points = []
        # Simple parser for M, L, A commands
        commands = re.findall(r'([MLAZmlaz])([^MLAZmlaz]*)', d)
        
        current_x, current_y = 0, 0
        
        for cmd, params in commands:
            params = params.strip()
            if not params and cmd.upper() != 'Z':
                continue
            
            if cmd.upper() == 'M':  # Move to
                coords = [float(x) for x in params.replace(',', ' ').split()]
                if len(coords) >= 2:
                    current_x, current_y = coords[0], coords[1]
                    px = int((current_x - min_x) * scale)
                    py = int((current_y - min_y) * scale)
                    points.append([px, py])
            
            elif cmd.upper() == 'L':  # Line to
                coords = [float(x) for x in params.replace(',', ' ').split()]
                if len(coords) >= 2:
                    current_x, current_y = coords[0], coords[1]
                    px = int((current_x - min_x) * scale)
                    py = int((current_y - min_y) * scale)
                    points.append([px, py])
            
            elif cmd.upper() == 'A':  # Arc - approximate with line segments
                coords = [float(x) for x in params.replace(',', ' ').split()]
                if len(coords) >= 7:
                    # For simplicity, just add the end point
                    # A proper implementation would interpolate the arc
                    current_x, current_y = coords[5], coords[6]
                    px = int((current_x - min_x) * scale)
                    py = int((current_y - min_y) * scale)
                    points.append([px, py])
            
            elif cmd.upper() == 'Z':  # Close path
                pass
        
        return points
    
    def contour_to_path(self, contour, scale, min_x, min_y, height):
        """Convert OpenCV contour to SVG path data."""
        # Simplify contour to reduce number of points
        epsilon = 0.5  # Adjust for more/less simplification
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        path_parts = []
        for i, point in enumerate(approx):
            # Convert pixel coordinates back to mm
            px, py = point[0]
            x = (px / scale) + min_x
            y = (py / scale) + min_y
            
            if i == 0:
                path_parts.append(f"M {x:.5f},{y:.5f}")
            else:
                path_parts.append(f"L {x:.5f},{y:.5f}")
        
        path_parts.append("Z")
        return " ".join(path_parts)


def main():
    parser = argparse.ArgumentParser(description="Convert a Gerber file to an SVG or PNG image.")
    parser.add_argument("input_file", help="Path to the input Gerber file.")
    parser.add_argument("--png", action="store_true", help="Output PNG instead of SVG.")
    args = parser.parse_args()

    # Auto-generate output filename
    output_format = 'png' if args.png else 'svg'
    output_file = args.input_file + '.' + output_format

    converter = GerberToSvg(args.input_file, output_file, output_format)
    converter.convert()


if __name__ == "__main__":
    main()
