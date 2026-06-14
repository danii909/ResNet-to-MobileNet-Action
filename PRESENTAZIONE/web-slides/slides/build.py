import os
import sys
import time
import argparse
from pathlib import Path

# Configuration of slides in order. 
# Set 'active': False to wrap a slide in <!-- --> in the output slides.html
SLIDES_CONFIG = [
    {"id": "s1", "active": True},
    {"id": "s2", "active": True},
    {"id": "s3", "active": True},
    {"id": "s4", "active": True},
    {"id": "s5", "active": True},
    {"id": "s6", "active": True},
    {"id": "s7", "active": True},
    {"id": "s8", "active": True},
    {"id": "s9", "active": True},
    {"id": "s10", "active": True},
    {"id": "s11", "active": True},
    {"id": "s_backup_sep", "active": True},
    {"id": "s12", "active": True},
    {"id": "s13", "active": True},
    {"id": "s14", "active": True},
    {"id": "s15", "active": False},  # Default commented out as in original
]

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
TEMPLATE_PATH = BASE_DIR / "template.html"
OUTPUT_PATH = BASE_DIR / "slides.html"

def compile_slides():
    try:
        if not TEMPLATE_PATH.exists():
            print(f"Error: Template file not found at {TEMPLATE_PATH}", file=sys.stderr)
            return False

        template_content = TEMPLATE_PATH.read_text(encoding="utf-8")
        
        slides_html_blocks = []
        for slide in SLIDES_CONFIG:
            slide_id = slide["id"]
            slide_file = SRC_DIR / f"{slide_id}.html"
            
            if not slide_file.exists():
                print(f"Warning: Slide file not found at {slide_file}", file=sys.stderr)
                continue
                
            slide_content = slide_file.read_text(encoding="utf-8")
            
            if not slide["active"]:
                # Wrap inactive slides in comments
                slide_content = f"<!--\n{slide_content}\n-->"
                
            slides_html_blocks.append(slide_content)
            
        # Combine slides with a clean indentation prefix (4 spaces)
        joined_slides = "\n\n".join(slides_html_blocks)
        
        # Replace the placeholder in the template
        output_content = template_content.replace("{{SLIDES}}", joined_slides)
        
        # Write slides.html in UTF-8
        OUTPUT_PATH.write_text(output_content, encoding="utf-8")
        return True
    except Exception as e:
        print(f"Error compiling slides: {e}", file=sys.stderr)
        return False

def get_mtimes():
    """Returns a dict mapping file path to modification time for all watched files."""
    mtimes = {}
    if TEMPLATE_PATH.exists():
        mtimes[TEMPLATE_PATH] = TEMPLATE_PATH.stat().st_mtime
    for slide in SLIDES_CONFIG:
        slide_file = SRC_DIR / f"{slide['id']}.html"
        if slide_file.exists():
            mtimes[slide_file] = slide_file.stat().st_mtime
    return mtimes

def main():
    parser = argparse.ArgumentParser(description="Compile individual slide files into a single slides.html.")
    parser.add_argument(
        "--watch", "-w", 
        action="store_true", 
        help="Watch for changes in template.html and src/ and automatically recompile."
    )
    args = parser.parse_args()

    print("Compiling slides.html...")
    if compile_slides():
        print("Successfully compiled slides.html")
    else:
        print("Failed to compile slides.html")
        sys.exit(1)

    if args.watch:
        print("\nWatch mode active. Monitoring template.html and src/...")
        print("Press Ctrl+C to exit.\n")
        
        # Initial modification times
        last_mtimes = get_mtimes()
        
        try:
            while True:
                time.sleep(1.0)
                current_mtimes = get_mtimes()
                
                changed_files = []
                for path, mtime in current_mtimes.items():
                    if path not in last_mtimes or mtime > last_mtimes[path]:
                        changed_files.append(path.name)
                
                if changed_files:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] Change detected in {', '.join(changed_files)}. Recompiling...")
                    if compile_slides():
                        print(f"[{timestamp}] Successfully recompiled slides.html")
                    else:
                        print(f"[{timestamp}] Recompilation failed")
                    last_mtimes = current_mtimes
        except KeyboardInterrupt:
            print("\nWatch stopped. Goodbye!")

if __name__ == "__main__":
    main()
