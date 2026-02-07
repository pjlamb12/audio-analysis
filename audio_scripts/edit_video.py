#!/usr/bin/env python3
import sys
import os
import argparse
import csv
import subprocess
from pathlib import Path
from rich.console import Console

# Auto-activate venv if not already active
script_dir = os.path.dirname(os.path.abspath(__file__))
# Venv is now one level up from the scripts
venv_dir = os.path.join(script_dir, "..", "venv")

if sys.platform == "win32":
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
else:
    venv_python = os.path.join(venv_dir, "bin", "python3")

# Check if we are running in the venv; if not, re-execute
if sys.prefix != os.path.abspath(venv_dir) and os.path.exists(venv_python):
    os.execv(venv_python, [venv_python] + sys.argv)

console = Console()

def get_unique_filepath(path: Path) -> Path:
    """Returns a unique filepath by appending a number if the file exists."""
    if not path.exists():
        return path
    
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    
    while True:
        new_path = parent / f"{stem}{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1

def edit_video(video_path, review_csv, output_path, blur_strength=10):
    """
    Edits the video to blur sections specified in the CSV.
    Uses ffmpeg filter: boxblur=luma_radius:luma_power:enable='between(t,start,end)'
    """
    if not video_path.exists():
        console.print(f"[bold red]Error:[/bold red] Video file not found: {video_path}")
        sys.exit(1)
        
    if not review_csv.exists():
        console.print(f"[bold red]Error:[/bold red] CSV file not found: {review_csv}")
        sys.exit(1)

    # Read CSV
    ranges = []
    with open(review_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row['start_seconds'])
                end = float(row['end_seconds'])
                ranges.append((start, end))
            except ValueError:
                continue

    if not ranges:
        console.print("[yellow]No ranges found in CSV to blur.[/yellow]")
        return

    # Construct Filter Complex
    # We can chain boxblur filters or use one complex filter.
    # If many ranges, chaining might be long. 
    # Better to use one boxblur with multiple enables? 
    # No, `enable` takes an expression. We can combine ranges with OR `+`.
    # between(t,s1,e1)+between(t,s2,e2)+...
    
    enable_expr = "+".join([f"between(t,{s},{e})" for s, e in ranges])
    
    filter_graph = f"boxblur={blur_strength}:1:enable='{enable_expr}'"
    
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", filter_graph,
        "-c:a", "copy", # Copy audio without re-encoding
        str(output_path)
    ]
    
    console.print(f"[green]Processing video...[/green]")
    console.print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        console.print(f"[bold green]Video saved to: {output_path}[/bold green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]FFmpeg Error:[/bold red] {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blur sections of a video based on a CSV review file.")
    parser.add_argument("video_file", type=Path, help="Path to the input video file.")
    parser.add_argument("review_csv", type=Path, help="Path to the review CSV file.")
    parser.add_argument("--output_file", type=Path, help="Path to save the output video.")
    parser.add_argument("--blur", type=int, default=20, help="Blur strength (default: 20).")

    args = parser.parse_args()
    
    if args.output_file:
        out_path = args.output_file
    else:
        out_path = args.video_file.parent / f"{args.video_file.stem}_blurred{args.video_file.suffix}"
        
    final_out_path = get_unique_filepath(out_path)
    
    edit_video(args.video_file, args.review_csv, final_out_path, args.blur)
