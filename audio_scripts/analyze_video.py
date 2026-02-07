#!/usr/bin/env python3
import sys
import os
import argparse
import csv
from pathlib import Path
import cv2
from nudenet import NudeDetector
from rich.console import Console
from rich.progress import Progress

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
    # Only re-execute if the venv python executable exists
    os.execv(venv_python, [venv_python] + sys.argv)

console = Console()

def format_time(seconds):
    """Formats seconds into HH:MM:SS string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(h), int(m), int(s))

def analyze_video(video_path, output_csv_path, threshold=0.5, frame_interval=1.0):
    """
    Analyzes a video for nudity using NudeNet.
    
    Args:
        video_path (Path): Path to video file.
        output_csv_path (Path): Path to output CSV.
        threshold (float): Confidence threshold.
        frame_interval (float): Process one frame every X seconds.
    """
    if not video_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {video_path}")
        sys.exit(1)

    console.print(f"[bold green]Analyzing video:[/bold green] {video_path} for nudity...")
    
    # Initialize Detector
    detector = NudeDetector()
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print(f"[bold red]Error:[/bold red] Could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    frame_step = int(fps * frame_interval)
    if frame_step < 1:
        frame_step = 1
        
    found_nudity = []
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing video...", total=total_frames)
        
        current_frame = 0
        while cap.isOpened():
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Perform detection on the frame
            # NudeNet expects path or image array (RGB)
            # OpenCV is BGR, convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect
            detections = detector.detect(rgb_frame)
            
            # Filter detections
            unsafe = [d for d in detections if d['score'] >= threshold and d['class'] in [
                'BUTTS', 'FEMALE_BREAST', 'FEMALE_GENITALIA', 'MALE_GENITALIA', 'ANUS'
            ]]
            
            if unsafe:
                timestamp = current_frame / fps
                for item in unsafe:
                    found_nudity.append({
                        'timestamp': timestamp,
                        'label': item['class'],
                        'score': item['score'],
                        'box': item['box'] # [x, y, w, h]
                    })
            
            current_frame += frame_step
            progress.update(task, completed=current_frame)
            
    cap.release()
    
    if found_nudity:
        console.print(f"[bold red]Found {len(found_nudity)} potential nudity frames.[/bold red]")
        
        # Aggregate logic: Combine adjacent frames into ranges?
        # For now, let's just dump the raw findings or maybe simple ranges.
        # Simple clustering: if frames are within 2*interval, merge?
        # Let's write raw 'points' for now, but edit_video might need ranges.
        # Let's create ranges: if timestamp T and T+interval both have nudity, it is continuous.
        
        ranges = []
        if found_nudity:
            # Sort by timestamp
            found_nudity.sort(key=lambda x: x['timestamp'])
            
            start = found_nudity[0]['timestamp']
            end = start
            labels = {found_nudity[0]['label']}
            
            for i in range(1, len(found_nudity)):
                curr = found_nudity[i]
                # If current timestamp is close to previous (within interval + small buffer)
                if curr['timestamp'] - end <= (frame_interval * 1.5):
                    end = curr['timestamp']
                    labels.add(curr['label'])
                else:
                    ranges.append((start, end, labels))
                    start = curr['timestamp']
                    end = start
                    labels = {curr['label']}
            ranges.append((start, end, labels))

        # Write CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["start_seconds", "end_seconds", "labels"])
            
            for start, end, lbls in ranges:
                # Add a small buffer to start and end for safety (e.g. 0.5s)
                s_buf = max(0, start - 0.5)
                e_buf = min(duration, end + 0.5)
                lbl_str = "|".join(sorted(list(lbls)))
                writer.writerow([f"{s_buf:.2f}", f"{e_buf:.2f}", lbl_str])
        
        console.print(f"[green]Review file saved to: {output_csv_path}[/green]")
        
    else:
        console.print("[bold green]No nudity detected.[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a video for nudity.")
    parser.add_argument("video_file", type=Path, help="Path to the video file.")
    parser.add_argument("--output_csv", type=Path, default="review_video.csv", help="Output CSV path.")
    parser.add_argument("--interval", type=float, default=1.0, help="Frame analysis interval in seconds (default: 1.0).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5).")

    args = parser.parse_args()
    
    analyze_video(args.video_file, args.output_csv, args.threshold, args.interval)
