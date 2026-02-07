#!/usr/bin/env python3
# edit_audio.py

import sys
import os
import platform

# Auto-activate venv if not already active
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_dir = os.path.join(script_dir, "venv")

if sys.platform == "win32":
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
else:
    venv_python = os.path.join(venv_dir, "bin", "python3")

if os.path.abspath(sys.prefix) != os.path.abspath(venv_dir):
    if os.path.exists(venv_python):
        try:
            os.execv(venv_python, [venv_python] + sys.argv)
        except OSError as e:
            print(f"Failed to activate venv: {e}")
            print("Running with system python.")
    else:
        print(f"Warning: 'venv' not found at {venv_dir}. Running with system python.")

import pandas as pd
import argparse
import subprocess
from pathlib import Path
from rich.console import Console

# Setup rich console for pretty printing
console = Console()

def get_unique_filepath(filepath: Path) -> Path:
    """
    Checks if a filepath exists. If it does, it appends a number
    to the filename until a unique path is found.
    """
    if not filepath.exists():
        return filepath

    parent = filepath.parent
    stem = filepath.stem
    suffix = filepath.suffix
    counter = 1

    while True:
        new_filepath = parent / f"{stem}{counter}{suffix}"
        if not new_filepath.exists():
            console.print(f"[bold yellow]Warning: Output file '{filepath.name}' already exists. Saving to '{new_filepath.name}' instead.[/bold yellow]")
            return new_filepath
        counter += 1

def edit_media_with_ffmpeg(media_path: Path, review_csv_path: Path, output_path: Path):
    """
    Silences sections of an audio track in an audio or video file using FFmpeg.

    Args:
        media_path (Path): Path to the original media file.
        review_csv_path (Path): Path to the CSV file with timestamps to silence.
        output_path (Path): Path to save the edited media file.
    """
    # --- Load Review File and Determine its Type ---
    console.print(f"Loading review file from [green]{review_csv_path}[/green]...")
    try:
        df = pd.read_csv(review_csv_path)
        if {'start', 'end'}.issubset(df.columns):
            start_col, end_col = 'start', 'end'
            console.print("Detected a [yellow]word censorship[/yellow] review file.")
        elif {'start_seconds', 'end_seconds'}.issubset(df.columns):
            start_col, end_col = 'start_seconds', 'end_seconds'
            console.print("Detected a [yellow]topic silencing[/yellow] review file.")
        else:
            console.print("[bold red]Error: Review CSV must contain either ('start', 'end') or ('start_seconds', 'end_seconds') columns.[/bold red]")
            return
            
    except FileNotFoundError:
        console.print(f"[bold red]Error: Review file not found at '{review_csv_path}'[/bold red]")
        return
        
    if df.empty:
        console.print("[bold yellow]Review file is empty. No edits will be applied.[/bold yellow]")
        return

    # --- Construct the FFmpeg Filter ---
    console.print(f"Building FFmpeg filter for [yellow]{len(df)}[/yellow] sections...")
    filter_parts = []
    for _, row in df.iterrows():
        start_sec = row[start_col]
        end_sec = row[end_col]
        filter_parts.append(f"volume=enable='between(t,{start_sec},{end_sec})':volume=0")
    
    audio_filter_string = ",".join(filter_parts)

    # --- Determine Correct Codecs and Mapping ---
    input_extension = media_path.suffix.lower()
    output_extension = output_path.suffix.lower()
    video_formats = ['.mp4', '.mkv', '.mov', '.avi', '.webm']
    
    command = ["ffmpeg", "-threads", "0", "-y", "-i", str(media_path)]

    if input_extension in video_formats:
        console.print("Detected a [yellow]video file[/yellow]. Video stream will be copied.")
        command.extend(["-c:v", "copy"]) # Copy video stream without re-encoding
        command.extend(["-map", "0:v:0?"]) # Map the first video stream, if it exists
        command.extend(["-map", "0:a:0?"]) # Map the first audio stream, if it exists
    else:
        command.extend(["-map", "0:a"]) # For audio-only files

    # Determine audio codec based on output file type
    if output_extension == ".mp3":
        audio_codec = "libmp3lame"
    else: # Default to AAC for .mp4, .m4b, .m4a, etc.
        audio_codec = "aac"

    command.extend([
        "-af", audio_filter_string,
        "-c:a", audio_codec,
        "-ac", "2", # THIS IS THE FIX: Downmix audio to 2 channels (stereo)
        "-b:a", "192k",
        str(output_path)
    ])
    
    console.print("\n[bold cyan]Executing FFmpeg...[/bold cyan]")
    console.print("This may take a long time. FFmpeg progress will be shown below.")

    try:
        process = subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        console.print("\n[bold red]FFmpeg failed to process the file. Please review the output above for errors.[/bold red]")
        return
    except FileNotFoundError:
        console.print("\n[bold red]Error: 'ffmpeg' command not found.[/bold red]")
        return

    console.print("\n[bold green]Success![/bold green] Editing complete.")
    console.print(f"Output file saved to: [cyan]{output_path}[/cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silence sections of an audio track in a media file using a review CSV and FFmpeg.")
    parser.add_argument("media_file", type=Path, help="Path to the original audio or video file to edit.")
    parser.add_argument("review_csv", type=Path, help="Path to the review CSV file.")
    parser.add_argument("--output_file", type=Path, help="Path to save the new, edited media file.")
    
    args = parser.parse_args()
    
    p = args.media_file
    default_output_path = p.parent / f"{p.stem}_edited{p.suffix}"
    unique_output_path = get_unique_filepath(args.output_file if args.output_file else default_output_path)

    if not p.exists():
        console.print(f"[bold red]Error: Media file not found at '{p}'[/bold red]")
    else:
        edit_media_with_ffmpeg(p, args.review_csv, unique_output_path)
