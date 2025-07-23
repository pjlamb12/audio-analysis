# edit_audio.py

import pandas as pd
import argparse
import subprocess
from pathlib import Path
from rich.console import Console

# Setup rich console for pretty printing
console = Console()

def edit_with_ffmpeg(audio_path: Path, review_csv_path: Path, output_audio_path: Path):
    """
    Silences sections of an audio file using FFmpeg, based on a review CSV.
    This script can handle review files for both individual words and entire topics.

    Args:
        audio_path (Path): Path to the original audio file.
        review_csv_path (Path): Path to the CSV file with timestamps to silence.
        output_audio_path (Path): Path to save the edited audio file.
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
        console.print("No output file will be created.")
        return

    # --- Construct the FFmpeg Filter ---
    console.print(f"Building FFmpeg filter for [yellow]{len(df)}[/yellow] sections...")
    filter_parts = []
    for _, row in df.iterrows():
        start_sec = row[start_col]
        end_sec = row[end_col]
        filter_parts.append(f"volume=enable='between(t,{start_sec},{end_sec})':volume=0")
    
    audio_filter_string = ",".join(filter_parts)

    # --- Determine Correct Codec Based on Output Extension ---
    output_extension = output_audio_path.suffix.lower()
    if output_extension == ".mp3":
        audio_codec = "libmp3lame"
    elif output_extension in [".m4b", ".m4a", ".mp4"]:
        audio_codec = "aac"
    else:
        console.print(f"[bold yellow]Warning: Unknown output format '{output_extension}'. Defaulting to 'aac' codec.[/bold yellow]")
        audio_codec = "aac"

    # --- Build and Run the FFmpeg Command ---
    command = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-map", "0:a",              # Select only the audio stream
        "-map_metadata", "0",       # Copy all metadata from the input
        "-af", audio_filter_string, # Apply the silencing audio filter
        "-c:a", audio_codec,        # Use the determined audio codec
        "-b:a", "128k",             # Set a reasonable bitrate
        str(output_audio_path)
    ]
    
    console.print("\n[bold cyan]Executing FFmpeg...[/bold cyan]")
    console.print("This may take a long time for large files. FFmpeg will print its own progress below.")

    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        console.print("\n[bold red]--- FFmpeg Error ---[/bold red]")
        console.print("FFmpeg failed to process the file. Here is the error output:")
        console.print(f"[red]{e.stderr}[/red]")
        return
    except FileNotFoundError:
        console.print("\n[bold red]Error: 'ffmpeg' command not found.[/bold red]")
        console.print("Please ensure FFmpeg is installed and accessible on your system's PATH.")
        return

    console.print("\n[bold green]Success![/bold green] Editing complete.")
    console.print(f"Output file saved to: [cyan]{output_audio_path}[/cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silence sections of an audio file (for words or topics) using a review CSV and FFmpeg.")
    parser.add_argument("audio_file", type=Path, help="Path to the original audio file to edit.")
    parser.add_argument("review_csv", type=Path, help="Path to the review CSV file from either analyze_audio.py or find_topics.py.")
    parser.add_argument("--output_file", type=Path, help="Path to save the new, edited audio file.")
    
    args = parser.parse_args()

    if args.output_file:
        output_file = args.output_file
    else:
        p = args.audio_file
        output_file = p.parent / f"{p.stem}_edited{p.suffix}"

    if not args.audio_file.exists():
        console.print(f"[bold red]Error: Audio file not found at '{args.audio_file}'[/bold red]")
    else:
        edit_with_ffmpeg(args.audio_file, args.review_csv, output_file)
