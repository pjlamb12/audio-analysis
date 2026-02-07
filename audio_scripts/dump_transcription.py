#!/usr/bin/env python3
# dump_transcription.py

import sys
import os
import platform

# Auto-activate venv if not already active
script_dir = os.path.dirname(os.path.abspath(__file__))
# Venv is now one level up from the scripts
venv_dir = os.path.join(script_dir, "..", "venv")

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

import whisper
import argparse
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

def format_time(seconds: float) -> str:
    """Converts seconds into HH:MM:SS format."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def dump_transcription(audio_path: Path, output_txt_path: Path, no_speech_thresh: float, logprob_thresh: float, temp: float):
    """
    Transcribes an entire audio file and saves the full, timestamped
    transcription to a text file for debugging.
    """
    with console.status("[bold cyan]Starting full transcription...[/bold cyan]") as status:
        status.update("[bold cyan]Loading Whisper model...[/bold cyan]")
        import torch
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

        console.print(f"Using device: [bold yellow]{device}[/bold yellow]")
        model = whisper.load_model("base", device=device)

    console.print(f"Transcribing audio from [green]{audio_path}[/green]... (Live Output)")
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        temperature=temp,
        no_speech_threshold=no_speech_thresh,
        logprob_threshold=logprob_thresh,
        verbose=True
    )
    console.print("[bold green]Transcription complete![/bold green]")

    with console.status("[bold cyan]Saving dump file...[/bold cyan]") as status:
        status.update(f"Writing full transcription to [green]{output_txt_path}[/green]...")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Full Transcription for: {audio_path.name}\n")
            f.write("="*40 + "\n\n")

            all_words = []
            if result.get('segments'):
                for segment in result['segments']:
                    all_words.extend(segment.get('words', []))

            if not all_words:
                f.write("No words were transcribed from this audio file.")
                return

            for word_info in all_words:
                if 'word' in word_info:
                    hms_time = format_time(word_info['start'])
                    start_sec = word_info['start']
                    end_sec = word_info['end']
                    word = word_info['word']
                    f.write(f"[{hms_time}] (Start: {start_sec:.2f}, End: {end_sec:.2f}) {word}\n")
        
    console.print(f"\n[bold green]Success![/bold green] Full transcription saved.")
    console.print(f"You can now open and search the file: [bold cyan]{output_txt_path}[/bold cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an entire audio file to a text file for debugging.")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to transcribe.")
    parser.add_argument("--output_file", type=Path, help="Path to save the output text file.")
    parser.add_argument("--no_speech_threshold", type=float, default=0.6, help="Threshold for VAD. Lower values are more aggressive. (Default: 0.6)")
    parser.add_argument("--logprob_threshold", type=float, default=-1.0, help="Log probability threshold. (Default: -1.0)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling. (Default: 0.1)")
    
    args = parser.parse_args()
    
    initial_output_path = args.output_file if args.output_file else Path("transcription_dump.txt")
    unique_output_path = get_unique_filepath(initial_output_path)

    if not args.audio_file.exists():
        console.print(f"[bold red]Error: Audio file not found at '{args.audio_file}'[/bold red]")
    else:
        dump_transcription(args.audio_file, unique_output_path, args.no_speech_threshold, args.logprob_threshold, args.temperature)
