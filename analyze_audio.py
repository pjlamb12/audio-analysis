#!/usr/bin/env python3
# analyze_audio.py

import sys
import os
import platform

# Auto-activate venv if not already active
# Determine the likely path for the venv python executable based on OS
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_dir = os.path.join(script_dir, "venv")

if sys.platform == "win32":
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
else:
    venv_python = os.path.join(venv_dir, "bin", "python3")

# Check if we are running from the venv
# On Windows, sys.prefix usually points to the venv root. On Unix, it might too.
# A robust check is to see if the venv python matches the current executable.
# However, sys.executable can be slightly different (lowercase on windows, resolved symlinks etc).
# Simpler check: is the venv_prefix inside sys.prefix?
if os.path.abspath(sys.prefix) != os.path.abspath(venv_dir):
    if os.path.exists(venv_python):
        # Re-execute the script with the venv python
        # On Windows, we need to quote arguments if they contain spaces, but execv handles logical args list
        try:
            os.execv(venv_python, [venv_python] + sys.argv)
        except OSError as e:
            print(f"Failed to activate venv: {e}")
            print("Running with system python.")
    else:
        print(f"Warning: 'venv' not found at {venv_dir}. Running with system python.")

import whisper
import whisper.timing
import torch
import subprocess
import numpy as np

# --- Monkeypatch for MPS Support ---
# Fixes: TypeError: Cannot convert a MPS Tensor to float64 dtype
def _patched_dtw(x):
    # If the tensor is on MPS, move to CPU before converting to double (float64)
    if hasattr(x, "device") and x.device.type == "mps":
        return whisper.timing.dtw_cpu(x.cpu().double().numpy())
    return whisper.timing.dtw_cpu(x.double().cpu().numpy())

# Only patch if using MPS, though it's safe to always patch as the function checks device
whisper.timing.dtw = _patched_dtw
# -----------------------------------
import pandas as pd
import argparse
from pathlib import Path
from rich.console import Console
import string

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

def get_audio_duration(file: Path) -> float:
    """Get the duration of an audio file using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        str(file)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        console.print(f"[bold red]Error getting duration: {e}[/bold red]")
        sys.exit(1)

def load_audio_chunk(file: Path, start_time: float, duration: float, sr: int = 16000):
    """
    Load a chunk of audio using ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", str(file),
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ss", str(start_time),
        "-t", str(duration),
        "-"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        audio = np.frombuffer(result.stdout, np.int16).flatten().astype(np.float32) / 32768.0
        return audio
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]FFmpeg error: {e.stderr.decode()}[/bold red]")
        raise

def analyze(audio_path: Path, words_path: Path, output_csv_path: Path, no_speech_thresh: float, logprob_thresh: float, temp: float, model_name: str, language: str = None):
    """
    Transcribes an audio file and finds timestamps of specified words.

    Args:
        audio_path (Path): Path to the input audio file.
        words_path (Path): Path to the text file with words to censor.
        output_csv_path (Path): Path to save the output CSV review file.
        no_speech_thresh (float): Threshold for detecting non-speech segments.
        logprob_thresh (float): Threshold for token log probability.
        temp (float): Temperature for transcription randomness.
        model_name (str): Name of the whisper model to use (e.g., 'base', 'medium').
    """
    with console.status("[bold cyan]Starting analysis...[/bold cyan]", spinner="dots") as status:
        # --- Load Model ---
        status.update(f"[bold cyan]Loading whisper model ({model_name})...[/bold cyan]")
        import torch
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            
        console.print(f"Using device: [bold yellow]{device}[/bold yellow] | Model: [bold yellow]{model_name}[/bold yellow]")
        model = whisper.load_model(model_name, device=device)

        # --- Load Banned Words ---
        status.update(f"Loading words to censor from [green]{words_path}[/green]")
        with open(words_path, "r", encoding='utf-8') as f:
            banned_words = {line.strip().lower() for line in f if line.strip()}
        
        if not banned_words:
            console.print("[bold red]Error: Banned words file is empty.[/bold red]")
            return
        console.print(f"Found [yellow]{len(banned_words)}[/yellow] words to search for.")

    # --- Transcribe Audio with Advanced Options (Chunked) ---
    console.print(f"Loading audio info from [green]{audio_path}[/green]...")
    
    # Process in 30-minute chunks to avoid OOM
    CHUNK_DURATION = 30 * 60  # 30 minutes in seconds
    
    total_duration = get_audio_duration(audio_path)
    total_chunks = int(total_duration // CHUNK_DURATION) + 1
    
    console.print(f"Audio Duration: {format_time(total_duration)} | Splitting into [yellow]{total_chunks}[/yellow] chunks of 30 mins.")

    all_words = []
    
    for i in range(total_chunks):
        chunk_offset_seconds = i * CHUNK_DURATION
        # Load only the necessary chunk
        # Add a small buffer (e.g. 1 sec) if needed, but strict 30m should be fine with independent processing
        # Note: whisper might need context, but for independent chunks we just accept boundaries might be cut.
        # Ideally, we overlap, but the original script didn't seem to overlap logic explicitly other than what whisper does internally?
        # The original script sliced `full_audio[start:end]`. That is a hard cut. So my logic preserves that behavior.
        
        chunk_data = load_audio_chunk(audio_path, chunk_offset_seconds, CHUNK_DURATION)
        
        console.print(f"\n[bold cyan]Processing Chunk {i+1}/{total_chunks}[/bold cyan] (Starts at {format_time(chunk_offset_seconds)})")
        
        # Transcribe chunk
        result = model.transcribe(
            chunk_data, 
            word_timestamps=True,
            temperature=temp,
            no_speech_threshold=no_speech_thresh,
            logprob_threshold=logprob_thresh,
            verbose=True,
            language=language
        )
        
        # Merge results with offset
        if result.get('segments'):
            for segment in result['segments']:
                for word_data in segment.get('words', []):
                    # Adjust timestamp by the chunk's start time
                    word_data['start'] += chunk_offset_seconds
                    word_data['end'] += chunk_offset_seconds
                    all_words.append(word_data)
        
    console.print("[bold green]Transcription complete![/bold green]")

    with console.status("[bold cyan]Processing matches...[/bold cyan]", spinner="dots") as status:
        # --- Find Matches ---
        status.update("Searching for banned words in transcription...")
        found_words = []
        # all_words is already populated from the chunking loop above

        for i, word_data in enumerate(all_words):
            if 'word' not in word_data:
                continue
            
            word_to_check = word_data['word'].strip().lower().strip(string.punctuation)
            if word_to_check in banned_words:
                start_seconds = word_data['start']
                start_idx = max(0, i - 5)
                end_idx = min(len(all_words), i + 6)
                context_words = [w.get('word', '') for w in all_words[start_idx:end_idx]]
                context = " ".join(context_words)

                found_words.append({
                    'start': start_seconds,
                    'hms_timestamp': format_time(start_seconds),
                    'end': word_data['end'],
                    'word': word_data['word'],
                    'context': context
                })

        if not found_words:
            console.print("[bold yellow]No banned words were found in the audio file.[/bold yellow]")
            return

        # --- Create and Save Review File ---
        status.update(f"Saving {len(found_words)} instances to review file...")
        df = pd.DataFrame(found_words)
        df = df[['start', 'hms_timestamp', 'end', 'word', 'context']]
        df.to_csv(output_csv_path, index=False)

    console.print(f"\n[bold green]Success![/bold green] Analysis complete.")
    console.print(f"Please review the file: [bold cyan]{output_csv_path}[/bold cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an audio file for specific words and generate a timestamp review file.")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to analyze.")
    parser.add_argument("--words_file", type=Path, default="banned_words.txt", help="Path to the text file containing words to censor.")
    parser.add_argument("--output_csv", type=Path, default="review.csv", help="Path to save the output review CSV file.")
    parser.add_argument("--no_speech_threshold", type=float, default=0.6, help="Threshold for VAD. Lower values are more aggressive in finding speech. (Default: 0.6)")
    parser.add_argument("--logprob_threshold", type=float, default=-1.0, help="Log probability threshold. (Default: -1.0)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling. Set to 0 for deterministic results. (Default: 0.1)")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model to use (tiny, base, small, medium, large). Default: medium")
    parser.add_argument("--language", type=str, default="en", help="Force language (e.g. 'en', 'zh'). Default: en")
    
    args = parser.parse_args()
    unique_output_path = get_unique_filepath(args.output_csv)

    if not args.audio_file.exists():
        console.print(f"[bold red]Error: Audio file not found at '{args.audio_file}'[/bold red]")
    elif not args.words_file.exists():
        console.print(f"[bold red]Error: Words file not found at '{args.words_file}'[/bold red]")
    else:
        analyze(args.audio_file, args.words_file, unique_output_path, args.no_speech_threshold, args.logprob_threshold, args.temperature, args.model, args.language)
