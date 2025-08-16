# analyze_audio.py

import whisper
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

def analyze(audio_path: Path, words_path: Path, output_csv_path: Path, no_speech_thresh: float, logprob_thresh: float, temp: float):
    """
    Transcribes an audio file and finds timestamps of specified words.

    Args:
        audio_path (Path): Path to the input audio file.
        words_path (Path): Path to the text file with words to censor.
        output_csv_path (Path): Path to save the output CSV review file.
        no_speech_thresh (float): Threshold for detecting non-speech segments.
        logprob_thresh (float): Threshold for token log probability.
        temp (float): Temperature for transcription randomness.
    """
    with console.status("[bold cyan]Starting analysis...[/bold cyan]", spinner="dots") as status:
        # --- Load Model ---
        status.update("[bold cyan]Loading whisper model...[/bold cyan]")
        model = whisper.load_model("base")

        # --- Load Banned Words ---
        status.update(f"Loading words to censor from [green]{words_path}[/green]")
        with open(words_path, "r", encoding='utf-8') as f:
            banned_words = {line.strip().lower() for line in f if line.strip()}
        
        if not banned_words:
            console.print("[bold red]Error: Banned words file is empty.[/bold red]")
            return
        console.print(f"Found [yellow]{len(banned_words)}[/yellow] words to search for.")

        # --- Transcribe Audio with Advanced Options ---
        status.update(f"Transcribing audio from [green]{audio_path}[/green]... (This can take a long time)")
        result = model.transcribe(
            str(audio_path), 
            word_timestamps=True,
            temperature=temp,
            no_speech_threshold=no_speech_thresh,
            logprob_threshold=logprob_thresh
        )
        console.print("[bold green]Transcription complete![/bold green]")

        # --- Find Matches ---
        status.update("Searching for banned words in transcription...")
        found_words = []
        all_words = []
        if result.get('segments'):
            for segment in result['segments']:
                all_words.extend(segment.get('words', []))

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
    
    args = parser.parse_args()
    unique_output_path = get_unique_filepath(args.output_csv)

    if not args.audio_file.exists():
        console.print(f"[bold red]Error: Audio file not found at '{args.audio_file}'[/bold red]")
    elif not args.words_file.exists():
        console.print(f"[bold red]Error: Words file not found at '{args.words_file}'[/bold red]")
    else:
        analyze(args.audio_file, args.words_file, unique_output_path, args.no_speech_threshold, args.logprob_threshold, args.temperature)
