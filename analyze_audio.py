# analyze_audio.py

import whisper
import pandas as pd
import argparse
from pathlib import Path
from rich.console import Console

# Setup rich console for pretty printing
console = Console()

def format_time(seconds: float) -> str:
    """Converts seconds into HH:MM:SS format."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def analyze(audio_path: Path, words_path: Path, output_csv_path: Path):
    """
    Transcribes an audio file and finds timestamps of specified words.

    Args:
        audio_path (Path): Path to the input audio file.
        words_path (Path): Path to the text file with words to censor.
        output_csv_path (Path): Path to save the output CSV review file.
    """
    with console.status("[bold cyan]Starting analysis...[/bold cyan]") as status:
        # --- Load Model ---
        status.update("[bold cyan]Loading whisper model... (This may take a moment)[/bold cyan]")
        model = whisper.load_model("base")

        # --- Load Banned Words ---
        status.update(f"Loading words to censor from [green]{words_path}[/green]")
        with open(words_path, "r") as f:
            banned_words = {line.strip().lower() for line in f if line.strip()}
        
        if not banned_words:
            console.print("[bold red]Error: Banned words file is empty.[/bold red]")
            return
        console.print(f"Found [yellow]{len(banned_words)}[/yellow] words to search for.")

        # --- Transcribe Audio ---
        status.update(f"Transcribing audio from [green]{audio_path}[/green]... (This can take a long time)")
        result = model.transcribe(
            str(audio_path), 
            word_timestamps=True,
            temperature=(0.0, 0.2, 0.4, 0.6)
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
            
            word = word_data['word'].strip().lower()
            if word in banned_words:
                start_seconds = word_data['start']
                start_idx = max(0, i - 5)
                end_idx = min(len(all_words), i + 6)
                context_words = [w.get('word', '') for w in all_words[start_idx:end_idx]]
                context = " ".join(context_words)

                # Append all data, including the newly formatted time
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
        # Reorder columns to ensure 'hms_timestamp' is in a logical place
        df = df[['start', 'hms_timestamp', 'end', 'word', 'context']]
        df.to_csv(output_csv_path, index=False)

    console.print(f"\n[bold green]Success![/bold green] Analysis complete.")
    console.print(f"Please review the file: [bold cyan]{output_csv_path}[/bold cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an audio file for specific words and generate a timestamp review file.")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to analyze (e.g., input.mp3).")
    parser.add_argument("--words_file", type=Path, default="banned_words.txt", help="Path to the text file containing words to censor.")
    parser.add_argument("--output_csv", type=Path, default="review.csv", help="Path to save the output review CSV file.")
    
    args = parser.parse_args()

    if not args.audio_file.exists():
        console.print(f"[bold red]Error: Audio file not found at '{args.audio_file}'[/bold red]")
    elif not args.words_file.exists():
        console.print(f"[bold red]Error: Words file not found at '{args.words_file}'[/bold red]")
    else:
        analyze(args.audio_file, args.words_file, args.output_csv)