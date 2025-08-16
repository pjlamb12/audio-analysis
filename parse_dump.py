# parse_dump.py

import argparse
import re
from pathlib import Path
import pandas as pd
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

def parse_dump_file(dump_path: Path, words_path: Path, output_csv_path: Path):
    """
    Parses a transcription dump file to find banned words and creates a review CSV.

    Args:
        dump_path (Path): Path to the transcription dump .txt file.
        words_path (Path): Path to the text file with words to censor.
        output_csv_path (Path): Path to save the output review CSV file.
    """
    # --- 1. Load Banned Words ---
    console.print(f"Loading banned words from [green]{words_path}[/green]...")
    try:
        with open(words_path, "r", encoding='utf-8') as f:
            banned_words = {line.strip().lower() for line in f if line.strip()}
        if not banned_words:
            console.print(f"[bold red]Error: Banned words file '{words_path}' is empty.[/bold red]")
            return
    except FileNotFoundError:
        console.print(f"[bold red]Error: Banned words file not found at '{words_path}'[/bold red]")
        return
    
    console.print(f"Found [yellow]{len(banned_words)}[/yellow] words to search for.")

    # --- 2. Parse the Dump File ---
    console.print(f"Parsing transcription dump file from [green]{dump_path}[/green]...")
    all_words_data = []
    # Regex to capture start, end, and the word from each line
    line_pattern = re.compile(r"\[\d{2}:\d{2}:\d{2}\] \(Start: ([\d.]+), End: ([\d.]+)\) (.*)")
    
    try:
        with open(dump_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = line_pattern.match(line)
                if match:
                    all_words_data.append({
                        'start': float(match.group(1)),
                        'end': float(match.group(2)),
                        'word': match.group(3).strip()
                    })
    except FileNotFoundError:
        console.print(f"[bold red]Error: Transcription dump file not found at '{dump_path}'[/bold red]")
        return

    if not all_words_data:
        console.print(f"[bold red]Error: No valid word entries found in '{dump_path}'.[/bold red]")
        return

    # --- 3. Find Matches and Generate Context ---
    console.print("Searching for banned words in parsed data...")
    found_words = []
    for i, word_info in enumerate(all_words_data):
        # Strip punctuation from the word before checking against the banned list
        word_to_check = word_info['word'].lower().strip(string.punctuation)
        if word_to_check in banned_words:
            # Generate context from the surrounding words in our parsed list
            start_idx = max(0, i - 5)
            end_idx = min(len(all_words_data), i + 6)
            context_words = [w['word'] for w in all_words_data[start_idx:end_idx]]
            context = " ".join(context_words)

            found_words.append({
                'start': word_info['start'],
                'hms_timestamp': format_time(word_info['start']),
                'end': word_info['end'],
                'word': word_info['word'], # Save the original word with punctuation
                'context': context
            })

    if not found_words:
        console.print("[bold yellow]No banned words were found in the dump file.[/bold yellow]")
        return

    # --- 4. Create and Save Review File ---
    console.print(f"Found [yellow]{len(found_words)}[/yellow] instances. Saving to review file...")
    df = pd.DataFrame(found_words)
    df = df[['start', 'hms_timestamp', 'end', 'word', 'context']]
    df.to_csv(output_csv_path, index=False)

    console.print(f"\n[bold green]Success![/bold green] Review file created.")
    console.print(f"You can now use this file with edit_audio.py: [bold cyan]{output_csv_path}[/bold cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a transcription dump file to find banned words.")
    parser.add_argument("dump_file", type=Path, help="Path to the transcription dump .txt file.")
    parser.add_argument("--words_file", type=Path, default="banned_words.txt", help="Path to the text file containing words to censor.")
    parser.add_argument("--output_csv", type=Path, default="review.csv", help="Path to save the output review CSV file.")
    
    args = parser.parse_args()

    # Get a unique path for the output CSV to avoid overwriting
    unique_output_path = get_unique_filepath(args.output_csv)

    parse_dump_file(args.dump_file, args.words_file, unique_output_path)
