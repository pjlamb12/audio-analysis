# find_topics.py

import sys
import os

# Auto-activate venv if not already active
if sys.prefix != os.path.abspath(os.path.join(os.path.dirname(__file__), "venv")):
    venv_python = os.path.join(os.path.dirname(__file__), "venv", "bin", "python3")
    if os.path.exists(venv_python):
        # Re-execute the script with the venv python
        os.execv(venv_python, [venv_python] + sys.argv)
    else:
        print("Warning: 'venv' not found. Running with system python.")

import whisper
import pandas as pd
import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import track
from transformers import pipeline

# Setup rich console for pretty printing
console = Console()

def format_time(seconds: float) -> str:
    """Converts seconds into HH:MM:SS format."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def analyze_for_topics(audio_path: Path, topics_path: Path, output_csv_path: Path):
    """
    Analyzes an audio file for specific topics using a zero-shot model.

    Args:
        audio_path (Path): Path to the input audio file.
        topics_path (Path): Path to the text file with topics to search for.
        output_csv_path (Path): Path to save the output CSV review file.
    """
    with console.status("[bold cyan]Starting analysis...[/bold cyan]") as status:
        # --- 1. Load Topics and Models ---
        status.update(f"Loading topics from [green]{topics_path}[/green]")
        try:
            with open(topics_path, "r") as f:
                topics = [line.strip() for line in f if line.strip()]
            if not topics:
                console.print(f"[bold red]Error: Topics file '{topics_path}' is empty or contains no valid topics.[/bold red]")
                return
        except FileNotFoundError:
            console.print(f"[bold red]Error: Topics file not found at '{topics_path}'[/bold red]")
            return
        
        console.print(f"Found [yellow]{len(topics)}[/yellow] topics to search for.")

        status.update("[bold cyan]Loading Whisper model...[/bold cyan]")
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        console.print(f"Using device: [bold yellow]{device}[/bold yellow]")
        whisper_model = whisper.load_model("base", device=device)

        status.update("[bold cyan]Loading Zero-Shot Classification model... (May download on first run)[/bold cyan]")
        device_id = 0 if device == "mps" else -1
        # Note: 'mps' string for device in transformers pipeline might require pytorch setup, usually integer for GPU or 'cpu'
        # For HuggingFace pipeline, device=0 uses the first GPU. For MPS it's often supported via string or mps device object.
        # Let's use the explicit device argument for pipeline which generally accepts "mps" strings in newer versions.
        classifier = pipeline("zero-shot-classification", device=device)

    # --- 2. Transcribe Audio ---
    console.print(f"Transcribing audio from [green]{audio_path}[/green]... (Live Output)")
    transcription_result = whisper_model.transcribe(str(audio_path), word_timestamps=True, verbose=True)
    console.print("[bold green]Transcription complete![/bold green]")

    with console.status("[bold cyan]Analyzing text...[/bold cyan]", spinner="dots") as status:
        # --- 3. Chunk Transcription into Segments ---
        status.update("Chunking transcription into 90-second segments...")
        all_words = []
        if transcription_result.get('segments'):
            for segment in transcription_result['segments']:
                all_words.extend(segment.get('words', []))

        if not all_words:
            console.print("[bold red]Could not extract any words from the audio.[/bold red]")
            return

        chunk_duration = 90
        text_chunks = []
        current_chunk_text = ""
        chunk_start_time = all_words[0]['start']

        for word_info in all_words:
            current_chunk_text += word_info['word'] + " "
            if word_info['end'] - chunk_start_time > chunk_duration:
                text_chunks.append({
                    'start': chunk_start_time,
                    'end': word_info['end'],
                    'text': current_chunk_text.strip()
                })
                current_chunk_text = ""
                chunk_start_time = word_info['end']
        
        if current_chunk_text:
            text_chunks.append({
                'start': chunk_start_time,
                'end': all_words[-1]['end'],
                'text': current_chunk_text.strip()
            })

        console.print(f"Created [yellow]{len(text_chunks)}[/yellow] text chunks for analysis.")

        # --- 4. Analyze Chunks for Topics ---
        found_topics = []
        confidence_threshold = 0.70

        for chunk in track(text_chunks, description="Analyzing chunks for topics..."):
            results = classifier(chunk['text'], candidate_labels=topics)
            
            top_score = results['scores'][0]
            top_label = results['labels'][0]

            if top_score >= confidence_threshold:
                found_topics.append({
                    'start_seconds': chunk['start'],
                    'hms_timestamp': format_time(chunk['start']),
                    'end_seconds': chunk['end'],
                    'topic': top_label,
                    'confidence': f"{top_score:.2%}",
                    'text_segment': chunk['text']
                })
        
        if not found_topics:
            console.print("[bold yellow]No sections matching the topics were found with high confidence.[/bold yellow]")
            return

        # --- 5. Save Results ---
        status.update("Saving results to review file...")
        df = pd.DataFrame(found_topics)
        df.to_csv(output_csv_path, index=False)

    console.print(f"\n[bold green]Success![/bold green] Topic analysis complete.")
    console.print(f"Please review the file: [bold cyan]{output_csv_path}[/bold cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an audio file for specific topics and find their timestamps.")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file to analyze (e.g., input.mp3 or input.m4b).")
    parser.add_argument("--topics_file", type=Path, default="topics.txt", help="Path to a text file containing topics, one per line.")
    parser.add_argument("--output_csv", type=Path, default="review_topics.csv", help="Path to save the output topic review CSV file.")
    
    args = parser.parse_args()
    
    if not args.audio_file.exists():
        console.print(f"[bold red]Error: Audio file not found at '{args.audio_file}'[/bold red]")
    else:
        analyze_for_topics(args.audio_file, args.topics_file, args.output_csv)
