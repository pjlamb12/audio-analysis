# Audio Analysis Tools

A collection of tools to transcribe audio, find bad words or specific topics, and automatically edit them out using ffmpeg.

## Features

- **Analyze Audio**: Transcribes audio using OpenAI's Whisper and searches for specific "banned" words.
- **Find Topics**: Uses Whisper + Zero-Shot Classification to find timestamps for specific topics.
- **Edit Audio/Video**: Takes a CSV of timestamps (from the above tools) and silences those sections in the media file.
- **Hardware Acceleration**: Optimized for Apple Silicon (MPS) for faster transcription and inference.

## Setup

1.  **Environment**:
    The tools require Python 3. This project is configured to use a local virtual environment (`venv`) to manage dependencies and avoid system conflicts.

    To create the environment and install dependencies (run once):

    **macOS / Linux:**

    ```bash
    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt
    ```

    **Windows:**

    ```bash
    python -m venv venv
    venv\Scripts\pip install -r requirements.txt
    ```

2.  **FFmpeg**:
    You need `ffmpeg` installed on your system for editing media.
    ```bash
    brew install ffmpeg
    ```

## Usage

You can run the python scripts directly. They are configured to automatically use the `venv` environment.

### 1. Analyze Audio (Word Search)

Search for specific words listed in `banned_words.txt`.

```bash
python analyze_audio.py input_audio.mp3
```

_Output: `review.csv`_

### 2. Find Topics

Search for theoretical topics found in `topics.txt`.

```bash
python find_topics.py input_audio.mp3
```

_Output: `review_topics.csv`_

### 3. Edit Media

Silence the sections found in the review CSV.

```bash
python edit_audio.py input_audio.mp3 review.csv
```

_Output: `input_audio_edited.mp3`_

### 4. Debugging Tools

If you are having trouble with the analysis or just want to see the full transcription with timestamps:

**a. Dump Transcription**

Transcribe the entire file to a text file.

```bash
python dump_transcription.py input_audio.mp3
```

_Output: `transcription_dump.txt`_

**b. Parse Dump**

Search for banned words within the dump file (faster than re-transcribing).

```bash
python parse_dump.py transcription_dump.txt
```

_Output: `review.csv`_

## Optimizations

The scripts are optimized to use **GPU acceleration** where available:

- **macOS (Apple Silicon):** Uses **MPS** (Metal Performance Shaders).
- **Windows / Linux (NVIDIA):** Uses **CUDA** (if available).
- **CPU Fallback:** If no GPU is found, it falls back to CPU.

FFmpeg operations are also multi-threaded.
