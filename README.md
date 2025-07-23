# Audio Analysis Toolkit

This project provides a suite of Python scripts to analyze and manipulate audio files, particularly large audiobook files (`.mp3`, `.m4b`). It allows you to perform two main tasks:

1.  **Censor Specific Words**: Automatically find and silence specific words (e.g., profanity) throughout an audio file.

2.  **Find Topics**: Identify and timestamp sections of an audio file that discuss specific topics (e.g., "camping" or "space travel").

The toolkit is designed with a two-step, human-in-the-loop process: an **analysis script** first generates a review file, and a second **action script** applies the changes, giving you full control over the final output.

## Features

-   **Word Censorship**: Scans audio for a custom list of words and generates a review file with timestamps. A second script then silences these words in the final audio.

-   **Topic Analysis**: Uses a modern AI model to find segments related to a custom list of topics, creating a review file with timestamps and confidence scores.

-   **Large File Support**: Uses `ffmpeg` for robust processing of large audio files (>4GB) that other libraries can't handle.

-   **User-Friendly**: Provides clear command-line interfaces and progress indicators for long-running tasks.

-   **Configurable**: Uses simple `.txt` files to manage lists of banned words and topics.

## How It Works

This project combines several powerful open-source technologies:

-   **Whisper (from OpenAI)**: A state-of-the-art AI model used to transcribe audio into text with highly accurate word-level timestamps. This forms the foundation for all analysis.

-   **FFmpeg**: The industry-standard for audio/video manipulation. It's used to robustly process and edit even very large audio files without running out of memory.

-   **Zero-Shot Classification (from Hugging Face)**: A flexible AI model that can identify topics within a piece of text without being explicitly trained on them. This powers the topic-finding script.

## Setup and Installation

### 1. Prerequisite: FFmpeg

You must have `ffmpeg` installed on your system and accessible from your command line.

-   **macOS (using Homebrew)**:

    ```bash
    brew install ffmpeg
    ```

-   **Windows (using Chocolatey)**:

    ```bash
    choco install ffmpeg
    ```

-   **Linux (using apt)**:

    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```

### 2. Python Environment

This project requires Python 3.9 or newer.

**Clone the repository and create a virtual environment (recommended):**

```bash
git clone <your-repo-url>
cd audio-analysis-toolkit
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Python Packages

Install all the necessary Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

_Note:_ The first time you run a script, it may need to download AI models (a few GB), _which can take some time._

## Usage

The toolkit is divided into two main workflows: **Censoring Words** and **Finding Topics**.

### Workflow 1: Censoring Specific Words

This is a two-step process to find and remove words from your audio.

#### Step 1: Create a `banned_words.txt` File

Create a file named `banned_words.txt` in your project directory. Add one word or phrase per line. The search is case-insensitive.

```txt
# banned_words.txt
heck
darn
fudge
```

#### Step 2: Run the Analysis Script (`analyze_audio.py`)

This script transcribes the audio and creates a `review.csv` file listing every occurrence of the banned words.

**Command:**

```bash
python analyze_audio.py "path/to/your/audio.mp3"
```

This will generate `review.csv` with the following columns:

-   `start`: Start time of the word in seconds.
-   `hms_timestamp`: Start time in `HH:MM:SS` format for easy checking.
-   `end`: End time of the word in seconds.
-   `word`: The specific word that was found.
-   `context`: The surrounding words for context.

#### Step 3: Review and Edit `review.csv`

Open `review.csv` in a spreadsheet program. **Delete any rows** for words you _do not_ want to censor. Save the file.

#### Step 4: Run the Editing Script (`edit_audio.py`)

This script reads your approved `review.csv` and uses `ffmpeg` to create a new, silenced audio file.

**Command:**

```bash
python edit_audio.py "path/to/your/audio.mp3" "review.csv"
```

This will create a new file named `audio_edited.mp3` in the same directory.

### Workflow 2: Finding Topics

This process finds sections of the audio that are related to specific topics.

#### Step 1: Create a `topics.txt` File

Create a file named `topics.txt` in your project directory. Add one topic per line. These can be single words or short phrases.

```txt
# topics.txt
camping
outdoor survival
building a fire
space travel
celestial mechanics
```

#### Step 2: Run the Topic-Finding Script (`find_topics.py`)

This script transcribes the audio, breaks it into 90-second chunks, and analyzes each chunk for your topics.

**Command:**

```bash
python find_topics.py "path/to/your/audiobook.m4b"
```

You can also specify a different topics file:

```bash
python find_topics.py "path/to/your/audiobook.m4b" --topics_file "other_topics.txt"
```

This will generate `review_topics.csv` with the following columns:

-   `start_seconds`: Start time of the relevant section.
-   `hms_timestamp`: Start time in `HH:MM:SS` format.
-   `end_seconds`: End time of the section.
-   `topic`: The topic that was identified as the best match.
-   `confidence`: The model's confidence in the match (e.g., 85.12%).
-   `text_segment`: The transcribed text from that section.

#### Step 3: Run the Editing Script (`edit_audio.py`)

This script reads your approved `review_topics.csv` and uses `ffmpeg` to create a new, silenced audio file.

**Command:**

```bash
python edit_audio.py "path/to/your/audio.mp3" "review_topics.csv"
```

This will create a new file named `audio_edited.mp3` in the same directory.
