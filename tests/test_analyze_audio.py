import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add audio_scripts to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio_scripts')))

from analyze_audio import format_time, get_unique_filepath, analyze

def test_format_time():
    assert format_time(0) == "00:00:00"
    assert format_time(61) == "00:01:01"
    assert format_time(3661) == "01:01:01"
    assert format_time(3600) == "01:00:00"

def test_get_unique_filepath_no_conflict(tmp_path):
    # If file doesn't exist, return it as is
    p = tmp_path / "test.csv"
    assert get_unique_filepath(p) == p

def test_get_unique_filepath_conflict(tmp_path):
    # If file exists, return new path with number
    p = tmp_path / "test.csv"
    p.touch()
    
    expected = tmp_path / "test1.csv"
    assert get_unique_filepath(p) == expected
    
    # If test1.csv also exists
    expected.touch()
    expected_2 = tmp_path / "test2.csv"
    assert get_unique_filepath(p) == expected_2

@patch('analyze_audio.whisper')
@patch('analyze_audio.subprocess')
@patch('analyze_audio.console')
def test_analyze_no_banned_words_found(mock_console, mock_subprocess, mock_whisper, tmp_path):
    # Setup mocks
    mock_model = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    
    # Mock result from transcribe
    mock_model.transcribe.return_value = {
        'segments': [
            {'words': [{'word': 'hello', 'start': 0.0, 'end': 1.0}]}
        ]
    }
    
    # Mock audio duration (via subprocess call in strict sense, or mocked get_audio_duration if we patch it directly)
    # The script calls get_audio_duration which calls subprocess.run
    mock_run = MagicMock()
    mock_run.stdout = "10.0" # 10 seconds
    mock_subprocess.run.return_value = mock_run

    # Mock load_audio_chunk to return dummy numpy array
    # The script calls load_audio_chunk -> subprocess.run -> capture stdout
    # matched by mock_run above? No, mock_run is for ffprobe. ffmpeg call follows.
    # We might need side_effect for subprocess.run
    
    def subprocess_side_effect(*args, **kwargs):
        cmd = args[0]
        if cmd[0] == 'ffprobe':
            m = MagicMock()
            m.stdout = "10.0"
            return m
        elif cmd[0] == 'ffmpeg':
            m = MagicMock()
            # 16000 * 10 * 2 bytes (int16) roughly?
            # actually load_audio_chunk: np.frombuffer(result.stdout, np.int16)
            # return some bytes
            import numpy as np
            dummy_audio = np.zeros(16000 * 10, dtype=np.int16).tobytes()
            m.stdout = dummy_audio
            return m
        return MagicMock()

    mock_subprocess.run.side_effect = subprocess_side_effect

    # Create dummy files
    audio_file = tmp_path / "audio.mp3"
    audio_file.touch()
    
    words_file = tmp_path / "banned.txt"
    words_file.write_text("badword")
    
    output_csv = tmp_path / "output.csv"
    
    # Run analysis
    analyze(
        audio_path=audio_file,
        words_path=words_file,
        output_csv_path=output_csv,
        no_speech_thresh=0.6,
        logprob_thresh=-1.0,
        temp=0.1,
        model_name="base"
    )
    
    # Verify no csv created if no words found? 
    # analyze() creates csv ONLY if found_words is not empty.
    # We mocked transcription to have 'hello', banned is 'badword'.
    # So no match.
    assert not output_csv.exists()
    # Verify console printed something about success/failure?
    # Actually the script returns early if no found words:
    # console.print("[bold yellow]No banned words were found in the audio file.[/bold yellow]")
    
    # Check if that message was printed
    # args of call:
    # mock_console.print.assert_any_call("[bold yellow]No banned words were found in the audio file.[/bold yellow]")
    # Rich console print might be complex to match exactly string, but let's try basic validity.
    assert mock_model.transcribe.called

@patch('analyze_audio.whisper')
@patch('analyze_audio.subprocess')
@patch('analyze_audio.console')
def test_analyze_banned_words_found(mock_console, mock_subprocess, mock_whisper, tmp_path):
    # Setup mocks
    mock_model = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    
    # Mock result with banned word
    mock_model.transcribe.return_value = {
        'segments': [
            {'words': [
                {'word': 'hello', 'start': 0.0, 'end': 1.0},
                {'word': 'badword', 'start': 1.0, 'end': 2.0}
            ]}
        ]
    }
    
    mock_subprocess.run.side_effect = lambda *args, **kwargs: MagicMock(stdout=b"") if args[0][0] == 'ffmpeg' else MagicMock(stdout="10.0")

    audio_file = tmp_path / "audio.mp3"
    audio_file.touch()
    
    words_file = tmp_path / "banned.txt"
    words_file.write_text("badword")
    
    output_csv = tmp_path / "output.csv"
    
    # Run analysis
    analyze(
        audio_path=audio_file,
        words_path=words_file,
        output_csv_path=output_csv,
        no_speech_thresh=0.6,
        logprob_thresh=-1.0,
        temp=0.1,
        model_name="base"
    )
    
    # Verify csv created
    assert output_csv.exists()
    # Check content
    content = output_csv.read_text()
    assert "badword" in content
    assert "hello badword" in content # Context
