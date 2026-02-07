import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio_scripts')))

from dump_transcription import format_time, dump_transcription

def test_format_time():
    assert format_time(0) == "00:00:00"
    assert format_time(3665) == "01:01:05"

@patch('dump_transcription.whisper')
@patch('dump_transcription.console')
def test_dump_transcription(mock_console, mock_whisper, tmp_path):
    mock_model = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    
    mock_model.transcribe.return_value = {
        'segments': [
            {'words': [
                {'word': 'hello', 'start': 0.0, 'end': 1.0},
                {'word': 'world', 'start': 1.0, 'end': 2.0}
            ]}
        ]
    }
    
    audio_file = tmp_path / "audio.mp3"
    audio_file.touch()
    
    output_file = tmp_path / "dump.txt"
    
    dump_transcription(
        audio_path=audio_file,
        output_txt_path=output_file,
        no_speech_thresh=0.6,
        logprob_thresh=-1.0,
        temp=0.1
    )
    
    assert output_file.exists()
    content = output_file.read_text()
    assert "Full Transcription for: audio.mp3" in content
    assert "[00:00:00] (Start: 0.00, End: 1.00) hello" in content
    assert "[00:00:01] (Start: 1.00, End: 2.00) world" in content
