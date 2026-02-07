import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio_scripts')))

from find_topics import format_time, analyze_for_topics

def test_format_time():
    assert format_time(0) == "00:00:00"

@patch('find_topics.whisper')
@patch('find_topics.pipeline')
@patch('find_topics.console')
def test_analyze_for_topics(mock_console, mock_pipeline, mock_whisper, tmp_path):
    # Mock Whisper
    mock_whisper_model = MagicMock()
    mock_whisper.load_model.return_value = mock_whisper_model
    
    # Mock transcription result: simple words
    mock_whisper_model.transcribe.return_value = {
        'segments': [
            {'words': [{'word': 'This', 'start': 0.0, 'end': 0.5},
                       {'word': 'is', 'start': 0.5, 'end': 1.0},
                       {'word': 'a', 'start': 1.0, 'end': 1.5},
                       {'word': 'test', 'start': 1.5, 'end': 2.0}]}
        ]
    }
    
    # Mock classifier pipeline
    mock_classifier = MagicMock()
    mock_pipeline.return_value = mock_classifier
    
    # Mock classification result
    # It returns a dict with 'scores' and 'labels'
    mock_classifier.return_value = {
        'scores': [0.95],
        'labels': ['testing']
    }
    
    audio_file = tmp_path / "audio.mp3"
    audio_file.touch()
    
    topics_file = tmp_path / "topics.txt"
    topics_file.write_text("testing")
    
    output_csv = tmp_path / "topics.csv"
    
    analyze_for_topics(audio_file, topics_file, output_csv)
    
    assert output_csv.exists()
    content = output_csv.read_text()
    assert "testing" in content
    assert "95.00%" in content
