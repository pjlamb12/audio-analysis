import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio_scripts')))

from edit_audio import get_unique_filepath, edit_media_with_ffmpeg

def test_get_unique_filepath(tmp_path):
    p = tmp_path / "out.mp3"
    assert get_unique_filepath(p) == p
    
    p.touch()
    assert get_unique_filepath(p) == tmp_path / "out1.mp3"

@patch('edit_audio.subprocess')
@patch('edit_audio.console')
def test_edit_media_with_ffmpeg(mock_console, mock_subprocess, tmp_path):
    # Setup
    media_file = tmp_path / "input.mp3"
    media_file.touch()
    
    review_file = tmp_path / "review.csv"
    review_file.write_text("start,end,word,context\n0.0,1.0,bad,ctx")
    
    output_file = tmp_path / "output.mp3"
    
    edit_media_with_ffmpeg(media_file, review_file, output_file)
    
    assert mock_subprocess.run.called
    args, _ = mock_subprocess.run.call_args
    cmd = args[0]
    
    # Check if command looks correct
    assert cmd[0] == "ffmpeg"
    assert str(media_file) in cmd
    assert str(output_file) in cmd
    # Check filter
    assert any("volume=enable='between(t,0.0,1.0)':volume=0" in arg for arg in cmd)

@patch('edit_audio.subprocess')
@patch('edit_audio.console')
def test_edit_media_mixed_columns(mock_console, mock_subprocess, tmp_path):
    # Test support for start_seconds/end_seconds from topics
    media_file = tmp_path / "input.mp3"
    media_file.touch()
    
    review_file = tmp_path / "review_topics.csv"
    review_file.write_text("start_seconds,end_seconds,topic\n0.0,2.0,topic")
    
    output_file = tmp_path / "output.mp3"
    
    edit_media_with_ffmpeg(media_file, review_file, output_file)
    
    assert mock_subprocess.run.called
    args, _ = mock_subprocess.run.call_args
    # Check filter
    cmd = args[0]
    assert any("volume=enable='between(t,0.0,2.0)':volume=0" in arg for arg in cmd)
