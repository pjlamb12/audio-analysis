import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio_scripts')))

from edit_video import get_unique_filepath, edit_video

def test_get_unique_filepath(tmp_path):
    p = tmp_path / "video.mp4"
    assert get_unique_filepath(p) == p
    p.touch()
    assert get_unique_filepath(p) == tmp_path / "video1.mp4"

@patch('edit_video.subprocess')
@patch('edit_video.console')
def test_edit_video(mock_console, mock_subprocess, tmp_path):
    video_file = tmp_path / "video.mp4"
    video_file.touch()
    
    review_csv = tmp_path / "review.csv"
    review_csv.write_text("start_seconds,end_seconds,labels\n1.0,2.0,BUTTS")
    
    output_file = tmp_path / "output.mp4"
    
    edit_video(video_file, review_csv, output_file)
    
    assert mock_subprocess.run.called
    args, _ = mock_subprocess.run.call_args
    cmd = args[0]
    
    assert str(video_file) in cmd
    assert str(output_file) in cmd
    # Check filter
    # boxblur=...enable='between(t,1.0,2.0)'
    assert any("enable='between(t,1.0,2.0)'" in arg for arg in cmd)
