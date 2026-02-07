import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio_scripts')))

from analyze_video import format_time, analyze_video

def test_format_time():
    assert format_time(0) == "00:00:00"

@patch('analyze_video.cv2')
@patch('analyze_video.NudeDetector')
@patch('analyze_video.console')
def test_analyze_video(mock_console, mock_detector_cls, mock_cv2, tmp_path):
    # Mock NudeDetector instance
    mock_detector = MagicMock()
    mock_detector_cls.return_value = mock_detector
    
    # Mock VideoCapture
    mock_cap = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap
    
    # Setup video props
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == mock_cv2.CAP_PROP_FPS else 60.0 # 2 seconds, 60 frames
    
    # Mock reading frames
    # We need to simulate a few frames.
    # The loop runs while cap.isOpened(), checking ret.
    # We can use side_effect for read().
    # frame 0: no nudity
    # frame 30: nudity
    # frame 60: stop
    
    # Logic in script: `while cap.isOpened(): ... ret, frame = cap.read()`
    # We need to control the loop.
    # The script updates `current_frame` manually via `cap.set` if using that logic?
    # Ah, let's check script logic.
    # script: 
    # while cap.isOpened():
    #   cap.set(..., current_frame)
    #   ret, frame = cap.read()
    #   current_frame += frame_step
    
    # So we need mock_cap.read() to return (True, frame) then eventually (False, None).
    # But current_frame increments by frame_step (30).
    # So: read(0) -> True, read(30) -> True, read(60) -> False (since total is 60, aka 0-59. 60 is end)
    
    mock_frame = MagicMock()
    mock_cv2.cvtColor.return_value = mock_frame # Dummy RGB frame
    
    mock_cap.read.side_effect = [(True, mock_frame), (True, mock_frame), (False, None)]
    
    # Mock detections
    # calls detector.detect(rgb_frame)
    # 1st call: empty
    # 2nd call: found nudity
    
    mock_detector.detect.side_effect = [
        [], 
        [{'class': 'BUTTS', 'score': 0.8, 'box': [0,0,10,10]}]
    ]
    
    video_file = tmp_path / "video.mp4"
    video_file.touch()
    
    output_csv = tmp_path / "review.csv"
    
    analyze_video(video_file, output_csv, frame_interval=1.0)
    
    assert output_csv.exists()
    content = output_csv.read_text()
    # Timestamp should be approx 1.0s (frame 30 / fps 30)
    # Range writing logic: merging. start=1.0, end=1.0. 
    # Writer adds buffer: start-0.5, end+0.5 -> 0.5 to 1.5
    assert "0.50" in content
    assert "1.50" in content
    assert "BUTTS" in content
