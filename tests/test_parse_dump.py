import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio_scripts')))

from parse_dump import format_time, parse_dump_file

def test_format_time():
    assert format_time(3600) == "01:00:00"

@patch('parse_dump.console')
def test_parse_dump_file(mock_console, tmp_path):
    # Create dummy dump file
    dump_file = tmp_path / "dump.txt"
    dump_file.write_text(
        "Full Transcription\n\n"
        "[00:00:00] (Start: 0.00, End: 1.00) hello\n"
        "[00:00:01] (Start: 1.00, End: 2.00) badword\n"
        "[00:00:02] (Start: 2.00, End: 3.00) world\n"
    )
    
    words_file = tmp_path / "banned.txt"
    words_file.write_text("badword")
    
    output_csv = tmp_path / "review.csv"
    
    parse_dump_file(dump_file, words_file, output_csv)
    
    assert output_csv.exists()
    content = output_csv.read_text()
    assert "badword" in content
    # Context should be: hello badword world (roughly)
    # The script grabs surrounding words.
    assert "hello badword world" in content
