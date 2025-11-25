import unittest
import os
import tempfile
import shutil
from pathlib import Path

# Get repo root
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path to import transcriber
import sys
sys.path.insert(0, repo_root)

from transcriber import transcribe_audio, get_audio_duration


class TestTranscriber(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        # Find the test audio file
        test_audio_files = list(Path(repo_root).glob("test/test_2min_audio_*.mp3"))
        if not test_audio_files:
            raise FileNotFoundError("No test audio file found. Run create_test_audio.py first.")
        
        cls.test_audio_path = str(test_audio_files[0])
        cls.temp_dir = tempfile.mkdtemp()
        print(f"\nUsing test audio: {cls.test_audio_path}")
        print(f"Temp directory: {cls.temp_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
            print(f"\nCleaned up temp directory: {cls.temp_dir}")
    
    def test_get_audio_duration(self):
        """Test that audio duration can be retrieved."""
        duration = get_audio_duration(self.test_audio_path)
        
        self.assertIsNotNone(duration, "Duration should not be None")
        self.assertIsInstance(duration, float, "Duration should be a float")
        self.assertGreater(duration, 0, "Duration should be positive")
        self.assertLessEqual(duration, 125, "Duration should be ~120 seconds (2 min) with some tolerance")
        print(f"Audio duration: {duration:.2f} seconds")
    
    def test_transcribe_audio_basic(self):
        """Test basic transcription with default settings."""
        output_file = transcribe_audio(
            audio_path=self.test_audio_path,
            model_size="base.en",  # Use smallest model for faster testing
            device="cpu",
            compute_type="int8",
            output_dir=self.temp_dir
        )
        
        self.assertIsNotNone(output_file, "Output file should not be None")
        self.assertTrue(os.path.exists(output_file), "Output file should exist")
        
        # Check file content
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("Transcription", content, "File should contain transcription header")
        self.assertIn("Audio file:", content, "File should contain audio file path")
        self.assertIn("Full Plain Text Transcript:", content, "File should contain plain text section")
        self.assertTrue(len(content) > 100, "Transcript should have substantial content")
        
        print(f"Transcript saved to: {output_file}")
        print(f"Transcript length: {len(content)} characters")