import subprocess
import glob
import os

# Get the repo root (parent directory of the test folder)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Find the first MP3 file in the audio directory
audio_files = glob.glob(os.path.join(repo_root, "audio", "*.mp3"))
if not audio_files:
    print("No MP3 files found in audio/ directory")
    exit(1)

input_file = audio_files[0]
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = os.path.join(repo_root, "test", f"test_2min_audio_{base_name}.mp3")

print(f"Creating 2-minute test file from: {input_file}")
print(f"Output: {output_file}")

# Extract first 2 minutes (120 seconds) using ffmpeg
try:
    subprocess.run([
        "ffmpeg",
        "-i", input_file,
        "-t", "120",  # Duration in seconds
        "-acodec", "copy",  # Copy audio codec without re-encoding
        output_file
    ], check=True)
    print(f"\n✓ Success! Test audio saved to: {output_file}")
except subprocess.CalledProcessError as e:
    print(f"\n✗ Error: {e}")
    print("Make sure ffmpeg is installed. Install with: brew install ffmpeg")
except FileNotFoundError:
    print("\n✗ Error: ffmpeg not found")
    print("Install ffmpeg with: brew install ffmpeg")