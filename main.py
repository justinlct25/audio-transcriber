import glob
import os
from transcriber import transcribe_audio

# --- Configuration ---
# 1. AUDIO FILE PATH
# Find the first MP3 file in the audio directory
audio_files = glob.glob("audio/*.mp3")
if audio_files:
    AUDIO_FILE_PATH = audio_files[0]
    print(f"Found audio file: {AUDIO_FILE_PATH}")
else:
    print("No MP3 files found in audio/ directory")
    print("Please add an audio file to the audio/ directory")
    exit(1)

# 2. MODEL SIZE
#    - "large-v3": Most accurate, uses the most memory (best for 8GB+ VRAM/RAM)
#    - "medium.en": Great balance of speed/accuracy for English-only audio
#    - "base.en": Fastest, lowest memory usage, but less accurate
MODEL_SIZE = "medium.en"

# 3. DEVICE
#    - "cuda": Use this if you have an NVIDIA GPU (requires CUDA installation)
#    - "cpu": Use this if you do NOT have a dedicated GPU
DEVICE = "cpu"

# 4. COMPUTE TYPE
#    - For GPU (cuda): "float16" (standard) or "int8" (less VRAM, slightly faster)
#    - For CPU (cpu): "int8" is highly recommended for speed and memory efficiency
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# 5. OUTPUT DIRECTORY (optional)
# Set to None to save in the same directory as the audio file
# Or specify a path like "output/transcript"
OUTPUT_DIR = "output/transcript"


if __name__ == "__main__":
    print("="*50)
    print("Audio Transcriber")
    print("="*50)
    
    # Transcribe the audio file
    output_file = transcribe_audio(
        audio_path=AUDIO_FILE_PATH,
        model_size=MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        output_dir=OUTPUT_DIR
    )
    
    if output_file:
        print(f"\n✓ Success! Transcript saved to: {output_file}")
    else:
        print("\n✗ Transcription failed")