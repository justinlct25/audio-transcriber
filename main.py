import os
from transcriber import transcribe_audio
from utils import get_untranscribed_files

# --- Configuration ---
# 1. AUDIO DIRECTORY
AUDIO_DIR = "audio"

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

# 5. OUTPUT DIRECTORY
OUTPUT_DIR = "output/transcript"

# 6. PAUSE THRESHOLD
# Minimum pause (in seconds) to create a new paragraph after a sentence ends
# Lower = more paragraphs, Higher = fewer, longer paragraphs
# Default: 1.5 seconds
PAUSE_THRESHOLD = 1.5


if __name__ == "__main__":
    print("="*50)
    print("Audio Transcriber")
    print("="*50)
    
    # Find untranscribed audio files
    untranscribed_files = get_untranscribed_files(AUDIO_DIR, OUTPUT_DIR)
    
    if not untranscribed_files:
        print("\n✓ All audio files have been transcribed!")
        print(f"Check {OUTPUT_DIR} for transcripts.")
        exit(0)
    
    print(f"\nFound {len(untranscribed_files)} audio file(s) to transcribe:")
    for i, audio_path in enumerate(untranscribed_files, 1):
        print(f"  {i}. {os.path.basename(audio_path)}")
    print()
    
    # Transcribe each file
    success_count = 0
    failed_files = []
    
    for i, audio_path in enumerate(untranscribed_files, 1):
        print("="*50)
        print(f"Transcribing file {i}/{len(untranscribed_files)}")
        print(f"File: {os.path.basename(audio_path)}")
        print("="*50)
        
        output_file = transcribe_audio(
            audio_path=audio_path,
            model_size=MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            output_dir=OUTPUT_DIR,
            pause_threshold=PAUSE_THRESHOLD
        )
        
        if output_file:
            success_count += 1
            print(f"\n✓ Transcript saved to: {output_file}\n")
        else:
            failed_files.append(audio_path)
            print(f"\n✗ Failed to transcribe: {audio_path}\n")
    
    # Summary
    print("="*50)
    print("Transcription Summary")
    print("="*50)
    print(f"Total files processed: {len(untranscribed_files)}")
    print(f"Successfully transcribed: {success_count}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {os.path.basename(f)}")
    
    print("\n✓ All done!")