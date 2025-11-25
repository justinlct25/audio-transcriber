import os
import time
from faster_whisper import WhisperModel
from tqdm import tqdm

import glob

# --- Configuration ---
# 1. SET YOUR AUDIO FILE PATH:
# Find the first MP3 file in the audio directory
audio_files = glob.glob("audio/*.mp3")
if audio_files:
    AUDIO_FILE_PATH = audio_files[0]
    print(f"Found audio file: {AUDIO_FILE_PATH}")
else:
    AUDIO_FILE_PATH = "audio/your_file.mp3"

# 2. CHOOSE YOUR MODEL:
#    - "large-v3": Most accurate, uses the most memory (best for 8GB+ VRAM/RAM).
#    - "medium.en": Great balance of speed/accuracy for English-only audio.
#    - "base.en": Fastest, lowest memory usage, but less accurate.
MODEL_SIZE = "medium.en"

# 3. CONFIGURE HARDWARE:
#    - device="cuda": Use this if you have an NVIDIA GPU (requires CUDA installation).
#    - device="cpu": Use this if you do NOT have a dedicated GPU.
DEVICE = "cpu"  # Changed to CPU for Mac

# 4. CONFIGURE COMPUTE TYPE (for speed/memory trade-off):
#    - For GPU (cuda): "float16" (standard) or "int8" (less VRAM, slightly faster).
#    - For CPU (cpu): "int8" is highly recommended for speed and memory efficiency.
COMPUTE_TYPE = "float16" 

# Adjustments for CPU only:
if DEVICE == "cpu":
    COMPUTE_TYPE = "int8"


def get_audio_duration(audio_path):
    """Get audio duration in seconds (approximate from file info)."""
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        return float(result.stdout.strip())
    except:
        # If ffprobe is not available, return None
        return None


def transcribe_audio(audio_path, model_size, device, compute_type):
    """Loads the model and performs the transcription with progress bar."""
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'. Please check the AUDIO_FILE_PATH.")
        return

    print(f"Device: {device.upper()} | Compute Type: {compute_type}")
    print(f"Loading Whisper model '{model_size}'...")
    
    # Load the model. It downloads automatically on first run.
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have the required dependencies (like PyTorch and CUDA for GPU) installed correctly.")
        return

    print("Model loaded successfully. Starting transcription...")
    
    # Prepare output file name and create it immediately
    base_name = os.path.splitext(audio_path)[0]
    output_filename = f"{base_name}_transcript.txt"
    
    # Create the file with a header
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Transcription in progress...\n")
        f.write(f"Audio file: {audio_path}\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
    
    print(f"Output file created: {output_filename}")
    
    # Try to get audio duration for progress bar
    duration = get_audio_duration(audio_path)
    
    start_time = time.time()

    # The transcribe method automatically handles long audio by chunking it.
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,             # Controls search width for better accuracy
        vad_filter=True          # Recommended: removes silence/noise chunks
    )

    # Process segments with progress bar and write in real-time
    full_transcript = []
    
    # Create progress bar
    if duration:
        pbar = tqdm(total=duration, desc="Transcribing", unit="s", 
                   bar_format='{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]')
    else:
        pbar = tqdm(desc="Transcribing", unit=" segments")
    
    # Open file in append mode to keep writing as we transcribe
    with open(output_filename, "a", encoding="utf-8") as f:
        for segment in segments:
            # Write segment with timestamps immediately
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}"
            f.write(line + "\n")
            f.flush()  # Force write to disk immediately
            full_transcript.append(segment.text)
            
            # Update progress bar
            if duration:
                pbar.n = segment.end
                pbar.refresh()
            else:
                pbar.update(1)
        
        pbar.close()
        
        # Write the full, clean text version at the end of the file
        f.write("\n\n" + "="*50 + "\n\n")
        f.write("Full Plain Text Transcript:\n")
        f.write("".join(full_transcript).strip())

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("-" * 50)
    print("Transcription Complete!")
    print(f"Detected Language: {info.language} (Probability: {info.language_probability:.2f})")
    print(f"Total time taken: {elapsed_time:.2f} seconds.")
    print(f"Transcript saved to: {output_filename}")
    print("-" * 50)


if __name__ == "__main__":
    transcribe_audio(AUDIO_FILE_PATH, MODEL_SIZE, DEVICE, COMPUTE_TYPE)