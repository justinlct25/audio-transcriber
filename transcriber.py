import os
import time
from faster_whisper import WhisperModel
from tqdm import tqdm


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


def transcribe_audio(audio_path, model_size="medium.en", device="cpu", compute_type="int8", output_dir=None):
    """
    Transcribe audio file to text with timestamps.
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Whisper model size (large-v3, medium.en, base.en, etc.)
        device (str): Device to use (cuda or cpu)
        compute_type (str): Compute type (float16, int8, etc.)
        output_dir (str): Optional output directory. If None, uses same dir as audio file
    
    Returns:
        str: Path to the output transcript file, or None if failed
    """
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        return None

    print(f"Device: {device.upper()} | Compute Type: {compute_type}")
    print(f"Loading Whisper model '{model_size}'...")
    
    # Load the model
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have the required dependencies installed correctly.")
        return None

    print("Model loaded successfully. Starting transcription...")
    
    # Prepare output file name
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{base_name}_transcript.txt")
    else:
        audio_dir = os.path.dirname(audio_path)
        output_filename = os.path.join(audio_dir, f"{base_name}_transcript.txt")
    
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

    # Transcribe the audio
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
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
    
    return output_filename