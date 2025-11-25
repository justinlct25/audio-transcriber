import os
import time
from faster_whisper import WhisperModel
from tqdm import tqdm
import re


def create_output_file(audio_path, output_dir):
    """Create and return the output transcript file path."""
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{base_name}_transcript.txt")
    else:
        audio_dir = os.path.dirname(audio_path)
        output_filename = os.path.join(audio_dir, f"{base_name}_transcript.txt")
    
    # Create the file with a header
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Audio file: {audio_path}\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
    
    print(f"Output file created: {output_filename}")
    return output_filename


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


def ends_with_sentence_terminator(text):
    """Check if text ends with sentence-ending punctuation."""
    # Strip whitespace and check last character
    text = text.strip()
    return len(text) > 0 and text[-1] in '.!?'


def check_should_end_paragraph(current_segment, previous_segment, pause_threshold):
    """Determine if the current paragraph should end based on sentence terminator and pause."""
    # Check for sentence ending
    if ends_with_sentence_terminator(current_segment['text']):
        return True
    
    # Check for pause threshold
    if previous_segment is not None:
        pause = current_segment['start'] - previous_segment['end']
        if pause >= pause_threshold:
            return True
    
    return False


def remove_in_progress_marker(filename):
    """Remove the 'Transcription in progress...' marker from the end of the file."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove the progress marker if it exists
    if content.endswith("Transcription in progress...\n"):
        content = content[:-len("Transcription in progress...\n")]
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)


def write_paragraph(paragraph_segments, filename):
    """Write a paragraph and add the 'in progress' marker."""
    if not paragraph_segments:
        return
    
    # Remove old progress marker
    remove_in_progress_marker(filename)
    
    paragraph_start = paragraph_segments[0]['start']
    paragraph_end = paragraph_segments[-1]['end']
    paragraph_text = " ".join(seg['text'] for seg in paragraph_segments)
    
    # Write paragraph and add progress marker
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{paragraph_start:.2f}s -> {paragraph_end:.2f}s] {paragraph_text}\n\n")
        f.write("Transcription in progress...\n")
        f.flush()


def transcribe_audio(audio_path, model_size="medium.en", device="cpu", compute_type="int8", 
                     output_dir=None, pause_threshold=1.5):
    """
    Transcribe audio file to text grouped into natural paragraphs/sessions.
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Whisper model size (large-v3, medium.en, base.en, etc.)
        device (str): Device to use (cuda or cpu)
        compute_type (str): Compute type (float16, int8, etc.)
        output_dir (str): Optional output directory. If None, uses same dir as audio file
        pause_threshold (float): Minimum pause (seconds) between sentences to start new paragraph
    
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
    
    # Create output file
    output_filename = create_output_file(audio_path, output_dir)
    
    # Try to get audio duration for progress bar
    duration = get_audio_duration(audio_path)
    
    start_time = time.time()

    # Transcribe the audio
    segments_iter, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
    )

    # Process segments with progress bar and write in real-time
    all_segments = []
    current_paragraph = []
    paragraph_count = 0
    previous_segment = None
    
    # Create progress bar
    if duration:
        pbar = tqdm(total=duration, desc="Transcribing", unit="s", 
                   bar_format='{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]')
    else:
        pbar = tqdm(desc="Transcribing", unit=" segments")
    
    for segment in segments_iter:
        segment_data = {
            'start': segment.start,
            'end': segment.end,
            'text': segment.text.strip()
        }
        all_segments.append(segment_data)
        
        # Add to current paragraph
        current_paragraph.append(segment_data)
        
        # Check if we should end the current paragraph
        should_end_paragraph = check_should_end_paragraph(segment_data, previous_segment, pause_threshold)
        
        previous_segment = segment_data
        
        # Update progress bar
        if duration:
            pbar.n = segment.end
            pbar.refresh()
        else:
            pbar.update(1)
        
        # Write paragraph if it's complete
        if should_end_paragraph and current_paragraph:
            write_paragraph(current_paragraph, output_filename)
            paragraph_count += 1
            current_paragraph = []  # Start new empty paragraph
    
    pbar.close()
    
    # Write the last paragraph if it exists
    if current_paragraph:
        remove_in_progress_marker(output_filename)
        paragraph_start = current_paragraph[0]['start']
        paragraph_end = current_paragraph[-1]['end']
        paragraph_text = " ".join(seg['text'] for seg in current_paragraph)
        
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"[{paragraph_start:.2f}s -> {paragraph_end:.2f}s] {paragraph_text}\n\n")
        
        paragraph_count += 1
    else:
        # Remove progress marker if no final paragraph
        remove_in_progress_marker(output_filename)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("-" * 50)
    print("Transcription Complete!")
    print(f"Detected Language: {info.language} (Probability: {info.language_probability:.2f})")
    print(f"Total segments: {len(all_segments)}")
    print(f"Total paragraphs: {paragraph_count}")
    print(f"Total time taken: {elapsed_time:.2f} seconds.")
    print(f"Transcript saved to: {output_filename}")
    print("-" * 50)
    
    return output_filename