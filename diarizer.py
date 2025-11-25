import os
import glob
from pyannote.audio import Pipeline
from tqdm import tqdm

# --- Configuration ---
# 1. HUGGING FACE TOKEN (required for speaker diarization)
# Get from https://huggingface.co/settings/tokens
# Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
HUGGINGFACE_TOKEN = "your_hf_token_here"

# 2. Find transcript and audio files
transcript_files = glob.glob("audio/*_transcript.txt")
audio_files = glob.glob("audio/*.mp3")


def parse_transcript(transcript_path):
    """Parse the transcript file and extract segments with timestamps."""
    segments = []
    
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            # Look for lines with timestamps like [0.00s -> 5.12s] Text here
            if line.startswith("[") and "s -> " in line and "s]" in line:
                try:
                    # Extract timestamp and text
                    timestamp_part = line[line.index("[")+1:line.index("]")]
                    start_str, end_str = timestamp_part.split(" -> ")
                    start = float(start_str.replace("s", ""))
                    end = float(end_str.replace("s", ""))
                    text = line[line.index("]")+1:].strip()
                    
                    segments.append({
                        'start': start,
                        'end': end,
                        'text': text
                    })
                except:
                    continue
    
    return segments


def assign_speaker_to_segment(segment_start, segment_end, diarization):
    """Find which speaker is talking during this segment."""
    max_overlap = 0
    assigned_speaker = "Unknown"
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Calculate overlap between segment and speaker turn
        overlap_start = max(segment_start, turn.start)
        overlap_end = min(segment_end, turn.end)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        if overlap_duration > max_overlap:
            max_overlap = overlap_duration
            assigned_speaker = speaker
    
    return assigned_speaker


def format_speaker_name(speaker_id):
    """Convert SPEAKER_00 to Speaker 1, SPEAKER_01 to Speaker 2, etc."""
    if speaker_id and speaker_id.startswith("SPEAKER_"):
        number = int(speaker_id.split("_")[1]) + 1
        return f"Speaker {number}"
    return speaker_id


def add_speakers_to_transcript(transcript_path, audio_path, hf_token):
    """Add speaker labels to an existing transcript."""
    
    if not os.path.exists(transcript_path):
        print(f"Error: Transcript file not found at '{transcript_path}'")
        return
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        return
    
    if not hf_token or hf_token == "your_hf_token_here":
        print("Error: Please set your Hugging Face token in HUGGINGFACE_TOKEN")
        print("Get one at: https://huggingface.co/settings/tokens")
        print("Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        return
    
    print(f"Reading transcript: {transcript_path}")
    segments = parse_transcript(transcript_path)
    print(f"Found {len(segments)} segments")
    
    # Load speaker diarization model
    try:
        print("Loading speaker diarization model...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        print("Performing speaker diarization (this may take a while)...")
        diarization = pipeline(audio_path)
        print("Speaker diarization complete!")
    except Exception as e:
        print(f"Error performing speaker diarization: {e}")
        return
    
    # Create output filename
    base_name = os.path.splitext(transcript_path)[0]
    if base_name.endswith("_transcript"):
        base_name = base_name[:-11]  # Remove "_transcript"
    output_filename = f"{base_name}_transcript_with_speakers.txt"
    
    print(f"Adding speakers to transcript...")
    
    # Process segments and add speakers
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Transcription with Speaker Diarization\n")
        f.write(f"Audio file: {audio_path}\n")
        f.write(f"Original transcript: {transcript_path}\n")
        f.write("="*50 + "\n\n")
        
        full_transcript = []
        
        for segment in tqdm(segments, desc="Assigning speakers"):
            # Assign speaker
            speaker = assign_speaker_to_segment(segment['start'], segment['end'], diarization)
            speaker_label = format_speaker_name(speaker)
            
            # Write segment with speaker
            line = f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {speaker_label}: {segment['text']}"
            f.write(line + "\n")
            full_transcript.append(segment['text'])
        
        # Write the full, clean text version at the end
        f.write("\n\n" + "="*50 + "\n\n")
        f.write("Full Plain Text Transcript:\n")
        f.write(" ".join(full_transcript).strip())
    
    print("-" * 50)
    print("Speaker Assignment Complete!")
    print(f"Output saved to: {output_filename}")
    print("-" * 50)


if __name__ == "__main__":
    if not transcript_files:
        print("Error: No transcript files found in audio/ directory")
        print("Please run main.py first to generate a transcript")
    elif not audio_files:
        print("Error: No audio files found in audio/ directory")
    else:
        # Use the first transcript and audio file found
        transcript_path = transcript_files[0]
        audio_path = audio_files[0]
        
        print(f"Using transcript: {transcript_path}")
        print(f"Using audio: {audio_path}")
        
        add_speakers_to_transcript(transcript_path, audio_path, HUGGINGFACE_TOKEN)