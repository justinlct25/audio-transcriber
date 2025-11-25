import glob
import os


def get_untranscribed_files(audio_dir, output_dir):
    """
    Find audio files that haven't been transcribed yet.
    
    Args:
        audio_dir (str): Directory containing audio files
        output_dir (str): Directory containing transcript files
    
    Returns:
        List of audio file paths that need transcription
    """
    # Get all audio files
    audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    
    if not audio_files:
        return []
    
    # Check which ones already have transcripts
    untranscribed = []
    for audio_path in audio_files:
        # Get the expected transcript filename
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
        
        # If transcript doesn't exist, add to list
        if not os.path.exists(transcript_path):
            untranscribed.append(audio_path)
    
    return untranscribed