import torchaudio
import os
import argparse

def test_audio_file(file_path):
    """Test a single audio file"""
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)

        # Get file info
        file_size = os.path.getsize(file_path)
        duration = waveform.shape[1] / sample_rate

        print(f"✓ {os.path.basename(file_path)}")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Duration: {duration:.2f} seconds")
        print(f"  - Channels: {waveform.shape[0]}")
        print(f"  - File size: {file_size:,} bytes")
        print()

        return True

    except Exception as e:
        print(f"✗ Error loading {file_path}: {e}")
        print()
        return False

def test_audio_files(directory="separated_speakers"):
    """Test that the separated audio files can be loaded and played"""

    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

    if not audio_files:
        print(f"No .wav files found in directory '{directory}'")
        return

    print("Testing separated audio files:")
    print("=" * 40)

    for audio_file in audio_files:
        file_path = os.path.join(directory, audio_file)
        test_audio_file(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test audio files")
    parser.add_argument("path", nargs="?", help="Path to audio file or directory (default: separated_speakers directory)")

    args = parser.parse_args()

    if args.path:
        if os.path.isfile(args.path):
            print("Testing single audio file:")
            print("=" * 40)
            test_audio_file(args.path)
        elif os.path.isdir(args.path):
            test_audio_files(args.path)
        else:
            print(f"Error: Path '{args.path}' does not exist.")
    else:
        test_audio_files()