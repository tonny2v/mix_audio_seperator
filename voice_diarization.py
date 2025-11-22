#!/usr/bin/env python3
"""
Voice Diarization Script
References test_audio.py and t.py for audio processing and speaker diarization.
Includes ffmpeg conversion for unsupported audio formats.
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
import logging

# Import from existing modules
try:
    import torchaudio
    import torch
    from pyannote.audio import Pipeline
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install torchaudio torch pyannote.audio")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported audio formats by torchaudio/pyannote
SUPPORTED_FORMATS = ['.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac']

def check_ffmpeg():
    """Check if ffmpeg is available on the system"""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def convert_audio_format(input_path, output_path, target_format='wav'):
    """
    Convert audio file to target format using ffmpeg

    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save converted audio
        target_format (str): Target audio format (default: wav)

    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',  # Standard WAV codec
            '-ar', '16000',          # 16kHz sample rate (good for diarization)
            '-ac', '1',              # Mono audio
            '-y',                    # Overwrite output file
            output_path
        ]

        logger.info(f"Converting {input_path} to {target_format} format...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info(f"Successfully converted to {output_path}")
            return True
        else:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timed out")
        return False
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        return False

def validate_audio_file(file_path, check_pyannote=False):
    """
    Validate if audio file can be loaded by torchaudio and optionally pyannote

    Args:
        file_path (str): Path to audio file
        check_pyannote (bool): Whether to check pyannote compatibility

    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        logger.info(f"Audio validation successful: {sample_rate}Hz, {waveform.shape[0]} channels")

        # Additional pyannote-specific checks
        if check_pyannote:
            file_ext = Path(file_path).suffix.lower()

            # pyannote has issues with certain formats, recommend conversion
            problematic_formats = ['.m4a', '.aac', '.wma', '.mp4']
            if file_ext in problematic_formats:
                logger.warning(f"Format {file_ext} may have compatibility issues with pyannote")
                return False

            # Check for stereo vs mono - pyannote generally prefers mono
            if waveform.shape[0] > 1:
                logger.info(f"Stereo audio detected ({waveform.shape[0]} channels), conversion to mono recommended")

            # Check sample rate - pyannote works best with 16kHz
            if sample_rate != 16000:
                logger.info(f"Sample rate {sample_rate}Hz detected, 16kHz recommended for optimal performance")

        return True
    except Exception as e:
        logger.warning(f"Audio validation failed: {e}")
        return False

def prepare_audio_file(input_path, temp_dir=None):
    """
    Prepare audio file for diarization - convert if necessary

    Args:
        input_path (str): Path to input audio file
        temp_dir (str): Temporary directory for converted files

    Returns:
        str: Path to prepared audio file, None if failed
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file does not exist: {input_path}")
        return None

    file_ext = Path(input_path).suffix.lower()

    # Check file with pyannote compatibility requirements
    if file_ext in SUPPORTED_FORMATS and validate_audio_file(input_path, check_pyannote=True):
        logger.info(f"Using original file: {input_path}")
        return input_path

    # If format not supported or validation failed, convert with ffmpeg
    if not check_ffmpeg():
        logger.error("FFmpeg not available. Cannot convert unsupported audio format.")
        return None

    # Create temporary file for conversion
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    base_name = Path(input_path).stem
    converted_path = os.path.join(temp_dir, f"{base_name}_converted.wav")

    logger.info(f"Converting {file_ext} to WAV format for pyannote compatibility...")
    if convert_audio_format(input_path, converted_path):
        if validate_audio_file(converted_path):
            return converted_path
        else:
            logger.error("Converted audio file is still invalid")
            return None
    else:
        logger.error("Failed to convert audio file")
        return None

def initialize_diarization_pipeline(auth_token=None):
    """
    Initialize the pyannote speaker diarization pipeline

    Args:
        auth_token (str): Hugging Face authentication token

    Returns:
        Pipeline: Initialized diarization pipeline or None if failed
    """
    try:
        # Set torch.backends to avoid TF32 warnings (from t.py)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Initialize pipeline
        if auth_token:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            )
        else:
            # Try without auth token first
            try:
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            except Exception:
                logger.error("Authentication token required for pyannote/speaker-diarization-3.1")
                logger.info("Please provide --auth-token or set HF_TOKEN environment variable")
                return None

        pipeline.to(device)
        logger.info("Diarization pipeline initialized successfully")
        return pipeline

    except Exception as e:
        logger.error(f"Failed to initialize diarization pipeline: {e}")
        return None

def perform_diarization(pipeline, audio_path, output_dir=None):
    """
    Perform speaker diarization on audio file

    Args:
        pipeline: Initialized diarization pipeline
        audio_path (str): Path to audio file
        output_dir (str): Directory to save results

    Returns:
        dict: Diarization results or None if failed
    """
    try:
        logger.info(f"Processing audio file: {audio_path}")
        output = pipeline(audio_path)

        # Collect results
        results = []
        for turn, _, speaker in output.itertracks(yield_label=True):
            results.append({
                'start': turn.start,
                'end': turn.end,
                'duration': turn.end - turn.start,
                'speaker': speaker
            })
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

        # Save results to disk
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save RTTM format
        rttm_path = os.path.join(output_dir or '.', "audio.rttm")
        with open(rttm_path, "w") as rttm:
            output.write_rttm(rttm)
        logger.info(f"Results saved to {rttm_path}")

        # Save detailed results as JSON
        import json
        json_path = os.path.join(output_dir or '.', "diarization_results.json")
        with open(json_path, "w") as f:
            json.dump({
                'audio_file': audio_path,
                'total_speakers': len(set(r['speaker'] for r in results)),
                'segments': results
            }, f, indent=2)
        logger.info(f"Detailed results saved to {json_path}")

        return {
            'segments': results,
            'rttm_file': rttm_path,
            'json_file': json_path,
            'total_speakers': len(set(r['speaker'] for r in results))
        }

    except Exception as e:
        logger.error(f"Error during diarization: {e}")
        return None

def test_separated_audio_files(audio_dir="separated_speakers"):
    """
    Test separated audio files using functionality from test_audio.py

    Args:
        audio_dir (str): Directory containing separated audio files
    """
    if not os.path.exists(audio_dir):
        logger.warning(f"Directory '{audio_dir}' does not exist")
        return

    # Import and use test_audio functionality
    sys.path.append('.')
    try:
        import test_audio

        logger.info(f"Testing separated audio files in {audio_dir}")
        test_audio.test_audio_files(audio_dir)
    except ImportError as e:
        logger.error(f"Could not import test_audio module: {e}")
        # Fallback implementation
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        if not audio_files:
            logger.info(f"No .wav files found in {audio_dir}")
            return

        logger.info(f"Found {len(audio_files)} audio files:")
        for audio_file in audio_files:
            file_path = os.path.join(audio_dir, audio_file)
            if test_audio_file(file_path):
                print(f"✓ {audio_file}")
            else:
                print(f"✗ {audio_file}")

def test_audio_file(file_path):
    """Test a single audio file (from test_audio.py)"""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        file_size = os.path.getsize(file_path)
        duration = waveform.shape[1] / sample_rate

        logger.info(f"✓ {os.path.basename(file_path)}")
        logger.info(f"  - Sample rate: {sample_rate} Hz")
        logger.info(f"  - Duration: {duration:.2f} seconds")
        logger.info(f"  - Channels: {waveform.shape[0]}")
        logger.info(f"  - File size: {file_size:,} bytes")

        return True
    except Exception as e:
        logger.error(f"✗ Error loading {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Voice Diarization Script with Audio Conversion Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python voice_diarization.py input.wav
  python voice_diarization.py input.mp4 --auth-token hf_xxxx
  python voice_diarization.py input.flac --output-dir results
  python voice_diarization.py --test-separated separated_speakers/
        """
    )

    parser.add_argument("input", nargs="?",
                       help="Input audio/video file path")
    parser.add_argument("--auth-token",
                       help="Hugging Face authentication token")
    parser.add_argument("--output-dir", "-o",
                       help="Output directory for results")
    parser.add_argument("--temp-dir",
                       help="Temporary directory for conversions")
    parser.add_argument("--test-separated",
                       help="Test separated audio files (path to directory)")
    parser.add_argument("--keep-temp", action="store_true",
                       help="Keep temporary files")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get auth token from argument or environment
    auth_token = args.auth_token or os.getenv('HF_TOKEN')

    # Handle testing separated audio files
    if args.test_separated:
        test_separated_audio_files(args.test_separated)
        return

    # Require input file for diarization
    if not args.input:
        parser.error("Input audio file is required")

    # Initialize diarization pipeline
    logger.info("Initializing diarization pipeline...")
    pipeline = initialize_diarization_pipeline(auth_token)
    if not pipeline:
        logger.error("Failed to initialize diarization pipeline")
        sys.exit(1)

    # Prepare audio file (convert if necessary)
    temp_dir = None
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    prepared_audio = prepare_audio_file(args.input, temp_dir)
    if not prepared_audio:
        logger.error("Failed to prepare audio file")
        sys.exit(1)

    try:
        # Perform diarization
        logger.info("Starting speaker diarization...")
        results = perform_diarization(pipeline, prepared_audio, args.output_dir)

        if results:
            logger.info(f"Diarization completed successfully!")
            logger.info(f"Total speakers detected: {results['total_speakers']}")
            logger.info(f"Total segments: {len(results['segments'])}")
            logger.info(f"Results saved to: {args.output_dir or 'current directory'}")
        else:
            logger.error("Diarization failed")
            sys.exit(1)

    finally:
        # Clean up temporary files unless --keep-temp specified
        if not args.keep_temp and temp_dir and temp_dir != args.temp_dir:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()