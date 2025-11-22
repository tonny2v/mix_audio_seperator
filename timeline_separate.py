#!/usr/bin/env python3
"""
Combined Timeline Voice Separation Script
Runs both diarization and timeline-preserving separation in one command.
"""

import os
import sys
import argparse
import tempfile
import shutil
import logging
from pathlib import Path

# Import existing modules
try:
    from voice_diarization import (
        initialize_diarization_pipeline,
        prepare_audio_file,
        perform_diarization,
        validate_audio_file
    )
    from separate_speakers import (
        load_diarization_results,
        separate_speakers_with_timeline,
        apply_audio_enhancement
    )
    import torchaudio
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_combined_timeline_separation(input_file, output_dir=None, auth_token=None,
                                   min_segment_duration=1.0, audio_format='wav',
                                   enhance_audio=True, cleanup_temp=True):
    """
    Run both diarization and timeline separation in one go

    Args:
        input_file (str): Path to input audio file
        output_dir (str): Output directory for separated files
        auth_token (str): Hugging Face authentication token
        min_segment_duration (float): Minimum duration for speech segments
        audio_format (str): Output audio format
        enhance_audio (bool): Whether to apply audio enhancement
        cleanup_temp (bool): Whether to cleanup temporary files

    Returns:
        dict: Results and output file paths
    """

    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not validate_audio_file(input_file):
        raise ValueError(f"Invalid audio file: {input_file}")

    # Setup directories
    if output_dir is None:
        base_name = Path(input_file).stem
        output_dir = f"{base_name}_timeline_separated"

    os.makedirs(output_dir, exist_ok=True)

    # Create temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix="timeline_separation_")

    try:
        logger.info("=" * 60)
        logger.info("COMBINED TIMELINE VOICE SEPARATION")
        logger.info("=" * 60)

        # Step 1: Diarization
        logger.info("Step 1: Running speaker diarization...")

        # Initialize pipeline
        logger.info("Initializing diarization pipeline...")
        pipeline = initialize_diarization_pipeline(auth_token)
        if not pipeline:
            raise RuntimeError("Failed to initialize diarization pipeline")

        # Prepare audio file
        logger.info("Preparing audio file...")
        prepared_audio = prepare_audio_file(input_file, temp_dir)
        if not prepared_audio:
            raise RuntimeError("Failed to prepare audio file")

        # Perform diarization
        logger.info("Performing speaker diarization...")
        diarization_results = perform_diarization(pipeline, prepared_audio, temp_dir)

        if not diarization_results:
            raise RuntimeError("Diarization failed")

        logger.info(f"Diarization completed: {diarization_results['total_speakers']} speakers found")
        logger.info(f"Total speech segments: {len(diarization_results['segments'])}")

        # Step 2: Timeline Separation
        logger.info("\nStep 2: Creating timeline-separated audio files...")

        # Load segments from JSON
        json_path = diarization_results['json_file']
        segments = load_diarization_results(json_path=json_path)

        if not segments:
            raise RuntimeError("Failed to load diarization results")

        # Perform timeline separation
        output_files = separate_speakers_with_timeline(
            input_file, segments, output_dir,
            min_segment_duration=min_segment_duration,
            audio_format=audio_format
        )

        if not output_files:
            raise RuntimeError("Timeline separation failed")

        # Step 3: Audio Enhancement (optional)
        # Note: Enhancement is built into the separation function

        # Create summary
        summary = {
            'input_file': input_file,
            'output_directory': output_dir,
            'total_speakers': diarization_results['total_speakers'],
            'total_segments': len(diarization_results['segments']),
            'output_files': output_files,
            'separation_method': 'timeline_preserving',
            'audio_enhancement': enhance_audio
        }

        # Save summary
        import json
        summary_path = os.path.join(output_dir, "timeline_separation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("SEPARATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Speakers separated: {diarization_results['total_speakers']}")
        # Count total files across all speakers
        total_files = sum(len(files) for files in output_files.values())
        logger.info(f"Timeline-aligned files created: {total_files}")

        # Audio enhancement note is already logged above

        logger.info(f"\nTimeline-separated files:")
        for speaker, files in output_files.items():
            for output_file in files:
                logger.info(f"  - {os.path.basename(output_file)}")

        logger.info(f"\nSummary saved to: {summary_path}")

        return summary

    finally:
        # Cleanup temporary files
        if cleanup_temp and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary files: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(
        description="Combined Timeline Voice Separation - Diarization + Timeline Separation in One Command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with timeline preservation
  python timeline_separate.py input.wav

  # With custom output directory and audio enhancement
  python timeline_separate.py input.wav --output-dir separated --enhance

  # With authentication token
  python timeline_separate.py input.wav --auth-token hf_xxxx

  # Custom audio format
  python timeline_separate.py input.wav --format flac
        """
    )

    parser.add_argument("input",
                       help="Input audio file path")
    parser.add_argument("--output-dir", "-o",
                       help="Output directory for separated files (default: auto-generated)")
    parser.add_argument("--auth-token",
                       help="Hugging Face authentication token")
    parser.add_argument("--format", "-f", default="wav", choices=["wav", "flac", "mp3"],
                       help="Output audio format (default: wav)")
    parser.add_argument("--min-duration", type=float, default=1.0,
                       help="Minimum duration for speech segments in seconds (default: 1.0)")
    parser.add_argument("--no-enhance", action="store_true",
                       help="Skip audio enhancement")
    parser.add_argument("--keep-temp", action="store_true",
                       help="Keep temporary files")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get auth token from argument or environment
    auth_token = args.auth_token or os.getenv('HF_TOKEN')

    try:
        results = run_combined_timeline_separation(
            input_file=args.input,
            output_dir=args.output_dir,
            auth_token=auth_token,
            min_segment_duration=args.min_duration,
            audio_format=args.format,
            enhance_audio=not args.no_enhance,
            cleanup_temp=not args.keep_temp
        )

        print(f"\n✅ Timeline separation completed successfully!")
        print(f"📁 Output: {results['output_directory']}")
        total_files = sum(len(files) for files in results['output_files'].values())
        print(f"🎵 Files created: {total_files}")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()