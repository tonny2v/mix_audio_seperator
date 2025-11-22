#!/usr/bin/env python3
"""
Speaker Separation Script
Separates individual speakers from mixed audio files based on diarization results.
Works in conjunction with voice_diarization.py to extract individual speaker segments.
"""

import os
import sys
import argparse
import json
import tempfile
import shutil
import logging
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple

# Audio processing imports
try:
    import torchaudio
    import torch
    import soundfile as sf
    from pyannote.audio import Pipeline
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install torchaudio torch soundfile pyannote.audio")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_diarization_results(rttm_path: str = None, json_path: str = None) -> Optional[List[Dict]]:
    """
    Load diarization results from RTTM or JSON file

    Args:
        rttm_path (str): Path to RTTM file
        json_path (str): Path to JSON file

    Returns:
        List of diarization segments or None if failed
    """
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data.get('segments', [])
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")

    if rttm_path and os.path.exists(rttm_path):
        try:
            # Parse RTTM format
            segments = []
            with open(rttm_path, 'r') as f:
                for line in f:
                    if line.startswith('SPEAKER'):
                        parts = line.strip().split()
                        # RTTM format: SPEAKER filename 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        speaker = parts[7]
                        segments.append({
                            'start': start_time,
                            'end': start_time + duration,
                            'duration': duration,
                            'speaker': speaker
                        })
            return segments
        except Exception as e:
            logger.error(f"Error parsing RTTM file: {e}")

    return None

def load_audio_file(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and return waveform and sample rate

    Args:
        audio_path (str): Path to audio file

    Returns:
        Tuple of (waveform, sample_rate)
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        logger.info(f"Loaded audio: {sample_rate}Hz, {waveform.shape[0]} channels, {waveform.shape[1]/sample_rate:.2f}s")
        return waveform, sample_rate
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        raise

def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert multi-channel audio to mono by averaging channels

    Args:
        waveform (torch.Tensor): Input waveform

    Returns:
        torch.Tensor: Mono waveform
    """
    if waveform.shape[0] > 1:
        logger.info(f"Converting {waveform.shape[0]} channels to mono")
        return torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def extract_audio_segment(waveform: torch.Tensor, sample_rate: int,
                         start_time: float, end_time: float) -> torch.Tensor:
    """
    Extract audio segment from waveform

    Args:
        waveform (torch.Tensor): Full audio waveform
        sample_rate (int): Sample rate
        start_time (float): Start time in seconds
        end_time (float): End time in seconds

    Returns:
        torch.Tensor: Extracted audio segment
    """
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Ensure indices are within bounds
    start_sample = max(0, start_sample)
    end_sample = min(waveform.shape[1], end_sample)

    if start_sample >= end_sample:
        return torch.zeros(1, 1)  # Return empty segment if invalid range

    return waveform[:, start_sample:end_sample]

def apply_audio_enhancement(segment: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Apply basic audio enhancement to the segment

    Args:
        segment (torch.Tensor): Audio segment
        sample_rate (int): Sample rate

    Returns:
        torch.Tensor: Enhanced audio segment
    """
    # Skip if segment is too short
    if segment.shape[1] < 100:
        return segment

    # Simple noise reduction by normalizing
    if torch.std(segment) > 0:
        segment = segment / torch.std(segment) * 0.1  # Normalize amplitude

    # Add small fade in/out to avoid clicks
    fade_samples = min(1024, segment.shape[1] // 10)

    if fade_samples > 0 and segment.shape[1] > fade_samples * 2:
        # Fade in
        fade_in = torch.linspace(0, 1, fade_samples).to(segment.device)
        segment[:, :fade_samples] *= fade_in

        # Fade out
        fade_out = torch.linspace(1, 0, fade_samples).to(segment.device)
        segment[:, -fade_samples:] *= fade_out

    return segment

def merge_speaker_segments(segments: List[torch.Tensor], sample_rate: int,
                          gap_duration: float = 0.5) -> torch.Tensor:
    """
    Merge multiple segments from the same speaker with gaps

    Args:
        segments (List[torch.Tensor]): List of audio segments
        sample_rate (int): Sample rate
        gap_duration (float): Duration of silence between segments in seconds

    Returns:
        torch.Tensor: Merged audio with gaps
    """
    if not segments:
        return torch.zeros(1, 0)

    gap_samples = int(gap_duration * sample_rate)
    gap_silence = torch.zeros(1, gap_samples)

    merged_segments = []
    for i, segment in enumerate(segments):
        merged_segments.append(segment)
        if i < len(segments) - 1:  # Add gap between segments
            merged_segments.append(gap_silence)

    return torch.cat(merged_segments, dim=1)

def save_audio_segment(segment: torch.Tensor, sample_rate: int,
                      output_path: str, format: str = 'wav') -> bool:
    """
    Save audio segment to file

    Args:
        segment (torch.Tensor): Audio segment to save
        sample_rate (int): Sample rate
        output_path (str): Output file path
        format (str): Audio format ('wav', 'flac', 'mp3')

    Returns:
        bool: True if saved successfully
    """
    try:
        # Ensure segment is not empty
        if segment.shape[1] == 0:
            logger.warning(f"Skipping empty segment: {output_path}")
            return False

        # Convert to numpy for soundfile
        audio_data = segment.squeeze().cpu().numpy()

        # Save using soundfile for better format support
        sf.write(output_path, audio_data, sample_rate, format=format)
        logger.info(f"Saved: {output_path} ({segment.shape[1]/sample_rate:.2f}s)")
        return True

    except Exception as e:
        logger.error(f"Error saving {output_path}: {e}")
        return False

def separate_speakers(audio_path: str, diarization_results: List[Dict],
                     output_dir: str = None,
                     merge_segments: bool = True,
                     min_segment_duration: float = 1.0,
                     audio_format: str = 'wav') -> Dict[str, List[str]]:
    """
    Separate speakers from mixed audio based on diarization results

    Args:
        audio_path (str): Path to input audio file
        diarization_results (List[Dict]): List of diarization segments
        output_dir (str): Output directory for separated speakers (auto-generated if None)
        merge_segments (bool): Whether to merge segments per speaker
        min_segment_duration (float): Minimum segment duration in seconds
        audio_format (str): Output audio format

    Returns:
        Dict mapping speaker IDs to list of output file paths
    """
    # Create output directory based on audio filename if not specified
    if output_dir is None:
        audio_name = Path(audio_path).stem
        output_dir = f"{audio_name}_separated"

    os.makedirs(output_dir, exist_ok=True)

    # Load audio file
    waveform, sample_rate = load_audio_file(audio_path)

    # Convert to mono for consistent processing
    waveform = convert_to_mono(waveform)

    # Group segments by speaker
    speaker_segments = {}
    for segment in diarization_results:
        speaker = segment['speaker']
        duration = segment['duration']

        # Skip very short segments
        if duration < min_segment_duration:
            logger.debug(f"Skipping short segment: {speaker}, {duration:.2f}s")
            continue

        if speaker not in speaker_segments:
            speaker_segments[speaker] = []

        speaker_segments[speaker].append(segment)

    logger.info(f"Found {len(speaker_segments)} speakers")
    for speaker, segments in speaker_segments.items():
        total_duration = sum(s['duration'] for s in segments)
        logger.info(f"Speaker {speaker}: {len(segments)} segments, {total_duration:.2f}s total")

    # Extract and save audio segments
    output_files = {}

    for speaker, segments in speaker_segments.items():
        speaker_files = []

        if merge_segments:
            # Extract all segments for this speaker and merge them
            audio_segments = []
            for segment in segments:
                audio_segment = extract_audio_segment(
                    waveform, sample_rate, segment['start'], segment['end']
                )
                audio_segment = apply_audio_enhancement(audio_segment, sample_rate)
                audio_segments.append(audio_segment)

            # Merge segments with small gaps
            merged_audio = merge_speaker_segments(audio_segments, sample_rate)

            # Save merged audio
            output_path = os.path.join(output_dir, f"speaker_{speaker}.{audio_format}")
            if save_audio_segment(merged_audio, sample_rate, output_path, audio_format):
                speaker_files.append(output_path)

        else:
            # Save each segment separately
            for i, segment in enumerate(segments):
                audio_segment = extract_audio_segment(
                    waveform, sample_rate, segment['start'], segment['end']
                )
                audio_segment = apply_audio_enhancement(audio_segment, sample_rate)

                output_path = os.path.join(
                    output_dir,
                    f"speaker_{speaker}_segment_{i+1:03d}.{audio_format}"
                )
                if save_audio_segment(audio_segment, sample_rate, output_path, audio_format):
                    speaker_files.append(output_path)

        output_files[speaker] = speaker_files

    # Save summary information
    summary_path = os.path.join(output_dir, "separation_summary.json")
    summary = {
        'input_audio': audio_path,
        'total_speakers': len(speaker_segments),
        'speakers': {},
        'settings': {
            'merge_segments': merge_segments,
            'min_segment_duration': min_segment_duration,
            'audio_format': audio_format
        }
    }

    for speaker, files in output_files.items():
        segments_info = []
        for segment in speaker_segments[speaker]:
            segments_info.append({
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration']
            })

        summary['speakers'][speaker] = {
            'segment_count': len(speaker_segments[speaker]),
            'output_files': [os.path.basename(f) for f in files],
            'segments': segments_info
        }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Separation summary saved to: {summary_path}")
    return output_files

def run_diarization_and_separation(audio_path: str, output_dir: str = None,
                                  auth_token: str = None, **kwargs) -> Dict[str, List[str]]:
    """
    Run diarization and separation in one step

    Args:
        audio_path (str): Path to input audio file
        output_dir (str): Output directory (auto-generated if None)
        auth_token (str): Hugging Face auth token
        **kwargs: Additional arguments for separation

    Returns:
        Dict mapping speaker IDs to list of output file paths
    """
    # Create output directory based on audio filename if not specified
    if output_dir is None:
        audio_name = Path(audio_path).stem
        output_dir = f"{audio_name}_separated"

    try:
        # Import voice_diarization module
        import voice_diarization

        logger.info("Running speaker diarization...")
        diarization_dir = os.path.join(output_dir, "diarization")

        # Initialize diarization pipeline
        pipeline = voice_diarization.initialize_diarization_pipeline(auth_token)
        if not pipeline:
            raise Exception("Failed to initialize diarization pipeline")

        # Prepare audio file
        prepared_audio = voice_diarization.prepare_audio_file(audio_path)
        if not prepared_audio:
            raise Exception("Failed to prepare audio file")

        # Perform diarization
        results = voice_diarization.perform_diarization(pipeline, prepared_audio, diarization_dir)
        if not results:
            raise Exception("Diarization failed")

        # Load diarization results
        segments = results['segments']
        logger.info(f"Found {len(segments)} diarization segments")

        # Separate speakers
        return separate_speakers(audio_path, segments, output_dir, **kwargs)

    except ImportError:
        logger.error("voice_diarization.py not found. Please run diarization first.")
        return {}
    except Exception as e:
        logger.error(f"Error in diarization and separation: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(
        description="Speaker Separation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python separate_speakers.py input.wav --rttm audio.rttm
  python separate_speakers.py input.wav --json diarization_results.json
  python separate_speakers.py input.wav --auto-separate --auth-token hf_xxxx
  python separate_speakers.py input.wav --segments diarization_segments.json --output-dir output
        """
    )

    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("--rttm", help="RTTM file from diarization")
    parser.add_argument("--json", help="JSON file from diarization")
    parser.add_argument("--segments", help="JSON file with segment list")
    parser.add_argument("--output-dir", "-o",
                       help="Output directory for separated speakers (auto-generated based on filename if not specified)")
    parser.add_argument("--auto-separate", action="store_true",
                       help="Run diarization and separation in one step")
    parser.add_argument("--auth-token", help="Hugging Face authentication token")
    parser.add_argument("--merge-segments", action="store_true", default=True,
                       help="Merge segments per speaker (default: True)")
    parser.add_argument("--no-merge", action="store_false", dest="merge_segments",
                       help="Don't merge segments - keep them separate")
    parser.add_argument("--min-duration", type=float, default=1.0,
                       help="Minimum segment duration in seconds (default: 1.0)")
    parser.add_argument("--format", choices=['wav', 'flac', 'mp3'], default='wav',
                       help="Output audio format (default: wav)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check if input file exists
    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")

    # Get auth token
    auth_token = args.auth_token or os.getenv('HF_TOKEN')

    # Handle auto-separate mode
    if args.auto_separate:
        results = run_diarization_and_separation(
            args.input, args.output_dir, auth_token,
            merge_segments=args.merge_segments,
            min_segment_duration=args.min_duration,
            audio_format=args.format
        )
    else:
        # Load diarization results
        segments = load_diarization_results(args.rttm, args.json or args.segments)
        if not segments:
            parser.error("No valid diarization results found. Please provide --rttm, --json, or --auto-separate")

        # Separate speakers
        results = separate_speakers(
            args.input, segments, args.output_dir,
            merge_segments=args.merge_segments,
            min_segment_duration=args.min_duration,
            audio_format=args.format
        )

    # Determine actual output directory for display
    actual_output_dir = args.output_dir
    if actual_output_dir is None:
        actual_output_dir = f"{Path(args.input).stem}_separated"

    # Display results
    if results:
        print(f"\nSpeaker separation completed!")
        print(f"Output directory: {actual_output_dir}")
        print(f"Total speakers: {len(results)}")

        for speaker, files in results.items():
            print(f"\nSpeaker {speaker}:")
            for file_path in files:
                print(f"  - {os.path.basename(file_path)}")
    else:
        print("Speaker separation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()