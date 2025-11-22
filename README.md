# Mix Audio Separator

A comprehensive toolkit for separating mixed audio tracks into individual components using speaker diarization and source separation techniques.

## Overview

This project provides tools for:
- **Speaker Diarization**: Identifying who spoke when in an audio recording
- **Source Separation**: Extracting individual speakers from mixed audio
- **Audio Processing**: Converting and preparing audio files for analysis

## Features

- 🎤 **Multi-format Support**: WAV, FLAC, MP3, M4A, and more with automatic conversion
- 🔊 **Speaker Diarization**: Using state-of-the-art pyannote.audio models
- ✂️ **Audio Separation**: Extract individual speakers based on diarization results
- 🔄 **Batch Processing**: Process multiple audio files
- 📊 **Detailed Results**: Export results in multiple formats (JSON, RTTM, SRT, etc.)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mix_audio_seperator
```

2. Install dependencies:
```bash
pip install -r requirements_diarization.txt
```

3. Set up Hugging Face token (required for pyannote models):
```bash
export HF_TOKEN="your_huggingface_token_here"
```

## Quick Start

### Option 1: Automatic Diarization + Separation

```bash
# Run both diarization and separation in one step
python separate_speakers.py input.wav --auto-separate --auth-token $HF_TOKEN
```

### Option 2: Manual Two-Step Process

**Step 1: Run speaker diarization**
```bash
python voice_diarization.py input.wav --output-dir results --auth-token $HF_TOKEN
```

**Step 2: Separate speakers using diarization results**
```bash
python separate_speakers.py input.wav --json results/diarization_results.json
```

## Scripts Overview

### `voice_diarization.py`
Main script for speaker diarization using pyannote.audio models.

**Usage:**
```bash
python voice_diarization.py input.wav --output-dir results --auth-token hf_xxxx
```

**Features:**
- Automatic audio format conversion using FFmpeg
- Audio validation and preprocessing
- Export results in RTTM and JSON formats

### `separate_speakers.py`
Script for extracting individual speakers from mixed audio based on diarization results.

**Usage:**
```bash
# Using JSON results from diarization
python separate_speakers.py input.wav --json diarization_results.json

# Using RTTM file
python separate_speakers.py input.wav --rttm audio.rttm

# Auto-mode (runs diarization + separation)
python separate_speakers.py input.wav --auto-separate --auth-token hf_xxxx
```

**Features:**
- Merge segments per speaker or keep them separate
- Audio enhancement and noise reduction
- Multiple output formats (WAV, FLAC, MP3)
- Detailed separation summary

### `test_audio.py`
Utility script for testing and validating audio files.

```bash
python test_audio.py audio_directory/
```

### `run_diarization_example.py`
Example script demonstrating the complete workflow.

## Requirements

### Python Dependencies
- `torch`
- `torchaudio`
- `pyannote.audio`
- `soundfile`
- `numpy`

### System Dependencies
- `ffmpeg` (required for audio format conversion)

### Authentication
- Hugging Face token with access to pyannote models (required)

## File Formats

### Input Audio Support
- **Direct Support**: WAV, FLAC, MP3, OGG
- **Via Conversion**: M4A, AAC, MP4, WMA (requires FFmpeg)

### Output Formats
- **Audio**: WAV, FLAC, MP3
- **Diarization**: RTTM, JSON
- **Transcripts**: SRT, VTT, TXT, TSV

## Project Structure

```
mix_audio_seperator/
├── README.md                          # This file
├── requirements_diarization.txt       # Python dependencies
├── voice_diarization.py              # Speaker diarization script
├── separate_speakers.py              # Audio separation script
├── test_audio.py                     # Audio testing utilities
├── run_diarization_example.py        # Example workflow
├── data/                             # Sample audio files (optional)
└── results/                          # Output directory (created automatically)
    ├── audio.rttm                    # Diarization results in RTTM format
    ├── diarization_results.json      # Detailed diarization results
    └── [audio_name]_separated/       # Separated speaker files
        ├── speaker_SPEAKER_00.wav     # Individual speaker files
        ├── separation_summary.json    # Separation metadata
        └── ...
```

## Examples

### Basic Usage
```bash
# Diarize an audio file
python voice_diarization.py meeting.wav --output-dir meeting_analysis

# Separate speakers using diarization results
python separate_speakers.py meeting.wav --json meeting_analysis/diarization_results.json
```

### Advanced Usage
```bash
# Convert audio and run diarization (handles unsupported formats)
python voice_diarization.py recording.mp4 --auth-token $HF_TOKEN --output-dir results

# Separate speakers with custom settings
python separate_speakers.py recording.wav --json results/diarization_results.json \
    --output-dir separated_speakers \
    --format flac \
    --no-merge \
    --min-duration 2.0
```

### Testing
```bash
# Test separated audio files
python voice_diarization.py --test-separated separated_speakers/

# Validate individual audio files
python test_audio.py audio_directory/
```

## Troubleshooting

### Common Issues

1. **"Authentication token required"**
   - Set `HF_TOKEN` environment variable or use `--auth-token`
   - Ensure you have access to pyannote models on Hugging Face

2. **"Unsupported audio format"**
   - Install FFmpeg: `sudo apt-get install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS)
   - The script will attempt automatic conversion

3. **CUDA out of memory**
   - Run with CPU-only: Set `CUDA_VISIBLE_DEVICES=""`
   - Use smaller audio files or reduce quality settings

### Performance Tips

- Use 16kHz mono audio for optimal performance
- SSD storage improves processing speed
- GPU acceleration available with CUDA

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!