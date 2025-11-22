# Example Audio Files

This directory contains sample audio files for testing the audio separation toolkit.

## Audio Files

| File | Format | Duration | Description |
|------|--------|----------|-------------|
| `alice.wav` | WAV | ~9 seconds | Single speaker audio sample |
| `mix.flac` | FLAC | ~30 seconds | Mixed audio with multiple speakers |
| `female_annie.m4a` | M4A | ~8 seconds | Female voice sample |
| `male.m4a` | M4A | ~8 seconds | Male voice sample |

## Sample Results

- `diarization_results/` - Example diarization output directory
- `audio.rttm` - Sample RTTM format diarization results

## Quick Test Commands

Try these commands to test the toolkit with the example files:

### 1. Test Diarization (Requires HF Token)
```bash
# Using the mixed audio file
python voice_diarization.py examples/mix.flac --output-dir examples/test_output --auth-token $HF_TOKEN

# Using single speaker audio
python voice_diarization.py examples/alice.wav --output-dir examples/test_output --auth-token $HF_TOKEN
```

### 2. Test Audio Separation
```bash
# Using sample RTTM file
python separate_speakers.py examples/mix.flac --rttm examples/audio.rttm --output-dir examples/separated_speakers

# Auto-mode (runs diarization + separation)
python separate_speakers.py examples/mix.flac --auto-separate --auth-token $HF_TOKEN
```

### 3. Test Audio Files
```bash
# Validate audio files by loading them with diarization (quick check)
python voice_diarization.py examples/alice.wav --output-dir examples/test_validation --auth-token $HF_TOKEN
```

### 4. Format Conversion Test
```bash
# Test automatic format conversion (M4A → WAV)
python voice_diarization.py examples/female_annie.m4a --output-dir examples/test_output --auth-token $HF_TOKEN
```

## Expected Output

After running the commands above, you should see:

1. **Diarization Results**:
   - `examples/test_output/audio.rttm`
   - `examples/test_output/diarization_results.json`

2. **Separated Speakers**:
   - `examples/separated_speakers/speaker_SPEAKER_00.wav`
   - `examples/separated_speakers/separation_summary.json`

3. **Console Output**:
   - Speaker identification information
   - Timing information for each speech segment
   - Audio file validation results

## Notes

- Some example files may require audio format conversion (handled automatically)
- Make sure you have FFmpeg installed for M4A file support
- HF_TOKEN environment variable must be set for pyannote model access
- Processing time varies based on audio length and system resources