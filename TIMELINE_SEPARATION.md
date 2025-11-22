# Timeline-Preserving Speaker Separation

This enhancement allows you to separate male and female voices while maintaining the original timeline of the audio.

## What it does

Unlike traditional speaker separation that creates separate audio files containing only the speech segments for each speaker (with compressed timeline), timeline-preserving separation creates output files that:

- Have the **same duration** as the original audio
- Contain the target speaker's voice **only during their speaking times**
- Have **silence** during all other periods
- Maintain perfect **temporal alignment** with the original audio

## Use Cases

This is useful for:
- Creating individual voice tracks for video editing while maintaining sync
- Analyzing specific speaker contributions in context
- Creating karaoke-style vocal isolation
- Audio processing where timing is critical
- Voice recognition training with preserved temporal context

## Usage

### Basic Usage

```bash
# Separate with timeline preservation
python separate_speakers.py input.wav --json diarization_results.json --timeline

# Using RTTM file
python separate_speakers.py input.wav --rttm audio.rttm --timeline

# Specify output format and directory
python separate_speakers.py input.wav --json diarization.json --timeline --format wav --output-dir timeline_output
```

### Complete Workflow

1. **First run diarization** to identify speakers:
   ```bash
   python voice_diarization.py input.wav --auth-token hf_your_token --output-dir diarization_results
   ```

2. **Then run timeline separation**:
   ```bash
   python separate_speakers.py input.wav --json diarization_results/diarization_results.json --timeline
   ```

### Demo Script

Run the included demo:
```bash
python demo_timeline_separation.py
```

## Output Files

The timeline separation creates:

- `speaker_SPEAKER_00_timeline.wav` - Speaker 0 with preserved timeline
- `speaker_SPEAKER_01_timeline.wav` - Speaker 1 with preserved timeline
- `timeline_separation_summary.json` - Detailed summary with timing information

## Example Output Structure

```
your_audio_timeline_separated/
├── speaker_SPEAKER_00_timeline.wav
├── speaker_SPEAKER_01_timeline.wav
└── timeline_separation_summary.json
```

## Technical Details

### How it Works

1. **Load Original Audio**: The complete original audio file is loaded
2. **Create Silence Base**: Create a silent audio track the same length as the original
3. **Extract Segments**: For each speaker segment, extract from original audio
4. **Apply Enhancement**: Clean up the audio segment
5. **Place in Timeline**: Insert the segment at the correct time position
6. **Save Result**: Export timeline-preserved audio

### Audio Processing

- Original audio is converted to mono for consistent processing
- Audio enhancement includes noise reduction and fade in/out
- Sample rate is preserved from original audio
- Silent periods are absolute silence (0 amplitude)

### File Naming Convention

- `speaker_{SPEAKER_ID}_timeline.{format}`
- Output directory: `{filename}_timeline_separated`

## Comparison with Traditional Separation

| Feature | Traditional | Timeline-Preserving |
|---------|-------------|-------------------|
| Output Duration | Compressed | Same as Original |
| Silent Periods | Removed | Preserved |
| Temporal Sync | Lost | Maintained |
| Use Case | Speech analysis | Audio production |
| File Size | Smaller | Same as input |

## Requirements

Same requirements as the original voice separation toolkit:
- Python 3.7+
- torch
- torchaudio
- soundfile
- pyannote.audio

## Limitations

- Currently not supported with `--auto-separate` mode
- Requires pre-existing diarization results
- Mono output (original stereo converted to mono)
- Processing time increases with audio length

## Troubleshooting

**"Timeline mode not supported with auto-separate yet"**
- Run diarization first, then use timeline mode separately

**Empty output files**
- Check diarization results for valid segments
- Verify audio file can be loaded
- Use `--verbose` flag for detailed logging

**Audio sync issues**
- Ensure diarization timing is accurate
- Check sample rate consistency
- Verify original audio quality