#!/usr/bin/env python3
"""
Example script demonstrating how to use the voice_diarization.py script
"""

import os
import subprocess
import sys

def main():
    print("Voice Diarization Example")
    print("=" * 40)

    # Check if we have an audio file to process
    audio_file = "mix.flac"  # Default audio file from t.py

    if not os.path.exists(audio_file):
        print(f"Audio file '{audio_file}' not found.")
        print("Please place an audio file in the current directory or specify the path.")
        print("\nUsage examples:")
        print("python run_diarization_example.py input.wav")
        print("python run_diarization_example.py input.mp4 --auth-token hf_xxxx")
        return

    # Build command
    cmd = [
        sys.executable, "voice_diarization.py", audio_file,
        "--output-dir", "diarization_results",
        "--verbose"
    ]

    # Add auth token if provided as environment variable
    auth_token = os.getenv('HF_TOKEN')
    if auth_token:
        cmd.extend(["--auth-token", auth_token])

    print(f"Running command: {' '.join(cmd)}")
    print()

    try:
        # Run the diarization
        result = subprocess.run(cmd, check=True)

        if result.returncode == 0:
            print("\nDiarization completed successfully!")
            print("Check the 'diarization_results' directory for output files.")

            # Display results if available
            rttm_file = "diarization_results/audio.rttm"
            json_file = "diarization_results/diarization_results.json"

            if os.path.exists(rttm_file):
                print(f"\nRTTM results saved to: {rttm_file}")

            if os.path.exists(json_file):
                print(f"JSON results saved to: {json_file}")

                # Display summary from JSON
                import json
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    print(f"\nSummary:")
                    print(f"- Audio file: {data['audio_file']}")
                    print(f"- Total speakers: {data['total_speakers']}")
                    print(f"- Total segments: {len(data['segments'])}")

                    print("\nSpeaker segments:")
                    for i, segment in enumerate(data['segments'][:10]):  # Show first 10
                        print(f"  {i+1}. {segment['speaker']}: {segment['start']:.1f}s - {segment['end']:.1f}s")

                    if len(data['segments']) > 10:
                        print(f"  ... and {len(data['segments']) - 10} more segments")

    except subprocess.CalledProcessError as e:
        print(f"Error running diarization: {e}")
        return
    except FileNotFoundError:
        print("Error: voice_diarization.py not found in current directory")
        return

if __name__ == "__main__":
    main()