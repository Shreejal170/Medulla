"""
Local audio extraction and spectrogram generation experiment runner.

This is a simple testing/experimentation script for local development.
It directly uses utilities without architectural layers, focusing on:
- Audio extraction from video
- Audio chunking into temporal segments
- Spectrogram generation for visual analysis
- Output structured results for downstream pipeline integration
"""

import argparse
import uuid
import sys
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.domain.models.ingestion import VideoIngestionEvent
from src.utils.audio_helper import (
    extract_audio,
    chunk_audio,
    generate_spectrograms,
    cleanup_all_audio
)


def main(
    video_path: str,
    chunk_duration: float = 5.0,
    cleanup: bool = False
) -> None:
    """
    Run local audio extraction and spectrogram generation experiment.

    Pipeline:
    1. Extract audio from video (mono, 16kHz, WAV)
    2. Chunk audio into temporal segments
    3. Generate Mel spectrograms for each chunk
    4. Output structured metadata for downstream processing
    5. Optional cleanup of temporary files

    Args:
        video_path: Path to input video file
        chunk_duration: Audio chunk duration in seconds (default: 5.0)
        cleanup: Whether to clean up temp files after extraction
    """

    # Create ingestion event
    event = VideoIngestionEvent(
        video_id=uuid.uuid4().hex[:8],
        file_path=video_path,
        sampling_fps=0.5  # Not used for audio, but included for consistency
    )

    logger.info(f"Starting audio extraction for video: {video_path}")

    try:
        # Step 1: Extract audio from video
        logger.info("Step 1/3: Extracting audio from video...")
        audio_file = extract_audio(
            video_id=event.video_id,
            file_path=event.file_path
        )
        logger.info(f"✓ Audio extracted: {audio_file}")

        # Step 2: Chunk audio into segments
        logger.info("Step 2/3: Chunking audio into temporal segments...")
        audio_chunks = chunk_audio(
            audio_file_path=audio_file,
            video_id=event.video_id,
            chunk_duration_sec=chunk_duration
        )
        logger.info(f"✓ Audio chunked into {len(audio_chunks)} segments")

        # Step 3: Generate spectrograms
        logger.info("Step 3/3: Generating Mel spectrograms...")
        audio_chunks_with_spectrograms = generate_spectrograms(audio_chunks)
        logger.info(f"✓ Spectrograms generated for {len(audio_chunks_with_spectrograms)} chunks")

        # Create extraction data model
        from src.domain.models.analysis import AudioExtractionData
        extraction_data = AudioExtractionData(
            video_id=event.video_id,
            extracted_audio_chunks=audio_chunks_with_spectrograms
        )

        # Output results as JSON
        output_json = json.dumps(
            extraction_data.model_dump(mode='json'),
            indent=2
        )
        print("\n" + "=" * 80)
        print("AUDIO EXTRACTION RESULTS")
        print("=" * 80)
        print(output_json)

        # Print summary
        print("\n" + "-" * 80)
        print("SUMMARY")
        print("-" * 80)
        print(f"Video ID:              {extraction_data.video_id}")
        print(f"Audio Chunks:          {extraction_data.total_audio_chunks}")
        print(f"Chunk Duration:        {chunk_duration} seconds")

        if audio_chunks_with_spectrograms:
            total_duration = audio_chunks_with_spectrograms[-1].end_sec
            print(f"Total Audio Duration:  {total_duration:.2f} seconds")

        print("\nChunk Details:")
        for i, chunk in enumerate(audio_chunks_with_spectrograms, 1):
            duration = chunk.end_sec - chunk.start_sec
            print(
                f"  [{i}] {chunk.chunk_id}: "
                f"{chunk.start_sec:.2f}s - {chunk.end_sec:.2f}s "
                f"({duration:.2f}s)"
            )
            print(f"      Audio:      {chunk.audio_chunk_path}")
            print(f"      Spectrogram: {chunk.spectrogram_image_path}")

        print("\n" + "=" * 80)
        print("NEXT STEPS FOR DOWNSTREAM TEAMS:")
        print("=" * 80)
        print("1. Read spectrogram_image_path from JSON output")
        print("2. Load spectrogram PNG image using PIL/OpenCV")
        print("3. Convert image to base64 using base64.b64encode()")
        print("4. Include base64 in multimodal LLM request as image content")
        print("5. Include metadata (chunk_id, timestamps) in LLM prompt context")
        print("6. Process results alongside frame-level analysis for complete video understanding")
        print("=" * 80)

        # Optional cleanup
        if cleanup:
            logger.info("\nCleaning up temporary audio files...")
            cleanup_all_audio()
            logger.info("Cleanup complete")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Processing error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local audio extraction and spectrogram generation experiment"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=5.0,
        help="Audio chunk duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up temporary audio files after extraction"
    )

    args = parser.parse_args()
    main(
        video_path=args.video,
        chunk_duration=args.chunk_duration,
        cleanup=args.cleanup
    )
