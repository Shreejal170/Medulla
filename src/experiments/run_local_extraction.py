"""
Local video frame extraction experiment runner.

This is a simple testing/experimentation script for local development.
It directly uses utilities without architectural layers.
"""

import argparse
import uuid
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.models.ingestion import VideoIngestionEvent
from utils.ffmpeg_helper import extract_frames, cleanup_temp_frames


def main(video_path: str, cleanup: bool = False) -> None:
    """
    Run local frame extraction experiment.
    
    Args:
        video_path: Path to input video file
        cleanup: Whether to clean up temp frames after extraction
    """
    
    # Create ingestion event
    event = VideoIngestionEvent(
        video_id=uuid.uuid4().hex[:8],
        file_path=video_path,
        sampling_fps=0.5
    )
    
    logger.info(f"Starting extraction for video: {video_path}")
    
    # Extract frames directly using utility
    extraction_data = extract_frames(
        video_id=event.video_id,
        file_path=event.file_path,
        sampling_fps=event.sampling_fps
    )
    
    # Output results as JSON
    output_json = json.dumps(
        extraction_data.model_dump(),
        indent=2
    )
    print(output_json)
    
    # Print summary
    print(f"\nExtraction complete.")
    print(f"Frames extracted: {extraction_data.total_frames}")
    
    # Optional cleanup
    if cleanup:
        logger.info("Cleaning up temporary frames...")
        cleanup_temp_frames()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local video frame extraction experiment")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--cleanup", action="store_true", help="Clean up temp frames after extraction")
    
    args = parser.parse_args()
    main(args.video, cleanup=args.cleanup)
