import argparse
import uuid
import sys
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

from domain.models.ingestion import (
    VideoIngestionEvent
)

from adapters.input.ffmpeg_extractor import (
    FFmpegExtractor
)

from adapters.output.console_publisher import (
    ConsolePublisher
)

from application.services.extract_frames_usecase import (
    ExtractFramesUseCase
)


def main(video_path: str):

    ingestion_event = VideoIngestionEvent(
        video_id=uuid.uuid4().hex[:8],
        file_path=video_path,
        sampling_fps=0.5
    )

    extractor = FFmpegExtractor()

    publisher = ConsolePublisher()

    usecase = ExtractFramesUseCase(
        extractor=extractor,
        publisher=publisher
    )

    result = usecase.execute(
        ingestion_event
    )

    print(
        f"\nExtraction complete."
        f"\nFrames extracted: "
        f"{result.total_frames}\n"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True
    )

    args = parser.parse_args()

    main(args.video)
    