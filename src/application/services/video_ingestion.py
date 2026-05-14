import logging
from src.ports.output.message_publisher import MessagePublisherPort
from src.domain.models.analysis import VideoExtractionData

# ffmpeg_helper for extracting frames
from src.utils import ffmpeg_helper

logger = logging.getLogger(__name__)

class VideoIngestionService:
    """
    Application service that orchestrates the video ingestion pipeline.
    Now directly coupled to ffmpeg_helper for extraction.
    """
    def __init__(self, publisher: MessagePublisherPort):
        self.publisher = publisher

    async def ingest_video(self, video_id: str, file_path: str) -> None:
        logger.info(f"Starting ingestion process for video: {video_id}")

        try:
            # 1. The utility now handles extraction AND building the Pydantic model
            extraction_payload = ffmpeg_helper.extract_frames(
                video_id=video_id,
                file_path=file_path,
                sampling_fps=0.5,
                output_dir="./data/temp_frames" 
            )

            logger.info(f"ETL Complete. {extraction_payload.total_frames} frames extracted.")

            # 2. Publish the payload directly to Kafka
            await self.publisher.publish(
                topic="frames-ready-for-ai",
                message=extraction_payload
            )

            logger.info(f"Successfully published data to Kafka for {video_id}")

        except Exception as e:
            logger.error(f"Ingestion pipeline failed for video {video_id}: {e}")
            raise
            