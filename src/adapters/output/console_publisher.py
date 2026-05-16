import json
import logging

from ports.output.frame_publisher_port import FramePublisherPort
from domain.models.analysis import VideoExtractionData


logger = logging.getLogger(__name__)


class ConsolePublisher(FramePublisherPort):
    """
    Console output publisher for local testing.

    Implements FramePublisherPort interface.
    Later, KafkaPublisher can be a drop-in replacement.

    Attrs:
        None
    """

    def publish(self, extraction_data: VideoExtractionData) -> None:
        """
        Publish extraction data to console (stdout).

        Args:
            extraction_data: VideoExtractionData to publish

        Raises:
            RuntimeError: If serialization fails
        """
        try:
            output_json = json.dumps(extraction_data.model_dump(), indent=2)
            print(output_json)
            logger.debug(f"Published data for video {extraction_data.video_id}")

        except TypeError as e:
            logger.error(f"Serialization error: {e}")
            raise RuntimeError(
                f"Failed to serialize VideoExtractionData: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected publishing error: {e}")
            raise RuntimeError(f"Publishing failed: {str(e)}") from e
