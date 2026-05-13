from ports.input.extract_frames_usecase_port import (
    ExtractFramesUseCasePort
)

from ports.output.frame_extractor_port import (
    FrameExtractorPort
)

from ports.output.frame_publisher_port import (
    FramePublisherPort
)

from domain.models.ingestion import VideoIngestionEvent

from domain.models.analysis import (
    VideoExtractionData
)


class ExtractFramesUseCase(
    ExtractFramesUseCasePort
):

    def __init__(
        self,
        extractor: FrameExtractorPort,
        publisher: FramePublisherPort
    ):
        self.extractor = extractor
        self.publisher = publisher

    def execute(
        self,
        event: VideoIngestionEvent
    ) -> VideoExtractionData:

        extraction_data = self.extractor.extract(
            video_id=event.video_id,
            file_path=event.file_path,
            sampling_fps=event.sampling_fps
        )

        self.publisher.publish(extraction_data)

        return extraction_data
    