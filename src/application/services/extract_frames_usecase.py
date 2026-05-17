from ports.input.extract_frames_usecase_port import ExtractFramesUseCasePort

from ports.output.frame_extractor_port import FrameExtractorPort

from ports.output.frame_publisher_port import FramePublisherPort

from domain.models.ingestion import VideoIngestionEvent

from domain.models.analysis import VideoExtractionData


class ExtractFramesUseCase(ExtractFramesUseCasePort):
    """Use case class to orchestrate the extraction of frames from a video and publishing the results.
    Attrs:
    extractor: An instance of FrameExtractorPort to handle frame extraction logic.
    publisher: An instance of FramePublisherPort to handle publishing the extracted data.
    """

    def __init__(self, extractor: FrameExtractorPort, publisher: FramePublisherPort):
        """Initialize the ExtractFramesUseCase with the given extractor and publisher.
        Args:
            extractor: An instance of FrameExtractorPort to handle frame extraction logic.
            publisher: An instance of FramePublisherPort to handle publishing the extracted data.
        """
        self.extractor = extractor
        self.publisher = publisher

    def execute(self, event: VideoIngestionEvent) -> VideoExtractionData:
        """Executes the use case by extracting frames from the video and publishing the results.
        Args:
            event: A VideoIngestionEvent containing the details of the video to process.
        Returns:
            A VideoExtractionData object containing the results of the frame extraction.
        """

        extraction_data = self.extractor.extract(
            video_id=event.video_id,
            file_path=event.file_path,
            sampling_fps=event.sampling_fps,
        )

        self.publisher.publish(extraction_data)

        return extraction_data
