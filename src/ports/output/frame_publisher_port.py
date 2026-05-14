from abc import ABC, abstractmethod

from domain.models.analysis import VideoExtractionData


class FramePublisherPort(ABC):

    @abstractmethod
    def publish(
        self,
        extraction_data: VideoExtractionData
    ) -> None:
        pass
    