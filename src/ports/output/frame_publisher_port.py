from abc import ABC, abstractmethod

from domain.models.analysis import VideoExtractionData


class FramePublisherPort(ABC):
    """Abstract base class defining the interface for publishing extracted frame data.
    This port defines the contract for publishing the results of frame extraction in a structured format.
    """

    @abstractmethod
    def publish(self, extraction_data: VideoExtractionData) -> None:
        pass
