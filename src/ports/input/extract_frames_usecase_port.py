from abc import ABC, abstractmethod

from domain.models.ingestion import VideoIngestionEvent
from domain.models.analysis import VideoExtractionData


class ExtractFramesUseCasePort(ABC):
    """Abstract base class defining the interface for the ExtractFramesUseCase.
    This port defines the contract for executing the use case of extracting frames from a video and publishing the results.
    """

    @abstractmethod
    def execute(self, event: VideoIngestionEvent) -> VideoExtractionData:
        pass
