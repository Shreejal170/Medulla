from abc import ABC, abstractmethod

from domain.models.ingestion import VideoIngestionEvent
from domain.models.analysis import VideoExtractionData


class ExtractFramesUseCasePort(ABC):

    @abstractmethod
    def execute(
        self,
        event: VideoIngestionEvent
    ) -> VideoExtractionData:
        pass
    