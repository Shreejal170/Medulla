from abc import ABC, abstractmethod

from domain.models.analysis import VideoExtractionData


class FrameExtractorPort(ABC):
    """Abstract base class defining the interface for frame extraction from videos.
    This port defines the contract for extracting frames from a video file and returning the results in a
    structured format.
    """

    @abstractmethod
    def extract(
        self, video_id: str, file_path: str, sampling_fps: float
    ) -> VideoExtractionData:
        pass
