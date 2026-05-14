from abc import ABC, abstractmethod

from domain.models.analysis import VideoExtractionData


class FrameExtractorPort(ABC):

    @abstractmethod
    def extract(
        self,
        video_id: str,
        file_path: str,
        sampling_fps: float
    ) -> VideoExtractionData:
        pass
    