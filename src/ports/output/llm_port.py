from abc import ABC, abstractmethod
from ...domain.models.analysis import VideoExtractionData, FrameAnalysis


class LlmPort(ABC):
    """Interface for the LLM port to generate analysis results based on extracted video data."""

    @abstractmethod
    def generate_batch_frame_analysis(self, video_data: VideoExtractionData) -> FrameAnalysis:
        """Generates analysis results based on the provided video extraction data."""
        pass
