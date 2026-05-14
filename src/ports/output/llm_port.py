from abc import ABC, abstractmethod
from src.domain.models.analysis import VideoExtractionData, FrameAnalysis


class LlmPort(ABC):
    """Interface for the LLM port to generate analysis results based on extracted video data."""

    @abstractmethod
    def generate(self, video_data: VideoExtractionData) -> FrameAnalysis:
        """Generates analysis results based on the provided video extraction data."""
        pass

    @abstractmethod
    def get_visual_evidence(self, analyses: list[FrameAnalysis]) -> list:
        """Stage 2: Generates qualitative forensic artifacts for suspicious frames."""
        pass