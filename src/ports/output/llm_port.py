from abc import ABC, abstractmethod
from src.domain.models.analysis import VideoExtractionData, FrameAnalysis


class LlmPort(ABC):
    """Interface for the LLM port to generate analysis results based on extracted video data."""

    @abstractmethod
    async def generate_frame_analysis(self, prompt: str, image_data, frame_id: str) -> FrameAnalysis:
        """Generates analysis results for a single frame based on the provided data."""
        pass
