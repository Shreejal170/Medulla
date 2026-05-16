from abc import ABC, abstractmethod
from src.domain.models.analysis import VideoExtractionData, FrameAnalysis


class LlmPort(ABC):
    """Interface for the LLM port to generate analysis results based on extracted video data."""

    @abstractmethod
    async def generate_frame_analysis(
        self, prompt: str, image_data, frame_id: str
    ) -> FrameAnalysis:
        """Generates analysis results for a single frame based on the provided data.
        Args:
            prompt: The text prompt to guide the LLM's analysis.
            image_data: The raw image data of the frame to be analyzed.
            frame_id: Unique identifier for the frame being analyzed.
        Returns:
            A FrameAnalysis object containing the analysis results for the frame.
        """
        pass

    @abstractmethod
    def get_visual_evidence(self, analyses: list[FrameAnalysis]) -> list:
        """Stage 2: Generates qualitative forensic artifacts for suspicious frames.
        Args:
            analyses: A list of FrameAnalysis objects that have been analyzed and deemed suspicious.
        Returns:
            A list of SynthesisArtifact objects containing detailed forensic evidence extracted by the LLM.
        """
        pass
