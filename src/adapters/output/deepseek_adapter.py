import logging
from typing import List
from pydantic import BaseModel
from src.domain.models.analysis import (
    FrameAnalysis,
    SynthesisArtifact,
    VideoMetrics,  # Imported in case you need it later
    VideoAnalysisResult,  # Imported in case you need it later
)
from src.ports.output.llm_port import LlmPort
from src.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# 1. Define a strict wrapper schema for the Stage 2 extraction
class ForensicReport(BaseModel):
    artifacts: List[SynthesisArtifact]


class DeepseekLlmAdapter(LlmPort):
    """Adapter class to integrate with the DeepSeek LLM via OpenAI compatible SDK using Instructor.
    Attrs:
        llm_client: The instructor-patched AsyncOpenAI client configured for DeepSeek.
        model_name: The specific DeepSeek model to use for analysis.
    """

    def __init__(self, llm_client, model_name: str = "deepseek-v4-pro"):
        # llm_client must be the instructor-patched AsyncOpenAI client
        """Initialize the DeepSeek LLM Adapter with the given client and model name.

        Args:
            llm_client: An instance of the instructor-patched AsyncOpenAI client configured for DeepSeek.
            model_name: The specific DeepSeek model to use for analysis (default: "deepseek-v4-pro").
        """

        self.llm_client = llm_client
        self.model_name = model_name

    async def generate_frame_analysis(
        self, prompt: str, image_data: str, frame_id: str
    ) -> FrameAnalysis:
        """Generates results based on the provided data using DeepSeek.
        Args:
            prompt: The text prompt to guide the LLM's analysis.
            image_data: The raw image data (base64 string) of the frame to analyze.
            frame_id: The unique identifier for the frame being analyzed.
        Returns:
            A FrameAnalysis object containing the analysis results.
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                }
            ]

            # INSTRUCTOR MAGIC: Notice we use `response_model`, NOT `response_format`
            frame_analysis = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_model=FrameAnalysis,
            )

            # Enforce the frame_id just in case the LLM forgot to include it
            frame_analysis.frame_id = frame_id

            return frame_analysis

        except Exception as e:
            logger.error(
                f"Error generating frame analysis with DeepSeek: {str(e)}",
                exc_info=True,
            )
            return FrameAnalysis(
                frame_id=frame_id,
                is_authentic=False,
                confidence_score=0.0,
                synthesis_artifacts=[],
            )

    async def get_visual_evidence(
        self, analyses: list[FrameAnalysis]
    ) -> list[SynthesisArtifact]:
        """
        Stage 2 Forensic LLM Call: Extracts synthesis artifacts
        (lighting, anatomy) and affected regions to justify an anomalous verdict.


        Args:
            analyses: A list of FrameAnalysis objects from Stage 1, which may contain suspicious frames.
        Returns:
            A list of SynthesisArtifact objects containing detailed forensic evidence extracted by DeepSeek.
        """
        try:
            # Filter for anomalous or suspicious frames to optimize token usage
            suspicious_frames = [f for f in analyses if f.confidence_score < -0.5]
            if not suspicious_frames:
                logger.info("No suspicious frames found for forensic extraction.")
                return []

            forensic_prompt = (
                f"Analyze these {len(suspicious_frames)} flagged frames for structural anomalies. "
                "Identify visual artifact types (e.g., lighting mismatch, anatomical inconsistencies), "
                "provide a detailed text description, and isolate the coordinates via [ymin, xmin, ymax, xmax]. "
            )

            # INSTRUCTOR MAGIC: We pass our ForensicReport wrapper model
            report = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": forensic_prompt}],
                response_model=ForensicReport,
            )

            logger.info(
                f"Successfully extracted {len(report.artifacts)} forensic artifacts via DeepSeek."
            )

            # Return just the raw list of SynthesisArtifact objects
            return report.artifacts

        except Exception as e:
            logger.error(
                f"Error generating forensic artifact summary with DeepSeek: {str(e)}",
                exc_info=True,
            )
            return []
