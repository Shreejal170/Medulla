from src.domain.models.analysis import (
    VideoAnalysisResult,
    FrameAnalysis,
    VideoMetrics,
    SynthesisArtifact,
)
from src.ports.output.llm_port import LlmPort
from src.core.logging_config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


class GeminiLlmAdapter(LlmPort):
    """Adapter class to integrate with the Gemini LLM for generating frame analysis results."""

    def __init__(self, llm_client, model_name: str = "gemini-3.1-flash-lite"):
        self.llm_client = llm_client
        self.model_name = model_name

    async def generate_frame_analysis(
        self, prompt: str, image_data, frame_id: str
    ) -> FrameAnalysis:
        """Generates results based on the provided data using the Gemini LLM."""
        try:
            content = [prompt, image_data]  # Adjust content structure
            response = self.llm_client.models.generate_content(
                model=self.model_name, contents=content
            )
            # Parse response.text as JSON (assuming LLM returns JSON)
            import json

            result = json.loads(response.text)

            # Map LLM response artifacts (with "type" field) to SynthesisArtifact objects (with "artifact_type" field)
            artifacts = []
            for artifact_data in result.get("synthesis_artifacts", []):
                artifact = SynthesisArtifact(
                    artifact_type=artifact_data.get("type"),
                    description=artifact_data.get("description"),
                    region=artifact_data.get("region"),
                    evidence_weight=artifact_data.get("evidence_weight"),
                )
                artifacts.append(artifact)

            frame_analysis = FrameAnalysis(
                frame_id=frame_id,
                is_authentic=result.get("is_authentic", False),
                confidence_score=result.get("confidence_score", 0.0),
                synthesis_artifacts=artifacts,
            )
            return frame_analysis
        except Exception as e:
            logger.error(f"Error generating frame analysis: {str(e)}", exc_info=True)
            return FrameAnalysis(frame_id="unknown", is_authentic=False, confidence_score=0.0, synthesis_artifacts=[])
        
        
    def get_visual_evidence(self, analyses: list[FrameAnalysis]) -> list:
        """
        Stage 2 Forensic LLM Call: Extracts synthesis artifacts 
        (lighting, anatomy) and affected regions to justify an anomalous verdict.
        """
        try:
            # Filter for anomalous or suspicious frames to optimize token usage
            suspicious_frames = [f for f in analyses if f.confidence_score < -0.5]
            if not suspicious_frames:
                logger.info("No suspicious frames found for forensic extraction.")
                return []
                
            # Construct a dynamic prompt referencing the specific failed frames
            forensic_prompt = (
                f"Analyze these {len(suspicious_frames)} flagged frames for structural anomalies. "
                "Identify visual artifact types (e.g., lighting mismatch, anatomical inconsistencies), "
                "provide a detailed text description, and isolate the coordinates via [ymin, xmin, ymax, xmax]."
            )
            
            response = self.llm_client.generate(forensic_prompt)
            artifacts = response.get("artifacts", [])
            
            logger.info(f"Successfully extracted {len(artifacts)} forensic artifacts.")
            return artifacts

        except Exception as e:
            # Fault Tolerance: Return empty array so application service degrades gracefully
            logger.error(f"Error generating forensic artifact summary: {str(e)}", exc_info=True)
            return []
