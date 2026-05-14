from Medulla.src.domain.models.analysis import VideoExtractionData,VideoAnalysisResult,FrameAnalysis, VideoMetrics, ExtractedFrame
from Medulla.src.application.prompts.FrameAnalysisPrompt import FrameAnalysisPrompt
from Medulla.src.ports.output.llm_port import LlmPort
from src.core.logging_config import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)



class GeminiLlmAdapter(LlmPort):
    """Adapter class to integrate with the Gemini LLM for generating frame analysis results."""
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    def generate_frame_analysis(self, prompt:str ) -> FrameAnalysis:
        """Generates  results based on the provided data using the Gemini LLM."""
        try:
            response = self.llm_client.generate(prompt)
            # Assuming the response is in the format: {"is_authentic": true/false, "confidence_score": float, "synthesis_artifacts": [SynthesisArticats]}
            frame_analysis = FrameAnalysis(
                frame_id="unknown",  # In a real implementation, this should be set to the actual frame ID
                is_authentic=response.get("is_authentic", False),
                confidence_score=response.get("confidence_score", 0.0),
                synthesis_artifacts=response.get("synthesis_artifacts", [])
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
