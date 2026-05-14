from src.domain.models.analysis import VideoExtractionData,VideoAnalysisResult,FrameAnalysis, VideoMetrics, ExtractedFrame
from src.application.prompts.frame_analysis_prompt import FrameAnalysisPrompt
from src.ports.output.llm_port import LlmPort
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
        
        
    def generate_analysis_summary(self, analyses: list[FrameAnalysis], prompt: str) -> list[dict]:
        """
        Generates a detailed forensic summary of visual artifacts.
        
        This satisfies the Stage 2 requirement: Extracting synthesis artifacts 
        (lighting, anatomy) and affected regions to justify the verdict.
        """
        try:
            # The prompt here should be the specialized Forensic Analysis Prompt
            response = self.llm_client.generate(prompt)
            
            # Assuming the response is a list of artifacts: 
            # [{"type": "...", "description": "...", "region": [ymin, xmin, ymax, xmax]}]
            artifacts = response.get("artifacts", [])
            
            if not artifacts:
                logger.info("No visual artifacts identified by the forensic LLM call.")
                return []

            logger.info(f"Successfully extracted {len(artifacts)} forensic artifacts.")
            return artifacts

        except Exception as e:
            # Resilience: Partial failure of forensic synthesis does not crash the pipeline
            logger.error(f"Error generating forensic artifact summary: {str(e)}", exc_info=True)
            return []
        
        