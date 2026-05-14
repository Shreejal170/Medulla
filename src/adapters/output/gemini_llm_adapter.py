from ...domain.models.analysis import VideoAnalysisResult,FrameAnalysis, VideoMetrics
from ...ports.output.llm_port import LlmPort
from ...core.logging_config import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)



class GeminiLlmAdapter(LlmPort):
    """Adapter class to integrate with the Gemini LLM for generating frame analysis results."""
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
        
    async def generate(self, prompt:str, images:str, frame_id:str) -> FrameAnalysis:
        """Generates  results based on the provided data using the Gemini LLM."""
        try:
            content = [prompt] + "Now analyze this image to give me the final output: "+images
            response = await self.llm_client.generate(content)        
            # Assuming the response is in the format: {"is_authentic": true/false, "confidence_score": float, "synthesis_artifacts": [SynthesisArticats]}
            frame_analysis = FrameAnalysis(
                frame_id=frame_id,
                is_authentic=response.get("is_authentic", False),
                confidence_score=response.get("confidence_score", 0.0),
                synthesis_artifacts=response.get("synthesis_artifacts", [])
            )
            return frame_analysis
        except Exception as e:
            logger.error(f"Error generating frame analysis: {str(e)}", exc_info=True)
            return FrameAnalysis(frame_id=frame_id, is_authentic=False, confidence_score=0.0, synthesis_artifacts=[])

    def generate_analysis_summary(self, video_analysis: VideoAnalysisResult, prompt:str) -> VideoMetrics:
        """Generates a concise summary of the overall analysis results for the video, highlighting reasoning for the"""
        pass
        
        