from src.domain.models.analysis import VideoAnalysisResult,FrameAnalysis, VideoMetrics, SynthesisArtifact
from src.ports.output.llm_port import LlmPort
from src.core.logging_config import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)



class GeminiLlmAdapter(LlmPort):
    """Adapter class to integrate with the Gemini LLM for generating frame analysis results."""
    def __init__(self, llm_client, model_name: str = 'gemini-1.5-flash'):
        self.llm_client = llm_client
        self.model_name = model_name
    
        
    async def generate_frame_analysis(self, prompt:str, image_data, frame_id:str) -> FrameAnalysis:
        """Generates results based on the provided data using the Gemini LLM."""
        try:
            content = [prompt, image_data]  # Adjust content structure
            response = self.llm_client.models.generate_content(
                model=self.model_name,
                contents=content
            )
            # Parse response.text as JSON (assuming LLM returns JSON)
            import json
            result = json.loads(response.text)
            
            # Map LLM response artifacts (with "type" field) to SynthesisArtifact objects (with "artifact_type" field)
            artifacts = []
            for artifact_data in result.get("artifacts", []):
                artifact = SynthesisArtifact(
                    artifact_type=artifact_data.get("type"),
                    description=artifact_data.get("description"),
                    region=artifact_data.get("region"),
                    evidence_weight=artifact_data.get("evidence_weight")
                )
                artifacts.append(artifact)
            
            frame_analysis = FrameAnalysis(
                frame_id=frame_id,
                is_authentic=result.get("is_authentic", False),
                confidence_score=result.get("confidence_score", 0.0),
                synthesis_artifacts=artifacts
            )
            return frame_analysis
        except Exception as e:
            logger.error(f"Error generating frame analysis: {str(e)}", exc_info=True)
            return FrameAnalysis(frame_id=frame_id, is_authentic=False, confidence_score=0.0, synthesis_artifacts=[])

    def generate_analysis_summary(self, video_analysis: VideoAnalysisResult, prompt:str) -> VideoMetrics:
        """Generates a concise summary of the overall analysis results for the video, highlighting reasoning for the"""
        pass
    
    
    
    
    

        
        