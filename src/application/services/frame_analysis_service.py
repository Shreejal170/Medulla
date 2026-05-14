from src.domain.models.analysis import VideoExtractionData,VideoAnalysisResult,FrameAnalysis, VideoMetrics, ExtractedFrame
from Medulla.src.application.prompts.frame_analysis_prompt import FrameAnalysisPrompt
from src.ports.output.llm_port import LlmPort
import logging
from src.utils.image_loader import load_image
from src.core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class FrameAnalysisService:
    """Service class to handle the analysis of video frames using the LLM port."""

    def __init__(self, llm_port):
        self.llm_port = llm_port

    async def analyze_frame(self, video_extraction_data):
        """Analyzes the extracted frames from the video using the LLM port and returns the analysis results."""
        all_results = []

        frames = video_extraction_data.extracted_frames

        for frame in frames:
            logger.info(f"Analyzing frame {frame.frame_id} for video {video_extraction_data.video_id}")
            try:
                # Load the frame image
                image_data = load_image(frame.frame_path)

                # Generate the prompt for the LLM
                prompt = FrameAnalysisPrompt.SYSTEMPROMPT

                # Call the LLM port to get the analysis result for the frame
                result = await self.llm_port.generate_batch_frame_analysis(prompt, image_data, frame.frame_id)
                frame_analysis = FrameAnalysis(
                    frame_id=frame.frame_id,
                    is_authentic=result.is_authentic,
                    confidence_score=result.confidence_score,
                    synthesis_artifacts=result.synthesis_artifacts)
                all_results.append(frame_analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing frame {frame.frame_id}: {str(e)}", exc_info=True)
                all_results.append(FrameAnalysis(frame_id=frame.frame_id, is_authentic=False, confidence_score=0.0, synthesis_artifacts=[]))
            
        return VideoAnalysisResult(video_id=video_extraction_data.video_id, frame_analyses=all_results)