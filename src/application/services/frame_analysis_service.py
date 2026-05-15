from src.domain.models.analysis import (
    VideoExtractionData,
    VideoAnalysisResult,
    FrameAnalysis,
)
from src.ports.output.llm_port import LlmPort
import logging
from src.utils.image_loader import load_image
from src.core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class FrameAnalysisService:
    """Service class to handle the analysis of video frames using the LLM port."""

    def __init__(self, LlmPort):
        self.llm_port = LlmPort

    async def analyze_frame(self, video_extraction_data: VideoExtractionData) -> VideoAnalysisResult:
        """Analyzes the extracted frames from the video using the LLM port and returns the analysis results."""
        all_results = []

        frames = video_extraction_data.get('extracted_frames')
<<<<<<< HEAD
        print(frames)
        for frame in frames:


=======
        for frame in frames:

>>>>>>> dev
            logger.info(
                f"Analyzing frame {frame.get('frame_id')} for video {video_extraction_data.get('video_id')}"
            )

<<<<<<< HEAD
            print("*"*20)
            print("Debugging")
            print(type(frame))
            print(frame)
            
            try:
                # Load the frame image using .get()
=======
            try:
                # Load the frame image
>>>>>>> dev
                image_data = load_image(frame.get('frame_file_path'))

                # Build a plain-text prompt string for the Gemini adapter
                prompt = (
                    "You are an expert forensic AI image analyst.\n"
                    "Analyze the provided frame and return ONLY valid JSON with keys: "
                    "is_authentic (bool), confidence_score (float), synthesis_artifacts (list)."
                )

                # Call the LLM port — returns a FrameAnalysis object directly
                result = await self.llm_port.generate_frame_analysis(
                    prompt, image_data, frame.get('frame_id')
                )
<<<<<<< HEAD
                
                frame_analysis = FrameAnalysis(
                    frame_id=frame.get('frame_id'),
                    is_authentic=result.get('is_authentic'),
                    confidence_score=result.get('confidence_score'),
                    synthesis_artifacts=result.get('synthesis_artifacts'),
                )
                all_results.append(frame_analysis)
                print(all_results)

=======

                # result is already a FrameAnalysis object; use it directly
                all_results.append(result)
                logger.info(f"Frame {frame.get('frame_id')} analysed: authentic={result.is_authentic}")
>>>>>>> dev

            except Exception as e:
                logger.error(
                    f"Error analyzing frame {frame.get('frame_id')}: {str(e)}", exc_info=True
                )
                all_results.append(
                    FrameAnalysis(
                        frame_id=frame.get('frame_id'),
                        is_authentic=False,
                        confidence_score=0.0,
                        synthesis_artifacts=[],
                    )
                )
            
            break

<<<<<<< HEAD
            return VideoAnalysisResult(
                video_id=video_extraction_data.get('video_id'), frame_analyses=all_results
            )
=======
        # Return AFTER all frames are processed (was incorrectly inside the loop)
        return VideoAnalysisResult(
            video_id=video_extraction_data.get('video_id'), frame_analyses=all_results
        )
>>>>>>> dev
