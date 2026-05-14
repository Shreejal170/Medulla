from src.core.logging_config import setup_logging
from src.application.services.frame_analysis_service import FrameAnalysisService
from src.adapters.output.gemini_llm_adapter import GeminiLlmAdapter
from src.domain.models.analysis import VideoExtractionData, ExtractedFrame
import logging
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


setup_logging() 

logger = logging.getLogger(__name__)


def setup_dependencies():
    client = genai.Client(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))  # Initialize the Gemini LLM client
    return client
    

async def orchestrate_analytics():
    logger.info("Starting Orchestration Service...")
    client = setup_dependencies()
    llm_adapter = GeminiLlmAdapter(llm_client=client, model_name='gemini-3.1-flash-lite')
    frame_analysis_service = FrameAnalysisService(llm_port=llm_adapter)
    # Example usage with dummy video extraction data
    video_extraction_data = VideoExtractionData(
        video_id="video_123",
        extracted_frames=[
            ExtractedFrame(frame_id="frame_0001", frame_file_path="C:\\Users\\abira\\Documents\\veel_projects\\Medulla\\src\\utils\\sample_images\\ai1.jpg", timestamp_sec=0.033),
            ExtractedFrame(frame_id="frame_0002", frame_file_path="C:\\Users\\abira\\Documents\\veel_projects\\Medulla\\src\\utils\\sample_images\\ai2.jpg", timestamp_sec=1.5)
        ],
        audio_path=""
    )
    analysis_result = await frame_analysis_service.analyze_frame(video_extraction_data)
    logger.info(f"Analysis Result: {analysis_result}")
    print(analysis_result)