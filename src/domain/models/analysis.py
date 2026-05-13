from pydantic import BaseModel
from typing import List, Optional

class ExtractedFrame(BaseModel):
    """Model representing an extracted frame from a video."""
    frame_id: str
    file_path: str
    timestamp_sec: float
    

class VideoExtractionData(BaseModel):
    """Model representing the data extracted from the video for downstream analysis."""
    video_id: str
    frames: List[ExtractedFrame]
    audio_path: Optional[str] = None
    total_frames_extracted: int
    
    
    
class FrameAnalysis(BaseModel):
    """Model representing the analysis results for a single extracted frame."""
    frame_id: Annotated[str, Field(description="The unique identifier for the analyzed frame.", examples=[1])]
    is_authentic: Annotated[bool, Field(description="Indicates whether the frame is authentic or not.", examples=[True])]
    confidence_score: Annotated[float, Field(description="The confidence score of the authenticity prediction for the frame.", examples=[0.95])]
    synthesis_artifacts: Optional[Annotated[List[str], Field(description="A list of detected synthesis artifacts in the frame, if any.", examples=[["artifact1", "artifact2"]])]] = []


class VideoMetrics(BaseModel):
    """Model representing the overall metrics for the video analysis."""
    total_valid_frames : Annotated[int, Field(description="The total number of valid frames analyzed in the video.", examples=[100])]
    ai_frame_count: Annotated[int, Field(description="The number of frames identified as AI-generated in the video.", examples=[20])]
    authentic_frame_count: Annotated[int, Field(description="The number of frames identified as authentic in the video.", examples=[80])]
    uncertain_frame_count: Annotated[int, Field(description="The number of frames for which the authenticity could not be determined.", examples=[0])]
    average_confidence: Annotated[float, Field(description="The average confidence score across all analyzed frames in the video.", examples=[0.92])]