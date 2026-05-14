from pydantic import BaseModel, Field, field_validator, computed_field
from typing import List, Optional, Annotated

class ExtractedFrame(BaseModel):
    """Model representing an extracted frame from a video."""
    frame_id: Annotated[str, Field(description="The unique identifier for the extracted frame.", examples=[1,2,3])]
    frame_file_path: Annotated[str,Field(description="The file path where the extracted frame is stored.", examples=["/path/to/frame1.jpg"])]
    timestamp_sec: Annotated[float,Field(description="The timestamp in seconds corresponding to the extracted frame.",examples=[0.033, 1.5, 2.0])]

    

class VideoExtractionData(BaseModel):
    """Model representing the data extracted from the video for downstream analysis."""
    video_id: Annotated[str, Field(description="The unique identifier for the video.", examples=["video_123"])]
    extracted_frames: Annotated[List[ExtractedFrame], Field(description="A list of frames extracted from the video for analysis.",examples=[[{"frame_id": 1, "frame_file_path": "/path/to/frame1.jpg", "timestamp_sec": 0.033}]])]
    audio_path: Optional[Annotated[str, Field(description="The file path where the extracted audio is stored.", examples=["/path/to/audio.wav"])]]
    
    @computed_field
<<<<<<< HEAD
    @property
    def total_frames(self) -> int:
        """Computed property to get the total number of extracted frames."""
        return len(self.extracted_frames)
=======
    def total_frames(self) -> int:
        """Computed property to get the total number of extracted frames."""
        return len(self.extracted_frames)
    
    
>>>>>>> feat-analysis
    
class FrameAnalysis(BaseModel):
    """Model representing the analysis results for a single extracted frame."""
    frame_id: Annotated[str, Field(description="The unique identifier for the analyzed frame.", examples=[1])]
    is_authentic: Annotated[bool, Field(description="Indicates whether the frame is authentic or not.", examples=[True])]
    confidence_score: Annotated[float, Field(description="The confidence score of the authenticity prediction for the frame.", examples=[0.95])]
    synthesis_artifacts: Optional[Annotated[List[str], Field(description="A list of detected synthesis artifacts in the frame, if any.", examples=[["artifact1", "artifact2"]])]] = []

class VideoAnalysisResult(BaseModel):
    """Model representing the overall analysis results for a video."""
    video_id: Annotated[str, Field(description="The unique identifier for the analyzed video.", examples=["video_123"])]
    frame_analyses: Annotated[List[FrameAnalysis], Field(description="A list of analysis results for each extracted frame in the video.", examples=[[{"frame_id": 1, "is_authentic": True, "confidence_score": 0.75, "synthesis_artifacts": ["artifact1", "artifact2"]}]])]


class VideoAnalysisResult(BaseModel):
    """Model representing the overall analysis results for a video."""
    video_id: Annotated[str, Field(description="The unique identifier for the analyzed video.", examples=["video_123"])]
    frame_analyses: Annotated[List[FrameAnalysis], Field(description="A list of analysis results for each extracted frame in the video.", examples=[[{"frame_id": 1, "is_authentic": True, "confidence_score": 0.95, "synthesis_artifacts": ["artifact1", "artifact2"]}]])]


class VideoMetrics(BaseModel):
    """Model representing the overall metrics for the video analysis."""
    total_valid_frames : Annotated[int, Field(description="The total number of valid frames analyzed in the video.", examples=[100])]
    ai_frame_count: Annotated[int, Field(description="The number of frames identified as AI-generated in the video.", examples=[20])]
    authentic_frame_count: Annotated[int, Field(description="The number of frames identified as authentic in the video.", examples=[80])]
    uncertain_frame_count: Annotated[int, Field(description="The number of frames for which the authenticity could not be determined.", examples=[0])]
    average_confidence: Annotated[float, Field(description="The average confidence score across all analyzed frames in the video.", examples=[0.92])]
