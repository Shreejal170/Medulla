from pydantic import BaseModel, Field
from typing import Optional, List


class VideoIngestionEvent(BaseModel):
    """Model representing the event data for video ingestion.
    Attrs:
    video_id: Unique identifier for the video.
    file_path: Path to the video file.
    sampling_fps: Frame rate for video sampling (default: 0.5 fps).
    """

    video_id: str = Field(..., description="Unique identifier for the video")
    file_path: str = Field(..., description="Path to the video file")
    sampling_fps: float = Field(0.5, description="Frame rate for video sampling")
