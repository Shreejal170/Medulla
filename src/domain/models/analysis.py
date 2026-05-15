from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Annotated


class ExtractedFrame(BaseModel):
    """Model representing an extracted frame from a video."""
    frame_id: Annotated[
        str,
        Field(
            description="The unique identifier for the extracted frame.",
            examples=["frame_0001"],
        ),
    ]
    frame_file_path: Annotated[
        str,
        Field(
            description="The file path where the extracted frame is stored.",
            examples=["/path/to/frame1.jpg"],
        ),
    ]
    timestamp_sec: Annotated[
        float,
        Field(
            description="The timestamp in seconds corresponding to the extracted frame.",
            examples=[0.033, 1.5, 2.0],
        ),
    ]


class VideoExtractionData(BaseModel):
    """Model representing the data extracted from the video for downstream analysis."""
    video_id: Annotated[
        str,
        Field(
            description="The unique identifier for the video.", examples=["video_123"]
        ),
    ]
    extracted_frames: Annotated[
        List[ExtractedFrame],
        Field(
            description="A list of frames extracted from the video for analysis.",
            examples=[
                [
                    {
                        "frame_id": "frame_0001",
                        "frame_file_path": "/path/to/frame1.jpg",
                        "timestamp_sec": 0.033,
                    }
                ]
            ],
        ),
    ]
    audio_path: Optional[
        Annotated[
            str,
            Field(
                description="The file path where the extracted audio is stored.",
                examples=["/path/to/audio.wav"],
            ),
        ]
    ] = None
    @computed_field
    @property
    def total_frames(self) -> int:
        """Computed property to get the total number of extracted frames."""
        return len(self.extracted_frames)

class SynthesisArtifact(BaseModel):
    """Model representing a detected synthesis artifact in a frame."""

    artifact_type: Annotated[
        str,
        Field(
            description="The type of synthesis artifact detected.",
            examples=["blurring", "inconsistent lighting"],
        ),
    ]
    description: Annotated[
        str,
        Field(
            description="A brief description of the detected artifact.",
            examples=["Blurring detected around the edges of the face."],
        ),
    ]
    region: Optional[
        Annotated[
            List[int],
            Field(
                description="The coordinates of the region in the frame where the artifact was detected, in the format [x1, y1, x2, y2].",
                examples=[[100, 150, 200, 250]],
            ),
        ]
    ] = None
    evidence_weight: Optional[
        Annotated[
            float,
            Field(
                description="The weight of the evidence for this artifact in contributing to the authenticity prediction.",
                examples=[0.8],
            ),
        ]
    ] = None


class FrameAnalysis(BaseModel):
    """Model representing the analysis results for a single extracted frame."""
    frame_id: Annotated[str, Field(description="The unique identifier for the analyzed frame.", examples=["frame_0001","frame_0002"])]
    is_authentic: Annotated[bool, Field(description="Indicates whether the frame is authentic or not.", examples=[True])]
    confidence_score: Annotated[float, Field(description="The confidence score of the authenticity prediction for the frame.", examples=[0.95])]
    synthesis_artifacts: Optional[Annotated[List[SynthesisArtifact], Field(description="A list of detected synthesis artifacts in the frame, if any.", examples=[[{"artifact_type": "artifact1", "description": "description"}]])]] = []

class VideoAnalysisResult(BaseModel):

    """Model representing the overall analysis results for a video."""
<<<<<<< HEAD

    video_id: Annotated[str,Field(description="Id of the associated video.")]

=======

    video_id: Annotated[str,Field(description="Id of the associated video.")]
>>>>>>> dev
    frame_analyses : Annotated[List[FrameAnalysis],Field(description = "List of frames analyzed by the llm.")]

class VideoMetrics(BaseModel):
    """Model representing the overall metrics for the video analysis."""

    total_valid_frames: Annotated[
        int,
        Field(
            description="The total number of valid frames analyzed in the video.",
            examples=[100],
        ),
    ]
    ai_frame_count: Annotated[
        int,
        Field(
            description="The number of frames identified as AI-generated in the video.",
            examples=[20],
        ),
    ]
    authentic_frame_count: Annotated[
        int,
        Field(
            description="The number of frames identified as authentic in the video.",
            examples=[80],
        ),
    ]
    uncertain_frame_count: Annotated[
        int,
        Field(
            description="The number of frames for which the authenticity could not be determined.",
            examples=[0],
        ),
    ]
    average_confidence: Annotated[
        float,
        Field(
            description="The average confidence score across all analyzed frames in the video.",
            examples=[0.92],
        ),
    ]
    analysis_summary: Annotated[
        str,
        Field(
            description="A concise summary of the overall analysis results for the video, highlighting reasoning for the authenticity predictions.",
            examples=[
                "The video appears to be authentic with a high confidence score across all frames. No significant synthesis artifacts were detected."
            ],
        ),
    ]
