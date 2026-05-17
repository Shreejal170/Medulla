from pydantic import BaseModel

from src.domain.models.analysis import VideoMetrics


class RuleContext(BaseModel):
    """
    Immutable context shared across all decision rules.

    Attributes:
        video_id (str):
            Unique identifier of the video being analyzed.

        metrics (VideoMetrics):
            Aggregated metrics calculated from frame analyses.

        ai_pct (float):
            Percentage of frames classified as AI-generated.

        auth_pct (float):
            Percentage of frames classified as authentic.
    """

    video_id: str
    metrics: VideoMetrics
    ai_pct: float
    auth_pct: float
