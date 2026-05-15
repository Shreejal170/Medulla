from pydantic import BaseModel, Field
from typing import Optional, Annotated


class VideoAnalysisResult(BaseModel):
    video_id: Annotated[str, Field("Video Id to indentify individual videos")]
    is_authentic: Annotated[
        bool,
        Field(
            "Boolean value denoting the final verdict on the authenticity of the video"
        ),
    ]
    explanation: Annotated[
        str, Field("Description on why the the verdict was what it was")
    ]
    error: Optional[str] = None
