from enum import Enum
from typing import Optional

from pydantic import BaseModel


class VerdictStatus(str, Enum):
    """
    Represents the final classification state of a video.

    Attributes:
        AI_GENERATED: Indicates the video is likely synthetic.
        AUTHENTIC: Indicates the video is likely authentic.
        INCONCLUSIVE: Indicates insufficient confidence for a final verdict.
    """

    AI_GENERATED = "ai_generated"
    AUTHENTIC = "authentic"
    INCONCLUSIVE = "inconclusive"


class Verdict(BaseModel):
    """
    Represents the result produced by a decision rule.

    Attributes:
        status (VerdictStatus):
            Final classification category.

        is_authentic (bool):
            Boolean representation of authenticity.

        confidence (str):
            Confidence level associated with the verdict.

        explanation (str):
            Human-readable explanation describing why the verdict was chosen.

        error (Optional[str]):
            Optional error or warning message.
    """

    status: VerdictStatus
    is_authentic: bool
    confidence: str
    explanation: str
    error: Optional[str] = None
