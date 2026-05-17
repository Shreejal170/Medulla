import logging
from typing import Callable, Optional

from src.core.logging_config import setup_logging
from src.domain.models.analysis import VideoMetrics
from src.domain.models.output import VideoAnalysisResult
from src.domain.models.rule_context import RuleContext
from src.domain.models.verdict import Verdict, VerdictStatus

setup_logging()

logger = logging.getLogger(__name__)

# Minimum number of valid frames required
# before producing a strong authentic verdict.
MIN_FRAMES_FOR_STRONG_VERDICT = 5


# Type alias for decision rule functions
RuleFn = Callable[[RuleContext], Optional[Verdict]]


def rule_strong_ai(ctx: RuleContext) -> Optional[Verdict]:
    """
    Determines whether the video shows a strong AI-generated signal.

    Rule:
        - If AI-generated frames >= 70%, classify as AI-generated.

    Confidence:
        - High: >= 80% AI frames
        - Medium: 70%–79% AI frames

    Args:
        ctx (RuleContext):
            Precomputed context object containing frame metrics.

    Returns:
        Optional[Verdict]:
            Verdict object if rule matches,
            otherwise None.
    """

    if ctx.ai_pct < 0.70:
        return None

    confidence = "high" if ctx.ai_pct >= 0.80 else "medium"

    return Verdict(
        status=VerdictStatus.AI_GENERATED,
        is_authentic=False,
        confidence=confidence,
        explanation=(
            f"AI-generated ({confidence} confidence). "
            f"{ctx.ai_pct:.0%} of frames flagged as synthetic."
        ),
    )


def rule_strong_authentic(ctx: RuleContext) -> Optional[Verdict]:
    """
    Determines whether the video shows a strong authentic signal.

    Rule:
        - Authentic frames >= 30%
        - AI-generated frames < 70%
        - Minimum frame count satisfied

    Confidence:
        - High: >= 50% authentic frames
        - Medium: 30%–49% authentic frames

    Args:
        ctx (RuleContext):
            Precomputed context object containing frame metrics.

    Returns:
        Optional[Verdict]:
            Verdict object if rule matches,
            otherwise None.
    """

    total = ctx.metrics.total_valid_frames

    if (
        total < MIN_FRAMES_FOR_STRONG_VERDICT
        or ctx.auth_pct < 0.30
        or ctx.ai_pct >= 0.70
    ):
        return None

    confidence = "high" if ctx.auth_pct >= 0.50 else "medium"

    return Verdict(
        status=VerdictStatus.AUTHENTIC,
        is_authentic=True,
        confidence=confidence,
        explanation=(
            f"Authentic ({confidence} confidence). "
            f"{ctx.auth_pct:.0%} of frames verified as authentic."
        ),
    )


def rule_avg_confidence_fallback(ctx: RuleContext) -> Optional[Verdict]:
    """
    Applies fallback logic using average confidence score.

    Rule:
        - Average confidence > +0.5 → Authentic
        - Average confidence < -0.5 → AI-generated

    Confidence:
        - Always low confidence

    Args:
        ctx (RuleContext):
            Precomputed context object containing frame metrics.

    Returns:
        Optional[Verdict]:
            Verdict object if rule matches,
            otherwise None.
    """

    avg = ctx.metrics.average_confidence

    if avg > 0.5:
        return Verdict(
            status=VerdictStatus.AUTHENTIC,
            is_authentic=True,
            confidence="low",
            explanation=(
                f"Authentic (low confidence). Average confidence score: {avg:.2f}."
            ),
        )

    if avg < -0.5:
        return Verdict(
            status=VerdictStatus.AI_GENERATED,
            is_authentic=False,
            confidence="low",
            explanation=(
                f"AI-generated (low confidence). Average confidence score: {avg:.2f}."
            ),
        )

    return None


def rule_inconclusive(ctx: RuleContext) -> Optional[Verdict]:
    """
    Produces an inconclusive verdict when no other rule matches.

    This acts as the guaranteed terminal fallback rule.

    Args:
        ctx (RuleContext):
            Precomputed context object containing frame metrics.

    Returns:
        Optional[Verdict]:
            Inconclusive verdict.
    """

    return Verdict(
        status=VerdictStatus.INCONCLUSIVE,
        is_authentic=False,
        confidence="inconclusive",
        explanation=(
            f"Inconclusive. Mixed signals across frames "
            f"(AI: {ctx.ai_pct:.0%}, Auth: {ctx.auth_pct:.0%})."
        ),
        error="Inconclusive — manual review required.",
    )


# Ordered rule registry
RULES: list[RuleFn] = [
    rule_strong_ai,
    rule_strong_authentic,
    rule_avg_confidence_fallback,
    rule_inconclusive,
]


def run_decision_engine(
    video_id: str,
    metrics: VideoMetrics,
) -> tuple[VideoAnalysisResult, VideoMetrics]:
    """
    Executes the deterministic decision engine.

    The engine evaluates frame-level metrics against
    a priority-based rule registry to produce
    a final video authenticity verdict.

    Args:
        video_id (str):
            Unique identifier of the analyzed video.

        metrics (VideoMetrics):
            Aggregated metrics derived from frame analyses.

    Returns:
        tuple[VideoAnalysisResult, VideoMetrics]:
            A tuple containing:
                1. Final structured analysis result.
                2. Updated metrics object.
    """

    logger.info("[%s] Decision engine started", video_id)

    # Guard clause for empty frame sets
    if metrics.total_valid_frames == 0:
        err_msg = "Zero valid frames after processing."

        metrics.analysis_summary = err_msg

        return (
            VideoAnalysisResult(
                video_id=video_id,
                is_authentic=False,
                explanation=err_msg,
                error=err_msg,
            ),
            metrics,
        )

    try:
        total = metrics.total_valid_frames

        ctx = RuleContext(
            video_id=video_id,
            metrics=metrics,
            ai_pct=metrics.ai_frame_count / total,
            auth_pct=metrics.authentic_frame_count / total,
        )

        # Execute rules by priority order
        verdict = next(v for rule in RULES if (v := rule(ctx)) is not None)

        metrics.analysis_summary = (
            f"{verdict.explanation} | "
            f"Status: {verdict.status} | "
            f"Confidence: {verdict.confidence}"
        )

        return (
            VideoAnalysisResult(
                video_id=video_id,
                is_authentic=verdict.is_authentic,
                explanation=verdict.explanation,
                error=verdict.error,
            ),
            metrics,
        )

    except Exception as e:
        logger.error("[%s] Unexpected engine error: %s", video_id, e)

        fail_msg = "Decision engine encountered an unexpected error."

        return (
            VideoAnalysisResult(
                video_id=video_id,
                is_authentic=False,
                explanation=fail_msg,
                error=str(e),
            ),
            metrics,
        )
