from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from src.domain.models.analysis import VideoMetrics
from src.domain.models.output import VideoAnalysisResult
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Minimum frames required for Rule 2 to produce a medium/high verdict.
# Below this count, authentic signal is treated as insufficient evidence.
MIN_FRAMES_FOR_STRONG_VERDICT = 5


# ── Verdict status ─────────────────────────────────────────────────

class VerdictStatus(str, Enum):
    """
    Explicit status attached to every verdict produced by the rule registry.
    
    AI_GENERATED: Video is synthetic. Rules 1 or 3 (negative) fired.
    AUTHENTIC:    Video is real. Rules 2 or 3 (positive) fired.
    INCONCLUSIVE: No rule produced a confident verdict. Manual review needed.
    """
    AI_GENERATED = "ai_generated"
    AUTHENTIC    = "authentic"
    INCONCLUSIVE = "inconclusive"


# ── Intermediate verdict ───────────────────────────────────────────

@dataclass
class Verdict:
    """
    Intermediate result produced by a single rule function.
    Explanation focuses strictly on mathematical justification.
    """
    status: VerdictStatus
    is_authentic: bool
    confidence: str
    explanation: str
    error: Optional[str] = None


# ── Rule context ───────────────────────────────────────────────────

@dataclass
class RuleContext:
    """
    Immutable bundle of pre-computed values. artifact_summary removed 
    to decouple forensic text from pure decision logic.
    """
    video_id: str
    metrics: VideoMetrics
    ai_pct: float
    auth_pct: float


RuleFn = Callable[[RuleContext], Optional[Verdict]]


# ── Individual rules (Priority Execution) ──────────────────────────

def rule_strong_ai(ctx: RuleContext) -> Optional[Verdict]:
    """Rule 1 — Strong AI signal (>= 70% AI frames)."""
    if ctx.ai_pct < 0.70:
        return None

    confidence = "high" if ctx.ai_pct >= 0.80 else "medium"
    return Verdict(
        status=VerdictStatus.AI_GENERATED,
        is_authentic=False,
        confidence=confidence,
        explanation=f"AI-generated ({confidence} confidence). {ctx.ai_pct:.0%} of frames flagged as synthetic."
    )


def rule_strong_authentic(ctx: RuleContext) -> Optional[Verdict]:
    """Rule 2 — Strong authentic signal (>= 30% authentic, AI < 70%)."""
    total = ctx.metrics.total_valid_frames

    if total < MIN_FRAMES_FOR_STRONG_VERDICT or ctx.auth_pct < 0.30 or ctx.ai_pct >= 0.70:
        return None

    confidence = "high" if ctx.auth_pct >= 0.50 else "medium"
    return Verdict(
        status=VerdictStatus.AUTHENTIC,
        is_authentic=True,
        confidence=confidence,
        explanation=f"Authentic ({confidence} confidence). {ctx.auth_pct:.0%} of frames verified as authentic."
    )


def rule_avg_confidence_fallback(ctx: RuleContext) -> Optional[Verdict]:
    """Rule 3 — Fallback for values outside the neutral band (±0.5)."""
    avg = ctx.metrics.average_confidence

    if avg > 0.5:
        return Verdict(
            status=VerdictStatus.AUTHENTIC,
            is_authentic=True,
            confidence="low",
            explanation=f"Authentic (low confidence). Average confidence score: {avg:.2f}."
        )

    if avg < -0.5:
        return Verdict(
            status=VerdictStatus.AI_GENERATED,
            is_authentic=False,
            confidence="low",
            explanation=f"AI-generated (low confidence). Average confidence score: {avg:.2f}."
        )
    return None


def rule_inconclusive(ctx: RuleContext) -> Optional[Verdict]:
    """Rule 4 — Guaranteed terminal fallback for mixed signals."""
    return Verdict(
        status=VerdictStatus.INCONCLUSIVE,
        is_authentic=False,
        confidence="inconclusive",
        explanation=f"Inconclusive. Mixed signals across frames (AI: {ctx.ai_pct:.0%}, Auth: {ctx.auth_pct:.0%}).",
        error="Inconclusive — manual review required."
    )


# ── Rule registry (Ordered by Priority) ───────────────────────────

RULES: list[RuleFn] = [
    rule_strong_ai,               # Rule 1
    rule_strong_authentic,        # Rule 2
    rule_avg_confidence_fallback, # Rule 3
    rule_inconclusive,            # Rule 4
]


# ── Engine entry point (Defensive & Fault-Tolerant) ────────────────

def run_decision_engine(
    video_id: str,
    metrics: VideoMetrics
) -> tuple[VideoAnalysisResult, VideoMetrics]:
    """
    Evaluates video metrics against the rule registry.
    Returns structured results and updated metrics for downstream services.
    """
    logger.info("[%s] Decision engine started", video_id)

    # Guard — Graceful degradation for zero-frame scenarios
    if metrics.total_valid_frames == 0:
        err_msg = "Zero valid frames after processing."
        metrics.analsysis_summary = err_msg
        return VideoAnalysisResult(
            video_id=video_id,
            is_authentic=False,
            explanation=err_msg,
            error=err_msg
        ), metrics

    try:
        total = metrics.total_valid_frames
        ctx = RuleContext(
            video_id=video_id,
            metrics=metrics,
            ai_pct=metrics.ai_frame_count / total,
            auth_pct=metrics.authentic_frame_count / total
        )

        # Priority execution: Stop at the first rule that fires
        verdict = next(v for rule in RULES if (v := rule(ctx)) is not None)

        # Populate structured metadata for the metrics object
        metrics.analsysis_summary = (
            f"{verdict.explanation} | Status: {verdict.status} | Confidence: {verdict.confidence}"
        )

        return VideoAnalysisResult(
            video_id=video_id,
            is_authentic=verdict.is_authentic,
            explanation=verdict.explanation,
            error=verdict.error
        ), metrics

    except Exception as e:
        # Non-crashing pipeline: Return safe failure state
        logger.error("[%s] Unexpected engine error: %s", video_id, e)
        fail_msg = "Decision engine encountered an unexpected error."
        return VideoAnalysisResult(
            video_id=video_id,
            is_authentic=False,
            explanation=fail_msg,
            error=str(e)
        ), metrics