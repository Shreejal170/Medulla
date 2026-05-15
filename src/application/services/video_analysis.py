import logging
from typing import List
from src.application.services.decision_engine import run_decision_engine, VerdictStatus
from src.core.logging_config import setup_logging
from src.domain.models.analysis import VideoMetrics, FrameAnalysis
from src.domain.models.output import VideoAnalysisResult
from src.ports.output.llm_port import LlmPort

setup_logging()
logger = logging.getLogger(__name__)

class VideoAnalysisService:
    """
    Application Layer Orchestrator implementing Hexagonal Architecture boundaries.
    Co-ordinates the pure domain logic (Decision Engine) and external infrastructure ports.
    """
    def __init__(self, llm_gateway: LlmPort):
        self.llm_gateway = llm_gateway  # Port contract injected into the application

    async def execute_full_analysis(
        self, 
        video_id: str, 
        metrics: VideoMetrics, 
        analyses: List[FrameAnalysis]
    ) -> VideoAnalysisResult:
        """
        Coordinates the multi-stage forensic verification pipeline.
        Ensures explainability, fault tolerance, and graceful degradation.
        """
        logger.info("[%s] Orchestrating complete workflow logic", video_id)

        # Step 1: Execute Pure Domain Evaluation
        verdict, updated_metrics = run_decision_engine(video_id, metrics)

        # Base properties for the final payload contract
        final_explanation = verdict.explanation
        forensic_artifacts = []

        # Step 2: Conditional Stage 2 Forensic LLM Call (Explainability & Transparency)
        if verdict.status in [VerdictStatus.AI_GENERATED, VerdictStatus.INCONCLUSIVE]:
            try:
                logger.info("[%s] Positive/Mixed evaluation triggered Stage 2 LLM analysis", video_id)
                # Querying the outbound adapter contract
                forensic_artifacts = await self.llm_gateway.get_visual_evidence(analyses)
                final_explanation = self._append_artifacts_to_explanation(verdict.explanation, forensic_artifacts)
            except Exception as e:
                # Resilience: Infrastructure/API failure must NOT crash the mathematical result
                logger.error("[%s] Structural Stage 2 extraction encountered a partial failure: %s", video_id, e)
                final_explanation += " (Forensic artifact synthesis unavailable due to downstream connection error.)"

        return VideoAnalysisResult(
            video_id=video_id,
            is_authentic=verdict.is_authentic,
            explanation=final_explanation,
            error=verdict.error  # Non-null value transparently surfaces structural data context
        )

    def _append_artifacts_to_explanation(self, base_text: str, artifacts: List) -> str:
        """Assembles descriptive indicators and coordinates cleanly without leaking memory spaces."""
        if not artifacts:
            return base_text
            
        evidence_lines = ["\n\nSynthesis Artifacts Observed:"]
        for idx, item in enumerate(artifacts, 1):
            # Safe extraction assuming downstream structured model mapping or dict fallback
            description = getattr(item, 'description', item.get('description', 'Unknown anomaly'))
            region = getattr(item, 'region', item.get('region', 'N/A'))
            evidence_lines.append(f"- {idx}. {description} (Affected Region: {region})")
            
        return base_text + "\n".join(evidence_lines)
