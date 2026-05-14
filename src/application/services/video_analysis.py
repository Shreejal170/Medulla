from src.application.services.decision_engine import run_decision_engine
from src.domain.models.output import VideoAnalysisResult

class ForensicOrchestrator:
    def __init__(self, llm_gateway):
        self.llm_gateway = llm_gateway  # Port/Adapter for the Stage 2 LLM call

    async def execute_full_analysis(self, video_id, metrics, analyses):
        # 1. Decision Engine: Get the mathematical verdict (Rules 1-4)
        # This part handles 'Error Handling' if frames are zero
        verdict, updated_metrics = run_decision_engine(video_id, metrics)

        # 2. Explainability: Conditional Forensic LLM Call
        final_explanation = verdict.explanation
        forensic_artifacts = []

        if verdict.status in ["ai_generated", "inconclusive"]:
            try:
                # Stage 2 LLM Call: Extracts 'lighting mismatches', 'regions', etc.
                forensic_artifacts = await self.llm_gateway.get_visual_evidence(analyses)
                final_explanation = self._append_artifacts(verdict.explanation, forensic_artifacts)
            except Exception as e:
                # Resilience: Artifact failure does not crash the verdict
                logger.error(f"Forensic Stage 2 failed: {e}")
                final_explanation += " (Visual artifact synthesis unavailable due to API error.)"

        return VideoAnalysisResult(
            video_id=video_id,
            is_authentic=verdict.is_authentic,
            explanation=final_explanation,
            error=verdict.error  # Surfaces runtime errors in payload
        )

    def _append_artifacts(self, base_text, artifacts):
        """Appends descriptive indicators to justify the verdict."""
        if not artifacts: return base_text
        evidence = "\n\nSynthesis Artifacts Observed:\n" + "\n".join(
            [f"- {a.description} (Region: {a.region})" for a in artifacts]
        )
        return base_text + evidence