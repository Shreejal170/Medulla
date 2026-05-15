import json
import logging
from src.domain.models.analysis import (
    VideoAnalysisResult,
    FrameAnalysis,
    VideoMetrics,
    SynthesisArtifact,
)
from src.ports.output.llm_port import LlmPort
from src.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class DeepseekLlmAdapter(LlmPort):
    """Adapter class to integrate with the DeepSeek LLM via OpenAI compatible SDK."""

    def __init__(self, llm_client, model_name: str = "deepseek-v4-pro"):
        # llm_client is expected to be an instance of openai.AsyncClient 
        # configured with DeepSeek's base_url="https://api.deepseek.com"
        self.llm_client = llm_client
        self.model_name = model_name

    async def generate_frame_analysis(
        self, prompt: str, image_data: str, frame_id: str
    ) -> FrameAnalysis:
        """Generates results based on the provided data using DeepSeek."""
        try:
            # Note: For DeepSeek-VL or OpenAI-compatible vision endpoints, 
            # image_data must usually be a base64 encoded string.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + "\nRespond strictly in JSON format."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        }
                    ]
                }
            ]

            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"} # Enforces JSON output if supported
            )
            
            raw_text = response.choices[0].message.content

            # DeepSeek sometimes wraps JSON in markdown blocks even with json_object format
            if raw_text.startswith("```json"):
                raw_text = raw_text.strip("`").replace("json\n", "", 1)

            result = json.loads(raw_text)

            # Map LLM response artifacts to SynthesisArtifact objects
            artifacts = []
            for artifact_data in result.get("synthesis_artifacts", []):
                artifact = SynthesisArtifact(
                    artifact_type=artifact_data.get("type"),
                    description=artifact_data.get("description"),
                    region=artifact_data.get("region"),
                    evidence_weight=artifact_data.get("evidence_weight"),
                )
                artifacts.append(artifact)

            frame_analysis = FrameAnalysis(
                frame_id=frame_id,
                is_authentic=result.get("is_authentic", False),
                confidence_score=result.get("confidence_score", 0.0),
                synthesis_artifacts=artifacts,
            )
            return frame_analysis

        except Exception as e:
            logger.error(f"Error generating frame analysis with DeepSeek: {str(e)}", exc_info=True)
            return FrameAnalysis(frame_id="unknown", is_authentic=False, confidence_score=0.0, synthesis_artifacts=[])
        
        
    async def get_visual_evidence(self, analyses: list[FrameAnalysis]) -> list:
        """
        Stage 2 Forensic LLM Call: Extracts synthesis artifacts 
        (lighting, anatomy) and affected regions to justify an anomalous verdict.
        """
        try:
            # Filter for anomalous or suspicious frames to optimize token usage
            suspicious_frames = [f for f in analyses if f.confidence_score < -0.5]
            if not suspicious_frames:
                logger.info("No suspicious frames found for forensic extraction.")
                return []
                
            # Construct a dynamic prompt referencing the specific failed frames
            # DeepSeek excels at text reasoning, making it great for this step.
            forensic_prompt = (
                f"Analyze these {len(suspicious_frames)} flagged frames for structural anomalies. "
                "Identify visual artifact types (e.g., lighting mismatch, anatomical inconsistencies), "
                "provide a detailed text description, and isolate the coordinates via [ymin, xmin, ymax, xmax]. "
                "Respond strictly with a JSON object containing an 'artifacts' array."
            )
            
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": forensic_prompt}],
                response_format={"type": "json_object"}
            )
            
            raw_text = response.choices[0].message.content
            if raw_text.startswith("```json"):
                raw_text = raw_text.strip("`").replace("json\n", "", 1)
                
            result_json = json.loads(raw_text)
            artifacts = result_json.get("artifacts", [])
            
            logger.info(f"Successfully extracted {len(artifacts)} forensic artifacts via DeepSeek.")
            return artifacts

        except Exception as e:
            # Fault Tolerance: Return empty array so application service degrades gracefully
            logger.error(f"Error generating forensic artifact summary with DeepSeek: {str(e)}", exc_info=True)
            return []