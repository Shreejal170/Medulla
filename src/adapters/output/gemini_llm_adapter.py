from src.domain.models.analysis import (
    FrameAnalysis,
    SynthesisArtifact,
)
from src.ports.output.llm_port import LlmPort
from src.core.logging_config import setup_logging
from google.genai import types
import base64
import logging
import json
import re

setup_logging()
logger = logging.getLogger(__name__)


class GeminiLlmAdapter(LlmPort):
    """Adapter class to integrate with Google Gemini LLM using the official GenAI SDK.

    Attrs:
        llm_client: An instance of the GenAI Client configured with the appropriate API key.
        model_name: The specific Gemini model to use for analysis (default: "gemini-3.1-flash-lite").
    """

    def __init__(self, llm_client, model_name: str = "gemini-3.1-flash-lite"):
        """Initialize the Gemini LLM Adapter with the given client and model name.
        Args:
            llm_client: An instance of the GenAI Client configured with the appropriate API key.
            model_name: The specific Gemini model to use for analysis (default: "gemini-3.1-flash-lite").
        """
        self.llm_client = llm_client
        self.model_name = model_name

    def _build_contents(self, prompt: str, image_data) -> list[types.Content]:
        """
        Converts raw prompt + image_data into Gemini-native types.Content objects.
        Handles base64 strings, raw bytes, and file paths.

        Args:
            prompt: The text prompt to guide the LLM's analysis.
            image_data: The raw image data (base64 string, bytes, or file path) of the frame to analyze.
        Returns:
            A list of types.Content objects formatted for Gemini input.
        """
        # --- Build the image Part ---
        if isinstance(image_data, bytes):
            image_part = types.Part.from_bytes(
                data=image_data,
                mime_type="image/jpeg",
            )
        elif isinstance(image_data, str):
            # Could be a base64 string or a file path
            try:
                raw_bytes = base64.b64decode(image_data)
                image_part = types.Part.from_bytes(
                    data=raw_bytes,
                    mime_type="image/jpeg",
                )
            except Exception:
                # Treat as file path
                with open(image_data, "rb") as f:
                    image_part = types.Part.from_bytes(
                        data=f.read(),
                        mime_type="image/jpeg",
                    )
        else:
            # Already a Gemini Part or Image — pass through
            image_part = image_data

        return [
            # Gemini has no "system" role in contents — prepend system instruction
            # as a user turn, or use system_instruction in config (see below).
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    image_part,
                ],
            )
        ]

    @staticmethod
    def _clean_schema(schema: dict) -> dict:
        """
        Recursively remove 'examples' keys from a Pydantic JSON schema
        because Google GenAI's Schema type does not accept them.


        Args:
            schema: A JSON schema dictionary potentially containing 'examples' keys.
        Returns:
            A cleaned JSON schema dictionary with all 'examples' keys removed.
        """
        if isinstance(schema, dict):
            return {
                k: GeminiLlmAdapter._clean_schema(v)
                for k, v in schema.items()
                if k != "examples"
            }
        if isinstance(schema, list):
            return [GeminiLlmAdapter._clean_schema(item) for item in schema]
        return schema

    def _parse_response(self, response) -> dict:
        """
        Parses the LLM response into a dictionary.
        Args:
            response: The raw response from the LLM.
        Returns:
            A dictionary containing the parsed response.
        """
        if hasattr(response, "parsed") and response.parsed is not None:
            parsed = response.parsed
            return parsed if isinstance(parsed, dict) else parsed.__dict__
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    async def generate_frame_analysis(
        self, prompt: str, image_data, frame_id: str
    ) -> FrameAnalysis:
        """
        Generates results based on the provided data using Google Gemini.
        Args:
            prompt: The text prompt to guide the LLM's analysis.
            image_data: The raw image data (base64 string, bytes, or file path) of the frame to analyze.
            frame_id: The unique identifier for the frame being analyzed.
        Returns:
            A FrameAnalysis object containing the analysis results.
        """

        try:
            contents = self._build_contents(prompt, image_data)

            response = self.llm_client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=self._clean_schema(
                        FrameAnalysis.model_json_schema()
                    ),
                ),
            )

            result = self._parse_response(response)

            artifacts = [
                SynthesisArtifact(
                    artifact_type=a.get("artifact_type") or a.get("type"),
                    description=a.get("description"),
                    region=a.get("region"),
                    evidence_weight=a.get("evidence_weight"),
                )
                for a in result.get("synthesis_artifacts") or []
            ]

            frame_analysis = FrameAnalysis(
                frame_id=frame_id,
                is_authentic=result.get("is_authentic", False),
                confidence_score=result.get("confidence_score", 0.0),
                synthesis_artifacts=artifacts,
            )

            logger.debug("Frame analysis for '%s': %s", frame_id, frame_analysis)
            return frame_analysis

        except Exception as e:
            logger.error(
                f"Error generating frame analysis for frame '{frame_id}': {e}",
                exc_info=True,
            )
            return FrameAnalysis(
                frame_id=frame_id,
                is_authentic=False,
                confidence_score=0.0,
                synthesis_artifacts=[],
            )

    def get_visual_evidence(
        self, analyses: list[FrameAnalysis]
    ) -> list[SynthesisArtifact]:
        """
        Generates a forensic artifact summary based on the analyses of multiple frames.
        Args:
            analyses: A list of FrameAnalysis objects to evaluate for forensic evidence.
        Returns:
            A list of SynthesisArtifact objects representing the extracted forensic evidence.
        """

        try:
            suspicious_frames = [f for f in analyses if f.confidence_score < -0.5]
            if not suspicious_frames:
                logger.info("No suspicious frames found for forensic extraction.")
                return []

            forensic_prompt = (
                f"Analyze these {len(suspicious_frames)} flagged frames for structural anomalies. "
                "Identify visual artifact types (e.g., lighting mismatch, anatomical inconsistencies), "
                "provide a detailed text description, and isolate the region coordinates via [x1, y1, x2, y2]."
            )

            envelope_schema = {
                "type": "object",
                "properties": {
                    "artifacts": {
                        "type": "array",
                        "items": SynthesisArtifact.model_json_schema(),
                    }
                },
                "required": ["artifacts"],
            }

            response = self.llm_client.models.generate_content(
                model=self.model_name,
                contents=forensic_prompt,  # No image needed here
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=envelope_schema,
                ),
            )

            result = self._parse_response(response)
            artifacts = [
                SynthesisArtifact(
                    artifact_type=a.get("artifact_type") or a.get("type"),
                    description=a.get("description"),
                    region=a.get("region"),
                    evidence_weight=a.get("evidence_weight"),
                )
                for a in result.get("artifacts", [])
            ]

            logger.info(f"Extracted {len(artifacts)} forensic artifacts.")
            return artifacts

        except Exception as e:
            logger.error(
                f"Error generating forensic artifact summary: {e}", exc_info=True
            )
            return []
