from ...utils.image_loader import load_image
import logging

logger = logging.getLogger(__name__)


class FrameAnalysisPrompt:

    SAMPLE_IMAGE_PATH_1 = "src/utils/sample_images/ai1.png"
    SAMPLE_IMAGE_PATH_2 = "src/utils/sample_images/real2.jpg"

    SYSTEM_PROMPT = """
You are an expert forensic AI image analyst.

Return ONLY valid JSON.
"""

    @staticmethod
    def _load_sample_images():
        try:
            return (
                load_image(FrameAnalysisPrompt.SAMPLE_IMAGE_PATH_1),
                load_image(FrameAnalysisPrompt.SAMPLE_IMAGE_PATH_2),
            )
        except Exception:
            logger.error("Couldn't load sample images.")
            return "", ""

    @classmethod
    def build_messages(cls):

        image_1_b64, image_2_b64 = cls._load_sample_images()

        return [
            {
                "role": "system",
                "content": cls.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [

                    # Example 1 — authentic
                    {
                        "type": "text",
                        "text": "EXAMPLE 1: Analyze this frame."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_1_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": """
OUTPUT:
{
    "is_authentic": true,
    "confidence_score": 0.94,
    "synthesis_artifacts": []
}
"""
                    },

                    # Example 2 — AI generated
                    {
                        "type": "text",
                        "text": "EXAMPLE 2: Analyze this frame."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_2_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": """
OUTPUT:
{
    "is_authentic": false,
    "confidence_score": 0.97,
    "synthesis_artifacts": []
}
"""
                    },
                ],
            },
        ]
