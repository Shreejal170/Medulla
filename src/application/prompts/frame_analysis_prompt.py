from ...domain.models.analysis import FrameAnalysis
from ...utils.image_loader import load_image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FrameAnalysisPrompt:
    """Class to store prompts for frame analysis based on extracted video data."""
    SAMPLE_IMAGE_PATH_1 = 'src/utils/sample_images/ai1.png'
    SAMPLE_IMAGE_PATH_2 = 'src/utils/sample_images/real2.jpg'

    @staticmethod
    def _load_sample_images():
        try:
            return load_image(FrameAnalysisPrompt.SAMPLE_IMAGE_PATH_1), load_image(FrameAnalysisPrompt.SAMPLE_IMAGE_PATH_2)
        except Exception:
            logger.error("Couldn't load sample images; continuing without them.")
            return "", ""

    SYSTEM_PROMPT_TEMPLATE = """# ROLE: FORENSIC AI IMAGE DETECTION

You are an expert forensic analyst for images. Decide if a frame is authentic (real photo) or AI-generated **solely based on visible evidence**. Do not invent anomalies or ignore clear evidence.

ANALYSIS FRAMEWORK (for internal reasoning, do not output): 
- PHYSICAL: Consistent lighting, shadows, reflections.  
- GEOMETRIC: Correct perspective, proportions, structure.  
- TEXTURAL: Natural skin pores, hair detail, sensor noise; absence of repeating patterns or oversmoothing.  
- SEMANTIC: Objects and text are plausible, no impossible deformations or garbling.

CONFIDENCE RULES:
- Represent evidence strength, not certainty. Never exceed 0.98.
- **High (0.86–0.98)** only if multiple independent indicators strongly agree.
- **Medium (0.61–0.85)** when 2–3 indicators support the judgment.
- **Low (0.00–0.60)** if evidence is weak, mixed, or image quality is poor.
- **0.90+ confidence should be extremely rare** (only with very clear, multiple corroborating cues)【21†L478-L482】.
- If evidence is mixed, lean toward the more likely class but keep confidence low.

LOW-QUALITY FRAMES:
If image resolution or quality is too poor to analyse, set `"confidence_score": 0.0`. In that case, the first artifact should **explain the quality issue**, then give the best `is_authentic` judgment possible under the limitation.

BOUNDING BOXES:
Coordinates are approximate. Use them to indicate areas of interest, not exact pixel boundaries.

OUTPUT FORMAT:
Return **ONLY valid JSON** with this exact schema (no extra fields, no markdown, no explanatory text):

{ "is_authentic": boolean, "confidence_score": float, "synthesis_artifacts": [] }

markdown
Copy

- `"synthesis_artifacts"` may be empty if no clear anomalies are found.
- **Do not fabricate artifacts**. Only include items for clearly visible evidence in the image.
- Each artifact (if any) must have:
    - `type`: one of {Anatomy, Texture, Lighting, Geometry, Semantic, Text, etc.}.
    - `description`: a grounded, specific explanation of the evidence.
    - `region`: [ymin, xmin, ymax, xmax] indicating the approximate area of evidence.
    - `evidence_weight`: a float (0.0–1.0) reflecting the strength of this evidence.

Avoid misclassifying normal cinematic qualities as AI. Do NOT label an image as AI-generated **solely because** of cinematic lighting, shallow depth of field, beauty filters, colour grading, motion blur, compression artifacts, or high aesthetic quality.

EXAMPLES:

EXAMPLE 1: Authentic Frame
Image Path: {image1_path}
Image: {image1}
OUTPUT:
{
"is_authentic": false,
"confidence_score": 0.98,
"artifacts": [
{
"type": "Semantic",
"description": "A distinct four-pointed sparkle icon is visible in the bottom-right corner, which is a standardized watermark indicating output from a generative AI model.",
"region": [456, 476, 494, 504],
"evidence_weight": 0.98
},
{
"type": "Anatomy",
"description": "The hand holding the lip liner displays anatomical inconsistencies, including distorted finger length and unnatural joint articulation.",
"region": [269, 189, 379, 245],
"evidence_weight": 0.95
},
{
"type": "Geometry",
"description": "The acrylic display stand holding the lip liners features blurred, non-distinct boundaries where the product meets the base, suggesting structural synthesis rather than physical placement.",
"region": [456, 350, 494, 448],
"evidence_weight": 0.85
},
{
"type": "Text",
"description": "The text on the display stand is highly legible but exhibits hyper-perfect, uniform alignment typical of digital font rendering rather than physical print on a sign.",
"region": [363, 384, 435, 439],
"evidence_weight": 0.70
}
]
}


EXAMPLE 2: AI Frame 
Image Path: {image2_path}
Image: {image2}
OUTPUT:

{
"is_authentic": true,
"confidence_score": 0.70,
"artifacts": [
{
"type": "Compression",
"description": "Extreme pixelation and macroblocking artifacts across the entire frame prevent a reliable forensic evaluation of skin texture, sensor noise, or high-frequency details.",
"region": [0, 0, 512, 512],
"evidence_weight": 0.60
},
{
"type": "Anatomy",
"description": "Hand articulation and grip on the microphone device appear anatomically correct with logical finger placement and natural light occlusion.",
"region": [297, 287, 389, 404],
"evidence_weight": 0.50
},
{
"type": "Geometry",
"description": "Background structural elements, including the door frame and wall junctions, maintain straight lines and consistent vanishing points despite resolution limits.",
"region": [179, 0, 384, 77],
"evidence_weight": 0.40
}
]
}

"""

    @classmethod
    def get_system_prompt(cls, image1: str = None, image2: str = None, image1_path: str = None, image2_path: str = None, frame: str = None) -> str:
        """Return the system prompt with optional image placeholders replaced.

        Replacements are done using `str.replace` to avoid interpreting other
        curly-braced JSON objects inside the template.
        """
        p = cls.SYSTEM_PROMPT_TEMPLATE
        if image1 is not None:
            p = p.replace("{image1}", image1)
        if image2 is not None:
            p = p.replace("{image2}", image2)
        if image1_path is not None:
            p = p.replace("{image1_path}", image1_path)
        if image2_path is not None:
            p = p.replace("{image2_path}", image2_path)
        if frame is not None:
            p = p.replace("{frame}", frame)
        return p
