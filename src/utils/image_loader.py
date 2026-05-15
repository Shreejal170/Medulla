from PIL import Image
import base64
from io import BytesIO


def load_image(file_path: str) -> str:
    """Load an image file and return a base64-encoded PNG string.

    This function is synchronous to keep usage simple across the codebase.
    It returns an empty string on error.
    """
    try:
        with Image.open(file_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return ""
