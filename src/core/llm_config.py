import logging
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class LLMSettings(BaseSettings):
    """
    Configuration for LLM Services.
    Pydantic automatically reads these from environment variables or a .env file.
    """
    # Pydantic makes this strictly required by default.
    # If it is missing from the .env, the app will safely crash on startup.
    google_gemini_api_key: str
    # deepseek_api_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Safely ignores other unrelated variables in the .env file
    )

try:
    # Initialize a singleton instance to be imported across the AI services
    llm_settings = LLMSettings()
    logger.info("LLM Settings loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load LLM settings. Check your .env file. Error: {e}")
    raise