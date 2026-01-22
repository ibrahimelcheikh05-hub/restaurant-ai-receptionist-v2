"""
Settings
========
Central configuration management.

Loads configuration from environment variables and provides
typed access to settings.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class VocodeSettings:
    """Vocode/Telephony settings."""
    
    # Twilio
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_phone_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")
    
    # Server
    base_url: str = os.getenv("BASE_URL", "https://localhost:8000")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))


@dataclass
class AISettings:
    """AI/LLM settings."""
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    openai_timeout: float = float(os.getenv("OPENAI_TIMEOUT", "10.0"))


@dataclass
class SpeechSettings:
    """Speech (STT/TTS) settings."""
    
    # Deepgram STT
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    
    # Google Cloud TTS
    google_application_credentials: str = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        ""
    )


@dataclass
class LanguageSettings:
    """Language detection and translation settings."""
    
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "en")
    supported_languages: list = os.getenv(
        "SUPPORTED_LANGUAGES",
        "en,es,ar"
    ).split(",")
    min_confidence: float = float(os.getenv("MIN_LANGUAGE_CONFIDENCE", "0.7"))
    translation_timeout: float = float(os.getenv("TRANSLATION_TIMEOUT", "2.0"))


@dataclass
class DatabaseSettings:
    """Database settings."""
    
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://localhost/restaurant"
    )
    pool_min_size: int = int(os.getenv("DB_POOL_MIN_SIZE", "5"))
    pool_max_size: int = int(os.getenv("DB_POOL_MAX_SIZE", "20"))


@dataclass
class SessionSettings:
    """Call session settings."""
    
    max_call_duration: float = float(os.getenv("MAX_CALL_DURATION", "1800"))
    max_silence_duration: float = float(os.getenv("MAX_SILENCE_DURATION", "30"))
    max_ai_response_time: float = float(os.getenv("MAX_AI_RESPONSE_TIME", "10"))
    greeting_timeout: float = float(os.getenv("GREETING_TIMEOUT", "5"))
    transfer_timeout: float = float(os.getenv("TRANSFER_TIMEOUT", "30"))


@dataclass
class FeatureFlags:
    """Feature flags."""
    
    enable_language_detection: bool = os.getenv(
        "ENABLE_LANGUAGE_DETECTION",
        "true"
    ).lower() == "true"
    
    enable_translation: bool = os.getenv(
        "ENABLE_TRANSLATION",
        "true"
    ).lower() == "true"
    
    enable_transfer: bool = os.getenv(
        "ENABLE_TRANSFER",
        "true"
    ).lower() == "true"
    
    enable_upsell: bool = os.getenv(
        "ENABLE_UPSELL",
        "true"
    ).lower() == "true"
    
    enable_sms: bool = os.getenv(
        "ENABLE_SMS",
        "true"
    ).lower() == "true"
    
    enable_memory: bool = os.getenv(
        "ENABLE_MEMORY",
        "true"
    ).lower() == "true"


@dataclass
class Settings:
    """Master settings container."""
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Sub-settings
    vocode: VocodeSettings = VocodeSettings()
    ai: AISettings = AISettings()
    speech: SpeechSettings = SpeechSettings()
    language: LanguageSettings = LanguageSettings()
    database: DatabaseSettings = DatabaseSettings()
    session: SessionSettings = SessionSettings()
    features: FeatureFlags = FeatureFlags()
    
    def __post_init__(self):
        """Initialize sub-settings."""
        self.vocode = VocodeSettings()
        self.ai = AISettings()
        self.speech = SpeechSettings()
        self.language = LanguageSettings()
        self.database = DatabaseSettings()
        self.session = SessionSettings()
        self.features = FeatureFlags()
    
    def validate(self) -> list:
        """
        Validate settings.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check required settings
        if not self.vocode.twilio_account_sid:
            errors.append("TWILIO_ACCOUNT_SID not set")
        
        if not self.vocode.twilio_auth_token:
            errors.append("TWILIO_AUTH_TOKEN not set")
        
        if not self.ai.openai_api_key:
            errors.append("OPENAI_API_KEY not set")
        
        return errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "vocode": {
                "base_url": self.vocode.base_url,
                "host": self.vocode.host,
                "port": self.vocode.port
            },
            "ai": {
                "model": self.ai.openai_model,
                "max_tokens": self.ai.openai_max_tokens,
                "temperature": self.ai.openai_temperature,
                "timeout": self.ai.openai_timeout
            },
            "language": {
                "default": self.language.default_language,
                "supported": self.language.supported_languages
            },
            "session": {
                "max_call_duration": self.session.max_call_duration,
                "max_silence_duration": self.session.max_silence_duration
            },
            "features": {
                "language_detection": self.features.enable_language_detection,
                "translation": self.features.enable_translation,
                "transfer": self.features.enable_transfer,
                "upsell": self.features.enable_upsell,
                "sms": self.features.enable_sms
            }
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get global settings instance.
    
    Returns:
        Settings instance
    """
    global _settings
    
    if _settings is None:
        _settings = Settings()
        
        # Validate
        errors = _settings.validate()
        if errors:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Settings validation errors: {errors}"
            )
    
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment.
    
    Returns:
        New settings instance
    """
    global _settings
    _settings = None
    return get_settings()
