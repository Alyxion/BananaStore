"""Global configuration singleton for BananaStore.

Reads API keys from environment variables by default.  When embedded
(e.g. from the NiceGUI sample host), the caller can populate the
singleton *before* the first request so that keys don't have to live
in the process environment.

    from app.config import settings
    settings.OPENAI_API_KEY = "sk-..."
"""

import os
from typing import Optional


class Settings:
    """Lightweight mutable config — one global instance."""

    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    COST_LIMIT_USD: Optional[float] = None

    def get(self, name: str) -> Optional[str]:
        """Return the attribute value if set, otherwise fall back to env."""
        value = getattr(self, name, None)
        if value is not None:
            return value if not isinstance(value, float) else str(value)
        return os.getenv(name)


settings = Settings()
