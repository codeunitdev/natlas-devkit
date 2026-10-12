"""N-ATLaS DevKit: SDK, OpenAI-compatible API and playground for N-ATLaS.

N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation
and Digital Economy, and powered by Awarri Technologies.
"""

from .client import (
    ATTRIBUTION,
    DEFAULT_MODEL_ID,
    LANGUAGES,
    ChatResult,
    GenerationConfig,
    NAtlas,
)

__version__ = "0.1.0"
__all__ = [
    "NAtlas",
    "GenerationConfig",
    "ChatResult",
    "LANGUAGES",
    "DEFAULT_MODEL_ID",
    "ATTRIBUTION",
]
