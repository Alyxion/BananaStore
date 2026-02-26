from typing import Any

PROVIDER_CAPABILITIES: dict[str, dict[str, Any]] = {
    "azure_openai": {
        "label": "OpenAI",
        "qualities": ["auto", "low", "medium", "high"],
        "ratios": ["1:1", "3:2", "2:3"],
        "ratioSizes": {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536"},
        "formats": ["Photo"],
        "requiresKey": "AZURE_OPENAI_API_KEY",
    },
    "openai": {
        "label": "OpenAI",
        "qualities": ["auto", "low", "medium", "high"],
        "ratios": ["1:1", "3:2", "2:3"],
        "ratioSizes": {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536"},
        "formats": ["Photo", "Vector"],
        "formatQualities": {"Photo": ["auto", "low", "medium", "high"], "Vector": ["low", "medium", "high"]},
        "requiresKey": "OPENAI_API_KEY",
    },
    "google": {
        "label": "Google",
        "qualities": ["standard", "hd"],
        "ratios": ["1:1", "16:9", "9:16"],
        "ratioSizes": {"1:1": "1024x1024", "16:9": "1280x720", "9:16": "720x1280"},
        "formats": ["Photo"],
        "requiresKey": "GOOGLE_API_KEY",
    },
    "anthropic": {
        "label": "Anthropic",
        "qualities": ["low", "medium", "high"],
        "ratios": ["1:1", "3:2", "2:3"],
        "ratioSizes": {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536"},
        "formats": ["Vector"],
        "requiresKey": "ANTHROPIC_API_KEY",
    },
}
