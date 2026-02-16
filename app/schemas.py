from pydantic import BaseModel


class GenerateResponse(BaseModel):
    provider: str
    size: str
    quality: str
    ratio: str
    format: str
    image_data_url: str
    used_reference_images: int
    cost_usd: float | None = None


class FilenameRequest(BaseModel):
    description: str


class FilenameResponse(BaseModel):
    filename: str


class TranscriptionResponse(BaseModel):
    text: str


class DescribeImageRequest(BaseModel):
    image_data_url: str
    source_text: str = ""
    language: str = ""


class DescribeImageResponse(BaseModel):
    description: str


class TtsRequest(BaseModel):
    text: str
    language: str = ""


class CostLimitRequest(BaseModel):
    limit_usd: float | None = None


class CostSummary(BaseModel):
    total_usd: float
    limit_usd: float | None
    by_category: dict[str, float]
    by_provider: dict[str, float]
    entry_count: int
