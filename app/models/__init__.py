"""Domain models.

Plain dataclasses with no dependency on FastAPI, SQLite, or any search engine,
so the domain rules stay testable on their own.
"""

from app.models.analytics import FeedbackEvent, FeedbackEventType, SearchEvent
from app.models.job import Job, JobStatus, JobType
from app.models.product import EmbeddingStatus, Product, ProductImage
from app.models.source import Source, SourceStatus
from app.models.tenant import ApiKey, Tenant, TenantStatus
from app.models.vector import ModelVersion, VectorRecord

__all__ = [
    "ApiKey",
    "EmbeddingStatus",
    "FeedbackEvent",
    "FeedbackEventType",
    "Job",
    "JobStatus",
    "JobType",
    "ModelVersion",
    "Product",
    "ProductImage",
    "SearchEvent",
    "Source",
    "SourceStatus",
    "Tenant",
    "TenantStatus",
    "VectorRecord",
]
