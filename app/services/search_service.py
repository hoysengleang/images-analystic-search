from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from app.core.config import Settings
from app.core.errors import BadRequestError
from app.detection.base import DetectorProvider
from app.embedding.base import EmbeddingProvider
from app.embedding.manager import EmbeddingManager, get_configured_provider
from app.schemas.image import CropRectangle, DetectedRegion, ImageSource
from app.schemas.search import (
    BatchSearchGroup,
    BatchSearchResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.collection_metadata_service import (
    CollectionMetadata,
    CollectionMetadataService,
)
from app.services.image_loader import ImageLoader
from app.services.vector_service import VectorService
from app.utils.image_utils import (
    MIN_CROP_PIXELS,
    crop_to_normalized_box,
    validate_and_load_image,
)

if TYPE_CHECKING:
    from PIL import Image

#: Score reported for a region the caller drew themselves. They did not guess.
EXPLICIT_CROP_SCORE = 1.0


@dataclass(frozen=True)
class QueryVectors:
    """What one query image turned into, and where each vector came from.

    ``regions`` is either empty, meaning the whole image was embedded, or the
    same length as ``vectors`` and in the same order.
    """

    vectors: list[list[float]]
    regions: list[DetectedRegion] = field(default_factory=list)

    @property
    def is_whole_image(self) -> bool:
        return not self.regions


class SearchService:
    def __init__(
        self,
        *,
        settings: Settings,
        metadata_service: CollectionMetadataService,
        image_loader: ImageLoader,
        embedding_manager: EmbeddingManager,
        vector_service: VectorService,
        detector: Optional[DetectorProvider] = None,
    ) -> None:
        self.settings = settings
        self.metadata_service = metadata_service
        self.image_loader = image_loader
        self.embedding_manager = embedding_manager
        self.vector_service = vector_service
        # None when DETECTOR_PROVIDER is off, which is the default. Detection is
        # query-side only, so it can be switched on without reindexing.
        self.detector = detector

    def search(self, request: SearchRequest) -> SearchResponse:
        collection_metadata, embedding_provider = self._collection_context(
            request.collection_name
        )
        query = self._embed_source(
            embedding_provider,
            request.source,
            crop=request.crop,
            detect=request.detect,
            detect_prompt=request.detect_prompt,
        )
        return self._search_response(
            collection_metadata=collection_metadata,
            query=query,
            top_k=request.top_k,
            filters=request.filters,
            min_score=request.min_score,
        )

    def search_upload(
        self,
        *,
        collection_name: str,
        image_bytes: bytes,
        filename: str,
        top_k: Optional[int],
        min_score: Optional[float] = None,
        crop: Optional[CropRectangle] = None,
        detect: bool = False,
        detect_prompt: Optional[list[str]] = None,
    ) -> SearchResponse:
        collection_metadata, embedding_provider = self._collection_context(
            collection_name
        )

        validated_image = validate_and_load_image(
            image_bytes,
            max_size_mb=self.settings.max_image_size_mb,
            extension=filename,
            allowed_extensions=self.settings.allowed_image_extensions,
            max_pixels=self.settings.max_image_pixels,
        )
        try:
            query = self._query_from_image(
                embedding_provider,
                validated_image.image,
                crop=crop,
                detect=detect,
                detect_prompt=detect_prompt,
            )
        finally:
            validated_image.image.close()
        return self._search_response(
            collection_metadata=collection_metadata,
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

    def search_text(
        self,
        *,
        collection_name: str,
        query: str,
        top_k: Optional[int],
        min_score: Optional[float] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> SearchResponse:
        """Search stored images with words instead of a query image."""
        collection_metadata, embedding_provider = self._collection_context(
            collection_name
        )

        return self._search_response(
            collection_metadata=collection_metadata,
            query=QueryVectors([embedding_provider.embed_text(query)]),
            top_k=top_k,
            filters=filters,
            min_score=min_score,
        )

    def search_hybrid(
        self,
        *,
        collection_name: str,
        source: ImageSource,
        query: str,
        text_weight: float,
        top_k: Optional[int],
        min_score: Optional[float] = None,
        filters: Optional[dict[str, Any]] = None,
        crop: Optional[CropRectangle] = None,
        detect: bool = False,
        detect_prompt: Optional[list[str]] = None,
    ) -> SearchResponse:
        """Search with an image steered by a text description.

        Both encoders write into one vector space, so a weighted blend of the
        two query vectors expresses "this image, but ..." in a single search.
        """
        collection_metadata, embedding_provider = self._collection_context(
            collection_name
        )

        image_query = self._embed_source(
            embedding_provider,
            source,
            crop=crop,
            detect=detect,
            detect_prompt=detect_prompt,
        )
        text_vector = embedding_provider.embed_text(query)

        # Every region gets the same nudge, so "this bag, but in blue" works
        # whether one region was found or three.
        blended = QueryVectors(
            vectors=[
                self._blend_vectors(image_vector, text_vector, text_weight)
                for image_vector in image_query.vectors
            ],
            regions=image_query.regions,
        )

        return self._search_response(
            collection_metadata=collection_metadata,
            query=blended,
            top_k=top_k,
            filters=filters,
            min_score=min_score,
        )

    def search_batch(
        self,
        *,
        collection_name: str,
        sources: list[ImageSource],
        mode: str,
        top_k: Optional[int],
        min_score: Optional[float] = None,
        filters: Optional[dict[str, object]] = None,
    ) -> BatchSearchResponse:
        resolved_top_k = self._resolve_top_k(top_k)

        collection_metadata, embedding_provider = self._collection_context(
            collection_name
        )
        # Batch search does not take a crop or detection: with no region
        # controls each source yields exactly one whole-image vector.
        vectors = [
            self._embed_source(embedding_provider, source).vectors[0]
            for source in sources
        ]

        if mode == "average":
            average_vector = self._normalize_vector(self._average_vectors(vectors))
            results = self.vector_service.search(
                collection_name=collection_metadata.collection_name,
                vector=average_vector,
                top_k=resolved_top_k,
                filters=filters or {},
                min_score=min_score,
            )
            return BatchSearchResponse(
                collection_name=collection_metadata.collection_name,
                mode="average",
                top_k=resolved_top_k,
                results=results,
                groups=[],
            )

        groups = [
            BatchSearchGroup(
                source_index=index,
                source=source,
                results=self.vector_service.search(
                    collection_name=collection_metadata.collection_name,
                    vector=vector,
                    top_k=resolved_top_k,
                    filters=filters or {},
                    min_score=min_score,
                ),
            )
            for index, (source, vector) in enumerate(zip(sources, vectors))
        ]
        return BatchSearchResponse(
            collection_name=collection_metadata.collection_name,
            mode="separate",
            top_k=resolved_top_k,
            results=[],
            groups=groups,
        )

    def _collection_context(
        self,
        collection_name: str,
    ) -> tuple[CollectionMetadata, EmbeddingProvider]:
        collection_metadata = self.metadata_service.get(collection_name)
        return (
            collection_metadata,
            get_configured_provider(self.embedding_manager, collection_metadata),
        )

    def _search_response(
        self,
        *,
        collection_metadata: CollectionMetadata,
        query: QueryVectors,
        top_k: Optional[int],
        min_score: Optional[float] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> SearchResponse:
        resolved_top_k = self._resolve_top_k(top_k)
        results = self._search_all_regions(
            collection_name=collection_metadata.collection_name,
            query=query,
            resolved_top_k=resolved_top_k,
            filters=filters or {},
            min_score=min_score,
        )
        return SearchResponse(
            collection_name=collection_metadata.collection_name,
            top_k=resolved_top_k,
            results=results,
            query_regions=query.regions,
        )

    def _search_all_regions(
        self,
        *,
        collection_name: str,
        query: QueryVectors,
        resolved_top_k: int,
        filters: dict[str, Any],
        min_score: Optional[float],
    ) -> list[SearchResult]:
        """Search once per query region and fold the hits into one ranked list.

        An image keeps its single best score across the regions, which is the
        same `max` fold section 13 of the specification mandates for grouping a
        product's images. Doing it that way here means the rule does not have to
        be redone when product grouping lands above it.
        """
        candidate_k = self._candidate_top_k(resolved_top_k, len(query.vectors))
        best: dict[str, SearchResult] = {}

        for region_index, vector in enumerate(query.vectors):
            for result in self.vector_service.search(
                collection_name=collection_name,
                vector=vector,
                top_k=candidate_k,
                filters=filters,
                min_score=min_score,
            ):
                previous = best.get(result.id)
                if previous is not None and previous.score >= result.score:
                    continue
                best[result.id] = (
                    result
                    if query.is_whole_image
                    else result.model_copy(update={"matched_query_region": region_index})
                )

        # Ties break on id so the same query always returns the same order.
        ranked = sorted(best.values(), key=lambda hit: (-hit.score, hit.id))
        return ranked[:resolved_top_k]

    def _candidate_top_k(self, resolved_top_k: int, region_count: int) -> int:
        """How many hits to pull per region before the fold discards duplicates.

        With one query vector nothing can collapse, so over-fetching would only
        move bytes. With several, the same image can be the best match for more
        than one region, and without slack the merged list comes up short.
        """
        if region_count <= 1:
            return resolved_top_k
        return resolved_top_k * self.settings.search_candidate_multiplier

    def _embed_source(
        self,
        embedding_provider: EmbeddingProvider,
        source: ImageSource,
        *,
        crop: Optional[CropRectangle] = None,
        detect: bool = False,
        detect_prompt: Optional[list[str]] = None,
    ) -> QueryVectors:
        image = self.image_loader.load_from_source(source)
        try:
            return self._query_from_image(
                embedding_provider,
                image,
                crop=crop,
                detect=detect,
                detect_prompt=detect_prompt,
            )
        finally:
            image.close()

    def _query_from_image(
        self,
        embedding_provider: EmbeddingProvider,
        image: Image.Image,
        *,
        crop: Optional[CropRectangle],
        detect: bool,
        detect_prompt: Optional[list[str]],
    ) -> QueryVectors:
        """Decide what part of the query image to search, and embed it."""
        if crop is not None and detect:
            raise BadRequestError(
                message=(
                    "Send either a crop or detect, not both: a crop already "
                    "says which region to search"
                ),
                code="CROP_AND_DETECT_CONFLICT",
            )

        if crop is not None:
            return QueryVectors(
                vectors=[self._embed_region(embedding_provider, image, crop)],
                regions=[DetectedRegion(box=crop, score=EXPLICIT_CROP_SCORE)],
            )

        if detect:
            regions = self._detect_regions(image, detect_prompt)
            if regions:
                return QueryVectors(
                    vectors=[
                        self._embed_region(embedding_provider, image, region.box)
                        for region in regions
                    ],
                    regions=regions,
                )
            # Finding nothing is an ordinary answer, not a failure. Searching
            # the whole image is a worse query but a much better outcome than
            # refusing to answer, and the empty query_regions says what happened.

        return QueryVectors([embedding_provider.embed_image(image)])

    def _detect_regions(
        self,
        image: Image.Image,
        detect_prompt: Optional[list[str]],
    ) -> list[DetectedRegion]:
        if self.detector is None:
            raise BadRequestError(
                message=(
                    "This server has no object detector configured; send an "
                    "explicit crop instead"
                ),
                code="DETECTION_NOT_AVAILABLE",
            )

        regions = self.detector.detect(
            image,
            prompts=detect_prompt or self.settings.detector_prompts,
            max_regions=self.settings.max_query_regions,
            min_score=self.settings.detector_min_score,
        )
        return self._large_enough(image, regions)

    @staticmethod
    def _large_enough(
        image: Image.Image,
        regions: Sequence[DetectedRegion],
    ) -> list[DetectedRegion]:
        """Drop boxes with too few pixels to embed.

        A caller who sends a tiny crop gets an error, because they asked for
        something impossible. A detector that returns one is the server's own
        doing, so the region is discarded quietly instead.
        """
        width, height = image.size
        return [
            region
            for region in regions
            if region.box.width * width >= MIN_CROP_PIXELS
            and region.box.height * height >= MIN_CROP_PIXELS
        ]

    @staticmethod
    def _embed_region(
        embedding_provider: EmbeddingProvider,
        image: Image.Image,
        box: CropRectangle,
    ) -> list[float]:
        cropped = crop_to_normalized_box(image, **box.model_dump())
        try:
            return embedding_provider.embed_image(cropped)
        finally:
            cropped.close()

    def _resolve_top_k(self, top_k: Optional[int]) -> int:
        """Fall back to DEFAULT_TOP_K and refuse anything above MAX_TOP_K."""
        resolved_top_k = top_k or self.settings.default_top_k

        if resolved_top_k > self.settings.max_top_k:
            raise BadRequestError(
                message="top_k exceeds the configured maximum",
                code="TOP_K_TOO_LARGE",
                details={
                    "top_k": resolved_top_k,
                    "max_top_k": self.settings.max_top_k,
                },
            )

        return resolved_top_k

    def _blend_vectors(
        self,
        image_vector: list[float],
        text_vector: list[float],
        text_weight: float,
    ) -> list[float]:
        image_weight = 1.0 - text_weight
        return self._normalize_vector(
            [
                image_weight * image_value + text_weight * text_value
                for image_value, text_value in zip(image_vector, text_vector)
            ]
        )

    def _average_vectors(self, vectors: list[list[float]]) -> list[float]:
        vector_count = len(vectors)
        dimension = len(vectors[0])
        return [
            sum(vector[index] for vector in vectors) / vector_count
            for index in range(dimension)
        ]

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            raise BadRequestError(
                message="Query vector cannot be zero",
                code="ZERO_QUERY_VECTOR",
                details={},
            )
        return [value / norm for value in vector]
