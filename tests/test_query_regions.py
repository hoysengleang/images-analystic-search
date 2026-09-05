"""Query-side region search: cropping, detection, and folding regions together.

The point of all this is one behaviour: a photograph of a bag on a cluttered
table should be searched as a bag, not as a table. These tests pin the parts of
that which are easy to get subtly wrong and impossible to notice — a crop that
lands a pixel off, a region fold that drops the best score, or a detector
failure that turns into no results instead of a whole-image search.
"""

from pathlib import Path

import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import BadRequestError
from app.embedding.base import EmbeddingProvider
from app.schemas.collection import CollectionModelConfig
from app.schemas.image import CropRectangle, DetectedRegion
from app.schemas.search import SearchRequest, SearchResult
from app.services.collection_metadata_service import CollectionMetadataService
from app.services.search_service import SearchService
from app.utils.image_utils import MIN_CROP_PIXELS, crop_to_normalized_box

QUERY_SOURCE = {"type": "url", "value": "https://example.com/query.jpg"}


# --- Crop geometry ----------------------------------------------------


@pytest.mark.parametrize(
    ("image_size", "box", "expected_size"),
    [
        ((1000, 500), (0.31, 0.44, 0.22, 0.30), (220, 150)),
        ((1000, 500), (0.0, 0.0, 1.0, 1.0), (1000, 500)),
        ((1000, 500), (0.7, 0.5, 0.3, 0.5), (300, 250)),
        ((640, 640), (0.25, 0.25, 0.5, 0.5), (320, 320)),
    ],
)
def test_crop_maps_fractions_onto_pixels(image_size, box, expected_size) -> None:
    x, y, width, height = box
    image = Image.new("RGB", image_size)

    cropped = crop_to_normalized_box(image, x=x, y=y, width=width, height=height)

    assert cropped.size == expected_size


def test_crop_takes_the_region_the_caller_asked_for() -> None:
    """Size alone would pass even if the box were in the wrong corner."""
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    image.paste(Image.new("RGB", (50, 50), (255, 0, 0)), (50, 0))

    top_right = crop_to_normalized_box(image, x=0.5, y=0.0, width=0.5, height=0.5)
    top_left = crop_to_normalized_box(image, x=0.0, y=0.0, width=0.5, height=0.5)

    assert top_right.getpixel((10, 10)) == (255, 0, 0)
    assert top_left.getpixel((10, 10)) == (0, 0, 0)


def test_a_crop_with_almost_no_pixels_is_refused() -> None:
    """Usually means the caller sent pixels where the API asks for fractions."""
    image = Image.new("RGB", (1000, 500))

    with pytest.raises(BadRequestError) as exc_info:
        crop_to_normalized_box(image, x=0.0, y=0.0, width=0.005, height=0.5)

    assert exc_info.value.code == "CROP_TOO_SMALL"
    assert exc_info.value.details["min_pixels"] == MIN_CROP_PIXELS


# --- Fakes ------------------------------------------------------------


class FakeImageLoader:
    def __init__(self, size=(640, 480)) -> None:
        self.size = size
        self.sources = []

    def load_from_source(self, source):
        self.sources.append(source)
        return Image.new("RGB", self.size)


class RecordingEmbeddingProvider(EmbeddingProvider):
    """Returns a different vector per image size, so crops are distinguishable."""

    provider_name = "openclip"

    def __init__(self) -> None:
        super().__init__(model_name="ViT-B-32", model_pretrained="none", vector_size=3)
        self.embedded_sizes = []

    def embed_image(self, image: Image.Image) -> list[float]:
        self.embedded_sizes.append(image.size)
        return [float(image.width), float(image.height), 0.0]


class FakeEmbeddingManager:
    def __init__(self) -> None:
        self.provider = RecordingEmbeddingProvider()

    def get_provider(self, **kwargs):
        return self.provider


class ScriptedVectorService:
    """Returns a caller-supplied result set per search call, in order."""

    def __init__(self, result_sets) -> None:
        self.result_sets = list(result_sets)
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.result_sets) - 1)
        return list(self.result_sets[index])


class ScriptedDetector:
    def __init__(self, regions, *, error=None) -> None:
        self.regions = regions
        self.error = error
        self.calls = []

    def detect(self, image, *, prompts, max_regions, min_score):
        self.calls.append(
            {"prompts": list(prompts), "max_regions": max_regions, "min_score": min_score}
        )
        if self.error is not None:
            raise self.error
        return list(self.regions)


def hit(result_id: str, score: float) -> SearchResult:
    return SearchResult(id=result_id, score=score)


def region(x, y, width, height, score=0.9, label="a product") -> DetectedRegion:
    return DetectedRegion(
        box=CropRectangle(x=x, y=y, width=width, height=height),
        score=score,
        label=label,
    )


def make_service(
    tmp_path: Path,
    *,
    result_sets=((hit("img_001", 0.9),),),
    detector=None,
    image_size=(640, 480),
    **setting_overrides,
):
    metadata_service = CollectionMetadataService(tmp_path / "collections.json")
    metadata_service.save(
        collection_name="products",
        model=CollectionModelConfig(
            provider="openclip",
            name="ViT-B-32",
            pretrained="none",
            vector_size=3,
            distance="cosine",
        ),
    )
    image_loader = FakeImageLoader(image_size)
    embedding_manager = FakeEmbeddingManager()
    vector_service = ScriptedVectorService(result_sets)
    service = SearchService(
        settings=Settings(default_top_k=3, max_top_k=100, **setting_overrides),
        metadata_service=metadata_service,
        image_loader=image_loader,
        embedding_manager=embedding_manager,
        vector_service=vector_service,
        detector=detector,
    )
    return service, embedding_manager.provider, vector_service


def search(service, **fields) -> object:
    return service.search(
        SearchRequest.model_validate(
            {"collection_name": "products", "source": QUERY_SOURCE, **fields}
        )
    )


# --- Crop and detect through the service ------------------------------


def test_without_a_crop_the_whole_image_is_searched(tmp_path: Path) -> None:
    service, provider, _ = make_service(tmp_path)

    response = search(service)

    assert provider.embedded_sizes == [(640, 480)]
    assert response.query_regions == []
    assert response.results[0].matched_query_region is None


def test_a_crop_narrows_what_gets_embedded(tmp_path: Path) -> None:
    service, provider, _ = make_service(tmp_path)

    response = search(service, crop={"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5})

    assert provider.embedded_sizes == [(320, 240)]
    # The caller drew the box, so it comes back as the region that was searched.
    assert len(response.query_regions) == 1
    assert response.query_regions[0].box.width == 0.5
    assert response.results[0].matched_query_region == 0


def test_detection_searches_each_region_it_finds(tmp_path: Path) -> None:
    detector = ScriptedDetector([region(0.0, 0.0, 0.5, 0.5), region(0.5, 0.5, 0.5, 0.5)])
    service, provider, vector_service = make_service(tmp_path, detector=detector)

    response = search(service, detect=True)

    assert provider.embedded_sizes == [(320, 240), (320, 240)]
    assert len(vector_service.calls) == 2
    assert len(response.query_regions) == 2


def test_detection_falls_back_to_the_whole_image_when_it_finds_nothing(
    tmp_path: Path,
) -> None:
    """Refusing to answer would be worse than answering with a weaker query."""
    service, provider, vector_service = make_service(
        tmp_path, detector=ScriptedDetector([])
    )

    response = search(service, detect=True)

    assert provider.embedded_sizes == [(640, 480)]
    assert len(vector_service.calls) == 1
    assert response.query_regions == []
    assert response.results


def test_regions_too_small_to_embed_are_dropped_not_raised(tmp_path: Path) -> None:
    """A useless box from our own detector is not the caller's mistake."""
    detector = ScriptedDetector(
        [region(0.0, 0.0, 0.001, 0.001), region(0.2, 0.2, 0.5, 0.5)]
    )
    service, provider, _ = make_service(tmp_path, detector=detector)

    response = search(service, detect=True)

    assert provider.embedded_sizes == [(320, 240)]
    assert len(response.query_regions) == 1


def test_detect_prompts_default_to_the_configured_ones(tmp_path: Path) -> None:
    detector = ScriptedDetector([region(0.0, 0.0, 0.5, 0.5)])
    service, _, _ = make_service(
        tmp_path,
        detector=detector,
        detector_prompts=["a product"],
        detector_min_score=0.25,
        max_query_regions=2,
    )

    search(service, detect=True)
    search(service, detect=True, detect_prompt=["a handbag"])

    assert detector.calls[0]["prompts"] == ["a product"]
    assert detector.calls[0]["min_score"] == 0.25
    assert detector.calls[0]["max_regions"] == 2
    assert detector.calls[1]["prompts"] == ["a handbag"]


def test_crop_and_detect_together_is_refused(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path, detector=ScriptedDetector([]))

    with pytest.raises(BadRequestError) as exc_info:
        search(
            service,
            detect=True,
            crop={"x": 0.0, "y": 0.0, "width": 0.5, "height": 0.5},
        )

    assert exc_info.value.code == "CROP_AND_DETECT_CONFLICT"


def test_detect_without_a_configured_detector_says_so(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path, detector=None)

    with pytest.raises(BadRequestError) as exc_info:
        search(service, detect=True)

    assert exc_info.value.code == "DETECTION_NOT_AVAILABLE"


# --- Folding several regions into one ranked list ---------------------


def test_an_image_keeps_its_best_score_across_regions(tmp_path: Path) -> None:
    """The same max fold the specification mandates for grouping a product."""
    detector = ScriptedDetector([region(0.0, 0.0, 0.5, 0.5), region(0.5, 0.5, 0.5, 0.5)])
    service, _, _ = make_service(
        tmp_path,
        detector=detector,
        result_sets=[
            [hit("bag", 0.41), hit("mug", 0.88)],
            [hit("bag", 0.93), hit("mug", 0.30)],
        ],
    )

    response = search(service, detect=True, top_k=5)

    scores = {result.id: result.score for result in response.results}
    assert scores == {"bag": 0.93, "mug": 0.88}
    # Ranked by the folded score, not by the order the regions were searched.
    assert [result.id for result in response.results] == ["bag", "mug"]


def test_a_result_names_the_region_that_matched_it(tmp_path: Path) -> None:
    detector = ScriptedDetector([region(0.0, 0.0, 0.5, 0.5), region(0.5, 0.5, 0.5, 0.5)])
    service, _, _ = make_service(
        tmp_path,
        detector=detector,
        result_sets=[[hit("bag", 0.41)], [hit("bag", 0.93)]],
    )

    response = search(service, detect=True)

    # Region 1 produced the winning score, so that is the one reported.
    assert response.results[0].matched_query_region == 1


def test_several_regions_over_fetch_so_the_fold_has_slack(tmp_path: Path) -> None:
    detector = ScriptedDetector([region(0.0, 0.0, 0.5, 0.5), region(0.5, 0.5, 0.5, 0.5)])
    service, _, vector_service = make_service(
        tmp_path, detector=detector, search_candidate_multiplier=4
    )

    response = search(service, detect=True, top_k=5)

    assert [call["top_k"] for call in vector_service.calls] == [20, 20]
    # The caller is still told what they asked for, not what was fetched.
    assert response.top_k == 5


def test_a_single_region_does_not_over_fetch(tmp_path: Path) -> None:
    """Nothing can collapse with one query vector, so slack would be waste."""
    service, _, vector_service = make_service(tmp_path, search_candidate_multiplier=4)

    search(service, top_k=5)

    assert [call["top_k"] for call in vector_service.calls] == [5]


def test_the_fold_still_honours_top_k(tmp_path: Path) -> None:
    detector = ScriptedDetector([region(0.0, 0.0, 0.5, 0.5), region(0.5, 0.5, 0.5, 0.5)])
    service, _, _ = make_service(
        tmp_path,
        detector=detector,
        result_sets=[
            [hit(f"a{index}", 0.9 - index / 100) for index in range(10)],
            [hit(f"b{index}", 0.8 - index / 100) for index in range(10)],
        ],
    )

    response = search(service, detect=True, top_k=4)

    assert len(response.results) == 4


def test_min_score_still_reaches_every_region_search(tmp_path: Path) -> None:
    """It is a per-region threshold, applied before the fold sees anything."""
    detector = ScriptedDetector([region(0.0, 0.0, 0.5, 0.5), region(0.5, 0.5, 0.5, 0.5)])
    service, _, vector_service = make_service(tmp_path, detector=detector)

    search(service, detect=True, min_score=0.42)

    assert [call["min_score"] for call in vector_service.calls] == [0.42, 0.42]


def test_batch_search_is_unaffected_by_regions(tmp_path: Path) -> None:
    """Batch takes no region controls, so each source stays one whole image."""
    service, provider, vector_service = make_service(tmp_path)

    service.search_batch(
        collection_name="products",
        sources=[QUERY_SOURCE, QUERY_SOURCE],
        mode="separate",
        top_k=5,
    )

    assert provider.embedded_sizes == [(640, 480), (640, 480)]
    assert [call["top_k"] for call in vector_service.calls] == [5, 5]


def test_only_the_strongest_region_is_searched_by_default() -> None:
    """One region, deliberately, and the reason is a relevance measurement.

    The fold keeps each product's best score across regions, so it is monotonic:
    an extra region can raise a product's score but never lower it. A weak region
    therefore lifts whatever it happens to resemble. Measured on a bag
    photographed beside a laptop, searching the strongest region left the bag
    0.257 clear of the runner-up, while adding a second 0.09-confidence region
    lifted the laptop and cut the lead to 0.073 - worse than not detecting at
    all. Raising this is a "find everything in the photo" choice.
    """
    assert Settings(_env_file=None).max_query_regions == 1


def test_the_configured_region_cap_reaches_the_detector(tmp_path: Path) -> None:
    detector = ScriptedDetector([region(0.0, 0.0, 0.5, 0.5)])
    service, _, _ = make_service(tmp_path, detector=detector, max_query_regions=1)

    search(service, detect=True)

    assert detector.calls[0]["max_regions"] == 1
