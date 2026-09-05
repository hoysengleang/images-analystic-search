from app.core.errors import BadRequestError
from app.detection.base import DetectorProvider
from app.detection.providers.owlvit_onnx_provider import OWLViTONNXProvider

#: DETECTOR_PROVIDER value that turns query-side detection off entirely.
DETECTION_DISABLED = "none"


class DetectorProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, type[DetectorProvider]] = {}

    def register(self, provider_class: type[DetectorProvider]) -> None:
        provider_name = provider_class.provider_name.strip().lower()
        self._providers[provider_name] = provider_class

    def get(self, provider_name: str) -> type[DetectorProvider]:
        normalized_name = provider_name.strip().lower()
        try:
            return self._providers[normalized_name]
        except KeyError as exc:
            raise BadRequestError(
                message=f"Unknown detector provider: {provider_name}",
                code="UNKNOWN_DETECTOR_PROVIDER",
                details={"provider": provider_name},
            ) from exc

    def list_provider_names(self) -> list[str]:
        return sorted(self._providers)


detector_registry = DetectorProviderRegistry()
detector_registry.register(OWLViTONNXProvider)


def get_detector_registry() -> DetectorProviderRegistry:
    return detector_registry
