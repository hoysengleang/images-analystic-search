from pathlib import Path

from app.core.errors import BadRequestError


def resolve_safe_image_path(path_value: str, allowed_root: Path) -> Path:
    if not path_value or not path_value.strip():
        raise BadRequestError(
            message="Image path is required",
            code="INVALID_IMAGE_PATH",
            details={},
        )

    allowed_root_resolved = allowed_root.resolve()
    candidate = Path(path_value.strip())

    if not candidate.is_absolute():
        candidate = allowed_root_resolved / candidate

    resolved_candidate = candidate.resolve()

    if not _is_relative_to(resolved_candidate, allowed_root_resolved):
        raise BadRequestError(
            message="Image path must be inside the allowed image root",
            code="IMAGE_PATH_NOT_ALLOWED",
            details={
                "path": path_value,
                "allowed_root": str(allowed_root_resolved),
            },
        )

    return resolved_candidate


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
