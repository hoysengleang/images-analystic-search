"""Shared onnxruntime plumbing for the providers that run exported models.

The embedding provider and the object detector open sessions and load CLIP
tokenizers in exactly the same way; only the model and the error codes differ.
Keeping the mechanics here means one place understands execution-provider
selection and the awkward details, such as ONNX Runtime having no int64 ArgMax
on its CPU provider.

Error codes are passed in by the caller rather than derived, so every code a
client can receive is still greppable at the call site that raises it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.errors import ServiceUnavailableError

#: ONNX tensor types an integer input may declare, mapped to the NumPy dtype
#: the session expects to be handed.
INTEGER_DTYPES = {"tensor(int32)": np.int32, "tensor(int64)": np.int64}

#: The two spellings the end-of-text token carries across CLIP tokenizers.
END_OF_TEXT_TOKENS = ("<|endoftext|>", "<end_of_text>")

#: Preference order. The accelerator is used when the installed runtime has one,
#: which mirrors how the PyTorch providers pick a device.
EXECUTION_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def load_session(model_path: Path, *, missing_code: str, failed_code: str) -> Any:
    """Open an ONNX model, or explain precisely why it could not be opened."""
    try:
        import onnxruntime
    except ImportError as exc:
        raise ServiceUnavailableError(
            message=(
                "This provider needs onnxruntime; install it with "
                "pip install -r requirements-onnx.txt"
            ),
            code="ONNX_RUNTIME_NOT_INSTALLED",
        ) from exc

    if not model_path.is_file():
        raise ServiceUnavailableError(
            message="ONNX model file not found",
            code=missing_code,
            details={"path": str(model_path)},
        )

    available = onnxruntime.get_available_providers()
    execution_providers = [name for name in EXECUTION_PROVIDERS if name in available]

    try:
        return onnxruntime.InferenceSession(
            str(model_path),
            providers=execution_providers or None,
        )
    except Exception as exc:
        raise ServiceUnavailableError(
            message="Could not load ONNX model",
            code=failed_code,
            details={"path": str(model_path), "error": str(exc)},
        ) from exc


def load_clip_tokenizer(tokenizer_path: Path, *, failed_code: str) -> tuple[Any, int]:
    """Return a CLIP tokenizer and the id of its end-of-text token."""
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ServiceUnavailableError(
            message=(
                "Text input needs the tokenizers package; install it with "
                "pip install -r requirements-onnx.txt"
            ),
            code="ONNX_TOKENIZER_NOT_INSTALLED",
        ) from exc

    if not tokenizer_path.is_file():
        raise ServiceUnavailableError(
            message="Tokenizer file not found",
            code=failed_code,
            details={"path": str(tokenizer_path)},
        )

    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:
        raise ServiceUnavailableError(
            message="Could not load the tokenizer",
            code=failed_code,
            details={"path": str(tokenizer_path), "error": str(exc)},
        ) from exc

    end_of_text_id = next(
        (
            token_id
            for token_id in (tokenizer.token_to_id(token) for token in END_OF_TEXT_TOKENS)
            if token_id is not None
        ),
        None,
    )
    if end_of_text_id is None:
        raise ServiceUnavailableError(
            message="Tokenizer has no end-of-text token, so it is not a CLIP one",
            code=failed_code,
            details={
                "path": str(tokenizer_path),
                "expected_one_of": list(END_OF_TEXT_TOKENS),
            },
        )

    return tokenizer, end_of_text_id


def integer_dtype_of(session: Any, *, index: int = 0) -> Any:
    """Match the integer width an input was exported with.

    ONNX Runtime has no int64 ArgMax on its CPU provider, and CLIP's text
    encoder argmaxes token ids to locate end-of-text, so exports are normally
    int32. Reading the declared type rather than assuming one keeps an int64
    export working wherever it does run.
    """
    return INTEGER_DTYPES.get(session.get_inputs()[index].type, np.int64)


def trailing_dimension_of(session: Any, *, index: int, axis: int, default: int) -> int:
    """Read a fixed dimension off an input, falling back when it is dynamic.

    Exports pin everything but the batch axis, so this is normally a plain
    integer; a fully dynamic export gets the caller's default.
    """
    shape = session.get_inputs()[index].shape
    if len(shape) < abs(axis):
        return default

    size = shape[axis]
    return size if isinstance(size, int) and size > 0 else default
