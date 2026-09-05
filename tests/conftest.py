import pytest

from app.db import connect, migrate
from app.main import app


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    """Keep one test's dependency overrides from leaking into the next."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def db(tmp_path):
    """A migrated SQLite database on disk, one per test."""
    connection = connect(tmp_path / "app.db")
    migrate(connection)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def clip_tokenizer_file(tmp_path_factory):
    """A tiny stand-in for a CLIP tokenizer, on disk.

    Both ONNX-backed providers load a real `tokenizers` file, so the contract
    tests need one; building it is much faster than downloading the genuine
    49408-token vocabulary, and the parts under test are the special tokens and
    the padding, not the merges.
    """
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers, processors

    start_of_text, end_of_text = "<|startoftext|>", "<|endoftext|>"
    words = ["red", "running", "shoe", "leather", "bag", "blue", "word", "handbag"]
    # Id 0 stays free: the embedding provider pads with it, so nothing real may
    # claim it.
    vocab = {start_of_text: 1, end_of_text: 2, "[UNK]": 3}
    vocab.update({word: index for index, word in enumerate(words, start=4)})

    tokenizer = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{start_of_text} $A {end_of_text}",
        special_tokens=[(start_of_text, 1), (end_of_text, 2)],
    )

    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    tokenizer.save(str(path))
    return path
