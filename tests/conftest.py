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
