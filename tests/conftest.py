import os
import shutil
from pathlib import Path

import pytest

from policypilot.config import PROJECT_ROOT, Settings
from policypilot.retrieval import PolicyIndex

# Tests use the fast, download-free embedder and no reranker, so they run anywhere in seconds.
os.environ["PP_EMBEDDER"] = "hashing"
os.environ["PP_RERANK"] = "off"


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    data = tmp_path_factory.mktemp("data")
    for sub in ("policies", "regulations", "eval"):
        shutil.copytree(PROJECT_ROOT / "data" / sub, data / sub)
    s = Settings()
    s.data_dir = Path(data)
    s.embedder, s.rerank, s.mode = "hashing", "off", "offline"
    return s


@pytest.fixture(scope="session")
def index(settings) -> PolicyIndex:
    return PolicyIndex.build(settings, force=True)
