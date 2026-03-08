from __future__ import annotations

from pathlib import Path

from cc_optimize.adapter.config_reader import read_config
from cc_optimize.models.candidate import ConfigCandidate


def generate_seed(repo_path: Path) -> ConfigCandidate:
    return read_config(repo_path)
