from pathlib import Path

from gpuprof.constants import RESULTS_ROOT


def baseline_dir() -> Path:
    return RESULTS_ROOT / "baseline"


def best_dir() -> Path:
    return RESULTS_ROOT / "best"


def search_dir_latest() -> Path:
    return RESULTS_ROOT / "search" / "latest"
