from __future__ import annotations

from pathlib import Path
from huggingface_hub import snapshot_download


REPO_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    local_dir = models_dir / "tinyllama"

    models_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} -> {local_dir}")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    # Must-have checks
    must_have = ["config.json", "tokenizer.json"]
    for name in must_have:
        p = local_dir / name
        print(f"{'OK' if p.exists() else 'MISSING'}: {p}")

    total_bytes = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file())
    print(f"Total size: {total_bytes/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
