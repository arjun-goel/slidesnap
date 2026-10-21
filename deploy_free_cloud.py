"""
Automated 100% Free Public Cloud Deployment Script for SlideSnap.
Deploys the Dockerized FastAPI + Computer Vision Worker to Hugging Face Spaces (Free 2 vCPU, 16 GB RAM).
Loads HF_TOKEN safely from ~/.env without exposing secrets in logs.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def deploy_to_huggingface_space() -> None:
    load_env_file(Path.home() / ".env")
    load_env_file(Path(__file__).resolve().parent / ".env")

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found in ~/.env")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed in .venv")
        sys.exit(1)

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    space_name = os.environ.get("HF_SPACE_NAME", "slidesnap")
    repo_id = f"{username}/{space_name}"

    print(f"Creating / verifying free public Hugging Face Space: {repo_id} ...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )

    project_root = Path(__file__).resolve().parent
    print("Uploading SlideSnap codebase & Dockerfile to Hugging Face Space...")
    api.upload_folder(
        folder_path=str(project_root),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            ".venv/**",
            "**/__pycache__/**",
            "backend/storage/**",
            ".git/**",
            ".pytest_cache/**",
            "*.db",
            ".env",
        ],
    )

    public_url = f"https://huggingface.co/spaces/{repo_id}"
    direct_app_url = f"https://{username}-{space_name}.hf.space"
    print("\n============================================================")
    print("SUCCESS! SlideSnap is now deployed to the public cloud (FREE)")
    print(f"Space Dashboard URL : {public_url}")
    print(f"Direct Public Web UI: {direct_app_url}")
    print("============================================================")


if __name__ == "__main__":
    deploy_to_huggingface_space()
