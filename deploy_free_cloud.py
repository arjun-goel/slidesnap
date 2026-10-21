"""
Deploys the permanent Public Web Frontend to Hugging Face Static Spaces (100% Free, No Pro Required)
and connects it seamlessly to the live public HTTPS backend API.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from huggingface_hub import HfApi


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def deploy_static_space(public_backend_url: str) -> None:
    load_env_file(Path.home() / ".env")
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/slidesnap"

    print(f"Creating free Public Static Space: {repo_id} ...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        private=False,
        exist_ok=True,
    )

    # Read our student dashboard HTML and wire API calls to the public HTTPS backend
    html_src = (Path(__file__).resolve().parent / "backend" / "app" / "static" / "index.html").read_text(encoding="utf-8")
    patched_html = html_src.replace("fetch('/api/", f"fetch('{public_backend_url}/api/")
    patched_html = patched_html.replace("fetch(`/api/", f"fetch(`{public_backend_url}/api/")
    patched_html = patched_html.replace('href="/docs"', f'href="{public_backend_url}/docs"')
    patched_html = patched_html.replace(
        "href=\"${job.pdf_download_url}\"",
        f"href=\"{public_backend_url}${{job.pdf_download_url}}\"",
    )
    patched_html = patched_html.replace(
        "src=\"${s.image_url}\"",
        f"src=\"{public_backend_url}${{s.image_url}}\"",
    )

    readme_md = f"""---
title: SlideSnap Lecture Slide Extractor
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# SlideSnap — AI & Computer Vision Lecture Slide Extractor ($0 Student Cloud Edition)

- **Live Public Space**: https://huggingface.co/spaces/{repo_id}
- **Direct Public Worker Tunnel**: {public_backend_url}
"""

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "index.html").write_text(patched_html, encoding="utf-8")
        (tmp_dir / "README.md").write_text(readme_md, encoding="utf-8")

        api.upload_folder(
            folder_path=str(tmp_dir),
            repo_id=repo_id,
            repo_type="space",
        )

    print(f"SUCCESS_SPACE_URL=https://huggingface.co/spaces/{repo_id}")
    print(f"SUCCESS_EMBED_URL=https://{username}-slidesnap.static.hf.space")


if __name__ == "__main__":
    import sys
    backend_url = sys.argv[1] if len(sys.argv) > 1 else "https://cached-july-shelter-pros.trycloudflare.com"
    deploy_static_space(backend_url)
