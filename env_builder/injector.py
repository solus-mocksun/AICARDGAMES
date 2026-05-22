"""
env_builder/injector.py

Takes a validated GameConfig dict and a template (local folder, local HTML
file, or GitHub repo URL) and writes the config into the right place.

Supports three template sources:

  GitHub URL  — "https://github.com/user/repo"
    Downloads the repo as a ZIP, extracts it, copies it, writes config.json.

  Local folder — "/path/to/CardGameTable"
    Copies the folder and writes config.json. (Style A — multi-file)

  Local HTML file — "/path/to/template.html"
    Copies the file and replaces <script id="game-config">. (Style B — single file)

Usage:
    from env_builder.injector import inject_config

    # From GitHub:
    out = inject_config(config, "https://github.com/solus-mocksun/Card_Game_Table_Templete")

    # From local folder:
    out = inject_config(config, "/path/to/CardGameTable", output_dir="games/")
"""

from __future__ import annotations

import io
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


# Matches <script id="game-config" ...>...</script> for single-file style
_CONFIG_BLOCK_RE = re.compile(
    r'(<script\s+id=["\']game-config["\'][^>]*>)(.*?)(</script>)',
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_config(
    config: dict,
    template: str | Path,
    output_dir: str | Path = "games/",
    output_filename: str | None = None,
) -> Path:
    """
    Inject a GameConfig dict into a copy of the template.

    template can be:
      - A GitHub URL  : "https://github.com/user/repo"
      - A local folder: "/path/to/CardGameTable"
      - A local .html : "/path/to/template.html"

    Returns the path to the output folder or file.
    """
    template_str = str(template)

    if template_str.startswith("https://github.com") or template_str.startswith("http://github.com"):
        return _inject_from_github(config, template_str, output_dir, output_filename)

    template_path = Path(template_str)
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    if template_path.is_dir():
        return _inject_multfile(config, template_path, output_dir, output_filename)
    else:
        return _inject_singlefile(config, template_path, output_dir, output_filename)


def extract_config(path: str | Path) -> dict:
    """
    Read config back out of a generated game folder or HTML file.
    """
    path = Path(path)
    if path.is_dir():
        cfg_file = path / "config.json"
        if not cfg_file.exists():
            raise ValueError(f"No config.json found in {path}")
        return json.loads(cfg_file.read_text(encoding="utf-8"))
    else:
        html = path.read_text(encoding="utf-8")
        m = _CONFIG_BLOCK_RE.search(html)
        if not m:
            raise ValueError(f"No <script id=\"game-config\"> block found in {path}")
        return json.loads(m.group(2).strip())


# ---------------------------------------------------------------------------
# GitHub download
# ---------------------------------------------------------------------------

def _inject_from_github(
    config: dict,
    github_url: str,
    output_dir: str | Path,
    output_name: str | None,
) -> Path:
    """
    Download the GitHub repo as a ZIP, extract it, treat it like a local folder.
    """
    # Normalise URL — strip trailing slash and .git
    base_url = github_url.rstrip("/").removesuffix(".git")

    # Try main branch first, then master
    for branch in ("main", "master"):
        zip_url = f"{base_url}/archive/refs/heads/{branch}.zip"
        try:
            print(f"Downloading template from GitHub ({branch})...", flush=True)
            with urllib.request.urlopen(zip_url, timeout=30) as resp:
                zip_data = resp.read()
            break
        except Exception:
            continue
    else:
        raise ConnectionError(
            f"Could not download template from {github_url}\n"
            "Make sure the repo is public and the URL is correct."
        )

    # Extract to a temp directory
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            zf.extractall(tmp)

        # GitHub ZIPs extract into a single subfolder: reponame-branchname/
        extracted_dirs = [
            d for d in Path(tmp).iterdir() if d.is_dir()
        ]
        if not extracted_dirs:
            raise ValueError("Downloaded ZIP appears to be empty")

        template_dir = extracted_dirs[0]
        return _inject_multfile(config, template_dir, output_dir, output_name)


# ---------------------------------------------------------------------------
# Style A — multi-file folder
# ---------------------------------------------------------------------------

def _inject_multfile(
    config: dict,
    template_dir: Path,
    output_dir: str | Path,
    output_name: str | None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        game_name = config.get("game_name", "game")
        output_name = re.sub(r"[^a-zA-Z0-9]+", "_", game_name).strip("_").lower()

    dest = output_dir / output_name

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(template_dir, dest)

    (dest / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Style B — single HTML file
# ---------------------------------------------------------------------------

def _inject_singlefile(
    config: dict,
    template_file: Path,
    output_dir: str | Path,
    output_filename: str | None,
) -> Path:
    html = template_file.read_text(encoding="utf-8")

    if not _CONFIG_BLOCK_RE.search(html):
        raise ValueError(
            f"Template {template_file} has no <script id=\"game-config\"> block."
        )

    json_str = json.dumps(config, indent=2)

    def _replacer(m: re.Match) -> str:
        return f"{m.group(1)}\n{json_str}\n{m.group(3)}"

    new_html = _CONFIG_BLOCK_RE.sub(_replacer, html)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        game_name = config.get("game_name", "game")
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", game_name).strip("_").lower()
        output_filename = f"{safe_name}.html"

    output_path = output_dir / output_filename
    output_path.write_text(new_html, encoding="utf-8")
    return output_path
