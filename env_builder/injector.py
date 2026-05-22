"""
env_builder/injector.py

Takes a validated GameConfig dict and a template folder (or a single
self-contained HTML file) and writes the config into the right place.

Supports two template styles:

  Style A — Multi-file (your CardGameTable setup):
    index.html + engine.js + style.css + config.json + server.js
    The injector copies the whole folder and overwrites config.json.

  Style B — Single-file HTML (original spec):
    One .html file with <script id="game-config"> embedded inside.
    The injector copies the file and replaces the JSON block.

The style is detected automatically.

Usage:
    from env_builder.injector import inject_config

    # Multi-file template folder:
    output_dir = inject_config(config, "CardGameTable/", output_dir="games/")

    # Single-file HTML:
    output_path = inject_config(config, "template.html", output_dir="games/")
"""

from __future__ import annotations

import json
import re
import shutil
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
    template_path: str | Path,
    output_dir: str | Path = "games/",
    output_filename: str | None = None,
) -> Path:
    """
    Inject a GameConfig dict into a copy of the template.

    Auto-detects whether template_path is a folder (multi-file style)
    or a single .html file (embedded style).

    Args:
        config:           Validated GameConfig dict.
        template_path:    Path to the template folder OR a single .html file.
        output_dir:       Parent directory where the output is written.
        output_filename:  Override the output folder/file name.
                          Defaults to the sanitised game name.

    Returns:
        Path to the output — either a folder (multi-file) or .html file (single).

    Raises:
        FileNotFoundError: If template_path does not exist.
        ValueError:        If a single .html has no <script id="game-config"> block.
    """
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    if template_path.is_dir():
        return _inject_multfile(config, template_path, output_dir, output_filename)
    else:
        return _inject_singlefile(config, template_path, output_dir, output_filename)


def extract_config(path: str | Path) -> dict:
    """
    Read config back out of a generated game.
    Works with both a game folder (reads config.json) and a single .html file.
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
# Style A — multi-file folder (CardGameTable style)
# ---------------------------------------------------------------------------

def _inject_multfile(
    config: dict,
    template_dir: Path,
    output_dir: str | Path,
    output_name: str | None,
) -> Path:
    """
    Copy the template folder into output_dir/<game_name>/ and
    overwrite config.json with the new config.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name is None:
        game_name = config.get("game_name", "game")
        output_name = re.sub(r"[^a-zA-Z0-9]+", "_", game_name).strip("_").lower()

    dest = output_dir / output_name

    # Copy the whole template folder fresh each time
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(template_dir, dest)

    # Overwrite config.json with the new game config
    config_path = dest / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return dest


# ---------------------------------------------------------------------------
# Style B — single self-contained HTML file
# ---------------------------------------------------------------------------

def _inject_singlefile(
    config: dict,
    template_file: Path,
    output_dir: str | Path,
    output_filename: str | None,
) -> Path:
    """
    Copy the template .html and replace the <script id="game-config"> block.
    """
    html = template_file.read_text(encoding="utf-8")

    if not _CONFIG_BLOCK_RE.search(html):
        raise ValueError(
            f"Template {template_file} has no <script id=\"game-config\"> block. "
            "Make sure the template follows the required structure."
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
