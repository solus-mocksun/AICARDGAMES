"""
env_builder/injector.py

Takes a validated GameConfig dict and a template.html file,
replaces the <script id="game-config"> block with the new JSON,
and writes the result as a new .html file.

Usage:
    from env_builder.injector import inject_config

    output_path = inject_config(config, "template.html", output_dir="games/")
    # → writes "games/hearts.html", returns the path
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# Matches the entire content of <script id="game-config" ...>...</script>
# including any existing JSON inside it. Non-greedy to avoid spanning multiple
# script tags.
_CONFIG_BLOCK_RE = re.compile(
    r'(<script\s+id=["\']game-config["\'][^>]*>)(.*?)(</script>)',
    re.DOTALL | re.IGNORECASE,
)


def inject_config(
    config: dict,
    template_path: str | Path,
    output_dir: str | Path = ".",
    output_filename: str | None = None,
) -> Path:
    """
    Inject a GameConfig dict into a copy of template.html.

    Args:
        config:           Validated GameConfig dict.
        template_path:    Path to the template.html file.
        output_dir:       Directory where the new .html file is written.
        output_filename:  Override the output filename. If omitted, the
                          game name is used (e.g. "hearts.html").

    Returns:
        Path to the written output file.

    Raises:
        FileNotFoundError: If template_path does not exist.
        ValueError:        If the template has no <script id="game-config"> block.
    """
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    html = template_path.read_text(encoding="utf-8")

    # Verify the placeholder exists
    if not _CONFIG_BLOCK_RE.search(html):
        raise ValueError(
            f"Template {template_path} has no <script id=\"game-config\"> block. "
            "Make sure the template follows the required structure."
        )

    # Pretty-print the JSON with 2-space indent so it's readable in the source
    json_str = json.dumps(config, indent=2)

    # Replace the content between the script tags, preserving the tags themselves
    def _replacer(m: re.Match) -> str:
        open_tag = m.group(1)
        close_tag = m.group(3)
        return f"{open_tag}\n{json_str}\n      {close_tag}"

    new_html = _CONFIG_BLOCK_RE.sub(_replacer, html)

    # Determine output path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        game_name = config.get("game_name", "game")
        # Sanitise: lowercase, replace spaces/special chars with underscores
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", game_name).strip("_").lower()
        output_filename = f"{safe_name}.html"

    output_path = output_dir / output_filename
    output_path.write_text(new_html, encoding="utf-8")

    return output_path


def extract_config(html_path: str | Path) -> dict:
    """
    Read a previously-generated game .html and extract the GameConfig JSON.
    Useful for inspecting or editing an existing generated game.

    Returns the config as a dict.
    """
    html_path = Path(html_path)
    html = html_path.read_text(encoding="utf-8")

    m = _CONFIG_BLOCK_RE.search(html)
    if not m:
        raise ValueError(f"No <script id=\"game-config\"> block found in {html_path}")

    json_str = m.group(2).strip()
    return json.loads(json_str)
