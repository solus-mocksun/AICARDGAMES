"""
env_builder/build.py

One-shot pipeline: rulebook → GameConfig → validated → game.html

Usage (CLI):
    python -m env_builder.build hearts.txt --template template.html
    python -m env_builder.build hearts.txt --template template.html --output-dir games/
    python -m env_builder.build --text "Snap is..." --template template.html

Usage (Python API):
    from env_builder.build import build_game

    html_path = build_game(
        rulebook="rulebooks/hearts.txt",
        template="template.html",
        output_dir="games/",
    )
    print(f"Game ready: {html_path}")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from env_builder.parser import parse_rulebook
from env_builder.validator import validate_config, print_validation_report
from env_builder.injector import inject_config


GITHUB_TEMPLATE = "https://github.com/solus-mocksun/Card_Game_Table_Templete"

def build_game(
    rulebook: Optional[str] = None,
    text: Optional[str] = None,
    template: str = GITHUB_TEMPLATE,
    output_dir: str = "games/",
    output_filename: Optional[str] = None,
    save_config: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Full pipeline: rulebook → parse → validate → inject → .html file.

    Args:
        rulebook:        Path to .txt rulebook file.
        text:            Inline rulebook text (alternative to rulebook).
        template:        Path to the HTML template file.
        output_dir:      Directory to write the output .html into.
        output_filename: Override filename (default: game_name.html).
        save_config:     Also save the GameConfig as a .json file alongside the HTML.
        verbose:         Print progress to stdout.

    Returns:
        Path to the generated .html file.
    """
    # Step 1: Parse
    if verbose:
        print("=" * 60)
        print("Step 1/3 — Parsing rulebook with Claude")
        print("=" * 60)

    config = parse_rulebook(filepath=rulebook, text=text, verbose=verbose)

    # Step 2: Validate
    if verbose:
        print("\n" + "=" * 60)
        print("Step 2/3 — Validating GameConfig")
        print("=" * 60)

    errors = validate_config(config)
    if errors:
        print(f"\n✗ Validation failed — {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        raise ValueError(
            f"Generated config for '{config.get('game_name', '?')}' "
            f"failed validation with {len(errors)} error(s). "
            "Check the output above."
        )
    elif verbose:
        print(f"✓ Config valid — {config['game_name']}")

    # Step 3: Inject into template
    if verbose:
        print("\n" + "=" * 60)
        print("Step 3/3 — Injecting config into template")
        print("=" * 60)

    html_path = inject_config(
        config,
        template=template,
        output_dir=output_dir,
        output_filename=output_filename,
    )

    if verbose:
        print(f"✓ Game HTML written: {html_path}")

    # Optionally save the JSON config alongside the HTML
    if save_config:
        json_path = html_path.with_suffix(".json")
        json_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        if verbose:
            print(f"✓ Config JSON saved: {json_path}")

    if verbose:
        print("\nDone. Open the HTML file in a browser to play.")

    return html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Build a playable card game HTML from a rulebook text file"
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("rulebook", nargs="?", help="Path to .txt rulebook file")
    src.add_argument("--text", help="Inline rulebook text")

    p.add_argument(
        "--template",
        default="https://github.com/solus-mocksun/Card_Game_Table_Templete",
        help="Template source: GitHub URL or local path (default: Card_Game_Table_Templete repo)",
    )
    p.add_argument(
        "--output-dir",
        default="games/",
        dest="output_dir",
        help="Directory to write the game .html (default: games/)",
    )
    p.add_argument(
        "--output-filename",
        dest="output_filename",
        help="Override output filename (default: game_name.html)",
    )
    p.add_argument(
        "--save-config",
        action="store_true",
        dest="save_config",
        help="Also save the GameConfig as a .json file",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = p.parse_args()

    html_path = build_game(
        rulebook=args.rulebook,
        text=args.text,
        template=args.template,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        save_config=args.save_config,
        verbose=not args.quiet,
    )

    print(f"\nReady: {html_path}")


if __name__ == "__main__":
    main()
