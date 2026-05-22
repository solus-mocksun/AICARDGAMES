"""
env_builder/parser.py

Reads a card game rulebook (file path or raw text) and uses the Claude API
to produce a valid GameConfig JSON that can be injected into the HTML template.

Usage:
    from env_builder.parser import parse_rulebook

    config = parse_rulebook("rulebooks/hearts.txt")
    config = parse_rulebook(text="Players take turns playing cards...")

Set your key in .env:
    ANTHROPIC_API_KEY=your-key-from-console.anthropic.com
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# Auto-load .env file so you don't need to export anything in the terminal
def _load_dotenv() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

_load_dotenv()

import anthropic

# ---------------------------------------------------------------------------
# System prompt — large and stable, marked for prompt caching so repeated
# calls (retries, re-parses) don't re-tokenise the whole thing each time.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a card game rule parser. Your job is to read a card game rulebook and output a single valid JSON object that configures an HTML card game table engine.

## Output Format

Output ONLY a raw JSON object — no markdown fences, no explanation, no commentary. The JSON must be parseable by json.loads() without any preprocessing.

## GameConfig Schema

Every field marked REQUIRED must be present. Optional fields can be omitted.

{
  "game_name": string,                     // REQUIRED — full name of the game
  "decks": 1 | 2 | 4 | 6 | 8,             // REQUIRED — number of 52-card decks
  "include_jokers": boolean,               // REQUIRED
  "players": integer 1-6,                  // REQUIRED — number of players

  "zones": [                               // REQUIRED — at least 1 zone
    {
      "id": string,                        // REQUIRED — unique snake_case identifier
      "type": "pile" | "spread" | "fan" | "row" | "grid",  // REQUIRED
      "label": string,                     // optional — shown on table
      "position": { "x": "0%-100%", "y": "0%-100%" },      // REQUIRED
      "face": "up" | "down" | "mixed" | "hand",             // REQUIRED
      "max_cards": integer,                // optional
      "visible_count": 0 | 1 | 2 | 3      // optional, default 1, pile type only
    }
  ],

  "deal": [                                // REQUIRED — how cards are dealt at start
    {
      "to": zone_id,                       // REQUIRED
      "cards": integer | "remaining",      // REQUIRED
      "face": "up" | "down"               // REQUIRED
    }
  ],

  "valid_moves": [                         // REQUIRED — at least 1 rule
    {
      "from": zone_id | "any",            // REQUIRED
      "to": zone_id | "any",             // REQUIRED
      "rule": string,                     // REQUIRED — see Rule Vocabulary below
      "move_stack": boolean               // optional, default false
    }
  ],

  "win_condition": string,                // REQUIRED — see Win Condition Vocabulary

  "actions": [                            // optional — buttons on the table
    {
      "label": string,
      "trigger": string,
      "position": { "x": "0%-100%", "y": "0%-100%" }
    }
  ],

  "category_hint": string                 // optional — gambling/trick_taking/melding/etc
}

## Zone Types
- pile: cards stack on top of each other, only top visible. Use for: draw decks, stock piles, discard.
- spread: cards fan out horizontally, all visible. Use for: open hands, community cards.
- fan: cards fanned at an angle as if held in hand. Use for: the human player's own hand.
- row: cards in a horizontal line, fully spaced. Use for: foundation piles, tableau.
- grid: cards in a rectangular grid. Use for: memory/matching games.

## Zone face values
- up: all cards face-up (visible to all)
- down: all cards face-down
- mixed: bottom cards face-down, top card face-up
- hand: all face-up but only the owning player can see them (private hand)

## Rule Vocabulary (valid_moves.rule) — use exactly one:
"any", "top_card_only", "same_suit", "same_rank",
"ascending_same_suit", "descending_same_suit", "descending_alternating_color",
"ascending_any_suit", "empty_zone_only", "king_to_empty", "ace_to_empty",
"higher_rank", "lower_rank", "no_rule"

## Win Condition Vocabulary — use exactly one:
"all_cards_in_foundation", "empty_hand", "highest_score", "lowest_score",
"target_score", "all_foundations_complete", "no_valid_moves"

## Layout Guidelines
- Table is 100% wide x 100% tall
- Top (y 10-30%): opponent/dealer hands
- Middle (y 40-60%): shared zones — deck, discard, community cards
- Bottom (y 65-85%): human player hand
- Buttons: y 90-95%
- Deck/stock piles: x 15% or x 80% (left or right side)
- Multi-player: player 0 bottom, player 1 top, player 2 left (x:10%), player 3 right (x:90%)

## Important Rules
1. Every zone_id used in deal and valid_moves must exist in zones
2. deal instructions are executed in order
3. For solitaire: use "remaining" for the last deal instruction
4. Zone ids must be snake_case strings (no spaces, no hyphens)
5. Percentage values must include the % sign: "50%" not "50"
"""

# ---------------------------------------------------------------------------
# Few-shot example — also cached since it never changes
# ---------------------------------------------------------------------------

FEW_SHOT_USER = """RULEBOOK:
War is a simple two-player card game. The deck is split evenly between two players (26 cards each), both face-down. Each turn, both players flip the top card of their pile simultaneously. The player with the higher rank wins both cards and places them at the bottom of their pile. If the cards are equal rank, it's "war" — each player places 3 face-down cards and 1 face-up card, and the higher face-up card wins all cards. Play continues until one player has all 52 cards (they win) or a player runs out of cards (they lose)."""

FEW_SHOT_ASSISTANT = """{
  "game_name": "War",
  "decks": 1,
  "include_jokers": false,
  "players": 2,
  "zones": [
    {
      "id": "player_pile",
      "type": "pile",
      "label": "Your Pile",
      "position": {"x": "35%", "y": "75%"},
      "face": "down",
      "visible_count": 2
    },
    {
      "id": "opponent_pile",
      "type": "pile",
      "label": "Opponent Pile",
      "position": {"x": "35%", "y": "20%"},
      "face": "down",
      "visible_count": 2
    },
    {
      "id": "player_play",
      "type": "pile",
      "label": "Your Card",
      "position": {"x": "55%", "y": "62%"},
      "face": "up",
      "visible_count": 0
    },
    {
      "id": "opponent_play",
      "type": "pile",
      "label": "Opponent Card",
      "position": {"x": "55%", "y": "35%"},
      "face": "up",
      "visible_count": 0
    }
  ],
  "deal": [
    {"to": "player_pile", "cards": 26, "face": "down"},
    {"to": "opponent_pile", "cards": 26, "face": "down"}
  ],
  "valid_moves": [
    {"from": "player_pile", "to": "player_play", "rule": "top_card_only"}
  ],
  "win_condition": "empty_hand",
  "actions": [
    {
      "label": "Flip Card",
      "trigger": "end_turn",
      "position": {"x": "50%", "y": "92%"}
    },
    {
      "label": "New Game",
      "trigger": "deal_new",
      "position": {"x": "85%", "y": "92%"}
    }
  ],
  "category_hint": "gambling"
}"""

DEFAULT_MODEL = "claude-opus-4-7"


def parse_rulebook(
    filepath: Optional[str] = None,
    text: Optional[str] = None,
    verbose: bool = True,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Parse a card game rulebook into a GameConfig dict using the Claude API.

    Args:
        filepath: Path to a .txt rulebook file. Mutually exclusive with text.
        text:     Raw rulebook text string. Mutually exclusive with filepath.
        verbose:  Stream tokens to stdout while generating (default True).
        model:    Claude model to use. Default: claude-opus-4-7.

    Returns:
        A dict matching the GameConfig schema, ready for json.dumps().

    Raises:
        ValueError: If neither or both of filepath/text are provided.
        ValueError: If the Claude response is not valid JSON.
        FileNotFoundError: If filepath does not exist.
    """
    if filepath is None and text is None:
        raise ValueError("Provide either filepath= or text=")
    if filepath is not None and text is not None:
        raise ValueError("Provide either filepath= or text=, not both")

    if filepath is not None:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Rulebook not found: {filepath}")
        rulebook_text = path.read_text(encoding="utf-8").strip()
    else:
        rulebook_text = text.strip()

    if not rulebook_text:
        raise ValueError("Rulebook text is empty")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Add it to your .env file: ANTHROPIC_API_KEY=your-key\n"
            "Get a key at: https://console.anthropic.com"
        )

    client = anthropic.Anthropic(api_key=api_key)

    if verbose:
        print(f"Parsing rulebook with Claude ({model})...\n", flush=True)

    collected_text = ""

    with client.messages.stream(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache the big schema prompt
            }
        ],
        messages=[
            # Few-shot example — cached since it never changes
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": FEW_SHOT_USER,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": FEW_SHOT_ASSISTANT,
            },
            # The actual rulebook
            {
                "role": "user",
                "content": rulebook_text,
            },
        ],
    ) as stream:
        for event in stream:
            if (
                hasattr(event, "type")
                and event.type == "content_block_delta"
                and hasattr(event.delta, "type")
                and event.delta.type == "text_delta"
            ):
                chunk = event.delta.text
                collected_text += chunk
                if verbose:
                    print(chunk, end="", flush=True)

        final_message = stream.get_final_message()

    if verbose:
        print("\n", flush=True)

    # Extract text from final message (skip thinking blocks)
    raw_output = ""
    for block in final_message.content:
        if hasattr(block, "text"):
            raw_output = block.text
            break

    if not raw_output:
        raise ValueError("Claude returned no text content")

    raw_output = raw_output.strip()

    # Strip markdown fences if Claude added them despite instructions
    if raw_output.startswith("```"):
        lines = raw_output.split("\n")
        raw_output = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        raw_output = raw_output.strip()

    try:
        config = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Claude output was not valid JSON.\n"
            f"Error: {e}\n"
            f"Raw output:\n{raw_output}"
        )

    if verbose:
        print(f"Game:    {config.get('game_name', '?')}")
        print(f"Players: {config.get('players', '?')}")
        print(f"Zones:   {[z['id'] for z in config.get('zones', [])]}")
        print(f"Actions: {[a['label'] for a in config.get('actions', [])]}")

    return config


# ---------------------------------------------------------------------------
# CLI:
#   python -m env_builder.parser hearts.txt
#   python -m env_builder.parser --text "Snap is a game where..."
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Parse a card game rulebook into a GameConfig JSON"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("filepath", nargs="?", help="Path to .txt rulebook file")
    group.add_argument("--text", help="Inline rulebook text (use quotes)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Claude model (default: {DEFAULT_MODEL})")
    p.add_argument("--output", "-o",
                   help="Save JSON to this file (default: print to stdout)")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress streaming output")
    args = p.parse_args()

    config = parse_rulebook(
        filepath=args.filepath,
        text=args.text,
        verbose=not args.quiet,
        model=args.model,
    )

    output_json = json.dumps(config, indent=2)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"\nSaved to {args.output}")
    else:
        print("\n" + output_json)


if __name__ == "__main__":
    main()
