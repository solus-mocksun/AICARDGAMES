"""
env_builder/parser.py

Reads a card game rulebook (file path or raw text) and uses the Claude API
to produce a valid GameConfig JSON that can be injected into the HTML template.

Usage:
    from env_builder.parser import parse_rulebook

    config = parse_rulebook("rulebooks/hearts.txt")
    config = parse_rulebook(text="Players take turns playing cards...")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import anthropic

# ---------------------------------------------------------------------------
# The system prompt is large and stable — mark it for prompt caching so
# repeated calls (e.g. retries, re-parsing) don't re-tokenise the whole thing.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a card game rule parser. Your job is to read a card game rulebook and output a single valid JSON object that configures an HTML card game table engine.

## Output Format

Output ONLY a raw JSON object — no markdown fences, no explanation, no commentary. The JSON must be parseable by json.loads() without any preprocessing.

## GameConfig Schema

Every field marked REQUIRED must be present. Optional fields can be omitted.

```
{
  "game_name": string,                     // REQUIRED — full name of the game
  "decks": 1 | 2 | 4 | 6 | 8,             // REQUIRED — number of 52-card decks
  "include_jokers": boolean,               // REQUIRED
  "players": integer 1-6,                 // REQUIRED — number of players

  "zones": [                              // REQUIRED — at least 1 zone
    {
      "id": string,                       // REQUIRED — unique snake_case identifier
      "type": "pile" | "spread" | "fan" | "row" | "grid",  // REQUIRED
      "label": string,                    // optional — shown on table
      "position": { "x": "0%-100%", "y": "0%-100%" },     // REQUIRED
      "face": "up" | "down" | "mixed" | "hand",            // REQUIRED
      "max_cards": integer,               // optional
      "visible_count": 0 | 1 | 2 | 3     // optional, default 1, pile type only
    }
  ],

  "deal": [                               // REQUIRED — how cards are dealt at start
    {
      "to": zone_id,                      // REQUIRED
      "cards": integer | "remaining",    // REQUIRED
      "face": "up" | "down"              // REQUIRED
    }
  ],

  "valid_moves": [                        // REQUIRED — at least 1 rule
    {
      "from": zone_id | "any",           // REQUIRED
      "to": zone_id | "any",             // REQUIRED
      "rule": string,                    // REQUIRED — see Rule Vocabulary below
      "move_stack": boolean              // optional, default false
    }
  ],

  "win_condition": string,               // REQUIRED — see Win Condition Vocabulary

  "actions": [                           // optional — buttons on the table
    {
      "label": string,                   // REQUIRED if action present
      "trigger": string,                 // REQUIRED if action present
      "position": { "x": "0%-100%", "y": "0%-100%" }  // REQUIRED if action present
    }
  ],

  "category_hint": string                // optional — gambling/trick_taking/melding/etc
}
```

## Zone Type Guide

- **pile** — cards stack face-down on top of each other, only top visible. Use for: draw decks, stock piles, discard piles.
- **spread** — cards fan out horizontally, all visible, overlap if >7. Use for: hands being displayed openly, community cards in a row.
- **fan** — cards fanned at a slight angle as if held in hand. Use for: the human player's own hand.
- **row** — cards placed in a horizontal line, fully spaced. Use for: foundation piles, tableau columns in solitaire.
- **grid** — cards in a rectangular grid. Use for: memory/matching games.

## Zone face values

- **up** — all cards face-up (visible to all)
- **down** — all cards face-down
- **mixed** — bottom cards face-down, top card face-up (like a dealt hand where one card is hidden)
- **hand** — all face-up but only the owning player can see them (private hand)

## Rule Vocabulary (valid_moves.rule)

Use exactly one of these strings:
- "any" — any card can move freely
- "top_card_only" — only the top card of the source zone can move
- "same_suit" — card must match suit of the top card in target
- "same_rank" — card must match rank of the top card in target
- "ascending_same_suit" — rank must be 1 higher, same suit (for building foundations A→K)
- "descending_same_suit" — rank must be 1 lower, same suit
- "descending_alternating_color" — rank 1 lower, opposite color (red/black alternating)
- "ascending_any_suit" — rank 1 higher, any suit
- "empty_zone_only" — only legal if target zone is empty
- "king_to_empty" — only Kings can be placed in empty zones
- "ace_to_empty" — only Aces can start this zone (foundation start)
- "higher_rank" — any card of strictly higher rank
- "lower_rank" — any card of strictly lower rank
- "no_rule" — always valid (drag anywhere)

## Win Condition Vocabulary

Use exactly one of these strings:
- "all_cards_in_foundation" — all cards in zones marked foundation
- "empty_hand" — player hand zone is empty
- "highest_score" — player with most points wins
- "lowest_score" — player with fewest points wins
- "target_score" — first to reach a score threshold
- "all_foundations_complete" — all foundation zones have 13 cards each
- "no_valid_moves" — game ends when no moves are possible

## Layout Guidelines

Position zones using percentage coordinates:
- Table is 100% wide × 100% tall
- Top area (y: 10%-30%): opponent/dealer hands
- Middle area (y: 40%-60%): shared zones (deck, discard, community)
- Bottom area (y: 65%-85%): player hand
- Buttons: y: 90%-95%
- Spread zones centered horizontally (x: 50%) unless otherwise specified
- Deck/stock piles: x: 15% or x: 80% (left or right side)

## Action Trigger Vocabulary

Common trigger strings for action buttons:
- "hit" — draw a card (Blackjack)
- "stand" — end turn without drawing
- "deal_new" — reset and redeal
- "flip_stock" — flip stock pile to discard (Klondike Solitaire)
- "end_turn" — pass turn to next player
- "draw_card" — draw from deck

## Multi-player Layout

For games with multiple players:
- Player 0 (human): bottom of table (y: 70%-80%)
- Player 1: top of table (y: 15%-25%)
- Player 2: left side (x: 10%, y: 50%)
- Player 3: right side (x: 90%, y: 50%)
- Shared/center zones: middle of table (x: 50%, y: 45%-55%)

## Important Rules

1. Every zone_id used in "deal" and "valid_moves" must exist in "zones"
2. "deal" instructions are executed in order — earlier entries deal first
3. For solitaire games: use "remaining" for the last deal instruction to put leftover cards in the stock
4. For trick-taking games: use "end_turn" action and "empty_hand" win condition
5. Zone ids must be snake_case strings (no spaces, no hyphens)
6. Percentage values must include the % sign: "50%" not "50"
"""

# ---------------------------------------------------------------------------
# Few-shot examples help Claude understand the output format clearly
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


def parse_rulebook(
    filepath: Optional[str] = None,
    text: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Parse a card game rulebook into a GameConfig dict.

    Args:
        filepath: Path to a .txt rulebook file. Mutually exclusive with text.
        text:     Raw rulebook text string. Mutually exclusive with filepath.
        verbose:  Stream tokens to stdout while generating (default True).

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

    # Read file if path given
    if filepath is not None:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Rulebook not found: {filepath}")
        rulebook_text = path.read_text(encoding="utf-8").strip()
    else:
        rulebook_text = text.strip()

    if not rulebook_text:
        raise ValueError("Rulebook text is empty")

    # Build the user message
    user_message = f"RULEBOOK:\n{rulebook_text}"

    client = anthropic.Anthropic()

    if verbose:
        print("Parsing rulebook with Claude...\n", flush=True)

    # ---------------------------------------------------------------------------
    # Streaming call with:
    # - claude-opus-4-7 (best reasoning for structured extraction)
    # - adaptive thinking (lets Claude reason through complex rules)
    # - prompt caching on the large stable system prompt
    # ---------------------------------------------------------------------------
    collected_text = ""

    with client.messages.stream(
        model="claude-opus-4-7",
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
            # Few-shot example — also cacheable since it's stable
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
            # Actual rulebook
            {
                "role": "user",
                "content": user_message,
            },
        ],
    ) as stream:
        for event in stream:
            # Stream text tokens to stdout so the user can watch it generate
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

        # Get the final complete message after streaming
        final_message = stream.get_final_message()

    if verbose:
        print("\n", flush=True)

    # Extract the text content from the response
    raw_output = ""
    for block in final_message.content:
        if hasattr(block, "text"):
            raw_output = block.text
            break

    if not raw_output:
        raise ValueError("Claude returned no text content")

    # Strip markdown fences if Claude added them despite instructions
    raw_output = raw_output.strip()
    if raw_output.startswith("```"):
        lines = raw_output.split("\n")
        # Remove first line (```json or ```) and last line (```)
        raw_output = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    # Parse JSON
    try:
        config = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Claude output was not valid JSON.\n"
            f"Error: {e}\n"
            f"Raw output:\n{raw_output}"
        )

    if verbose:
        print(f"Game: {config.get('game_name', '?')}")
        print(f"Players: {config.get('players', '?')}")
        print(f"Zones: {[z['id'] for z in config.get('zones', [])]}")
        print(f"Actions: {[a['label'] for a in config.get('actions', [])]}")

    return config


# ---------------------------------------------------------------------------
# CLI usage: python -m env_builder.parser hearts.txt
#            python -m env_builder.parser --text "Snap is a game where..."
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Parse a card game rulebook into a GameConfig JSON"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("filepath", nargs="?", help="Path to .txt rulebook file")
    group.add_argument("--text", help="Inline rulebook text (use quotes)")
    p.add_argument(
        "--output", "-o", help="Save JSON to this file (default: print to stdout)"
    )
    p.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress streaming output"
    )
    args = p.parse_args()

    config = parse_rulebook(
        filepath=args.filepath,
        text=args.text,
        verbose=not args.quiet,
    )

    output_json = json.dumps(config, indent=2)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"\nSaved to {args.output}")
    else:
        print("\n" + output_json)


if __name__ == "__main__":
    main()
