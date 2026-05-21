"""
env_builder/validator.py

Validates a GameConfig dict against the HTML template schema.
Catches missing fields, unknown zone IDs, bad percentages, etc.
before the config gets injected into the template.

Usage:
    from env_builder.validator import validate_config, ValidationError

    errors = validate_config(config)
    if errors:
        for e in errors: print(e)
    else:
        print("Config is valid")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Allowed vocabulary — mirrors TEMPLATE_REQUIREMENTS.txt exactly
# ---------------------------------------------------------------------------

VALID_ZONE_TYPES = {"pile", "spread", "fan", "row", "grid"}
VALID_FACE_VALUES = {"up", "down", "mixed", "hand"}
VALID_RULES = {
    "any",
    "top_card_only",
    "same_suit",
    "same_rank",
    "ascending_same_suit",
    "descending_same_suit",
    "descending_alternating_color",
    "ascending_any_suit",
    "empty_zone_only",
    "king_to_empty",
    "ace_to_empty",
    "higher_rank",
    "lower_rank",
    "no_rule",
}
VALID_WIN_CONDITIONS = {
    "all_cards_in_foundation",
    "empty_hand",
    "highest_score",
    "lowest_score",
    "target_score",
    "all_foundations_complete",
    "no_valid_moves",
}
VALID_DECK_COUNTS = {1, 2, 4, 6, 8}


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.field}] {self.message}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_percent(value: Any) -> bool:
    """Return True if value is a string like '50%' with 0 <= n <= 100."""
    if not isinstance(value, str):
        return False
    if not value.endswith("%"):
        return False
    try:
        n = float(value[:-1])
        return 0.0 <= n <= 100.0
    except ValueError:
        return False


def _check_position(pos: Any, path: str, errors: list) -> None:
    if not isinstance(pos, dict):
        errors.append(ValidationError(path, "position must be an object {x, y}"))
        return
    for axis in ("x", "y"):
        if axis not in pos:
            errors.append(ValidationError(f"{path}.{axis}", f"missing '{axis}' coordinate"))
        elif not _is_percent(pos[axis]):
            errors.append(ValidationError(
                f"{path}.{axis}",
                f"must be a percentage string like '50%', got: {pos[axis]!r}"
            ))


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_config(config: dict) -> list[ValidationError]:
    """
    Validate a GameConfig dict. Returns a list of ValidationErrors.
    Empty list means the config is valid.
    """
    errors: list[ValidationError] = []

    if not isinstance(config, dict):
        errors.append(ValidationError("root", "Config must be a JSON object (dict)"))
        return errors

    # ---- Top-level required fields ----------------------------------------

    # game_name
    if "game_name" not in config:
        errors.append(ValidationError("game_name", "Missing required field"))
    elif not isinstance(config["game_name"], str) or not config["game_name"].strip():
        errors.append(ValidationError("game_name", "Must be a non-empty string"))

    # decks
    if "decks" not in config:
        errors.append(ValidationError("decks", "Missing required field"))
    elif config["decks"] not in VALID_DECK_COUNTS:
        errors.append(ValidationError("decks", f"Must be one of {sorted(VALID_DECK_COUNTS)}, got {config['decks']!r}"))

    # include_jokers
    if "include_jokers" not in config:
        errors.append(ValidationError("include_jokers", "Missing required field"))
    elif not isinstance(config["include_jokers"], bool):
        errors.append(ValidationError("include_jokers", "Must be true or false"))

    # players
    if "players" not in config:
        errors.append(ValidationError("players", "Missing required field"))
    elif not isinstance(config["players"], int) or not (1 <= config["players"] <= 6):
        errors.append(ValidationError("players", f"Must be an integer 1-6, got {config['players']!r}"))

    # win_condition
    if "win_condition" not in config:
        errors.append(ValidationError("win_condition", "Missing required field"))
    elif config["win_condition"] not in VALID_WIN_CONDITIONS:
        errors.append(ValidationError(
            "win_condition",
            f"Unknown value {config['win_condition']!r}. Valid: {sorted(VALID_WIN_CONDITIONS)}"
        ))

    # ---- zones ------------------------------------------------------------

    if "zones" not in config:
        errors.append(ValidationError("zones", "Missing required field"))
        zone_ids: set[str] = set()
    elif not isinstance(config["zones"], list) or len(config["zones"]) == 0:
        errors.append(ValidationError("zones", "Must be a non-empty array"))
        zone_ids = set()
    else:
        zone_ids = set()
        ids_seen: set[str] = set()

        for i, zone in enumerate(config["zones"]):
            prefix = f"zones[{i}]"

            if not isinstance(zone, dict):
                errors.append(ValidationError(prefix, "Each zone must be an object"))
                continue

            # id
            if "id" not in zone:
                errors.append(ValidationError(f"{prefix}.id", "Missing required field"))
            elif not isinstance(zone["id"], str) or not zone["id"].strip():
                errors.append(ValidationError(f"{prefix}.id", "Must be a non-empty string"))
            else:
                zid = zone["id"]
                if zid in ids_seen:
                    errors.append(ValidationError(f"{prefix}.id", f"Duplicate zone id {zid!r}"))
                ids_seen.add(zid)
                zone_ids.add(zid)

            # type
            if "type" not in zone:
                errors.append(ValidationError(f"{prefix}.type", "Missing required field"))
            elif zone["type"] not in VALID_ZONE_TYPES:
                errors.append(ValidationError(
                    f"{prefix}.type",
                    f"Unknown type {zone['type']!r}. Valid: {sorted(VALID_ZONE_TYPES)}"
                ))

            # position
            if "position" not in zone:
                errors.append(ValidationError(f"{prefix}.position", "Missing required field"))
            else:
                _check_position(zone["position"], f"{prefix}.position", errors)

            # face
            if "face" not in zone:
                errors.append(ValidationError(f"{prefix}.face", "Missing required field"))
            elif zone["face"] not in VALID_FACE_VALUES:
                errors.append(ValidationError(
                    f"{prefix}.face",
                    f"Unknown face value {zone['face']!r}. Valid: {sorted(VALID_FACE_VALUES)}"
                ))

            # optional: max_cards
            if "max_cards" in zone and not (
                isinstance(zone["max_cards"], int) and zone["max_cards"] > 0
            ):
                errors.append(ValidationError(f"{prefix}.max_cards", "Must be a positive integer"))

            # optional: visible_count
            if "visible_count" in zone and zone["visible_count"] not in (0, 1, 2, 3):
                errors.append(ValidationError(f"{prefix}.visible_count", "Must be 0, 1, 2, or 3"))

    # ---- deal -------------------------------------------------------------

    if "deal" not in config:
        errors.append(ValidationError("deal", "Missing required field"))
    elif not isinstance(config["deal"], list) or len(config["deal"]) == 0:
        errors.append(ValidationError("deal", "Must be a non-empty array"))
    else:
        remaining_used = False
        for i, instruction in enumerate(config["deal"]):
            prefix = f"deal[{i}]"

            if not isinstance(instruction, dict):
                errors.append(ValidationError(prefix, "Each deal instruction must be an object"))
                continue

            # to
            if "to" not in instruction:
                errors.append(ValidationError(f"{prefix}.to", "Missing required field"))
            elif instruction["to"] not in zone_ids:
                errors.append(ValidationError(
                    f"{prefix}.to",
                    f"Unknown zone id {instruction['to']!r}. Known zones: {sorted(zone_ids)}"
                ))

            # cards
            if "cards" not in instruction:
                errors.append(ValidationError(f"{prefix}.cards", "Missing required field"))
            else:
                cards_val = instruction["cards"]
                if cards_val == "remaining":
                    if remaining_used:
                        errors.append(ValidationError(
                            f"{prefix}.cards",
                            '"remaining" can only appear once in the deal array'
                        ))
                    remaining_used = True
                elif not isinstance(cards_val, int) or cards_val < 1:
                    errors.append(ValidationError(
                        f"{prefix}.cards",
                        f'Must be a positive integer or "remaining", got {cards_val!r}'
                    ))

            # face
            if "face" not in instruction:
                errors.append(ValidationError(f"{prefix}.face", "Missing required field"))
            elif instruction["face"] not in ("up", "down"):
                errors.append(ValidationError(
                    f"{prefix}.face",
                    f'Must be "up" or "down", got {instruction["face"]!r}'
                ))

    # ---- valid_moves -------------------------------------------------------

    if "valid_moves" not in config:
        errors.append(ValidationError("valid_moves", "Missing required field"))
    elif not isinstance(config["valid_moves"], list) or len(config["valid_moves"]) == 0:
        errors.append(ValidationError("valid_moves", "Must be a non-empty array"))
    else:
        for i, move in enumerate(config["valid_moves"]):
            prefix = f"valid_moves[{i}]"

            if not isinstance(move, dict):
                errors.append(ValidationError(prefix, "Each move must be an object"))
                continue

            # from
            if "from" not in move:
                errors.append(ValidationError(f"{prefix}.from", "Missing required field"))
            elif move["from"] != "any" and move["from"] not in zone_ids:
                errors.append(ValidationError(
                    f"{prefix}.from",
                    f"Unknown zone id {move['from']!r}. Use a zone id or 'any'"
                ))

            # to
            if "to" not in move:
                errors.append(ValidationError(f"{prefix}.to", "Missing required field"))
            elif move["to"] != "any" and move["to"] not in zone_ids:
                errors.append(ValidationError(
                    f"{prefix}.to",
                    f"Unknown zone id {move['to']!r}. Use a zone id or 'any'"
                ))

            # rule
            if "rule" not in move:
                errors.append(ValidationError(f"{prefix}.rule", "Missing required field"))
            elif move["rule"] not in VALID_RULES:
                errors.append(ValidationError(
                    f"{prefix}.rule",
                    f"Unknown rule {move['rule']!r}. Valid: {sorted(VALID_RULES)}"
                ))

            # move_stack (optional bool)
            if "move_stack" in move and not isinstance(move["move_stack"], bool):
                errors.append(ValidationError(f"{prefix}.move_stack", "Must be true or false"))

    # ---- actions (optional) -----------------------------------------------

    if "actions" in config:
        if not isinstance(config["actions"], list):
            errors.append(ValidationError("actions", "Must be an array"))
        else:
            for i, action in enumerate(config["actions"]):
                prefix = f"actions[{i}]"

                if not isinstance(action, dict):
                    errors.append(ValidationError(prefix, "Each action must be an object"))
                    continue

                if "label" not in action:
                    errors.append(ValidationError(f"{prefix}.label", "Missing required field"))
                elif not isinstance(action["label"], str):
                    errors.append(ValidationError(f"{prefix}.label", "Must be a string"))

                if "trigger" not in action:
                    errors.append(ValidationError(f"{prefix}.trigger", "Missing required field"))
                elif not isinstance(action["trigger"], str):
                    errors.append(ValidationError(f"{prefix}.trigger", "Must be a string"))

                if "position" not in action:
                    errors.append(ValidationError(f"{prefix}.position", "Missing required field"))
                else:
                    _check_position(action["position"], f"{prefix}.position", errors)

    return errors


def validate_or_raise(config: dict) -> None:
    """
    Validate config and raise a ValueError listing all errors if any are found.
    Use this when you want to fail fast.
    """
    errors = validate_config(config)
    if errors:
        msg = f"GameConfig has {len(errors)} error(s):\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)


def print_validation_report(config: dict) -> bool:
    """
    Print a human-readable validation report.
    Returns True if valid, False if errors found.
    """
    game_name = config.get("game_name", "unknown")
    errors = validate_config(config)

    if not errors:
        print(f"✓ {game_name} — config is valid")
        return True

    print(f"✗ {game_name} — {len(errors)} error(s) found:")
    for e in errors:
        print(f"  {e}")
    return False
