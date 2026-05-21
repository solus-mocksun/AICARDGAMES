"""Tests for env_builder — validator and injector (no API calls needed)."""

import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Validator tests — no API needed
# ---------------------------------------------------------------------------

class TestValidator:
    def _valid_config(self):
        """A minimal fully-valid config."""
        return {
            "game_name": "Test Game",
            "decks": 1,
            "include_jokers": False,
            "players": 2,
            "zones": [
                {
                    "id": "deck",
                    "type": "pile",
                    "position": {"x": "20%", "y": "50%"},
                    "face": "down",
                }
            ],
            "deal": [
                {"to": "deck", "cards": 52, "face": "down"}
            ],
            "valid_moves": [
                {"from": "deck", "to": "deck", "rule": "any"}
            ],
            "win_condition": "empty_hand",
        }

    def test_valid_config_has_no_errors(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        errors = validate_config(config)
        assert errors == []

    def test_missing_game_name(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        del config["game_name"]
        errors = validate_config(config)
        fields = [e.field for e in errors]
        assert "game_name" in fields

    def test_invalid_deck_count(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["decks"] = 3   # not in {1,2,4,6,8}
        errors = validate_config(config)
        assert any(e.field == "decks" for e in errors)

    def test_players_out_of_range(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["players"] = 10
        errors = validate_config(config)
        assert any(e.field == "players" for e in errors)

    def test_unknown_zone_type(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["zones"][0]["type"] = "circle"
        errors = validate_config(config)
        assert any("type" in e.field for e in errors)

    def test_bad_position_missing_percent(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["zones"][0]["position"]["x"] = "50"   # missing %
        errors = validate_config(config)
        assert any("position" in e.field for e in errors)

    def test_bad_position_out_of_range(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["zones"][0]["position"]["y"] = "150%"  # >100
        errors = validate_config(config)
        assert any("position" in e.field for e in errors)

    def test_unknown_zone_id_in_deal(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["deal"][0]["to"] = "nonexistent_zone"
        errors = validate_config(config)
        assert any("deal" in e.field for e in errors)

    def test_unknown_zone_id_in_valid_moves(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["valid_moves"][0]["from"] = "ghost_zone"
        errors = validate_config(config)
        assert any("valid_moves" in e.field for e in errors)

    def test_unknown_rule(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["valid_moves"][0]["rule"] = "made_up_rule"
        errors = validate_config(config)
        assert any("rule" in e.field for e in errors)

    def test_unknown_win_condition(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["win_condition"] = "collect_all_stars"
        errors = validate_config(config)
        assert any("win_condition" in e.field for e in errors)

    def test_duplicate_zone_ids(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["zones"].append({
            "id": "deck",   # same id as first zone
            "type": "pile",
            "position": {"x": "80%", "y": "50%"},
            "face": "up",
        })
        errors = validate_config(config)
        assert any("duplicate" in e.message.lower() for e in errors)

    def test_remaining_used_twice_is_error(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["deal"] = [
            {"to": "deck", "cards": "remaining", "face": "down"},
            {"to": "deck", "cards": "remaining", "face": "down"},
        ]
        errors = validate_config(config)
        assert any("remaining" in e.message for e in errors)

    def test_any_zone_id_is_valid_in_moves(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["valid_moves"][0]["from"] = "any"
        config["valid_moves"][0]["to"] = "any"
        errors = validate_config(config)
        assert errors == []

    def test_action_missing_position(self):
        from env_builder.validator import validate_config
        config = self._valid_config()
        config["actions"] = [
            {"label": "Hit", "trigger": "hit"}  # missing position
        ]
        errors = validate_config(config)
        assert any("position" in e.field for e in errors)

    def test_validate_or_raise_raises_on_error(self):
        from env_builder.validator import validate_or_raise
        with pytest.raises(ValueError, match="error"):
            validate_or_raise({"game_name": "broken"})

    def test_validate_or_raise_silent_on_valid(self):
        from env_builder.validator import validate_or_raise
        config = self._valid_config()
        validate_or_raise(config)   # should not raise


# ---------------------------------------------------------------------------
# Injector tests — no API needed, uses a minimal fake template
# ---------------------------------------------------------------------------

FAKE_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head><title>Card Game</title></head>
<body>
  <script id="game-config" type="application/json">
    {"game_name": "DEFAULT", "decks": 1}
  </script>
  <script>/* engine */</script>
</body>
</html>
"""


@pytest.fixture
def template_file(tmp_path):
    p = tmp_path / "template.html"
    p.write_text(FAKE_TEMPLATE)
    return p


class TestInjector:
    def _config(self):
        return {
            "game_name": "Hearts",
            "decks": 1,
            "include_jokers": False,
            "players": 4,
            "zones": [],
            "deal": [],
            "valid_moves": [],
            "win_condition": "lowest_score",
        }

    def test_output_file_is_created(self, template_file, tmp_path):
        from env_builder.injector import inject_config
        out = inject_config(self._config(), template_file, output_dir=tmp_path)
        assert out.exists()

    def test_output_filename_is_game_name(self, template_file, tmp_path):
        from env_builder.injector import inject_config
        out = inject_config(self._config(), template_file, output_dir=tmp_path)
        assert out.name == "hearts.html"

    def test_config_is_in_output(self, template_file, tmp_path):
        from env_builder.injector import inject_config
        config = self._config()
        out = inject_config(config, template_file, output_dir=tmp_path)
        html = out.read_text()
        assert '"Hearts"' in html

    def test_old_config_is_gone(self, template_file, tmp_path):
        from env_builder.injector import inject_config
        out = inject_config(self._config(), template_file, output_dir=tmp_path)
        html = out.read_text()
        assert '"DEFAULT"' not in html

    def test_engine_script_preserved(self, template_file, tmp_path):
        from env_builder.injector import inject_config
        out = inject_config(self._config(), template_file, output_dir=tmp_path)
        html = out.read_text()
        assert "/* engine */" in html

    def test_extract_config_round_trips(self, template_file, tmp_path):
        from env_builder.injector import inject_config, extract_config
        config = self._config()
        out = inject_config(config, template_file, output_dir=tmp_path)
        recovered = extract_config(out)
        assert recovered["game_name"] == "Hearts"
        assert recovered["players"] == 4

    def test_game_name_with_spaces(self, template_file, tmp_path):
        from env_builder.injector import inject_config
        config = self._config()
        config["game_name"] = "Gin Rummy"
        out = inject_config(config, template_file, output_dir=tmp_path)
        assert out.name == "gin_rummy.html"

    def test_missing_template_raises(self, tmp_path):
        from env_builder.injector import inject_config
        with pytest.raises(FileNotFoundError):
            inject_config(self._config(), "nonexistent.html", output_dir=tmp_path)

    def test_template_without_config_block_raises(self, tmp_path):
        from env_builder.injector import inject_config
        bad_template = tmp_path / "bad.html"
        bad_template.write_text("<html><body>no config block</body></html>")
        with pytest.raises(ValueError, match="game-config"):
            inject_config(self._config(), bad_template, output_dir=tmp_path)

    def test_custom_output_filename(self, template_file, tmp_path):
        from env_builder.injector import inject_config
        out = inject_config(
            self._config(), template_file,
            output_dir=tmp_path, output_filename="my_custom.html"
        )
        assert out.name == "my_custom.html"
