"""
pvp_server/room_manager.py

In-memory room registry for PvP sessions.
Each Room holds N WebSocket connections (Players).
No game logic lives here — the server is a pure relay.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _make_room_id() -> str:
    """Generate a random 6-character alphanumeric room code."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Player:
    ws: Any          # FastAPI WebSocket — typed as Any to avoid circular import
    name: str
    player_idx: int


@dataclass
class Room:
    room_id: str
    game_id: str
    max_players: int
    players: List[Player] = field(default_factory=list)
    state: str = "waiting"   # waiting | playing | finished

    def is_full(self) -> bool:
        return len(self.players) >= self.max_players

    def is_empty(self) -> bool:
        return len(self.players) == 0

    def player_names(self) -> List[str]:
        return [p.name for p in self.players]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class RoomManager:
    def __init__(self) -> None:
        self._rooms: Dict[str, Room] = {}

    # ── Room creation ────────────────────────────────────────────────────

    def create_room(self, game_id: str, max_players: int) -> str:
        room_id = _make_room_id()
        while room_id in self._rooms:
            room_id = _make_room_id()
        self._rooms[room_id] = Room(
            room_id=room_id,
            game_id=game_id,
            max_players=max_players,
        )
        return room_id

    # ── Lookups ──────────────────────────────────────────────────────────

    def get_room(self, room_id: str) -> Optional[Room]:
        return self._rooms.get(room_id.upper())

    def find_player(self, ws: Any) -> Optional[Tuple[Room, Player]]:
        """Return (room, player) for this WebSocket, or None."""
        for room in self._rooms.values():
            for player in room.players:
                if player.ws is ws:
                    return room, player
        return None

    def list_rooms(self) -> List[dict]:
        return [
            {
                "room_id": r.room_id,
                "game_id": r.game_id,
                "players": len(r.players),
                "max_players": r.max_players,
                "state": r.state,
            }
            for r in self._rooms.values()
        ]

    # ── Player removal ───────────────────────────────────────────────────

    def remove_player(self, ws: Any) -> Optional[Tuple[Room, Player]]:
        """
        Remove ws from whatever room it's in.
        Deletes the room if it becomes empty.
        Returns (room, player) if the player was found, else None.
        NOTE: 'room' is still returned even after players list is mutated;
              callers should check room.is_empty() before broadcasting.
        """
        for room in list(self._rooms.values()):
            for player in list(room.players):
                if player.ws is ws:
                    room.players.remove(player)
                    if room.is_empty():
                        del self._rooms[room.room_id]
                    return room, player
        return None

    # ── Broadcast ────────────────────────────────────────────────────────

    async def broadcast(
        self,
        room: Room,
        message: dict,
        exclude: Any = None,
    ) -> None:
        """Send message to all players in room (optionally skipping one ws)."""
        dead: List[Any] = []
        for player in list(room.players):
            if player.ws is exclude:
                continue
            try:
                await player.ws.send_json(message)
            except Exception:
                dead.append(player.ws)
        for ws in dead:
            self.remove_player(ws)
