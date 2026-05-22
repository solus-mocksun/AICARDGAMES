"""
pvp_server/ws_handler.py

WebSocket message handler for PvP rooms.

Protocol (client → server):
    { "type": "create_room", "game_id": "hearts", "max_players": 4, "name": "Alice" }
    { "type": "join_room",   "room_id": "ABC123", "name": "Bob" }
    { "type": "move",        "action": { ... } }
    { "type": "state_sync",  "state": { ... } }   # host pushes full state
    { "type": "chat",        "text": "hello" }
    { "type": "ping" }

Protocol (server → client):
    { "type": "room_created",  "room_id", "player_idx", "game_id", "max_players" }
    { "type": "joined",        "room_id", "player_idx", "game_id", "players", "max_players" }
    { "type": "player_joined", "name", "player_idx", "players", "player_count", "max_players" }
    { "type": "game_start",    "your_player_idx", "players", "room_id", "game_id" }
    { "type": "move",          "player_idx", "action" }
    { "type": "state_sync",    "from_player_idx", "state" }
    { "type": "chat",          "name", "player_idx", "text" }
    { "type": "player_left",   "name", "player_idx", "players", "player_count" }
    { "type": "error",         "message" }
    { "type": "pong" }
"""

from __future__ import annotations

from fastapi import WebSocket, WebSocketDisconnect

from pvp_server.room_manager import Player, RoomManager

# One shared instance for the whole process
manager = RoomManager()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def pvp_ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            await _handle(ws, data)
    except WebSocketDisconnect:
        await _on_disconnect(ws)
    except Exception:
        # Swallow transport errors; clean up anyway
        await _on_disconnect(ws)


# ---------------------------------------------------------------------------
# Message dispatch
# ---------------------------------------------------------------------------

async def _handle(ws: WebSocket, data: dict) -> None:  # noqa: C901
    msg_type = data.get("type")

    # ── create_room ──────────────────────────────────────────────────────
    if msg_type == "create_room":
        game_id     = str(data.get("game_id") or "unknown")
        max_players = max(2, min(int(data.get("max_players") or 2), 8))
        name        = _clean_name(data.get("name"), default="Player 1")

        room_id = manager.create_room(game_id=game_id, max_players=max_players)
        room    = manager.get_room(room_id)
        room.players.append(Player(ws=ws, name=name, player_idx=0))

        await ws.send_json({
            "type":        "room_created",
            "room_id":     room_id,
            "player_idx":  0,
            "game_id":     game_id,
            "max_players": max_players,
        })

    # ── join_room ────────────────────────────────────────────────────────
    elif msg_type == "join_room":
        room_id = (data.get("room_id") or "").strip().upper()
        name    = _clean_name(data.get("name"), default="Player")
        room    = manager.get_room(room_id)

        if not room:
            await ws.send_json({"type": "error", "message": f"Room '{room_id}' not found"})
            return
        if room.state == "playing":
            await ws.send_json({"type": "error", "message": "Game already in progress"})
            return
        if room.is_full():
            await ws.send_json({"type": "error", "message": "Room is full"})
            return

        # Make name unique within room
        existing = room.player_names()
        base, i = name, 2
        while name in existing:
            name = f"{base} {i}"
            i   += 1

        player_idx = len(room.players)
        room.players.append(Player(ws=ws, name=name, player_idx=player_idx))

        # Confirm to the joiner
        await ws.send_json({
            "type":        "joined",
            "room_id":     room_id,
            "player_idx":  player_idx,
            "game_id":     room.game_id,
            "players":     room.player_names(),
            "max_players": room.max_players,
        })

        # Notify everyone else
        await manager.broadcast(room, {
            "type":         "player_joined",
            "name":         name,
            "player_idx":   player_idx,
            "players":      room.player_names(),
            "player_count": len(room.players),
            "max_players":  room.max_players,
        }, exclude=ws)

        # Auto-start when full
        if room.is_full():
            room.state = "playing"
            names = room.player_names()
            for i, p in enumerate(room.players):
                await p.ws.send_json({
                    "type":           "game_start",
                    "your_player_idx": i,
                    "players":        names,
                    "room_id":        room_id,
                    "game_id":        room.game_id,
                })

    # ── move (relay to all other players) ───────────────────────────────
    elif msg_type == "move":
        result = manager.find_player(ws)
        if result:
            room, player = result
            await manager.broadcast(room, {
                "type":       "move",
                "player_idx": player.player_idx,
                "action":     data.get("action"),
            }, exclude=ws)

    # ── state_sync (host pushes authoritative snapshot) ─────────────────
    elif msg_type == "state_sync":
        result = manager.find_player(ws)
        if result:
            room, player = result
            await manager.broadcast(room, {
                "type":             "state_sync",
                "from_player_idx":  player.player_idx,
                "state":            data.get("state"),
            }, exclude=ws)

    # ── chat ─────────────────────────────────────────────────────────────
    elif msg_type == "chat":
        result = manager.find_player(ws)
        if result:
            room, player = result
            text = str(data.get("text") or "").strip()[:256]
            if text:
                await manager.broadcast(room, {
                    "type":       "chat",
                    "name":       player.name,
                    "player_idx": player.player_idx,
                    "text":       text,
                })

    # ── ping ─────────────────────────────────────────────────────────────
    elif msg_type == "ping":
        await ws.send_json({"type": "pong"})


# ---------------------------------------------------------------------------
# Disconnect cleanup
# ---------------------------------------------------------------------------

async def _on_disconnect(ws: WebSocket) -> None:
    result = manager.remove_player(ws)
    if not result:
        return
    room, player = result
    # Notify remaining players (room still exists if not empty)
    if not room.is_empty():
        await manager.broadcast(room, {
            "type":         "player_left",
            "name":         player.name,
            "player_idx":   player.player_idx,
            "players":      room.player_names(),
            "player_count": len(room.players),
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_name(raw, *, default: str) -> str:
    name = str(raw or "").strip()[:32]
    return name if name else default
