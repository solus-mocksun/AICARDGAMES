"""
web_server.py — FastAPI server for Card Game Builder

Serves the web UI, runs the build pipeline, and serves generated games.

Local:   python web_server.py  →  http://localhost:8000
Railway: set ANTHROPIC_API_KEY env var, deploy from GitHub
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import queue
import secrets
import sys
import tempfile
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse

from env_builder.build import build_game
from pvp_server.ws_handler import manager as pvp_manager, pvp_ws_endpoint

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Card Game Builder")


# ---------------------------------------------------------------------------
# Basic-Auth middleware
# Only active when AUTH_USER and AUTH_PASS are set as environment variables.
# Leave them unset for local development — auth is skipped automatically.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def basic_auth(request: Request, call_next) -> Response:
    auth_pass = os.environ.get("AUTH_PASS", "")

    # Auth disabled if AUTH_PASS is not set (safe local default)
    if not auth_pass:
        return await call_next(request)

    # WebSocket upgrade — skip (browser can't send Basic Auth on WS)
    if request.headers.get("upgrade", "").lower() == "websocket":
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Basic "):
        try:
            decoded  = base64.b64decode(auth_header[6:]).decode()
            _, _, password = decoded.partition(":")
            if secrets.compare_digest(password.encode(), auth_pass.encode()):
                return await call_next(request)
        except Exception:
            pass

    return Response(
        content="Unauthorized",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Card Game Builder"'},
    )


GAMES_DIR = Path("games")
GAMES_DIR.mkdir(exist_ok=True)

PUBLIC_DIR = Path("public")
PUBLIC_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Routes — UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("public/index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Routes — PvP
# ---------------------------------------------------------------------------

@app.websocket("/ws/pvp")
async def pvp_websocket(ws: WebSocket):
    await pvp_ws_endpoint(ws)


@app.get("/api/rooms")
async def list_rooms():
    """List all active PvP rooms (waiting or in-progress)."""
    return pvp_manager.list_rooms()


# ---------------------------------------------------------------------------
# Routes — game library
# ---------------------------------------------------------------------------

@app.get("/api/games")
async def list_games():
    games = []
    for d in sorted(GAMES_DIR.iterdir()) if GAMES_DIR.exists() else []:
        if not (d.is_dir() and (d / "index.html").exists()):
            continue
        cfg_file = d / "config.json"
        display_name = d.name.replace("_", " ").title()
        players = None
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text())
                display_name = cfg.get("game_name", display_name)
                players = cfg.get("players")
            except Exception:
                pass
        games.append({
            "id":      d.name,
            "name":    display_name,
            "players": players,
            "url":     f"/games/{d.name}/",
        })
    return games


# ---------------------------------------------------------------------------
# Routes — build pipeline (Server-Sent Events stream)
# ---------------------------------------------------------------------------

class _QueueWriter(io.TextIOBase):
    """Redirect stdout to a queue so we can stream it over SSE."""
    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, text: str) -> int:
        if text:
            self._q.put(text)
        return len(text)

    def flush(self) -> None:
        pass


@app.post("/api/build")
async def build_endpoint(
    text: str       = Form(None),
    file: UploadFile = File(None),
):
    if not text and (not file or not file.filename):
        raise HTTPException(400, "Provide either rulebook text or upload a file")

    async def event_stream():
        q: queue.Queue = queue.Queue()

        def run() -> None:
            old_stdout = sys.stdout
            sys.stdout = _QueueWriter(q)
            tmp_path = None
            try:
                filepath = None
                if file and file.filename:
                    suffix = Path(file.filename).suffix or ".txt"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.file.read())
                        tmp_path = tmp.name
                    filepath = tmp_path

                out_path = build_game(
                    text=text if not filepath else None,
                    rulebook=filepath,
                    output_dir=str(GAMES_DIR),
                    verbose=True,
                )
                q.put(f"__DONE__:{out_path.name}")
            except Exception as exc:
                q.put(f"__ERROR__:{exc}")
            finally:
                sys.stdout = old_stdout
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        while True:
            try:
                msg = q.get(timeout=0.15)
                if msg.startswith("__DONE__:"):
                    game_id = msg[9:]
                    payload = json.dumps({
                        "type":    "done",
                        "game_id": game_id,
                        "url":     f"/games/{game_id}/",
                    })
                    yield f"data: {payload}\n\n"
                    break
                elif msg.startswith("__ERROR__:"):
                    payload = json.dumps({"type": "error", "message": msg[10:]})
                    yield f"data: {payload}\n\n"
                    break
                else:
                    payload = json.dumps({"type": "log", "text": msg})
                    yield f"data: {payload}\n\n"
            except queue.Empty:
                if not thread.is_alive():
                    break
                await asyncio.sleep(0.05)
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Routes — serve generated game files
# ---------------------------------------------------------------------------

@app.get("/games/{game_name}")
async def game_redirect(game_name: str):
    return RedirectResponse(url=f"/games/{game_name}/")


@app.get("/games/{game_name}/")
async def game_index(game_name: str):
    p = GAMES_DIR / game_name / "index.html"
    if not p.exists():
        raise HTTPException(404, f"Game not found: {game_name}")
    return FileResponse(p)


@app.get("/games/{game_name}/{path:path}")
async def game_asset(game_name: str, path: str):
    p = GAMES_DIR / game_name / path
    if not p.exists():
        raise HTTPException(404, f"File not found: {path}")
    return FileResponse(p)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  Card Game Builder → http://localhost:{port}\n")
    uvicorn.run("web_server:app", host="0.0.0.0", port=port, reload=False)
