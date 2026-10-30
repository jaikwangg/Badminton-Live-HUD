"""HTTP operator endpoints described by Contract v1."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from hud.fusion.state import FusionState

_OPERATOR_DIR = Path(__file__).resolve().parents[3] / "web" / "operator"


def create_operator_app(state: FusionState) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("Install FastAPI and Uvicorn to run the operator API") from exc

    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return state.snapshot(drain_events=False)["health"]

    @app.post("/api/calibrate")
    async def calibrate(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            state.calibrate(payload["corners_px"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/players")
    async def players(payload: dict[str, Any]) -> dict[str, bool]:
        try:
            values = payload["players"]
            # `team` is optional for legacy clients and then defaults to the starting side.
            if any(
                not {"id", "name", "color", "side"}
                <= set(item)
                <= {"id", "name", "color", "team", "side"}
                for item in values
            ):
                raise ValueError("Each player must include id, name, color, team, and side")
            state.set_players(values, payload.get("mode", "singles"))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/score")
    async def score(payload: dict[str, Any]) -> dict[str, bool]:
        try:
            team = payload["team"] if "team" in payload else payload["side"]
            state.add_score(int(team), int(payload["delta"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/swap_ids")
    async def swap_ids(payload: dict[str, Any]) -> dict[str, bool]:
        try:
            state.swap_ids(int(payload["a"]), int(payload["b"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/point/confirm")
    async def confirm_point(payload: dict[str, Any]) -> dict[str, bool]:
        try:
            event_ts, accept = payload["event_ts"], payload["accept"]
            if type(event_ts) is not int or type(accept) is not bool:
                raise ValueError("event_ts must be an integer and accept must be a boolean")
            state.confirm_point(event_ts, accept)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/rally/reset")
    async def reset_rally() -> dict[str, bool]:
        state.reset_rally()
        return {"ok": True}

    # Mounted last so the API routes above take precedence; serves the page at `/`.
    app.mount("/", StaticFiles(directory=_OPERATOR_DIR, html=True), name="operator")
    return app


async def serve(config_path: str | None = None) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install Uvicorn to start the HTTP services") from exc
    from hud.config import load_config
    from hud.server.ws_server import create_ws_app, ingest_worker_messages

    config = load_config(config_path)
    # `fusion` is a plain dict from config; older configs without it keep the defaults.
    state = FusionState(fusion=getattr(config, "fusion", {}) or {})
    ws_config = uvicorn.Config(create_ws_app(state), host="0.0.0.0", port=config.server["ws_port"])
    http_config = uvicorn.Config(
        create_operator_app(state), host="0.0.0.0", port=config.server["http_port"]
    )
    ws_server = uvicorn.Server(ws_config)
    http_server = uvicorn.Server(http_config)
    await asyncio.gather(ws_server.serve(), http_server.serve(), ingest_worker_messages(state))


if __name__ == "__main__":
    asyncio.run(serve())
