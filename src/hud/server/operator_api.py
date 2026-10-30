"""HTTP operator endpoints described by Contract v1."""

from __future__ import annotations

import asyncio
from typing import Any

from hud.fusion.state import FusionState


def create_operator_app(state: FusionState) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
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
            if any(set(item) != {"id", "name", "color", "side"} for item in values):
                raise ValueError("Each player must include id, name, color, and side")
            state.set_players(values)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/score")
    async def score(payload: dict[str, Any]) -> dict[str, bool]:
        try:
            state.add_score(int(payload["side"]), int(payload["delta"]))
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

    return app


async def serve(config_path: str | None = None) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install Uvicorn to start the HTTP services") from exc
    from hud.config import load_config
    from hud.server.ws_server import create_ws_app, ingest_worker_messages

    config = load_config(config_path)
    state = FusionState()
    ws_config = uvicorn.Config(create_ws_app(state), host="0.0.0.0", port=config.server["ws_port"])
    http_config = uvicorn.Config(
        create_operator_app(state), host="0.0.0.0", port=config.server["http_port"]
    )
    ws_server = uvicorn.Server(ws_config)
    http_server = uvicorn.Server(http_config)
    await asyncio.gather(ws_server.serve(), http_server.serve(), ingest_worker_messages(state))


if __name__ == "__main__":
    asyncio.run(serve())
