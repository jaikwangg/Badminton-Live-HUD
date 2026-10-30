"""HTTP operator endpoints described by Contract v1."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from hud.bus.shm_ring import SharedFrameRing
from hud.config import AppConfig
from hud.fusion.state import FusionState

_OPERATOR_DIR = Path(__file__).resolve().parents[3] / "web" / "operator"


def read_frame_jpeg(config: AppConfig) -> bytes | None:
    """Encode the newest complete frame on the bus, or None when there is no bus or frame."""
    import cv2
    import numpy as np

    try:
        ring = SharedFrameRing(
            config.bus.name,
            config.source.width,
            config.source.height,
            config.bus.slots,
            create=False,
        )
    except (FileNotFoundError, ValueError):
        return None
    try:  # create=False attaches untracked, so closing never unlinks the capture's ring
        frame = ring.read_latest()
    except (ValueError, IndexError):  # struct.error is a ValueError: ring smaller than config
        return None
    finally:
        ring.close()
    if frame is None:
        return None
    image = np.frombuffer(frame.image, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    ok, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes() if ok else None


def create_operator_app(state: FusionState, config: AppConfig | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Response
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("Install FastAPI and Uvicorn to run the operator API") from exc

    app = FastAPI()
    app_config = config or AppConfig()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return state.snapshot(drain_events=False)["health"]

    @app.get("/api/frame.jpg")
    async def frame_jpg() -> Response:
        # Off the event loop: copying and encoding a full frame takes milliseconds.
        jpeg = await asyncio.to_thread(read_frame_jpeg, app_config)
        if jpeg is None:
            raise HTTPException(status_code=503, detail="No camera frame on the frame bus yet")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

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
    # load_config resolves fusion.calibration_path against the config file's directory.
    state = FusionState(config.fusion["calibration_path"], fusion=config.fusion)
    ws_config = uvicorn.Config(create_ws_app(state), host="0.0.0.0", port=config.server["ws_port"])
    http_config = uvicorn.Config(
        create_operator_app(state, config), host="0.0.0.0", port=config.server["http_port"]
    )
    ws_server = uvicorn.Server(ws_config)
    http_server = uvicorn.Server(http_config)
    await asyncio.gather(
        ws_server.serve(), http_server.serve(), ingest_worker_messages(state, config)
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    if sys.platform == "win32":  # zmq.asyncio cannot poll on the default Proactor loop
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(serve(parser.parse_args().config))
