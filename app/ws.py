"""WebSocket endpoint with token-based session auth."""

import base64
import logging
import traceback

from fastapi import HTTPException
from starlette.websockets import WebSocket, WebSocketDisconnect

from app.costs import SpendingLimitExceeded
from app.handlers import (
    handle_costs,
    handle_costs_history,
    handle_costs_limit,
    handle_describe_image,
    handle_generate,
    handle_providers,
    handle_suggest_filename,
    handle_transcribe,
    handle_tts,
)
from app.session import registry

logger = logging.getLogger("bananastore.ws")


async def ws_endpoint(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Token required")
        return

    session = await registry.get_session(token)
    if not session:
        await websocket.close(code=4001, reason="Invalid token")
        return

    # Lifecycle: on_connect hook — host can reject the connection
    if registry.on_connect:
        allowed = await registry.on_connect(session, websocket)
        if not allowed:
            await websocket.close(code=4003, reason="Connection rejected")
            return

    await websocket.accept()
    session.websocket = websocket

    # Send auth message with the session token
    await websocket.send_json({"type": "auth", "token": session.token})

    try:
        while True:
            msg = await websocket.receive_json()
            req_id = msg.get("id")
            action = msg.get("action")
            payload = msg.get("payload") or {}

            try:
                result = await _dispatch(session, action, payload)
                await websocket.send_json({"id": req_id, "ok": True, "result": result})
            except HTTPException as exc:
                await websocket.send_json({
                    "id": req_id, "ok": False,
                    "error": exc.detail, "code": exc.status_code,
                })
            except SpendingLimitExceeded as exc:
                await websocket.send_json({
                    "id": req_id, "ok": False,
                    "error": str(exc), "code": 429,
                    "limit": exc.limit, "current": exc.current, "attempted": exc.attempted,
                })
            except Exception as exc:
                logger.error("WS dispatch error: %s\n%s", exc, traceback.format_exc())
                await websocket.send_json({
                    "id": req_id, "ok": False,
                    "error": str(exc) or "Internal error", "code": 500,
                })
    except WebSocketDisconnect:
        pass
    finally:
        session.websocket = None
        if registry.on_disconnect:
            await registry.on_disconnect(session)


async def _dispatch(session, action: str, payload: dict) -> dict | list | bytes:
    tracker = session.tracker

    if action == "providers":
        return await handle_providers()

    elif action == "suggest-filename":
        return await handle_suggest_filename(payload.get("description", ""))

    elif action == "transcribe":
        audio_b64 = payload.get("audio_b64", "")
        audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
        return await handle_transcribe(
            audio_bytes=audio_bytes,
            filename=payload.get("filename", "voice.webm"),
            content_type=payload.get("content_type", "audio/webm"),
        )

    elif action == "describe-image":
        return await handle_describe_image(
            image_data_url=payload.get("image_data_url", ""),
            source_text=payload.get("source_text", ""),
            language=payload.get("language", ""),
        )

    elif action == "tts":
        audio_bytes = await handle_tts(
            text=payload.get("text", ""),
            language=payload.get("language", ""),
        )
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {"audio_b64": audio_b64}

    elif action == "generate":
        return await handle_generate(
            provider=payload.get("provider", ""),
            description=payload.get("description", ""),
            quality=payload.get("quality", ""),
            ratio=payload.get("ratio", ""),
            format=payload.get("format", "Photo"),
            model=payload.get("model", ""),
            reference_images=payload.get("reference_images", []),
            tracker=tracker,
        )

    elif action == "costs":
        return await handle_costs(tracker)

    elif action == "costs-history":
        return await handle_costs_history(tracker)

    elif action == "costs-limit":
        return await handle_costs_limit(tracker, payload.get("limit_usd"))

    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
