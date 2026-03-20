from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import WebSocket, WebSocketDisconnect

SESSION_TTL_SEC = 60
GC_INTERVAL_SEC = 10


def _looks_like_soulx_config(config: dict[str, Any]) -> bool:
    model_config = config.get("model_config") or {}
    return (
        model_config.get("task") == "state_prediction"
        and "glm_tokenizer_path" in model_config
        and "model_name" in model_config
    )


def is_soulx_duplug_model(model: str) -> bool:
    if "SoulX-Duplug" in model:
        return True

    candidate = Path(model).expanduser()
    config_path = candidate / "config.yaml"
    if config_path.exists():
        try:
            return _looks_like_soulx_config(yaml.safe_load(config_path.read_text()) or {})
        except Exception:
            return False

    # Keep detection narrow so unrelated model startups do not probe the network.
    return False


class TurnTakingEngine:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.contexts: dict[str, Any] | None = None

    def process(self, audio: np.ndarray) -> dict[str, Any]:
        if self.contexts is None:
            self.model.reset()
        else:
            self.model.restore_runtime(self.contexts)
        result = self.model.process(audio)
        self.contexts = self.model.snapshot_runtime()
        return result


class TurnSession:
    def __init__(self, engine: TurnTakingEngine) -> None:
        self.engine = engine
        self.last_state: str | None = None
        now = time.time()
        self.created_ts = now
        self.last_active_ts = now

    def touch(self) -> None:
        self.last_active_ts = time.time()

    def feed_audio(self, audio: np.ndarray) -> dict[str, Any]:
        self.touch()
        result = self.engine.process(audio)
        self.last_state = result["state"]
        return result


class SoulXDuplugTurnHandler:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.sessions: dict[str, TurnSession] = {}
        infer_config = getattr(getattr(model, "config", None), "infer_config", None)
        input_config = getattr(infer_config, "input", None)
        self.expected_chunk_size = input_config.get("chunk_size") if isinstance(input_config, dict) else None
        # The model snapshots per-session runtime but still mutates one shared
        # module instance, so requests need serialization to avoid cross-session
        # state corruption.
        self._model_lock = asyncio.Lock()

    async def session_gc_loop(self) -> None:
        while True:
            now = time.time()
            expired = [sid for sid, sess in self.sessions.items() if now - sess.last_active_ts > SESSION_TTL_SEC]
            for sid in expired:
                del self.sessions[sid]
            await asyncio.sleep(GC_INTERVAL_SEC)

    async def handle_session(self, ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if data.get("type") != "audio":
                    continue

                session_id = data.get("session_id")
                if not session_id:
                    continue

                try:
                    audio = np.frombuffer(base64.b64decode(data["audio"]), dtype=np.float32)
                except Exception:
                    continue
                if self.expected_chunk_size is not None and audio.size != self.expected_chunk_size:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "session_id": session_id,
                                "error": {
                                    "code": "invalid_audio_chunk_size",
                                    "message": (
                                        "Turn model /turn expects one float32 chunk per message "
                                        f"with exactly {self.expected_chunk_size} samples; got {audio.size}."
                                    ),
                                },
                                "ts": time.time(),
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue

                session = self.sessions.get(session_id)
                if session is None:
                    session = TurnSession(TurnTakingEngine(self.model))
                    self.sessions[session_id] = session

                async with self._model_lock:
                    state = session.feed_audio(audio)

                await ws.send_text(
                    json.dumps(
                        {
                            "type": "turn_state",
                            "session_id": session_id,
                            "state": state,
                            "ts": time.time(),
                        },
                        ensure_ascii=False,
                    )
                )
        except WebSocketDisconnect:
            return


class SoulXDuplugTurnModelDefinition:
    name = "soulx-duplug"
    supported_tasks = ("turn",)

    def matches(self, model: str) -> bool:
        return is_soulx_duplug_model(model)

    def create_handler(self, model: str) -> SoulXDuplugTurnHandler:
        from vllm_omni.model_executor.models.soulx_duplug import load_turn_model

        return SoulXDuplugTurnHandler(load_turn_model(model))


SOULX_DUPLUG_TURN_MODEL = SoulXDuplugTurnModelDefinition()
