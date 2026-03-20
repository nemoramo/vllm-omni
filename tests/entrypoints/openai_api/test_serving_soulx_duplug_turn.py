# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_omni"
    / "entrypoints"
    / "openai"
    / "serving_turn.py"
)
SOULX_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_omni"
    / "entrypoints"
    / "openai"
    / "serving_soulx_duplug_turn.py"
)
SOULX_MODULE_SPEC = importlib.util.spec_from_file_location("test_serving_soulx_duplug_adapter_module", SOULX_MODULE_PATH)
assert SOULX_MODULE_SPEC is not None and SOULX_MODULE_SPEC.loader is not None
SOULX_MODULE = importlib.util.module_from_spec(SOULX_MODULE_SPEC)
SOULX_MODULE_SPEC.loader.exec_module(SOULX_MODULE)

OPENAI_PACKAGE = types.ModuleType("vllm_omni.entrypoints.openai")
OPENAI_PACKAGE.__path__ = [str(MODULE_PATH.parent)]
sys.modules["vllm_omni.entrypoints.openai"] = OPENAI_PACKAGE
sys.modules["vllm_omni.entrypoints.openai.serving_soulx_duplug_turn"] = SOULX_MODULE

MODULE_SPEC = importlib.util.spec_from_file_location("test_serving_soulx_duplug_turn_module", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
TURN_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(TURN_MODULE)


class DummyTurnModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(infer_config=SimpleNamespace(input={"chunk_size": 2560}))
        self.reset()

    def reset(self) -> None:
        self._runtime = {"count": 0}

    def restore_runtime(self, state) -> None:
        self._runtime = dict(state or {"count": 0})

    def snapshot_runtime(self):
        return dict(getattr(self, "_runtime", {}))

    def process(self, audio):
        self._runtime["count"] += 1
        return {
            "state": "idle",
            "num_samples": int(audio.shape[0]),
            "count": self._runtime["count"],
            "asr_segment": "",
            "asr_buffer": "",
        }


def test_turn_model_resolution_detects_model_id() -> None:
    definition = TURN_MODULE.resolve_turn_model_definition("Soul-AILab/SoulX-Duplug-0.6B")
    assert definition is not None
    assert definition.name == "soulx-duplug"
    assert TURN_MODULE.get_turn_supported_tasks("Soul-AILab/SoulX-Duplug-0.6B") == {"turn"}
    assert TURN_MODULE.resolve_turn_model_definition("Qwen/Qwen3-0.6B") is None


def test_turn_model_resolution_detects_local_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_config:",
                "  task: state_prediction",
                "  glm_tokenizer_path: pretrained_models/glm-4-voice-tokenizer",
                "  model_name: pretrained_models/Qwen3-0.6B-expand_vocab_v2",
            ]
        )
    )

    definition = TURN_MODULE.resolve_turn_model_definition(str(tmp_path))
    assert definition is not None
    assert definition.name == "soulx-duplug"


def test_build_turn_app_smoke(monkeypatch) -> None:
    fake_module = types.ModuleType("vllm_omni.model_executor.models.soulx_duplug")
    fake_module.load_turn_model = lambda _: DummyTurnModel()
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.model_executor.models.soulx_duplug",
        fake_module,
    )
    args = Namespace(
        model="/tmp/Soul-AILab/SoulX-Duplug-0.6B",
        served_model_name=None,
    )
    definition = TURN_MODULE.resolve_turn_model_definition(args.model)
    assert definition is not None

    with TestClient(TURN_MODULE.build_turn_app(args, definition)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "healthy",
            "mode": "turn",
            "turn_model": "soulx-duplug",
        }

        supported_tasks = asyncio.run(client.app.state.engine_client.get_supported_tasks())
        assert supported_tasks == {"turn"}

        models = client.get("/v1/models")
        assert models.status_code == 200
        assert models.json() == {
            "data": [{"id": "/tmp/Soul-AILab/SoulX-Duplug-0.6B", "object": "model"}]
        }

        with client.websocket_connect("/turn") as websocket:
            audio = np.zeros(2560, dtype=np.float32)
            websocket.send_json(
                {
                    "type": "audio",
                    "session_id": "test-session",
                    "audio": base64.b64encode(audio.tobytes()).decode(),
                }
            )
            response = websocket.receive_json()

        assert response["type"] == "turn_state"
        assert response["session_id"] == "test-session"
        assert response["state"]["state"] == "idle"
        assert response["state"]["num_samples"] == 2560
        assert response["state"]["count"] == 1


def test_turn_sessions_restore_state_and_isolate_sessions(monkeypatch) -> None:
    fake_module = types.ModuleType("vllm_omni.model_executor.models.soulx_duplug")
    fake_module.load_turn_model = lambda _: DummyTurnModel()
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.model_executor.models.soulx_duplug",
        fake_module,
    )
    args = Namespace(
        model="/tmp/Soul-AILab/SoulX-Duplug-0.6B",
        served_model_name=None,
    )
    definition = TURN_MODULE.resolve_turn_model_definition(args.model)
    assert definition is not None

    with TestClient(TURN_MODULE.build_turn_app(args, definition)) as client:
        with client.websocket_connect("/turn") as websocket:
            audio = np.zeros(2560, dtype=np.float32)
            payload = {
                "type": "audio",
                "audio": base64.b64encode(audio.tobytes()).decode(),
            }

            websocket.send_json({**payload, "session_id": "session-a"})
            first = websocket.receive_json()
            websocket.send_json({**payload, "session_id": "session-a"})
            second = websocket.receive_json()
            websocket.send_json({**payload, "session_id": "session-b"})
            third = websocket.receive_json()

        assert first["state"]["count"] == 1
        assert second["state"]["count"] == 2
        assert third["state"]["count"] == 1


def test_turn_rejects_invalid_chunk_size(monkeypatch) -> None:
    fake_module = types.ModuleType("vllm_omni.model_executor.models.soulx_duplug")
    fake_module.load_turn_model = lambda _: DummyTurnModel()
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.model_executor.models.soulx_duplug",
        fake_module,
    )
    args = Namespace(
        model="/tmp/Soul-AILab/SoulX-Duplug-0.6B",
        served_model_name=None,
    )
    definition = TURN_MODULE.resolve_turn_model_definition(args.model)
    assert definition is not None

    with TestClient(TURN_MODULE.build_turn_app(args, definition)) as client:
        with client.websocket_connect("/turn") as websocket:
            bad_audio = np.zeros(1280, dtype=np.float32)
            websocket.send_json(
                {
                    "type": "audio",
                    "session_id": "test-session",
                    "audio": base64.b64encode(bad_audio.tobytes()).decode(),
                }
            )
            response = websocket.receive_json()

        assert response["type"] == "error"
        assert response["error"]["code"] == "invalid_audio_chunk_size"


def test_soulx_adapter_keeps_generic_error_contract() -> None:
    assert SOULX_MODULE.SOULX_DUPLUG_TURN_MODEL.supported_tasks == ("turn",)
    assert SOULX_MODULE.is_soulx_duplug_model("Soul-AILab/SoulX-Duplug-0.6B")
