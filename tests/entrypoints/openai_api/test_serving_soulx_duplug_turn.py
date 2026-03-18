# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import sys
import types
from argparse import Namespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai.serving_soulx_duplug_turn import (
    build_soulx_duplug_app,
    is_soulx_duplug_model,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DummyTurnModel:
    def reset(self) -> None:
        self._runtime = {}

    def restore_runtime(self, state) -> None:
        self._runtime = dict(state or {})

    def snapshot_runtime(self):
        return dict(getattr(self, "_runtime", {}))

    def process(self, audio):
        return {
            "state": "idle",
            "num_samples": int(audio.shape[0]),
            "asr_segment": "",
            "asr_buffer": "",
        }


def test_is_soulx_duplug_model_detects_model_id() -> None:
    assert is_soulx_duplug_model("Soul-AILab/SoulX-Duplug-0.6B")
    assert not is_soulx_duplug_model("Qwen/Qwen3-0.6B")


def test_is_soulx_duplug_model_detects_local_config(tmp_path) -> None:
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

    assert is_soulx_duplug_model(str(tmp_path))


def test_build_soulx_duplug_app_smoke(monkeypatch) -> None:
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

    with TestClient(build_soulx_duplug_app(args)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "healthy", "mode": "soulx-duplug-turn"}

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
