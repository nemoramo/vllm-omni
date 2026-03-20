from __future__ import annotations

import copy
import logging
import math
import random
import re
import string
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor

from .speech_tokenizer import WhisperVQEncoder

logger = logging.getLogger(__name__)


def split_cn_en(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|[0-9]+", text)


def check_en(text: str) -> bool:
    if not text:
        return False

    symbol_pattern = re.compile(
        r"[\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
        r"\u2000-\u206F"
        r"\u3000-\u303F"
        r"\uFF00-\uFFEF]"
    )
    for char in reversed(text):
        if char.isdigit() or symbol_pattern.match(char):
            continue
        return not ("\u4e00" <= char <= "\u9fff")
    return True


def get_lcs_substrings(s1: Sequence[Any], s2: Sequence[Any]) -> tuple[Sequence[Any], Sequence[Any]]:
    if not s1 or not s2:
        return s1, s2

    m = len(s1)
    n = len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if s1[i] == s2[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    max_len = dp[0][0]
    if max_len == 0:
        return s1, s2

    start_i = -1
    start_j = -1
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j] and dp[i][j] == max_len:
                start_i = i
                start_j = j
                break
        if start_i != -1:
            break
    return s1[start_i:], s2[start_j:]


def remove_leading_backchannel(text: str) -> str:
    backchannel_chars = {"嗯", "啊", "哦", "噢", "呃", "哎", "哼", "嘿"}
    punctuation_chars = {
        " ",
        ",",
        ".",
        "?",
        "!",
        "，",
        "。",
        "？",
        "！",
        "、",
        "；",
        ";",
        "…",
        ":",
        "：",
    }
    skip_chars = backchannel_chars.union(punctuation_chars)
    for idx, char in enumerate(text):
        if char not in skip_chars:
            return text[idx:]
    return ""


def zh_norm(text: str) -> str:
    return text


def zh_remove_punc(text: str) -> str:
    punctuation_all = string.punctuation + "，。！？；：“”‘’、（）【】《》…"
    for token in punctuation_all:
        text = text.replace(token, "")
    return text.replace("  ", " ").strip()


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _has_nested_key(config: dict[str, Any], *keys: str) -> bool:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return False
        cursor = cursor[key]
    return True


def _maybe_copy_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        copied: list[Any] = []
        for item in value:
            if isinstance(item, tuple):
                copied.append(tuple(_maybe_copy_array(x) for x in item))
            else:
                copied.append(_maybe_copy_array(item))
        return copied
    if isinstance(value, dict):
        return {k: _maybe_copy_array(v) for k, v in value.items()}
    return value


def _clone_cache(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_cache(item) for item in value)
    if isinstance(value, list):
        return [_clone_cache(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_cache(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _resolve_torch_dtype(device: str, precision: str) -> torch.dtype:
    if not device.startswith("cuda"):
        return torch.float32
    lowered = precision.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "16-mixed"}:
        return torch.float16
    return torch.float32


@dataclass
class SoulXDuplugModelConfig:
    task: str = "state_prediction"
    text_vocab_size: int = 151643
    original_vocab_size: int = 151669
    lm_vocab_size: int = 151936
    tokenizer_vocab_size: int = 203566
    added_audio_token_size: int = 51866
    added_special_token_size: int = 31
    special_token_start: int = 151643
    added_token_start: int = 151669
    added_audio_token_start: int = 151700
    total_vocab_size: int = field(init=False)
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    audio_pad_token_id: int = 151673
    asr_eos_token_id: int = 151674
    asr_bos_token_id: int = 151675
    user_complete_token_id: int = 151676
    user_backchannel_token_id: int = 151677
    user_incomplete_token_id: int = 151678
    assistant_backchannel_token_id: int = 151679
    user_idle_token_id: int = 151680
    user_nonidle_token_id: int = 151681
    assistant_interrupt_token_id: int = 151682
    audio_embed_dim: int = 1280
    llm_dim: int = 2048
    glm_tokenizer_path: str = "pretrained_models/glm-4-voice-tokenizer"
    model_name: str = "pretrained_models/Qwen3-1.7B-expand_vocab_v2"
    init_ckpt_path: str = ""
    init_ckpt_path_lora: str = ""
    sampling_rate: int = 16000
    chunk_size: int = 960
    extract_token_batch: int = 256
    enable_audio_mask: bool = False
    asr_repetition_penalty: float = 1.0
    num_beam: int = 3
    punctuation: bool = False
    max_chunk_token_length: int = 50
    max_token_length: int = 1500
    enable_cascade_asr: bool = True
    enable_lora: bool = False
    lora_task_type: str = "CAUSAL_LM"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    enable_projector: bool = True
    freeze_projector: bool = False
    embed_only: bool = False

    def __post_init__(self) -> None:
        self.total_vocab_size = (
            self.original_vocab_size
            + self.added_audio_token_size
            + self.added_special_token_size
        )


@dataclass
class SoulXDuplugInferConfig:
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    seed: int = 42
    precision: str = "bf16"
    sample_rate: int = 16000
    max_wait_num: int = 10
    max_mistake_num: int = 5
    far_field_threshold: float = 0.02
    developer_mode: bool = False
    single_round: bool = False
    config_path: str = "config/config.yaml"
    input: dict[str, Any] = field(
        default_factory=lambda: {
            "chunk_size": 2560,
            "audio_back_size": 15360,
            "audio_ahead_size": 640,
            "sample_rate": 16000,
            "chunk_token_len_small": 2,
        }
    )
    asr: dict[str, Any] = field(
        default_factory=lambda: {
            "model_name": "paraformer",
            "language": "auto",
            "max_chunk_token_length": 256,
        }
    )


@dataclass
class SoulXDuplugRunConfig:
    model_config: SoulXDuplugModelConfig = field(default_factory=SoulXDuplugModelConfig)
    infer_config: SoulXDuplugInferConfig = field(default_factory=SoulXDuplugInferConfig)


class NullASR:
    def recognize(self, audio_chunk: np.ndarray, sample_rate: int = 16000, **_: Any) -> str:
        return ""


class ParaformerASR:
    def __init__(self) -> None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.asr_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            model_revision="v2.0.4",
            device="cuda",
            disable_pbar=True,
            disable_update=True,
        )

    def recognize(self, audio_chunk: np.ndarray, sample_rate: int = 16000, **_: Any) -> str:
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)
        if sample_rate != 16000:
            import soxr

            audio_chunk = soxr.resample(audio_chunk, sample_rate, 16000)
        try:
            return self.asr_pipeline(audio_chunk)[0]["text"].strip()
        except Exception:
            return ""


class SenseVoiceASR:
    def __init__(self, language: str = "auto") -> None:
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self.sensevoice_model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=False,
            device="cuda",
            disable_pbar=True,
            disable_update=True,
        )
        self.language = language
        self._postprocess = rich_transcription_postprocess
        remove_set = {"😊", "😔", "😡", "😰", "🤢", "😮", "🎼", "👏", "😀", "😭", "🤧", "😷"}
        self._pattern = "[" + "".join(remove_set) + "]"

    def _clean_text(self, text: str) -> str:
        if not re.search(r"[\u4e00-\u9fff]|[a-zA-Z]", text):
            return ""
        return re.sub(self._pattern, "", text)

    def recognize(self, audio_chunk: np.ndarray, sample_rate: int = 16000, language: str | None = None) -> str:
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)
        if sample_rate != 16000:
            import soxr

            audio_chunk = soxr.resample(audio_chunk, sample_rate, 16000)
        if language is None:
            language = self.language
        try:
            result = self.sensevoice_model.generate(
                input=audio_chunk,
                cache={},
                language=language,
                use_itn=True,
                batch_size=16,
            )[0]["text"]
            return self._clean_text(self._postprocess(result).strip())
        except Exception:
            return ""


def resolve_model_root(model: str) -> Path:
    candidate = Path(model).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return Path(snapshot_download(model)).resolve()


def _resolve_model_asset_path(model_root: Path, configured_path: str) -> Path:
    configured = Path(configured_path)
    if configured.is_absolute():
        return configured
    parts = configured.parts
    if parts and parts[0] == "pretrained_models":
        configured = Path(*parts[1:])
    return (model_root / configured).resolve()


def _infer_llm_hidden_size(model_name: str) -> int | None:
    try:
        llm_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None
    return getattr(llm_config, "hidden_size", None)


def load_turn_run_config(model: str, device: str | None = None) -> tuple[Path, SoulXDuplugRunConfig]:
    model_root = resolve_model_root(model)
    config_path = model_root / "config.yaml"
    raw_cfg = yaml.safe_load(config_path.read_text()) or {}

    base = {
        "model_config": SoulXDuplugModelConfig().__dict__.copy(),
        "infer_config": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": 42,
            "precision": "bf16",
            "sample_rate": 16000,
            "max_wait_num": 10,
            "max_mistake_num": 5,
            "far_field_threshold": 0.02,
            "developer_mode": False,
            "single_round": False,
            "config_path": str(config_path),
            "input": SoulXDuplugInferConfig().input.copy(),
            "asr": SoulXDuplugInferConfig().asr.copy(),
        },
    }
    base["model_config"].pop("total_vocab_size", None)
    merged = _deep_update(base, raw_cfg)

    if device is not None:
        merged["infer_config"]["device"] = device

    model_config = SoulXDuplugModelConfig(**merged["model_config"])
    infer_config = SoulXDuplugInferConfig(**merged["infer_config"])
    model_config.glm_tokenizer_path = str(
        _resolve_model_asset_path(model_root, model_config.glm_tokenizer_path)
    )
    model_config.model_name = str(_resolve_model_asset_path(model_root, model_config.model_name))
    if (
        not _has_nested_key(raw_cfg, "model_config", "llm_dim")
        and (hidden_size := _infer_llm_hidden_size(model_config.model_name)) is not None
    ):
        model_config.llm_dim = hidden_size
    if (
        not _has_nested_key(raw_cfg, "model_config", "enable_lora")
        and bool(model_config.init_ckpt_path_lora)
    ):
        model_config.enable_lora = True
    if model_config.init_ckpt_path:
        model_config.init_ckpt_path = str(
            _resolve_model_asset_path(model_root, model_config.init_ckpt_path)
        )
    if model_config.init_ckpt_path_lora:
        model_config.init_ckpt_path_lora = str(
            _resolve_model_asset_path(model_root, model_config.init_ckpt_path_lora)
        )
    # The Hugging Face artifact omits these official service-time overrides.
    if not _has_nested_key(raw_cfg, "infer_config", "max_wait_num"):
        infer_config.max_wait_num = 8
    if not _has_nested_key(raw_cfg, "infer_config", "max_mistake_num"):
        infer_config.max_mistake_num = 3
    return model_root, SoulXDuplugRunConfig(model_config=model_config, infer_config=infer_config)


class EncoderProjector(nn.Module):
    def __init__(self, config: SoulXDuplugModelConfig) -> None:
        super().__init__()
        self.linear1 = nn.Linear(config.audio_embed_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(2048, config.llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x.contiguous())
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return self.linear3(x)


class SoulXDuplugStatePredictionModel(nn.Module):
    def __init__(self, config: SoulXDuplugRunConfig) -> None:
        super().__init__()
        self.config = config
        self.model_config = config.model_config
        self.infer_config = config.infer_config
        self.asr_eos_token_id = self.model_config.asr_eos_token_id
        self.sampling_rate = self.model_config.sampling_rate
        self.token_samples = int(0.08 * self.sampling_rate)

        self.glm_tokenizer = WhisperVQEncoder.from_pretrained(self.model_config.glm_tokenizer_path)
        self.glm_tokenizer.eval()
        for param in self.glm_tokenizer.parameters():
            param.requires_grad = False

        self.audio_projector = EncoderProjector(self.model_config) if self.model_config.enable_projector else None
        if self.audio_projector is not None and self.model_config.freeze_projector:
            self.audio_projector.eval()
            for param in self.audio_projector.parameters():
                param.requires_grad = False

        torch_dtype = _resolve_torch_dtype(
            self.infer_config.device,
            self.infer_config.precision,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        if self.model_config.enable_lora:
            peft_config = LoraConfig(
                task_type=self.model_config.lora_task_type,
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                lora_dropout=self.model_config.lora_dropout,
            )
            self.llm = get_peft_model(self.llm, peft_config)

        checkpoint_path = self.model_config.init_ckpt_path_lora or self.model_config.init_ckpt_path
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            try:
                self.load_state_dict(state_dict)
            except Exception:
                self.load_state_dict(state_dict, strict=False)

        if hasattr(self.llm.model, "embed_tokens"):
            self.embed_tokens_func = self.llm.model.embed_tokens
        elif hasattr(self.llm.model, "model") and hasattr(self.llm.model.model, "embed_tokens"):
            self.embed_tokens_func = self.llm.model.model.embed_tokens
        else:
            self.embed_tokens_func = self.llm.model.model.model.embed_tokens


class SoulXDuplugTurnModel:
    def __init__(self, model: str, device: str | None = None) -> None:
        self.model_root, self.config = load_turn_run_config(model, device=device)

        seed = self.config.infer_config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = self.config.infer_config.device
        self.model = SoulXDuplugStatePredictionModel(self.config).eval().to(self.device)
        self.model_dtype = next(self.model.llm.parameters()).dtype
        if self.model.audio_projector is not None:
            self.model.audio_projector = self.model.audio_projector.to(dtype=self.model_dtype)
        self.model.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.model_config.glm_tokenizer_path
        )

        if hasattr(self.model.llm.model, "embed_tokens"):
            self.embed_tokens_func = self.model.llm.model.embed_tokens
        elif hasattr(self.model.llm.model, "model") and hasattr(self.model.llm.model.model, "embed_tokens"):
            self.embed_tokens_func = self.model.llm.model.model.embed_tokens
        else:
            self.embed_tokens_func = self.model.llm.model.model.model.embed_tokens

        self.cascade_asr = self._build_asr()
        self._init_hyperparameters()
        self._init_prompt_embeds()
        self.reset()

    def _build_asr(self) -> Any:
        asr_cfg = self.config.infer_config.asr
        model_name = asr_cfg.get("model_name", "paraformer")
        try:
            if model_name == "sensevoice":
                return SenseVoiceASR(language=asr_cfg.get("language", "auto"))
            return ParaformerASR()
        except Exception as exc:
            logger.warning(
                "Failed to initialize SoulX cascade ASR '%s'; falling back to NullASR: %s",
                model_name,
                exc,
            )
            return NullASR()

    def _init_hyperparameters(self) -> None:
        self.sampling_rate = self.config.infer_config.input["sample_rate"]
        self.chunk_token_len_small = self.config.infer_config.input.get("chunk_token_len_small", 2)
        self.developer_mode = self.config.infer_config.developer_mode

    def _init_prompt_embeds(self) -> None:
        tokenizer = self.model.tokenizer
        device = self.device
        self.input_text_tokens = torch.tensor(
            tokenizer.encode("<|task_duplex_predict|><|punctuation_off|>")
        ).to(device)
        self.audio_pad_token = torch.tensor(tokenizer.encode("<|padding|>")).to(device)
        self.audio_bos_token = torch.tensor(tokenizer.encode("<|begin_of_sentence|>")).to(device)
        self.audio_eos_token = torch.tensor(tokenizer.encode("<|end_of_sentence|>")).to(device)
        self.action_speak_token = torch.tensor(tokenizer.encode("<|user_complete|>")).to(device)
        self.action_wait_token = torch.tensor(tokenizer.encode("<|user_incomplete|>")).to(device)
        self.non_idle_token = torch.tensor(tokenizer.encode("<|user_nonidle|>")).to(device)

        self.text_embeds = self.embed_tokens_func(self.input_text_tokens)
        self.audio_pad_embeds = self.embed_tokens_func(self.audio_pad_token)
        self.audio_bos_embeds = self.embed_tokens_func(self.audio_bos_token)
        self.audio_eos_embeds = self.embed_tokens_func(self.audio_eos_token)
        self.action_speak_embeds = self.embed_tokens_func(self.action_speak_token)
        self.action_wait_embeds = self.embed_tokens_func(self.action_wait_token)
        self.non_idle_embeds = self.embed_tokens_func(self.non_idle_token)

    def _log(self, *args: Any, **kwargs: Any) -> None:
        if self.developer_mode:
            print(*args, **kwargs)

    def reset(self) -> None:
        input_cfg = self.config.infer_config.input
        self.buffer = (
            np.random.randn(input_cfg["audio_back_size"] + input_cfg["audio_ahead_size"]) * 0.0001
        )
        self.buffer_for_asr = np.random.randn(int(1.6 * self.sampling_rate)) * 0.00001
        self.cascade_buffer = np.random.randn(int(3.2 * self.sampling_rate)) * 0.00001
        self.speech_detected = False
        self.past_state: dict[str, Any] | None = None
        self.history_chunks: list[tuple[np.ndarray, str]] = []
        self.wait_idle_cnt = 0
        self.monitoring_wait_silence = False

    def snapshot_runtime(self) -> dict[str, Any]:
        return {
            "buffer": _maybe_copy_array(self.buffer),
            "buffer_for_asr": _maybe_copy_array(self.buffer_for_asr),
            "cascade_buffer": _maybe_copy_array(self.cascade_buffer),
            "speech_detected": self.speech_detected,
            "past_state": _maybe_copy_array(self.past_state),
            "history_chunks": _maybe_copy_array(self.history_chunks),
            "wait_idle_cnt": self.wait_idle_cnt,
            "monitoring_wait_silence": self.monitoring_wait_silence,
        }

    def restore_runtime(self, state: dict[str, Any] | None) -> None:
        if not state:
            self.reset()
            return
        self.buffer = _maybe_copy_array(state.get("buffer", self.buffer))
        self.buffer_for_asr = _maybe_copy_array(state.get("buffer_for_asr", self.buffer_for_asr))
        self.cascade_buffer = _maybe_copy_array(state.get("cascade_buffer", self.cascade_buffer))
        self.speech_detected = state.get("speech_detected", False)
        self.past_state = _maybe_copy_array(state.get("past_state"))
        self.history_chunks = _maybe_copy_array(state.get("history_chunks", []))
        self.wait_idle_cnt = state.get("wait_idle_cnt", 0)
        self.monitoring_wait_silence = state.get("monitoring_wait_silence", False)

    def get_rms(self, audio_chunk: np.ndarray) -> float:
        if audio_chunk.dtype == np.int16:
            samples = audio_chunk.astype(np.float32) / 32768.0
        elif audio_chunk.dtype == np.uint8:
            samples = (audio_chunk.astype(np.float32) - 128.0) / 128.0
        else:
            samples = np.clip(audio_chunk.astype(np.float32), -1.0, 1.0)
        return float(np.sqrt(np.mean(samples**2)))

    def get_chunk(self) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        input_cfg = self.config.infer_config.input
        required = input_cfg["chunk_size"] + input_cfg["audio_back_size"] + input_cfg["audio_ahead_size"]
        if len(self.buffer) < required:
            return False, None, None, None

        audio_back = self.buffer[: input_cfg["audio_back_size"]].astype(np.float32)
        process_chunk = self.buffer[
            input_cfg["audio_back_size"] : input_cfg["audio_back_size"] + input_cfg["chunk_size"]
        ].astype(np.float32)
        audio_ahead = self.buffer[
            input_cfg["audio_back_size"] + input_cfg["chunk_size"] : required
        ].astype(np.float32)
        self.buffer = self.buffer[input_cfg["chunk_size"] :]
        return True, process_chunk, audio_back, audio_ahead

    def process(self, audio_chunk: np.ndarray) -> dict[str, Any]:
        assert audio_chunk.dtype == np.float32
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        predicted_state: dict[str, Any] = {
            "state": "blank",
            "asr_segment": "",
            "asr_buffer": "",
        }
        start_prediction, process_chunk, audio_back, audio_ahead = self.get_chunk()
        if start_prediction:
            t_start = time.time()
            predicted_state = self.state_predict(process_chunk, audio_back, audio_ahead)
            self._log(f"[Timing] Total chunk: {time.time() - t_start:.4f}s")
        return predicted_state

    def state_predict(
        self,
        process_chunk: np.ndarray,
        audio_back: np.ndarray,
        audio_ahead: np.ndarray,
    ) -> dict[str, Any]:
        state, delta_text, asr_buffer = self.infer(process_chunk, audio_back, audio_ahead)

        if (
            self.get_rms(process_chunk) < self.config.infer_config.far_field_threshold
            and not self.speech_detected
            and state == "<|user_nonidle|>"
        ):
            self.reset()
            return {"state": "idle", "asr_segment": "", "asr_buffer": ""}

        assert self.past_state is not None
        self.past_state["history_len"] += 1
        self.history_chunks.append((process_chunk.copy(), state))
        if len(self.history_chunks) > 5:
            self.history_chunks.pop(0)

        if state == "<|user_idle|>":
            if self.monitoring_wait_silence:
                self.wait_idle_cnt += 1
                if self.wait_idle_cnt >= self.config.infer_config.max_wait_num:
                    if self.speech_detected:
                        segment = self.cascade_asr.recognize(self.buffer_for_asr, self.sampling_rate)
                        self.reset()
                        return {
                            "state": "speak",
                            "text": segment,
                            "asr_segment": delta_text,
                            "asr_buffer": asr_buffer,
                        }
            if self.past_state["history_len"] > 200 and not self.monitoring_wait_silence and not self.speech_detected:
                self.reset()

        elif state == "<|user_nonidle|>":
            self.speech_detected = True
            if self.monitoring_wait_silence:
                self.monitoring_wait_silence = False
                self.wait_idle_cnt = 0

            to_concat: list[np.ndarray] = []
            if len(self.history_chunks) >= 2:
                _, prev_state = self.history_chunks[-2]
                if prev_state in ["<|user_idle|>", "<|user_backchannel|>"]:
                    candidates: list[np.ndarray] = []
                    for idx in range(len(self.history_chunks) - 2, -1, -1):
                        chunk, chunk_state = self.history_chunks[idx]
                        if chunk_state in ["<|user_idle|>", "<|user_backchannel|>"]:
                            candidates.append(chunk)
                            if len(candidates) >= 5:
                                break
                        else:
                            break
                    to_concat = candidates[::-1]

            if to_concat:
                self.buffer_for_asr = np.concatenate([self.buffer_for_asr, *to_concat, process_chunk])
            else:
                self.buffer_for_asr = np.concatenate([self.buffer_for_asr, process_chunk])
            return {
                "state": "nonidle",
                "asr_segment": delta_text,
                "asr_buffer": asr_buffer,
            }

        elif state == "<|user_backchannel|>":
            if not self.speech_detected:
                self.reset()
            if self.monitoring_wait_silence:
                self.wait_idle_cnt = 1

        elif state == "<|user_complete|>":
            if self.speech_detected:
                self.buffer_for_asr = np.concatenate([self.buffer_for_asr, process_chunk])
                segment = self.cascade_asr.recognize(self.buffer_for_asr, self.sampling_rate)
                self.reset()
                return {
                    "state": "speak",
                    "text": segment,
                    "asr_segment": delta_text,
                    "asr_buffer": asr_buffer,
                }
            self.reset()

        elif state == "<|user_incomplete|>":
            self.monitoring_wait_silence = True
            self.wait_idle_cnt = 0
            self.buffer_for_asr = np.concatenate([self.buffer_for_asr, process_chunk])
        else:
            self.reset()

        return {
            "state": "idle",
            "asr_segment": delta_text,
            "asr_buffer": asr_buffer,
        }

    @torch.no_grad()
    def infer(
        self,
        audio_chunk: np.ndarray,
        audio_back: np.ndarray,
        audio_ahead: np.ndarray,
    ) -> tuple[str, str, str]:
        self.cascade_buffer = np.concatenate([self.cascade_buffer, audio_chunk])
        if self.past_state is None:
            self.past_state = {
                "input_embeds": self.text_embeds,
                "past_key_values": None,
                "delta_text": [],
                "cascade_text": "",
                "state": "",
                "history_len": 0,
                "mistake_len": 0,
                "checkpoint": None,
            }

        audio_tokens = self._audio_to_tokens(audio_back, audio_chunk, audio_ahead)
        audio_embeds = self._tokens_to_embeds(audio_tokens)
        self.past_state["input_embeds"] = torch.cat(
            (self.past_state["input_embeds"], audio_embeds), dim=0
        ).unsqueeze(0)
        delta_text = self._asr(audio_embeds)
        state = self._state_predict(delta_text)
        return state, delta_text, self.past_state["cascade_text"]

    def _audio_to_tokens(
        self,
        audio_back: np.ndarray,
        audio_chunk: np.ndarray,
        audio_ahead: np.ndarray,
    ) -> list[int]:
        audio_segment = np.concatenate([audio_back, audio_chunk, audio_ahead], axis=0)
        start_index = len(audio_back) // self.model.token_samples
        end_index = min(
            start_index + 2,
            math.ceil(audio_segment.shape[0] / self.model.token_samples),
        )
        valid_range = (start_index, end_index)

        pooling_kernel_size = self.model.glm_tokenizer.config.pooling_kernel_size or 1
        stride = (
            self.model.glm_tokenizer.conv1.stride[0]
            * self.model.glm_tokenizer.conv2.stride[0]
            * pooling_kernel_size
            * self.model.feature_extractor.hop_length
        )
        features = self.model.feature_extractor(
            [audio_segment],
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
            device=self.device,
            padding="longest",
            pad_to_multiple_of=stride,
        ).to(self.device)
        outputs = self.model.glm_tokenizer(**features)
        speech_tokens = outputs.quantized_token_ids
        attention_mask = features.attention_mask[
            :,
            :: self.model.glm_tokenizer.conv1.stride[0] * self.model.glm_tokenizer.conv2.stride[0],
        ]
        attention_mask = attention_mask[
            :, :: self.model.glm_tokenizer.config.pooling_kernel_size
        ]
        speech_token = speech_tokens[0][attention_mask[0].bool()].tolist()
        return speech_token[valid_range[0] : valid_range[1]]

    def _tokens_to_embeds(self, tokens: list[int]) -> torch.Tensor:
        token_tensor = torch.tensor(tokens, device=self.device)
        embeds = self.model.glm_tokenizer.codebook(token_tensor)
        assert self.model.audio_projector is not None
        embeds = embeds.to(dtype=self.model_dtype)
        embeds = self.model.audio_projector(embeds)
        embeds = embeds.to(dtype=self.model_dtype)
        if embeds.shape[0] < self.chunk_token_len_small:
            embeds = torch.cat(
                (
                    embeds,
                    self.audio_pad_embeds.expand(self.chunk_token_len_small - embeds.shape[0], -1),
                ),
                dim=0,
            )
        return embeds

    def _asr(self, audio_embeds: torch.Tensor) -> str:
        assert self.past_state is not None
        outputs = self.model.llm(
            inputs_embeds=self.past_state["input_embeds"],
            past_key_values=self.past_state["past_key_values"],
            use_cache=True,
        )
        logits = outputs.logits[0]
        current_kv = outputs.past_key_values
        pred = torch.argmax(logits, -1)[-1]

        delta_text = ""
        need_correction = False
        corrected_prev_delta = ""

        if pred != self.model.asr_eos_token_id:
            full_text = remove_leading_backchannel(
                self.cascade_asr.recognize(self.cascade_buffer, self.sampling_rate)
            )
            history_text = self.past_state.get("cascade_text", "")
            norm_full_text = split_cn_en(zh_norm(zh_remove_punc(full_text.strip())))
            norm_history_text = split_cn_en(zh_norm(zh_remove_punc(history_text.strip())))
            backup_norm_full_text = norm_full_text.copy()
            backup_norm_history_text = norm_history_text.copy()

            if len(norm_full_text) >= 5 and len(norm_history_text) >= 5:
                norm_full_text, norm_history_text = get_lcs_substrings(
                    norm_full_text,
                    norm_history_text,
                )

            prev_delta = self.past_state["delta_text"][-1] if self.past_state["delta_text"] else ""
            prev_delta_split = split_cn_en(prev_delta)
            len_prev = len(prev_delta_split)

            if len_prev > len(norm_history_text):
                norm_full_text = backup_norm_full_text
                norm_history_text = backup_norm_history_text

            history_base = norm_history_text[:-len_prev] if len_prev > 0 else norm_history_text

            if len(norm_full_text) > len(norm_history_text):
                current_segment_in_full = norm_full_text[len(history_base) : len(history_base) + len_prev]
                if current_segment_in_full == prev_delta_split:
                    delta_text = "".join(
                        [(item + " ") if check_en(item) else item for item in norm_full_text[len(norm_history_text) :]]
                    ).strip()
                else:
                    need_correction = True
                    corrected_prev_delta = "".join(
                        [(item + " ") if check_en(item) else item for item in current_segment_in_full]
                    ).strip()
                    delta_text = "".join(
                        [
                            (item + " ") if check_en(item) else item
                            for item in norm_full_text[len(history_base) + len_prev :]
                        ]
                    ).strip()
            elif len(norm_full_text) == len(norm_history_text):
                current_segment_in_full = norm_full_text[len(history_base) :]
                if current_segment_in_full != prev_delta_split:
                    need_correction = True
                    corrected_prev_delta = "".join(
                        [(item + " ") if check_en(item) else item for item in current_segment_in_full]
                    ).strip()
            else:
                need_correction = True
                remainder = norm_full_text[len(history_base) :]
                corrected_prev_delta = ""
                delta_text = "".join(
                    [(item + " ") if check_en(item) else item for item in remainder]
                ).strip()

            self.past_state["cascade_text"] = "".join(
                [(item + " ") if check_en(item) else item for item in norm_full_text]
            ).strip()

        if need_correction and self.past_state["checkpoint"] is not None:
            self.past_state["past_key_values"] = _clone_cache(self.past_state["checkpoint"])
            embeds_list = []
            if corrected_prev_delta:
                ids = self.model.tokenizer.encode(corrected_prev_delta, add_special_tokens=False)
                if ids:
                    token_ids = torch.tensor(ids, dtype=torch.long).to(self.device)
                    embeds_list.append(self.embed_tokens_func(token_ids))
            embeds_list.append(self.audio_eos_embeds)
            embeds_list.append(self.non_idle_embeds)
            embeds_list.append(audio_embeds)

            correction_input = torch.cat(embeds_list, dim=0).unsqueeze(0)
            outputs = self.model.llm(
                inputs_embeds=correction_input,
                past_key_values=self.past_state["past_key_values"],
                use_cache=True,
            )
            self.past_state["past_key_values"] = outputs.past_key_values
            if self.past_state["delta_text"]:
                self.past_state["delta_text"][-1] = corrected_prev_delta
        else:
            self.past_state["past_key_values"] = current_kv

        self.past_state["checkpoint"] = _clone_cache(self.past_state["past_key_values"])
        self.past_state["delta_text"].append(delta_text)

        max_len = int(3.2 * self.sampling_rate)
        if len(self.cascade_buffer) > max_len:
            self.cascade_buffer = self.cascade_buffer[-max_len:]
            delta_list = self.past_state["delta_text"][-20:]
            total_tokens = sum(len(split_cn_en(text)) for text in delta_list)
            cascade_tokens = split_cn_en(self.past_state["cascade_text"])
            keep_tokens = cascade_tokens[-total_tokens:] if total_tokens > 0 else []
            self.past_state["cascade_text"] = "".join(
                [(item + " ") if check_en(item) else item for item in keep_tokens]
            ).strip()

        input_embeds_next = self.audio_eos_embeds.unsqueeze(0)
        if delta_text:
            ids = self.model.tokenizer.encode(delta_text, add_special_tokens=False)
            if ids:
                input_ids = torch.tensor(ids, dtype=torch.long).to(self.device)
                embeds = self.embed_tokens_func(input_ids).unsqueeze(0)
                input_embeds_next = torch.cat((embeds, input_embeds_next), dim=1)

        self.past_state["input_embeds"] = input_embeds_next
        return delta_text

    def _state_predict(self, delta_text: str) -> str:
        assert self.past_state is not None
        outputs = self.model.llm(
            inputs_embeds=self.past_state["input_embeds"],
            past_key_values=self.past_state["past_key_values"],
            use_cache=True,
        )
        logits = outputs.logits[0]
        pred = torch.argmax(logits, -1)[-1]
        state = self.model.tokenizer.decode(pred)

        if state == "<|user_nonidle|>" and not delta_text:
            self.past_state["mistake_len"] += 1
        else:
            self.past_state["mistake_len"] = 0

        if (
            self.past_state["state"] == "<|user_nonidle|>" and state == "<|user_idle|>"
        ) or self.past_state["mistake_len"] >= self.config.infer_config.max_mistake_num:
            if logits[-1, self.config.model_config.user_complete_token_id] > logits[
                -1, self.config.model_config.user_incomplete_token_id
            ]:
                state = "<|user_complete|>"
                self.past_state["input_embeds"] = self.action_speak_embeds
            else:
                state = "<|user_incomplete|>"
                self.past_state["input_embeds"] = self.action_wait_embeds
        else:
            self.past_state["input_embeds"] = self.embed_tokens_func(pred.unsqueeze(0))

        self.past_state["past_key_values"] = outputs.past_key_values
        self.past_state["state"] = state
        return state


def load_turn_model(model: str, device: str | None = None) -> SoulXDuplugTurnModel:
    return SoulXDuplugTurnModel(model=model, device=device)
