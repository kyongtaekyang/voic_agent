"""Korean Realtime Voice Server 실행 스크립트.

speech-to-speech 1.1.0 을 기반으로 쓰고, 패키지 파일은 수정하지 않은 채 두 지점만 교체한다.
  1) STT `nemotron-streaming` -> realtime_server.stt_streaming.NemotronTransformersSTTHandler (NeMo 불필요)
  2) LMOutputProcessor        -> realtime_server.korean_lm_processor.KoreanLMOutputProcessor (출력 검사 + 정규화)
나머지 옵션은 config/services.yaml 에서 만든다.

실행: .venv/bin/python -m realtime_server.main [--dry-run] [추가 speech-to-speech 옵션 ...]
"""

from __future__ import annotations

import ctypes
import dataclasses
import glob
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def preload_cuda12() -> None:
    """qwentts-cpp 의 ggml CUDA 라이브러리는 CUDA 12 용이다. venv 의 CUDA 12 런타임을 먼저 전역으로 올려 둔다."""
    site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    for pattern in ("nvidia/cuda_runtime/lib/libcudart.so.12*", "nvidia/cublas/lib/libcublasLt.so.12*",
                    "nvidia/cublas/lib/libcublas.so.12*"):
        for so in sorted(glob.glob(str(site / pattern)))[:1]:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)


def build_argv(cfg: dict) -> list[str]:
    llm, stt, tts, turn, feat, ports = (cfg[k] for k in ("llm", "stt", "tts", "turn", "features", "ports"))
    prompt = (ROOT / llm["system_prompt_file"]).read_text(encoding="utf-8").strip()
    argv = [
        "serve",
        "--host", "127.0.0.1", "--port", str(ports["realtime_server"]),
        "--stt", "nemotron-streaming",
        "--nemotron_streaming_model_name", stt["model"],
        "--nemotron_streaming_device", stt["device"],
        "--nemotron_streaming_language", stt["language"],
        "--llm_backend", "chat-completions",
        "--responses_api_base_url", f"http://127.0.0.1:{ports['gemma_vllm']}/v1",
        "--responses_api_api_key", "EMPTY",
        "--model_name", llm["served_name"],
        "--init_chat_role", "system",
        "--init_chat_prompt", prompt,
        "--chat_size", str(llm["chat_size"]),
        "--stream_batch_sentences", str(feat["stream_batch_sentences"]),
        "--tts", tts["backend"],
        "--qwen3_tts_model_name", tts["model"],
        "--qwen3_tts_backend", tts["qwen3_backend"],
        "--qwen3_tts_device", tts["device"],
        "--qwen3_tts_speaker", tts["speaker"],
        "--qwen3_tts_language", tts["language"],
        "--thresh", str(turn["vad_threshold"]),
        "--min_silence_ms", str(turn["min_silence_ms"]),
        "--smart_turn" if turn["smart_turn"] else "--no_smart_turn",
    ]
    for key in ("min_speech_ms", "speech_pad_ms", "speculative_reopen_ms"):
        if key in turn:
            argv += [f"--{key}", str(turn[key])]
    if llm.get("disable_thinking"):
        argv.append("--responses_api_disable_thinking")
    if feat.get("log_transcripts"):
        argv.append("--log_transcripts")
    return argv


def patch_speech_to_speech(cfg: dict) -> None:
    from speech_to_speech import backend_registry as br

    if not cfg["features"].get("log_transcripts"):
        # STT/TTS 처리기는 log_transcripts 와 무관하게 공용 console 에 "USER: ...", "ASSISTANT: ..." 를 찍는다.
        # 아동 발화 원문이 서비스 로그(ada-ops/logs)에 남지 않도록 그 console 을 끈다. logging 출력은 그대로 둔다.
        from speech_to_speech.pipeline.console import console
        console.quiet = True

    spec = br.STT_BACKENDS["nemotron-streaming"]
    br.STT_BACKENDS["nemotron-streaming"] = dataclasses.replace(
        spec,
        create_handler=br._simple_handler_factory(
            "realtime_server.stt_streaming", "NemotronTransformersSTTHandler", attach_speculative_turns=True),
        required_extra=None,
    )
    # chat-completions 백엔드는 텍스트 응답에 temperature/max_tokens 를 보내지 않는다(1.1.0 기준, CLI 옵션도 없음).
    # 아동 대상 짧은 답변을 보장하려고 요청마다 services.yaml 값을 넣는다.
    from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler

    sampling = {"temperature": float(cfg["llm"]["temperature"]), "max_tokens": int(cfg["llm"]["max_new_tokens"])}
    original = ChatCompletionsApiModelHandler._build_optional_kwargs

    def _build_optional_kwargs(self, req_tools, req_tool_choice):
        kwargs = original(self, req_tools, req_tool_choice)
        for key, value in sampling.items():
            kwargs.setdefault(key, value)
        return kwargs

    ChatCompletionsApiModelHandler._build_optional_kwargs = _build_optional_kwargs

    if cfg["features"].get("output_guard") or cfg["features"].get("text_normalizer"):
        # s2s_pipeline 은 핸들러 체인을 만들 때 함수 안에서 lm_output_processor 모듈로부터 import 하므로 그 모듈 속성을 바꾼다.
        from speech_to_speech.LLM import lm_output_processor
        from realtime_server.korean_lm_processor import KoreanLMOutputProcessor
        lm_output_processor.LMOutputProcessor = KoreanLMOutputProcessor
    from realtime_server import latency_log
    latency_log.install()  # 턴별 단계 지연을 events.jsonl 로 (개발자 화면)

    if cfg["features"].get("output_guard"):
        # 차단된 원문이 대화 이력(다음 턴 프롬프트)에 남지 않게 한다.
        from realtime_server import history_guard
        from realtime_server.output_guard import OutputGuard
        history_guard.install(OutputGuard.from_yaml(os.environ.get("KVA_SAFETY_RULES") or
                                                     ROOT / "config" / "safety_rules.yaml"))


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "services.yaml").read_text(encoding="utf-8"))
    extra = [a for a in sys.argv[1:] if a != "--dry-run"]
    os.environ.setdefault("KVA_STT_THREADS", str(cfg["stt"]["threads"]))
    os.environ.setdefault("KVA_MISMATCH_TURNS", str(cfg["stt"]["mismatch_turns"]))
    os.environ.setdefault("KVA_LOG_TRANSCRIPTS", "1" if cfg["features"].get("log_transcripts") else "0")
    argv = build_argv(cfg) + extra
    if "--dry-run" in sys.argv:
        print(" ".join(a if " " not in a and "\n" not in a else repr(a[:40] + "...") for a in argv))
        return
    preload_cuda12()
    patch_speech_to_speech(cfg)
    from speech_to_speech import cli
    sys.argv = ["speech-to-speech", *argv]
    cli.main()


if __name__ == "__main__":
    main()
