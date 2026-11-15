"""services.yaml 로 만든 argv 가 speech-to-speech 1.1.0 파서에서 모두 인식되는지 확인한다 (모델은 로드하지 않음)."""

import logging

import pytest
import yaml

from realtime_server import main as m

s2s = pytest.importorskip("speech_to_speech.s2s_pipeline")


def _argv():
    cfg = yaml.safe_load((m.ROOT / "config" / "services.yaml").read_text(encoding="utf-8"))
    return cfg, m.build_argv(cfg)


def test_argv_parses_without_ignored_options(caplog):
    cfg, argv = _argv()
    assert argv[0] == "serve"
    with caplog.at_level(logging.WARNING):
        parsed = s2s.parse_arguments(argv[1:], command="serve")
    assert "Ignoring options" not in caplog.text
    assert parsed.stt_backend.name == "nemotron-streaming"
    assert parsed.llm_backend.name == "chat-completions"
    assert parsed.tts_backend.name == "qwen3"
    llm = parsed.llm_backend.config
    assert llm["disable_thinking"] is True
    assert llm["model_name"] == cfg["llm"]["served_name"]
    assert llm["base_url"] == f"http://127.0.0.1:{cfg['ports']['gemma_vllm']}/v1"
    assert llm["stream_batch_sentences"] == cfg["features"]["stream_batch_sentences"]
    assert llm["init_chat_prompt"].startswith("당신은 한국어로")
    text = repr(parsed)
    for needle in (cfg["stt"]["language"], cfg["tts"]["speaker"]):
        assert needle in text, needle
    assert parsed.realtime_server_kwargs.port == cfg["ports"]["realtime_server"]


def test_patch_injects_sampling_and_replaces_handlers():
    cfg, _ = _argv()
    m.patch_speech_to_speech(cfg)
    from speech_to_speech import backend_registry as br
    from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler

    kwargs = ChatCompletionsApiModelHandler._build_optional_kwargs(None, None, None)
    assert kwargs["temperature"] == float(cfg["llm"]["temperature"])
    assert kwargs["max_tokens"] == int(cfg["llm"]["max_new_tokens"])
    assert br.STT_BACKENDS["nemotron-streaming"].required_extra is None
    from speech_to_speech.LLM import lm_output_processor

    assert lm_output_processor.LMOutputProcessor.__name__ == "KoreanLMOutputProcessor"
