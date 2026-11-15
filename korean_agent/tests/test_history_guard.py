"""차단된 문장이 대화 이력에 남지 않는지 확인한다."""

import pytest

from realtime_server.output_guard import OutputGuard

guard = OutputGuard.from_yaml()
REPL = guard.replacement


def test_sanitize_passes_clean_text_unchanged():
    text = "사과가 3개 있어요. 지금은 10:30이에요."
    assert guard.sanitize(text) == (text, None)


def test_sanitize_cuts_from_first_blocked_sentence():
    text = "사과가 3개 있어요. 너의 전화번호를 알려 줄래요? 이 문장은 나오면 안 돼요."
    cleaned, hit = guard.sanitize(text)
    assert cleaned == f"사과가 3개 있어요. {REPL}"
    assert hit.rule == "pii_request.ask_phone"


def test_sanitize_first_sentence_blocked():
    cleaned, hit = guard.sanitize("주소가 어디예요?\n다음 문제를 풀어요.")
    assert cleaned == REPL and hit is not None


def test_sanitize_empty():
    assert guard.sanitize("  ") == ("  ", None)


m = pytest.importorskip("speech_to_speech.LLM.base_openai_compatible_language_model")


def test_installed_to_chat_item_sanitizes(tmp_path, monkeypatch):
    from realtime_server import history_guard, korean_lm_processor

    monkeypatch.setattr(korean_lm_processor, "EVENT_LOG", tmp_path / "events.jsonl")
    history_guard.install(guard)
    history_guard.install(guard)  # 두 번 설치해도 한 번만 감싼다
    msg = m.AssistantMessage(id="item_1", content=[{"type": "output_text", "text": "좋아요. 너의 전화번호를 알려 줘."}])
    item = msg.to_chat_item()
    assert item.content[0].text == f"좋아요. {REPL}"
    assert msg.content[0].text.endswith("알려 줘.")  # 원본 객체는 바꾸지 않는다
    assert "history_sanitized" in (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    clean = m.AssistantMessage(id="item_2", content=[{"type": "output_text", "text": "토끼는 당근을 좋아해요."}])
    assert clean.to_chat_item().content[0].text == "토끼는 당근을 좋아해요."
