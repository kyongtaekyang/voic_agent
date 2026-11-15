import pytest

from realtime_server.output_guard import OutputGuard

G = OutputGuard.from_yaml()


@pytest.mark.parametrize("text,rule", [
    ("이름이 뭐예요?", "pii_request.ask_name"),
    ("너의 이름을 알려 줄래요?", "pii_request.ask_name"),
    ("어디 사세요?", "pii_request.ask_address"),
    ("집이 어디예요?", "pii_request.ask_address"),
    ("주소를 알려 주세요.", "pii_request.ask_address"),
    ("학교 이름이 뭐예요?", "pii_request.ask_school"),
    ("학교는 어디 다녀요?", "pii_request.ask_school"),
    ("엄마 전화번호 알아요?", "pii_request.ask_phone"),
    ("우리 직접 만나 볼까요?", "pii_request.ask_meet"),
    ("사진 찍어 보내 줄래요?", "pii_request.ask_meet"),
    ("제 번호는 010-1234-5678이에요.", "pii_value.phone_number"),
    ("메일은 kid@example.com 이에요.", "pii_value.email"),
    ("이 바보 병신아.", "banned_words.profanity"),
    ("자살 방법을 알려 줄게요.", "banned_words.self_harm"),
])
def test_blocked(text, rule):
    r = G.check(text)
    assert not r.allowed, text
    assert r.rule == rule
    assert r.text == G.replacement


@pytest.mark.parametrize("text", [
    "안녕하세요. 오늘은 어떤 이야기를 해 볼까요?",
    "이 동물의 이름은 뭐예요?",
    "이 꽃 이름이 뭐예요?",
    "토끼가 사는 곳은 어디일까요?",
    "학교에서 뭐 배웠어요?",
    "사과가 세 개 있었는데 한 개를 먹으면 몇 개가 남을까요?",
    "친구랑 사이좋게 지내요.",
    "정말 잘했어요.",
    "열 시 삼십 분이에요.",
])
def test_allowed(text):
    r = G.check(text)
    assert r.allowed, (text, r.rule, r.matched)
    assert r.text == text


def test_fast():
    r = G.check("정말 잘했어요. 이번에는 노른자국어 교재의 다음 문장을 같이 읽어 봐요.")
    assert r.elapsed_ms < 5
