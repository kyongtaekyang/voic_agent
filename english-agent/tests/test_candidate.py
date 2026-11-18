"""끼어들기·사용자 발화 비율 지표와 파인튜닝 후보 분류·개인정보 검사 시험 (프롬프트 2.5절). GPU 를 쓰지 않는다."""

import json

import pytest

from slow_path import run as sp
from slow_path.candidate import PiiScanner, classify, mask
from slow_path.turns import turn_taking
from test_slow_path import CFG, _Mock, make_session, mock_url  # noqa: F401  (fixture 재사용)


# 끼어들기, 사용자 발화 비율 --------------------------------------------------------------------------------------------

def test_turn_taking_no_overlap():
    t = turn_taking([(0.0, 2.0), (4.0, 5.0)], [(2.5, 3.5), (5.5, 6.5)])
    assert t["user_turns"] == 2 and t["agent_turns"] == 2
    assert t["user_speech_s"] == 3.0 and t["agent_speech_s"] == 2.0 and t["user_speech_ratio"] == 0.6
    assert t["barge_ins"] == 0 and t["barge_in_yield_rate"] is None and t["agent_cuts_user"] == 0


def test_barge_in_yielded_and_not():
    # 사용자가 3.0초에 끼어듦 → Agent 가 3.4초에 멈춤 (양보), 8.0초에 끼어듦 → Agent 는 10.0초까지 계속 (양보 안 함)
    t = turn_taking([(3.0, 4.0), (8.0, 9.0)], [(2.0, 3.4), (7.0, 10.0)])
    assert t["barge_ins"] == 2 and t["barge_ins_yielded"] == 1 and t["barge_in_yield_rate"] == 0.5


def test_agent_cuts_user_ignores_short_tail_overlap():
    # Agent 가 1.0초에 시작, 사용자는 3.0초까지 계속 → 끊음. 두 번째는 사용자 끝 0.2초 전에 시작 (짧은 겹침, 끊음 아님)
    t = turn_taking([(0.0, 3.0), (5.0, 6.0)], [(1.0, 1.5), (5.8, 7.0)])
    assert t["agent_cuts_user"] == 1 and t["agent_cuts_user_rate"] == 0.5
    assert turn_taking([], [])["user_speech_ratio"] is None


# 개인정보 검사 --------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scanner():
    return PiiScanner()


@pytest.mark.parametrize("text,rule", [
    ("Hi! My name is Jimin Park.", "self_name"),
    ("I live at 25 Maple street", "self_address"),
    ("My mom's phone number is secret", "self_contact"),
    ("It's 010-1234-5678.", "phone_number"),
    ("write to kid123@example.com", "email"),
    ("five five five one two one two", "spelled_digits"),
    ("My birthday is May fifth", "self_birthday"),
    ("I go to Hanbit elementary school", "self_school"),
])
def test_pii_detected(scanner, text, rule):
    assert rule in {f["rule"] for f in scanner.scan(text)}


@pytest.mark.parametrize("text", [
    "One two three four five six seven eight nine ten!",   # 숫자 세기 연습
    "ten nine eight seven six five four three two one",
    "What is your favorite animal? I like dinosaurs.",
    "The cat's name is Tom.",                                # 남의 이름 (자기 이름 아님)
    "We have 3 apples and 12 bananas.",
    "",
])
def test_pii_clean(scanner, text):
    assert scanner.scan(text) == []


def test_masked_output_does_not_repeat_value(scanner):
    found = scanner.scan("My name is Jimin and my number is 010 1234 5678")
    joined = json.dumps(found)
    assert "jimin" not in joined and "5678" not in joined
    assert mask("my name is jimin") == "my name is j***"
    assert mask("010 1234 5678") == "*** *** ***"


# 분류 ---------------------------------------------------------------------------------------------------------------

GOOD_METRICS = {
    "asr_turns": 10, "non_english_turns": 0, "errors": 0, "transcript_truncated": False,
    "safety": {"blocks": 0, "audit": {"violations_in_agent_audio": 0, "items": []}},
    "turn_taking": {"user_turns": 10, "user_speech_s": 60.0, "user_speech_ratio": 0.4, "agent_cuts_user_rate": 0.1},
}
CLEAN_TR = {"user": {"text": "I like dinosaurs."}, "agent": {"text": "Great! Tell me more."}}
CONSENT = {"consent": {"finetune": True}}


def test_classify_candidate(scanner):
    c = classify(CONSENT, GOOD_METRICS, CLEAN_TR, None, scanner)
    assert c["decision"] == "candidate" and c["reasons"] == [] and c["pii"] == []


def test_classify_requires_consent(scanner):
    c = classify({}, GOOD_METRICS, CLEAN_TR, None, scanner)
    assert c["decision"] == "rejected" and any("동의" in r for r in c["reasons"])
    assert classify({}, GOOD_METRICS, CLEAN_TR, {"require_consent": False}, scanner)["decision"] == "candidate"


def test_classify_pii_needs_review(scanner):
    tr = {"user": {"text": "My name is Jimin."}, "agent": {"text": "Nice to meet you."}}
    c = classify(CONSENT, GOOD_METRICS, tr, None, scanner)
    assert c["decision"] == "needs_review" and c["pii"][0]["source"] == "user" and c["pii"][0]["rule"] == "self_name"


@pytest.mark.parametrize("patch,word", [
    ({"transcript_truncated": True}, "잘림"),
    ({"errors": 2}, "오류"),
    ({"safety": {"blocks": 1, "audit": {"violations_in_agent_audio": 0}}}, "차단"),
    ({"safety": {"blocks": 0, "audit": {"violations_in_agent_audio": 1}}}, "사후 검사 위반"),
    ({"safety": {"blocks": 0}}, "사후 안전 검사 안 함"),
    ({"non_english_turns": 3}, "비영어"),
    ({"turn_taking": {"user_turns": 10, "user_speech_s": 60.0, "user_speech_ratio": 0.9, "agent_cuts_user_rate": 0.1}}, "비율"),
    ({"turn_taking": {"user_turns": 10, "user_speech_s": 60.0, "user_speech_ratio": 0.4, "agent_cuts_user_rate": 0.5}}, "끊은"),
    ({"turn_taking": {"user_turns": 1, "user_speech_s": 5.0, "user_speech_ratio": 0.4, "agent_cuts_user_rate": 0.0}}, "사용자 발화"),
])
def test_classify_rejects(scanner, patch, word):
    c = classify(CONSENT, GOOD_METRICS | patch, CLEAN_TR, None, scanner)
    assert c["decision"] == "rejected" and any(word in r for r in c["reasons"]), c["reasons"]


# 배치 연결 ----------------------------------------------------------------------------------------------------------

RELAXED = {"require_consent": True, "min_user_speech_s": 0, "min_user_turns": 1, "max_non_english_turn_ratio": 1.0,
           "user_speech_ratio": [0.0, 1.0], "max_agent_cuts_user_rate": 1.0}


def test_batch_writes_turn_taking_and_candidate(tmp_path, monkeypatch, mock_url):  # noqa: F811
    d = make_session(tmp_path, monkeypatch, block=False)
    _Mock.replies = ["My name is Jimin. Hello can you help me?", "Sure, I can."]
    assert sp.transcribe_stage(tmp_path, CFG | {"finetune_candidate": RELAXED}, mock_url, None)["done"] == 1

    m = json.loads((d / "metrics.json").read_text())
    tt = m["turn_taking"]
    assert tt["user_turns"] == 1 and tt["agent_turns"] == 1 and tt["barge_ins"] == 0
    assert tt["user_speech_ratio"] == pytest.approx(1.0 / 1.5, abs=0.03)

    # 동의 기록이 없으므로 rejected
    meta = json.loads((d / "meta.json").read_text())
    assert meta["finetune"] == "rejected"
    c = json.loads((d / "candidate.json").read_text())
    assert any("동의" in r for r in c["reasons"]) and c["pii"][0]["rule"] == "self_name"
    assert "jimin" not in (d / "candidate.json").read_text().lower()

    # 동의 기록을 붙이고 GPU 없이 다시 분류 → 개인정보가 있으므로 needs_review
    meta["consent"] = {"finetune": True}
    sp.save_meta(d, meta)
    assert sp.classify_stage(tmp_path, CFG | {"finetune_candidate": RELAXED}) == {"stage": "classify",
                                                                                   "decisions": {"needs_review": 1}}
    k = sp.report(tmp_path)
    assert k["finetune"] == {"candidate": 0, "needs_review": 1, "rejected": 0}
    assert k["barge_ins"] == 0 and k["user_speech_ratio_median"] == pytest.approx(0.667, abs=0.03)
    ft = json.loads((tmp_path / "_finetune_candidates.json").read_text())
    assert ft["needs_review"][0]["session"] == d.name and ft["needs_review"][0]["pii_rules"] == ["self_name"]


def test_cli_classify_on_empty_queue(tmp_path, capsys):
    assert sp.main(["classify", "--queue", str(tmp_path / "none")]) == 0
    assert json.loads(capsys.readouterr().out)["decisions"] == {}
