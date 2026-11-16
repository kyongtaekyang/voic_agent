from core.asr_monitor import LanguageMismatchDetector, hangul_ratio, split_lang_tag


def test_split_lang_tag():
    assert split_lang_tag("<ko-KR> 안녕하세요") == ("ko", "안녕하세요")
    assert split_lang_tag("hello") == (None, "hello")


def test_hangul_ratio():
    assert hangul_ratio("hello") == 0.0
    assert hangul_ratio("안녕") == 1.0
    assert hangul_ratio("123 ...") is None


def test_two_korean_turns_trigger_once():
    d = LanguageMismatchDetector(mismatch_turns=2)
    assert d.feed("안녕하세요") is None
    ev = d.feed("<ko-KR> 이거 영어로 뭐예요")
    assert ev["type"] == "eva.language_mismatch" and ev["detected_language"] == "ko"
    assert d.feed("한국어 계속") is None  # 영어가 나올 때까지 다시 내지 않음


def test_english_resets_streak():
    d = LanguageMismatchDetector(mismatch_turns=2)
    d.feed("안녕하세요")
    d.feed("Hello teacher")
    assert d.feed("다시 한국어") is None
    assert d.feed("또 한국어")["turns"] == 2


def test_empty_or_short_is_ignored():
    d = LanguageMismatchDetector(mismatch_turns=2)
    d.feed("안녕하세요")
    assert d.feed("") is None
    assert d.feed("이") is None  # 한 글자: 판단하지 않음, 연속 횟수도 그대로
    assert d.feed("한국어예요") is not None


def test_mixed_english_with_korean_word():
    d = LanguageMismatchDetector()
    assert d.classify("I like 김치 very much").english is True


def test_tag_wins_over_script():
    d = LanguageMismatchDetector()
    assert d.classify("<en-US> kimchi").english is True
    assert d.classify("<ja-JP> konnichiwa").english is False
