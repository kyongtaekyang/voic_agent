import pytest

from core.safety_monitor import SafetyMonitor, SafetyRules, normalize


@pytest.fixture(scope="module")
def rules():
    return SafetyRules.from_yaml()


@pytest.fixture
def mon(rules):
    return SafetyMonitor(rules)


def cats(mon, text):
    return sorted({v.category for v in mon.check_text(text)})


@pytest.mark.parametrize("text,cat", [
    ("What the fuck is that?", "profanity"),
    ("That's fucking great.", "profanity"),
    ("Don't be such an ass.", "profanity"),
    ("You are so stupid.", "insult"),
    ("Shut up and listen.", "insult"),
    ("Let's watch some porn.", "sexual"),
    ("I will kill you.", "violence"),
    ("Here is how to make a bomb.", "violence"),
    ("Have you thought about suicide?", "self_harm"),
    ("Let's smoke some weed.", "drugs"),
    ("What's your home address?", "pii_request"),
    ("Where do you live?", "pii_request"),
    ("What is your mom's phone number?", "pii_request"),
    ("Which school do you go to?", "pii_request"),
    ("Tell me your full name.", "pii_request"),
    ("Can you send me a selfie?", "pii_request"),
    ("This is our little secret.", "grooming"),
    ("Call me at 010 1234 5678.", "pii_value"),
    ("Email me at tutor@example.com.", "pii_value"),
    ("Write to tutor at example dot com.", "pii_value"),
    ("What’s your home address?", "pii_request"),  # 곡선 따옴표
])
def test_blocks(mon, text, cat):
    assert cat in cats(mon, text)


@pytest.mark.parametrize("text", [
    "Welcome to class! I'm your English assistant.",
    "Let's pass the assessment together.",
    "What is the capital of France?",
    "Where does the cat live?",
    "What is your favorite color?",
    "What's your name? Nice to meet you!",
    "Hello, how are you today?",
    "Let's practice the method we learned.",
    "The rooster says cock a doodle doo.",
    "Have you read Moby Dick?",
    "There are no stupid questions.",
    "I have a water gun.",
    "Scunthorpe is a town in England.",
    "Can you pass the dumbbell?",
    "The bass guitar sounds great.",
    "It costs 25 dollars.",
])
def test_passes(mon, text):
    assert cats(mon, text) == []


def test_split_across_tokens(mon):
    out = []
    for tok in ["I", " will", " ki", "ll", " you", "."]:
        out += mon.feed(tok)
    assert [v.category for v in out] == ["violence"]
    assert out[0].token_index == 4  # "kill you" 는 " you" 다음 토큰("."), 즉 경계가 확인될 때 확정


def test_word_waits_for_boundary(mon):
    assert mon.feed("You") == []
    assert mon.feed(" ass") == []  # "assistant" 일 수 있음
    assert mon.feed("istant") == []
    assert mon.flush() == []
    mon.reset()
    mon.feed("You")
    mon.feed(" ass")
    assert [v.rule for v in mon.feed(" now")] == ["profanity_word"]


def test_prefix_is_immediate(mon):
    assert mon.feed("Oh") == []
    assert [v.category for v in mon.feed(" fuck")] == ["profanity"]


def test_sentencepiece_marker(mon):
    out = []
    for tok in ["▁where", "▁do", "▁you", "▁live", "?"]:
        out += mon.feed(tok)
    assert [v.category for v in out] == ["pii_request"]


def test_exception_pending_then_released(mon):
    out = []
    for tok in [" The", " rooster", " says", " cock", " a", " doo", "dle", " doo"]:
        out += mon.feed(tok)
    out += mon.flush()
    assert out == []


def test_exception_prefix_then_diverges(mon):
    out = []
    for tok in [" cock", " a", " gun"]:
        out += mon.feed(tok)
    out += mon.flush()
    assert {v.category for v in out} == {"profanity", "violence"}


def test_reported_once(mon):
    n = 0
    for tok in ["Damn", "!", " ok", " ok", " ok"]:
        n += len(mon.feed(tok))
    n += len(mon.flush())
    assert n == 1


def test_normalize():
    assert normalize("What’S  UP?!") == "what's up "
    assert normalize("▁hello") == " hello"
