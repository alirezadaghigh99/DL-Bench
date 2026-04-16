from harness.extract import extract


def test_returns_none_for_empty():
    assert extract("") is None


def test_picks_python_fenced_block():
    r = extract("Sure, here you go:\n```python\ndef f():\n    return 1\n```\nThanks!")
    assert r is not None
    assert r.source == "fenced-python"
    assert "def f()" in r.code


def test_picks_longest_python_block():
    s = (
        "```python\ndef short(): pass\n```\n"
        "```python\ndef long():\n    x = 1\n    y = 2\n    return x + y\n```\n"
    )
    r = extract(s)
    assert r is not None
    assert "def long" in r.code
    assert "def short" not in r.code


def test_falls_back_to_plain_block():
    r = extract("explanation\n```\ndef g():\n    return 2\n```")
    assert r is not None
    assert r.source == "fenced-plain"


def test_skips_unparseable_block():
    r = extract("```python\ndef f(:::\n```")
    assert r is None


def test_falls_back_to_raw_when_no_fence():
    r = extract("def h():\n    return 3\n")
    assert r is not None
    assert r.source == "raw"
