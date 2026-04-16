from harness.evaluator import _split_test_spec, _strip_line_anchor


def test_strip_line_anchor():
    assert _strip_line_anchor("torchvision/x.py#L674") == "torchvision/x.py"
    assert _strip_line_anchor("a/b.py") == "a/b.py"


def test_split_test_spec_no_selector():
    p, s = _split_test_spec("tests/test_x.py")
    assert p == "tests/test_x.py" and s is None


def test_split_test_spec_with_class():
    p, s = _split_test_spec("tests/test_x.py::TestX")
    assert p == "tests/test_x.py" and s == "TestX"


def test_split_test_spec_with_class_and_method():
    p, s = _split_test_spec("tests/test_x.py::TestX::test_y")
    assert p == "tests/test_x.py" and s == "TestX::test_y"
