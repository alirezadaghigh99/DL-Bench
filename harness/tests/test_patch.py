import ast
import textwrap

import pytest

from harness.patch import PatchError, make_patched_source, patched_file


def _src(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def test_replaces_top_level_function():
    original = _src("""
        import math

        def add(a, b):
            return a + b

        def other():
            return 1
    """)
    llm = _src("""
        def add(a, b):
            # smarter version
            return a + b + 0
    """)
    out = make_patched_source(original, llm, "add", None)
    tree = ast.parse(out)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "add" in funcs and "other" in funcs
    assert "smarter version" in ast.unparse(funcs["add"])


def test_replaces_class_method_and_keeps_other_methods():
    original = _src("""
        class C:
            def a(self):
                return 1

            def b(self):
                return 2
    """)
    llm = _src("""
        def b(self):
            return 99
    """)
    out = make_patched_source(original, llm, "b", "C")
    tree = ast.parse(out)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    assert "a" in methods and "b" in methods
    assert "return 99" in ast.unparse(methods["b"])


def test_preserves_decorators_when_llm_drops_them():
    original = _src("""
        def cache(fn): return fn

        @cache
        def decorated(x):
            return x
    """)
    llm = _src("""
        def decorated(x):
            return x * 2
    """)
    out = make_patched_source(original, llm, "decorated", None)
    tree = ast.parse(out)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decorated")
    assert any(isinstance(d, ast.Name) and d.id == "cache" for d in fn.decorator_list)


def test_uses_llm_decorators_if_provided():
    original = _src("""
        def cache(fn): return fn
        def cache2(fn): return fn

        @cache
        def decorated(x): return x
    """)
    llm = _src("""
        @cache2
        def decorated(x):
            return x
    """)
    out = make_patched_source(original, llm, "decorated", None)
    tree = ast.parse(out)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decorated")
    names = [d.id for d in fn.decorator_list if isinstance(d, ast.Name)]
    assert names == ["cache2"]


def test_injects_only_new_imports():
    original = _src("""
        import math
        def f(): return math.pi
    """)
    llm = _src("""
        import math
        import statistics
        def f():
            return statistics.mean([1, 2, 3])
    """)
    out = make_patched_source(original, llm, "f", None)
    assert out.count("import math") == 1
    assert out.count("import statistics") == 1


def test_raises_when_function_missing_in_original():
    with pytest.raises(PatchError):
        make_patched_source("def other(): pass\n", "def f(): pass\n", "f", None)


def test_raises_when_class_missing():
    with pytest.raises(PatchError):
        make_patched_source("class A: pass\n", "def m(self): pass\n", "m", "B")


def test_raises_on_unparseable_llm_code():
    with pytest.raises(PatchError):
        make_patched_source("def f(): pass\n", "def f( bad python", "f", None)


def test_patched_file_restores_on_success(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("original\n", encoding="utf-8")
    with patched_file(p, "patched\n"):
        assert p.read_text(encoding="utf-8") == "patched\n"
    assert p.read_text(encoding="utf-8") == "original\n"
    assert not p.with_suffix(".py.dlbench-bak").exists()


def test_patched_file_restores_on_exception(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("original\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with patched_file(p, "patched\n"):
            raise RuntimeError("boom")
    assert p.read_text(encoding="utf-8") == "original\n"
