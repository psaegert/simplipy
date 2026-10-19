"""The hand-written `_core.pyi` stub names exactly the compiled surface.

The stub mirrors `rust/lib.rs` by hand, and it drifted twice: exports went missing from it
and deleted ones lingered. This compares the public names both ways, for each class and
for the module itself.
"""
import ast
from pathlib import Path

import pytest

from simplipy import _core

STUB = ast.parse(
    (Path(__file__).resolve().parents[1] / 'src' / 'simplipy' / '_core.pyi').read_text(encoding='utf-8'))


def _public(names) -> set[str]:
    return {n for n in names if not n.startswith('_')}


@pytest.mark.parametrize('name', ['Engine', 'CandidateLibrary'])
def test_class_members_match_the_stub(name: str) -> None:
    cls = next(n for n in STUB.body if isinstance(n, ast.ClassDef) and n.name == name)
    stubbed = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert _public(stubbed) == _public(dir(getattr(_core, name)))


def test_module_names_match_the_stub() -> None:
    stubbed: set[str] = set()
    for n in STUB.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            stubbed.add(n.name)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            stubbed.add(n.target.id)
    assert _public(stubbed) == _public(dir(_core))
