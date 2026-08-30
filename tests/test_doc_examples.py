"""mech-3: every fenced python block in the published docs EXECUTES, and every
literal output-comment in one is ASSERTED.

The release doctrine this mechanizes: a claim we publish either runs green in CI,
re-measures itself, or carries a dated scope. A ``<!-- docs-example: skip: reason -->``
comment immediately above a fence opts a block out WITH its reason on record
(unrunnable externals, cache-mutating operations); everything else must exec in a
per-file namespace (blocks build on earlier blocks in the same file, like a reader
following along). A bare-expression line whose following comment is
``# -> <python-literal>`` (or the README's ``# > <literal>``) is rewritten into an
equality assertion -- the printed claims are the strongest ones the docs make, so
they are exactly what must not drift.
"""
import ast
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ['README.md', 'docs/index.md', 'docs/rules.md',
         'docs/getting-started.md', 'docs/guides/simplify.md',
         'docs/guides/artifacts.md', 'docs/guides/masking.md',
         'docs/guides/trust.md', 'docs/guides/verify.md']

_FENCE = re.compile(
    r'(?P<directive><!--\s*docs-example:\s*skip:\s*(?P<reason>[^>]*?)\s*-->\s*\n)?'
    r'^```python\n(?P<body>.*?)^```', re.S | re.M)
_OUT = re.compile(r'^#\s*(?:->|>)\s*(?P<lit>.+?)\s*$')


def _literal(text: str):
    try:
        return True, ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return False, None


def _instrument(body: str) -> str:
    """Rewrite every top-level expression STATEMENT whose next line is a literal
    output-comment into an assert.

    Instrumenting PHYSICAL LINES missed every multi-line call, and a wrong claim was
    sitting in that hole: `docs/guides/masking.md` promised a tagged result from a
    two-line `masking.mask(...)` call that has returned explicit prefix since `simplify`
    became dialect-preserving. Parsing the block once and using each statement's
    `end_lineno` closes it -- a call spanning four lines is instrumented exactly as a
    one-liner is.
    """
    lines = body.split('\n')
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return body
    # statement START line (1-indexed) -> its END line, so a multi-line call is replaced
    # as one unit. Keying on the END emitted the first line twice -- once bare, leaving an
    # unclosed paren, and once inside the assert.
    spans = {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and node.end_lineno is not None:
            spans[node.lineno] = node.end_lineno

    out: list = []
    i = 0
    while i < len(lines):
        end = spans.get(i + 1)
        nxt = lines[end].strip() if end is not None and end < len(lines) else ''
        m = _OUT.match(nxt)
        ok, expected = _literal(m.group('lit')) if m else (False, None)
        if ok and end is not None:
            start = i + 1
            code = '\n'.join(lines[start - 1:end])
            code = code.split('#', 1)[0].rstrip()
            label = ' '.join(code.split())
            out.append(f'__doc_out = ({code})')
            out.append(f'assert __doc_out == {expected!r}, '
                       f'{f"doc example drifted: {label} -> "!r} + repr(__doc_out)')
            i = end + 1
            continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out)


@pytest.mark.parametrize('relpath', FILES)
def test_doc_examples_execute_and_their_outputs_hold(relpath, tmp_path, monkeypatch):
    from conftest import acj_config_path, require_or_skip
    require_or_skip(acj_config_path(), 'acj-4-3 asset not staged')
    monkeypatch.chdir(tmp_path)  # examples must not depend on or touch the repo cwd
    text = open(os.path.join(REPO, relpath)).read()
    namespace: dict = {}
    ran = 0
    for m in _FENCE.finditer(text):
        if m.group('directive'):
            assert m.group('reason').strip(), f'{relpath}: skip directive without a reason'
            continue
        code = _instrument(m.group('body'))
        try:
            exec(compile(code, f'{relpath}[block {ran}]', 'exec'), namespace)
        except Exception as ex:
            pytest.fail(f'{relpath} block failed: {type(ex).__name__}: {ex}\n--- block ---\n'
                        + m.group('body')[:400])
        ran += 1
    assert ran > 0, f'{relpath}: no executable examples found (vacuous)'
