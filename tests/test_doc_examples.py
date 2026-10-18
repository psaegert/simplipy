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
import io
import os
import re
import tokenize

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
    """A claim's literal: the comment after `->`, up to an optional note separated from
    it by two or more spaces (`# -> ['x0']   (the explicit form)`)."""
    text = re.split(r'\s{2,}', text.strip(), maxsplit=1)[0]
    try:
        return True, ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return False, None


def _instrument(body: str) -> tuple[str, list[str]]:
    """Rewrite every top-level expression STATEMENT that carries an output claim into an
    assert, and return the rewritten block with the claims that could not be asserted.

    A claim is a comment `# -> <python-literal>` (or the README's `# > <literal>`), either
    at the end of the statement's last line or alone on the line after it. A claim that
    follows no expression statement, or whose text is not a literal, is returned as a
    problem: a printed claim the test cannot check is a claim nobody checks.

    Parsing the block once and using each statement's `end_lineno` instruments a call
    spanning several lines exactly as a one-liner.
    """
    lines = body.split('\n')
    try:
        tree = ast.parse(body)
        comments = [tok for tok in tokenize.generate_tokens(io.StringIO(body).readline)
                    if tok.type == tokenize.COMMENT]
    except (SyntaxError, tokenize.TokenError):
        return body, []  # the exec reports the block itself
    # each expression statement, by the line it ENDS on: (start line, end line)
    ends = {node.end_lineno: node.lineno for node in tree.body
            if isinstance(node, ast.Expr) and node.end_lineno is not None}

    # where each line's comment starts: stripped from an instrumented statement
    comment_col = {tok.start[0]: tok.start[1] for tok in comments}

    problems: list[str] = []
    claims: dict = {}  # start line -> (end line, expected, the claim is inline)
    for tok in comments:
        m = _OUT.match(tok.string)
        if not m:
            continue
        line, col = tok.start
        inline = bool(lines[line - 1][:col].strip())
        end = line if inline else line - 1
        ok, expected = _literal(m.group('lit'))
        if end not in ends:
            problems.append(f'line {line}: claim {tok.string!r} follows no expression')
        elif not ok:
            problems.append(f'line {line}: claim {tok.string!r} is not a python literal')
        else:
            claims[ends[end]] = (end, expected, inline)

    out: list = []
    i = 0
    while i < len(lines):
        if i + 1 in claims:
            end, expected, inline = claims[i + 1]
            code = '\n'.join(lines[n - 1][:comment_col.get(n)] for n in range(i + 1, end + 1)).rstrip()
            label = ' '.join(code.split())
            out.append(f'__doc_out = ({code})')
            out.append(f'assert __doc_out == {expected!r}, '
                       f'{f"doc example drifted: {label} -> "!r} + repr(__doc_out)')
            i = end if inline else end + 1
            continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out), problems


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
        code, problems = _instrument(m.group('body'))
        assert not problems, f'{relpath} block {ran}: ' + '; '.join(problems)
        try:
            exec(compile(code, f'{relpath}[block {ran}]', 'exec'), namespace)
        except Exception as ex:
            pytest.fail(f'{relpath} block failed: {type(ex).__name__}: {ex}\n--- block ---\n'
                        + m.group('body')[:400])
        ran += 1
    assert ran > 0, f'{relpath}: no executable examples found (vacuous)'
