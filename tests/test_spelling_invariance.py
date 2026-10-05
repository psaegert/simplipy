"""Nothing reads the reader's spelling (owner 2026-10-02: "fix this properly"; plan v2 phase 1).

A printed token form is not only an answer. The interval certificates, the folds, the served
rules, the mining judge and the promotion refund print a state and read the tokens back, and
callers mask and compare token answers. So the reader's spelling (integer over decimal) lives
in the infix text alone, which nothing reads back, and every token form keeps the 0.14.7
spelling. Readable token forms are the job of a display function, not of the token printer.

Pinned here: the infix text and the token answer are one state at one price on the whole
corpus; the token answers and the mining judge keep the 0.14.7 spellings; and the source keeps
the reader's rule inside the infix printer.
"""
import json
import os
import re

import pytest

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = acj_config_path()
CORPUS = os.path.join(REPO, 'benchmarks', 'corpus', 'raw_skeletons_nv.json')
PI = '3.141592653589793'


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj-4-3 config not staged')
    return SimpliPyEngine.from_config(CONFIG)


def tokens(eng, text):
    return list(eng.simplify(list(eng._core.parse(text, True, False))))


class TestOneStateTwoSpellings:
    def test_the_infix_text_reads_back_to_the_token_answer_at_the_same_price(self, eng):
        from conftest import require_or_skip
        require_or_skip(CORPUS, 'tracked corpus not present')
        rows = json.load(open(CORPUS))
        for i, row in enumerate(rows):
            answer = list(eng.simplify(list(row)))
            text = eng.simplify(eng.to_infix(answer))
            assert tokens(eng, text) == answer, (i, text, answer)
            assert eng.complexity(text) == eng.complexity(answer), (i, text, answer)
        assert len(rows) == 400

    def test_the_reader_spelling_reads_back_to_the_token_answer_at_the_same_price(self, eng):
        # the corpus has no value the reader's spelling touches; these do (25 of 44)
        sources = [f'{c}*x1/({k}*{PI})' for c in (1, 3, 7) for k in range(1, 13)]
        sources += [f'x1*rootn(1/({k}*{PI}), 2)' for k in range(1, 7)]
        sources += ['100/101*x1', f'sin(x1/(2*{PI}))']
        long_integer = re.compile(r'(?<![\d.])\d{10,}(?![\d.])')
        respelled = 0
        for src in sources:
            answer = tokens(eng, src)
            text = eng.simplify(src)
            assert tokens(eng, text) == answer, (src, text, answer)
            assert eng.complexity(text) == eng.complexity(answer), (src, text, answer)
            respelled += any(t.isdigit() and len(t) >= 10 for t in answer) and not long_integer.search(text)
        assert respelled == 25, respelled


class TestTheTokenSpellingDoesNotMove:
    # (input, token answer = the 0.14.7 answer, infix answer)
    CASES = [
        (['/', '1', '*', '2', PI], ['/', '500000000000000', '3141592653589793'], '1/6.283185307179586'),
        (['/', '*', '/', '3', '5', 'pow', 'x1', '2', '*', '*', '*', '4', PI, 'x2', 'x3'],
         ['/', '*', '150000000000000', 'pow', 'x1', '2', '*', '3141592653589793', '*', 'x2', 'x3'],
         '3*x1^2/62.83185307179586/x2/x3'),
        (['*', 'x1', 'rootn', '/', '1', '*', '2', PI, '2'],
         ['*', 'x1', 'rootn', '/', '500000000000000', '3141592653589793', '2'],
         'x1*rootn(1/6.283185307179586, 2)'),
    ]

    @pytest.mark.parametrize('expr, answer, text', CASES)
    def test_token_and_infix_answers(self, eng, expr, answer, text):
        assert list(eng.simplify(expr)) == answer
        assert eng.simplify(eng.to_infix(expr)) == text

    @pytest.mark.parametrize('expr, answer, _', CASES)
    def test_the_mining_judge_answers_in_the_token_spelling(self, eng, expr, answer, _):
        # the judge's form is the miner's "mark to beat" and coverage's comparison spelling
        assert eng._core.ac_judge(expr, 48)[2] == answer

    @pytest.mark.parametrize('expr, answer, _', CASES)
    def test_the_tagged_answer_has_no_reader_spelling(self, eng, expr, answer, _):
        tagged = list(eng.simplify(eng.to_tagged(expr)))
        assert not any('6.283185307179586' in t or '62.83185307179586' in t for t in tagged), tagged


class TestReaderRuleStaysInTheInfixPrinter:
    """A source guard: the reader-facing spelling rule reached from the token printer, or an
    engine consumer of the infix text, fails here before it can move a verdict."""

    @staticmethod
    def source(path):
        full = os.path.join(REPO, path)
        if not os.path.exists(full):
            pytest.skip('source tree not present')
        return open(full).read().split('\n')

    def test_the_rule_is_named_only_inside_the_infix_printer(self):
        # any mention counts (a call, the function passed as a value, a wrapper, the infix
        # helper reused); only the rule's own definition may sit outside the infix printer
        lines = self.source('rust/ac/convert.rs')
        start = next(i for i, ln in enumerate(lines) if ln.startswith('pub fn to_infix_pretty('))
        end = next(i for i, ln in enumerate(lines) if ln.startswith('pub fn canonical_tokens('))
        tests = next(i for i, ln in enumerate(lines) if ln.startswith('mod tests'))
        names = re.compile(r'\b(ratio_spelling|infix_num)\b')
        uses = [i for i, ln in enumerate(lines[:tests])
                if names.search(ln.split('//')[0]) and not ln.startswith('fn ratio_spelling(')]
        assert uses and all(start < i < end for i in uses), \
            [(i + 1, lines[i].strip()) for i in uses if not start < i < end]

    def test_the_engine_prints_infix_only_for_the_infix_answers(self):
        calls = [ln for ln in self.source('rust/engine/ac.rs') if 'to_infix_pretty(&' in ln]
        assert len(calls) == 2, calls  # ac_simplify_infix and ac_simplify_infix_explore

    def test_the_library_asks_for_infix_only_in_public_simplify(self):
        hits = []
        for root, _, files in os.walk(os.path.join(REPO, 'src', 'simplipy')):
            for f in files:
                if f.endswith('.py'):
                    rel = os.path.relpath(os.path.join(root, f), REPO)
                    hits += [rel for ln in self.source(rel) if '_core.ac_simplify_infix' in ln]
        assert hits == ['src/simplipy/engine.py'], hits
