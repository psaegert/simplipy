"""The engine never reads the reader's spelling (owner 2026-10-02: "fix this properly").

A printed form is not only an answer. The interval certificates, the folds, the served rules,
the mining judge and the promotion refund all print a state and read the tokens back, so the
first reader-facing spelling rules (integer over decimal, even roots as `rootn`) moved their
verdicts: the even-root spelling let the zero-set certificate prove more and moved the corpus
pin. The printers therefore have two spellings of one state:

* KERNEL -- the one fixed internal spelling, which every engine consumer reads (it is the
  0.14.7 printer, byte for byte);
* DISPLAY -- the reader's spelling, printed only for the caller (public `simplify`'s explicit
  and infix answers) and never read back.

Pinned here: the two spellings denote one state at one price on the whole corpus; the kernel
spelling and the mining judge keep the 0.14.7 spellings; and the source routes the display
spelling to the public boundary only.
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


class TestOneStateTwoSpellings:
    def test_the_corpus_reads_back_to_the_same_state_at_the_same_price(self, eng):
        from conftest import require_or_skip
        require_or_skip(CORPUS, 'tracked corpus not present')
        differ = []
        for i, row in enumerate(json.load(open(CORPUS))):
            kernel = list(eng._simplify(list(row)))
            display = list(eng.simplify(list(row)))
            assert list(eng._simplify(display)) == kernel, (i, display, kernel)
            assert eng.complexity(display) == eng.complexity(kernel), (i, display, kernel)
            if display != kernel:
                differ.append(i)
        # anti-vacuity, pinned exactly (measured 2026-10-02): the rows whose display spelling
        # differs -- every one an even root below the fraction bar (R1)
        assert len(differ) == 17, differ


class TestTheKernelSpellingDoesNotMove:
    # (input, kernel = the 0.14.7 answer, display)
    CASES = [
        (['/', '1', '*', '2', PI], ['/', '500000000000000', '3141592653589793'],
         ['/', '1', '6.283185307179586']),
        (['inv', 'rootn', 'x0', '2'], ['inv', 'pow', 'x0', '/', '1', '2'], ['inv', 'rootn', 'x0', '2']),
        (['/', 'x1', 'rootn', 'x0', '2'], ['/', 'x1', 'pow', 'x0', '/', '1', '2'],
         ['/', 'x1', 'rootn', 'x0', '2']),
        (['rootn', '/', '1', '3', '2'], ['rootn', '/', '1', '3', '2'], ['inv', 'rootn', '3', '2']),
    ]

    @pytest.mark.parametrize('tokens, kernel, display', CASES)
    def test_kernel_and_display(self, eng, tokens, kernel, display):
        assert list(eng._simplify(tokens)) == kernel
        assert list(eng.simplify(tokens)) == display

    @pytest.mark.parametrize('tokens, kernel, _', CASES)
    def test_the_mining_judge_answers_in_the_kernel_spelling(self, eng, tokens, kernel, _):
        # the judge's form is the miner's "mark to beat" and coverage's comparison spelling
        assert eng._core.ac_judge(tokens, 48)[2] == kernel

    def test_the_kernel_spelling_takes_tokens_only(self, eng):
        with pytest.raises(TypeError):
            eng._simplify('1/rootn(x0, 2)')


class TestDisplayOnlyAtTheBoundary:
    """A source guard: the display spelling is requested in exactly these places. A new
    internal consumer that prints in it would fail here before it could move a verdict."""

    @staticmethod
    def lines(path, pattern):
        full = os.path.join(REPO, path)
        if not os.path.exists(full):
            pytest.skip('source tree not present')
        return [ln.strip() for ln in open(full) if re.search(pattern, ln)
                and not ln.strip().startswith('//')]

    def test_rust_requests_the_display_spelling_only_in_the_printers(self):
        for root, _, files in os.walk(os.path.join(REPO, 'rust')):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), REPO)
                if f.endswith('.rs') and rel not in ('rust/ac/convert.rs', 'rust/ac/expr.rs'):
                    assert not self.lines(rel, r'Spelling::Display'), rel

    def test_the_engine_prints_display_only_for_the_public_projection(self):
        assert self.lines('rust/engine/ac.rs', r'to_prefix_display\(') == [
            'AcForm::Display => to_prefix_display(&best, &bare),'] * 2
        # the infix text is display by definition; its only callers are the two infix answers
        assert len(self.lines('rust/engine/ac.rs', r'to_infix_pretty\(&')) == 2

    def test_python_asks_for_display_only_in_public_simplify(self):
        hits = []
        for root, _, files in os.walk(os.path.join(REPO, 'src', 'simplipy')):
            for f in files:
                if f.endswith('.py'):
                    rel = os.path.relpath(os.path.join(root, f), REPO)
                    hits += [(rel, ln) for ln in self.lines(rel, r"'display'")]
        assert hits == [('src/simplipy/engine.py',
                         "tokens, max_passes, rule_mode, 'display' if display and form == 'explicit' "
                         "else form, effort)")], hits
