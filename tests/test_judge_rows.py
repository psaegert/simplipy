"""The judge's f64 lane reads one matrix of rows, the shipped artifacts' mine's (seed 42, 1,024 rows
over two variables, confirmed), whoever asks: a verdict is the rule's. A mine of other rows judges
its picks as `verify_rule` judges them; its own rows only prefilter."""
import numpy as np

from simplipy import _core
from simplipy.verify import _contract, verify_rule


class _Exact(np.ndarray):
    """A draw that takes IEEE 754's exact operations only: any other numpy ufunc (its vectorized
    `power`, `exp`, `tanh`, ...) raises, and what an exact one gives stays guarded."""
    EXACT = frozenset({np.add, np.subtract, np.multiply, np.true_divide, np.negative, np.positive,
                       np.absolute, np.sqrt, np.equal, np.not_equal, np.less, np.less_equal,
                       np.greater, np.greater_equal, np.logical_and, np.logical_or, np.logical_not,
                       np.isfinite, np.isnan, np.isinf, np.signbit, np.copysign})

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc not in self.EXACT:
            raise AssertionError(f"a row value went through numpy's {ufunc.__name__}")

        def plain(x):
            return x.view(np.ndarray) if isinstance(x, _Exact) else x
        if 'out' in kwargs:
            kwargs['out'] = tuple(plain(o) for o in kwargs['out'])
        out = getattr(ufunc, method)(*(plain(x) for x in inputs), **kwargs)
        return out.view(_Exact) if isinstance(out, np.ndarray) else out


class _GuardedRng:
    """`default_rng(seed)`, every array it draws guarded (`_Exact`)."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def __getattr__(self, name: str):
        draw = getattr(self._rng, name)

        def guarded(*args, **kwargs):
            out = draw(*args, **kwargs)
            return out.view(_Exact) if isinstance(out, np.ndarray) else out
        return guarded


def test_a_row_is_a_draw_an_exact_operation_or_the_cores_arithmetic() -> None:
    """numpy's vectorized transcendentals are its own code on a machine with AVX-512, not the
    platform's math library, and give other bits there. A row is a draw of the bit generator, an
    exact operation on draws or the core's kernel, so draws that refuse every other numpy ufunc
    make the same rows, bit for bit."""
    guarded = np.asarray(_contract.mine_rows(512, 2, _GuardedRng(42)))
    assert guarded.tobytes() == _contract.mine_rows(512, 2, np.random.default_rng(42)).tobytes()


def test_the_witness_grid_is_the_cores_arithmetic() -> None:
    """The exists-witness search grid's powers of ten are the core's `pow`, as the rows' are."""
    exponents = np.linspace(-3, 12, 120)
    got = np.asarray(_contract._powers_of_ten(exponents.view(_Exact)))
    assert got.tolist() == [_core.kernel_binary_scalar('pow', 10.0, e) for e in exponents.tolist()]
    powers = {_core.kernel_binary_scalar('pow', 10.0, e)
              for e in np.concatenate([np.linspace(12, 308, 49)[1:], np.linspace(12, -3, 120),
                                       exponents]).tolist()}
    linear = set(np.linspace(-3, 3, 81).tolist())
    xs = np.asarray(_contract._XS).tolist()
    assert all(x in linear or abs(x) in powers for x in xs)
    assert powers <= {abs(x) for x in xs}


def test_the_judge_reads_one_matrix() -> None:
    rows = _contract.judge_rows()
    assert rows.shape == (3 * 1024, 2)
    assert np.array_equal(rows, _contract.mine_matrix(*_contract.DEFAULT_ROWS))
    assert not rows.flags.writeable
    assert not hasattr(_contract, 'set_rows')


def test_a_mine_of_other_rows_leaves_the_judges_rows_as_they_are(tmp_path) -> None:
    import yaml

    from simplipy import SimpliPyEngine
    ops = {
        "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
        "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
        "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
    }
    (tmp_path / "rules.json").write_text("[]")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"operators": ops, "rules": "rules.json"}))
    engine = SimpliPyEngine.from_config(str(tmp_path / "config.yaml"))
    before = _contract.judge_rows().copy()
    engine.find_rules(max_source_pattern_length=2, max_target_pattern_length=2, dummy_variables=1,
                      X=64, seed=3, verbose=False, output_file=str(tmp_path / "mined.json"),
                      modes="f64")
    assert np.array_equal(_contract.judge_rows(), before)
    # the two log-exp identities the reduced mine's own rows refused at one rounding, held alike
    for lhs, rhs in (('asin sinh asinh x0', 'asin x0'), ('log exp acos x0', 'acos x0')):
        assert verify_rule(lhs.split(), rhs.split())['tier'] == 'core'
