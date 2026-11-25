"""How an engine evaluates expressions: the explicit constant fold, the host libm's fingerprint, and
expressions compiled to Python callables."""

import hashlib
from types import CodeType, FunctionType
from typing import Callable, cast

import numpy as np

from simplipy._reading import _Reading
from simplipy._schema import _SPECIAL_LEAVES
from simplipy.utils import is_named_constant


class _Evaluation(_Reading):
    """Evaluation of expressions."""

    def evaluate_constants(
            self, expression: str | list[str] | tuple[str, ...] | np.ndarray) -> str | list[str]:
        """Fold every variable-free subtree to its f64 value -- EXPLICITLY, on request.

        This is the front door for the numeric evaluation :meth:`simplify` refuses to
        do. The engine folds only what it computes EXACTLY (rational arithmetic, in the
        constructors), because an f64 evaluation that lands exactly on a cheap literal
        while the true value differs at 1e-17 is not a simplification but a rounding. So
        ``simplify(['tan', '1'])`` returns ``tan 1`` in every mode, and this returns
        ``1.5574077246549023``.

        Having it as its own method is the whole point: the capability exists, it is
        named for what it does, it is opt-in, and it can never contaminate a canonical
        form. Someone who wants a number asks for one; someone canonicalising a training
        corpus does not get one by accident.

        A subtree is foldable when it contains no variable and no slot -- a
        ``<constant>`` is a FITTED degree of freedom, not a value, and so is a named
        constant (``<constant_NAME>``), so a subtree carrying one is left alone. Folding is
        MAXIMAL: the largest foldable subtrees go first, so ``tan(1) + x`` folds the
        ``tan(1)`` and stops.

        A subtree whose value is NOT FINITE is left unevaluated. Substituting
        ``float("inf")`` or ``float("nan")`` would inject a non-finite token into an
        expression that did not have one, which is the defect class that put a
        non-finite guard into flash-ansr; refusing is the safe direction and the caller
        can still evaluate the expression themselves.

        NEVER called by :meth:`simplify`, in any mode. Returns the input's dialect: infix
        text for a ``str``, otherwise the tagged or binary-prefix tokens it was given. To
        change the notation, convert the answer (:meth:`to_infix`, :meth:`to_tagged`,
        :meth:`to_prefix`).
        """
        in_form, tokens = self._read_input(expression)
        if not tokens:
            return '' if in_form == 'infix' else []
        # an infix `str` was already read (strictly) by `_read_input`
        prefix = self._core.to_prefix_syntactic(tokens) if in_form == 'tagged' else tokens
        self._core.check_form(prefix)

        arity = self.operator_arity
        # A variable is what `expression_variables` says it is, the engine's one definition.
        variables = set(self.expression_variables(prefix))

        def span(i: int) -> int:
            j = i + 1
            for _ in range(arity.get(prefix[i], 0)):
                j = span(j)
            return j

        def foldable(sub: list[str]) -> bool:
            return not any(t in variables or t == '<constant>' or is_named_constant(t)
                           or t[:1] in '_!?$' and t[1:].isdigit()
                           for t in sub)

        out: list[str] = []
        i = 0
        while i < len(prefix):
            j = span(i)
            sub = prefix[i:j]
            if len(sub) > 1 and foldable(sub):
                try:
                    value = float(np.asarray(
                        self._core.evaluate_batch(sub, ['x0'], [0.0], 1, [])).ravel()[0])
                except ValueError:  # an operator the evaluator has no kernel for: left as is
                    value = float('nan')
                if np.isfinite(value):
                    out.append(repr(value))
                    i = j
                    continue
            out.append(prefix[i])
            i += 1

        if in_form == 'tagged':
            return cast(list, self.to_tagged(out))
        if in_form == 'infix':
            return cast(str, self.to_infix(out))
        return out

    # The libm probe battery: expressions whose folded values are
    # decided by the SYSTEM libm the deployed folder calls -- the C symbols bound in
    # rust/numeric.rs resolve on THIS machine at load, so the exposure is runtime, not
    # build-time, and no wheel qualification can cover it. Chosen to discriminate real
    # builds: the cosh(acosh(k)) family (>= 5 shipped acj-4-3 rules exist only because
    # glibc 2.43 is 1 ulp wrong exactly there), argument-reduction-sensitive trig at a
    # large argument, atanh near its edge (libm-routed), exp/log/pow at
    # spread magnitudes, and ulp-amplifying compositions. Fixed and ordered: the
    # fingerprint must be comparable across sidecars.
    _LIBM_PROBES: tuple[tuple[str, ...], ...] = (
        ('cosh', 'acosh', '2'), ('cosh', 'acosh', '5'), ('cosh', 'acosh', '8'),
        ('cosh', 'acosh', 'np.e'), ('cosh', 'acosh', 'np.pi'),
        ('sin', '1'), ('sin', '2'), ('sin', '1000000000'),
        ('cos', '1'), ('cos', '1000000000'), ('tan', '1'), ('tan', '100'),
        ('asin', '0.5'), ('asin', '0.9999'), ('acos', '0.5'), ('atan', '2'),
        ('sinh', '1'), ('sinh', '20'), ('cosh', '1'), ('tanh', '1'),
        ('asinh', '2'), ('acosh', '1.5'), ('atanh', '0.5'), ('atanh', '0.999'),
        ('exp', '1'), ('exp', '40'), ('exp', '-40'), ('log', '2'), ('log', 'np.pi'),
        ('pow', 'np.pi', 'np.e'), ('pow', '2', '0.5'), ('pow', '10', '-7'),
        ('rootn', '2', '3'), ('sin', 'exp', '3'), ('log', 'cosh', '10'),
        ('tan', 'exp', '2'),
        # the 0.15 heads that call libm: erf, and atan2 off its exact cells
        ('erf', '0.5'), ('erf', '2'), ('atan2', '1', '2'), ('atan2', '-3', '0.5'),
        # besselj, whose port calls libm in each regime: AMOS's power series and its large-
        # argument expansion, a non-integer order, and the rotation through Y of a negative one
        ('besselj', '0', '1'), ('besselj', '1', '1000'), ('besselj', '2.5', '30'),
        ('besselj', '-1.5', '2'),
        # erfc and the other Bessel functions, ported as scipy ships them, in each regime:
        # erfc's `1 - erf` and its two rational tails; Y and K through AMOS (small and large
        # arguments, a large order's uniform expansion, Y's rotation of a negative order); I
        # through Temme's method, its asymptotic expansion and the uniform one above order 50
        ('erfc', '0.5'), ('erfc', '2'), ('erfc', '10'),
        ('bessely', '0', '1'), ('bessely', '1', '1000'), ('bessely', '-1.5', '2'),
        ('bessely', '100', '30'),
        ('besseli', '0.5', '1'), ('besseli', '1', '1000'), ('besseli', '60', '10'),
        ('besselk', '0', '1'), ('besselk', '2.5', '30'), ('besselk', '100', '30'),
    )

    def libm_fingerprint(self) -> str:
        """A sha256 fingerprint of the host's libm, computed through the deployed
        folding path.

        The transcendental folds go through the SYSTEM libm (the C symbols
        `rust/numeric.rs` binds), which resolves on the running machine -- so two
        hosts with identical wheels and identical `pip freeze` can mine different
        artifacts (measured: the `cosh(acosh(2)) -> 2` rule exists on glibc 2.43 and
        not on the publish host). The fingerprint makes that difference a recorded,
        comparable fact: it is stored in every mine's provenance sidecar
        (`environment.libm_fingerprint`), and two sidecars with different values are
        mines over different arithmetic.
        """
        assert self._core is not None
        lines = []
        for probe in self._LIBM_PROBES:
            # `None` is the core's own answer for a probe it cannot fold (an operator the
            # config lacks); an error is not an answer and propagates.
            out = self._core.evaluate_constant_subtree(list(probe))
            lines.append(f"{' '.join(probe)}={out}")
        return hashlib.sha256('\n'.join(lines).encode()).hexdigest()

    def expression_variables(
            self, expression: 'str | list[str] | tuple[str, ...] | np.ndarray') -> list[str]:
        """The free variables of an expression, IN ORDER OF FIRST APPEARANCE.

        Order is the contract, not an accident: it is the signature
        :meth:`as_callable` binds, so `f(*values)` has to mean what the expression
        reads. A token is a variable when it is none of the things that are not one --
        an operator, a numeric literal, a special constant, or a slot. Slots are
        excluded because a `<constant>` is a fitted degree of freedom, not an argument,
        and so is a named constant (``<constant_NAME>``): a parameter, never a variable.
        """
        from .utils import is_numeric_string
        tokens = self.to_prefix(expression)
        out: list[str] = []
        for t in tokens:
            if t in self._operators_config or t in out:
                continue
            if is_numeric_string(t) or t in _SPECIAL_LEAVES:
                continue
            # The PARENTHESIZED negative literal (`(-3)`) is the engine's own spelling and
            # is not caught by `is_numeric_string`; `!` is a slot sigil like the rest.
            # Missing both made 3,389 of the shipped artifact's 10,638 rule sides report a
            # literal as a free variable, and `as_callable` then compiled a bad signature.
            if t.startswith('(') and t.endswith(')') and is_numeric_string(t[1:-1]):
                continue
            if t.startswith(('<', '_', '$', '?', '!')):
                continue
            out.append(t)
        return out

    def to_realization(
            self, expression: 'str | list[str] | tuple[str, ...] | np.ndarray') -> list[str]:
        """The expression in REALIZATION notation: prefix tokens, operators spelled as
        the callables they run (``sin`` -> ``np.sin``).

        The fourth notation of the conversion family, and reversible like the other
        three: :meth:`to_prefix`, :meth:`to_infix` and :meth:`to_tagged` all accept
        realization-spelled input and read it back. It is what the compile pipeline
        consumes, and it is occasionally what you want directly -- to see which callable
        an operator actually resolves to, without running anything.

        Reading realization notation back needs an injective realization map; writing it
        does not, so this direction never refuses.
        """
        return self.operators_to_realizations(self.to_prefix(expression))

    def as_code(
            self, expression: 'str | list[str] | tuple[str, ...] | np.ndarray',
            variables: list[str] | None = None) -> CodeType:
        """Compile the expression to a Python code object.

        ``as_``, not ``to_``: this is TERMINAL. Every ``to_*`` in this API is a notation
        that converts back, and a code object has no syntax to recover -- naming it
        ``to_compiled`` would advertise a round-trip that cannot exist.

        Every literal is read as its nearest float64 (:func:`simplipy.utils.evaluator_literal`):
        an integer beyond 2^53 is written as a float, and one beyond float64's range as
        ``inf``. Integers up to 2^53 stay Python ``int``s, so arithmetic between two of them
        is exact integer arithmetic: ``sin(4294967296*4294967297)`` raises numpy's TypeError
        as before (the simplified form folds the product to one literal).

        A named constant (``<constant_NAME>``) is a PARAMETER of the compiled function, one
        per name, written ``constant_NAME`` in the code: with ``variables=None`` the
        signature is :meth:`expression_variables` followed by the named constants in order of
        first appearance (``<constant_a>*x1 + <constant_a>*x2`` compiles to
        ``lambda x1, x2, constant_a: ...``); an explicit ``variables`` list names each of them
        by its token, where the caller wants it. The anonymous ``<constant>`` is not a
        parameter of its own: name its slots first (:func:`simplipy.utils.explicit_constant_placeholders`).

        .. warning::
           Compiling runs the expression's realizations through :func:`compile`, so the
           usual trust rules apply -- the evaluation namespace is scoping, not a sandbox
           (:mod:`simplipy.trust`).
        """
        from .utils import codify, evaluator_literal
        prefix = self.to_prefix(expression)
        named: list[str] = []
        for t in prefix:
            if is_named_constant(t) and t not in named:
                named.append(t)
        if variables is None:
            variables = self.expression_variables(prefix) + named
        if named:
            missing = [t for t in named if t not in variables]
            if missing:
                raise ValueError(
                    f"the named constants {missing} are parameters of the compiled function: "
                    f"list them in `variables` (or pass variables=None)")
            identifier = {t: 'constant_' + t[len('<constant_'):-1] for t in named}
            clash = set(identifier.values()) & (set(variables) | set(prefix))
            if clash:
                raise ValueError(
                    f"the identifiers {sorted(clash)} of named constants are already names in "
                    f"the expression or in `variables`")
            prefix = [identifier.get(t, t) for t in prefix]
            variables = [identifier.get(v, v) for v in variables]
        # Every literal is read as its nearest float64 (`evaluator_literal`): an integer
        # beyond 2^53 would otherwise reach Python as an exact int, which numpy's ufuncs and
        # float arithmetic reject above int64 and float64's range.
        prefix = [evaluator_literal(t) for t in prefix]
        return codify(self.prefix_to_infix(prefix, realization=True), variables)

    def as_callable(
            self, expression: 'str | list[str] | tuple[str, ...] | np.ndarray',
            variables: list[str] | None = None) -> Callable[..., float]:
        """Compile the expression to a callable bound to THIS engine's namespace.

        The one-step form of :meth:`as_code` followed by :meth:`code_to_lambda`, which
        is the pipeline nearly every caller wanted. Terminal, for the reason given on
        :meth:`as_code`.
        """
        return self.code_to_lambda(self.as_code(expression, variables))

    def code_to_lambda(self, code: CodeType) -> Callable[..., float]:
        """Converts a Python code object into an executable lambda function.

        The compiled expression is bound to THIS engine's evaluation namespace: the
        modules its own realizations needed, plus ``np`` for the ``np.pi``/``np.e``
        token grammar. Until 0.13 every engine shared ``simplipy.engine``'s module
        globals, so one engine's imports were reachable from another's expressions,
        along with simplipy's own imports (``os``, ``importlib``, ...); since 0.13 each
        engine has its own namespace. Python still injects builtins, and a hostile EXPRESSION is
        still unsafe to compile -- the namespace is scoping, not a sandbox
        (:mod:`simplipy.trust`).

        Compatibility: this was a ``staticmethod`` before 0.13. Instance calls
        (``engine.code_to_lambda(code)``, the documented and only in-tree usage) are
        unaffected; a call on the CLASS now needs an instance.

        Parameters
        ----------
        code : CodeType
            The compiled code object to convert.

        Returns
        -------
        Callable[..., float]
            An executable lambda function.
        """
        return FunctionType(code, self._eval_namespace)()
