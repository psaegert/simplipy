"""How an engine reads and writes expressions: validation, the conversions between the prefix, infix
and tagged forms, and the realization notation."""

from typing import Literal, cast

import numpy as np

from simplipy._loading import _Loading
from simplipy._schema import _TAGGED_DIALECT_TOKENS, _token_of
from simplipy.utils import is_numeric_string


class _Reading(_Loading):
    """Validation and conversion of expressions."""

    def is_valid(self, prefix_expression: 'str | list[str] | tuple[str, ...] | np.ndarray',
                 verbose: bool = False) -> bool:
        """Checks if an expression is syntactically valid, in ANY of the three forms.

        An expression is valid if it reads as a well-formed expression over THIS
        engine's vocabulary: every operator gets exactly its arity of operands, a
        single root remains, every tag section closes, and every token passes the token
        grammar every reader applies (no reserved numeric spelling such as ``nan``,
        ``inf`` or ``1/0``, no empty or spaced token, at most 4,096 tokens).

        All three external forms are accepted and VERIFIED; the form is detected
        exactly as the conversion API detects it, by TYPE first and then by dialect:

        * an infix ``str`` (``'2*atan(x0)/3'``) -- read, then verified;
        * an explicit binary-prefix token sequence (``['*', '2', 'atan', 'x0']``);
        * a TAGGED token sequence (``['<mul>', '2', 'atan', 'x0', '</mul>']``) --
          the engine's own native serialization.

        A ``list``, ``tuple`` or 1-D ``ndarray`` is a token sequence; tag-bearing
        sequences are verified by the shared liberal parser (the arbiter for that
        dialect, exactly as in :meth:`simplify`), everything else by the arity
        oracle. Note the difference from :meth:`read_infix`, which TOLERATES
        undeclared vocabulary: ``read_infix('zeta(x0)')`` reads, while
        ``is_valid('zeta(x0)')`` is False if this engine declares no ``zeta`` (a defined
        function, such as the shipped configs' ``sqrt``, is valid) --
        this method verifies against the vocabulary rather than merely reading.

        Malformed input is a False VERDICT, never an exception -- that is what a
        predicate promises. Only a type that is no expression at all raises.

        One inherited tolerance is worth knowing about on the infix path: the
        reader silently DROPS unmatched OPENING parentheses (``'((x0'`` reads as
        ``x0``), so such a string validates. Unmatched CLOSING parentheses survive
        as a stray token and are rejected. A character the reader has no token for
        (``|``, ``!``, ``?``, ``~``, brackets, braces, quotes, non-ASCII letters) is
        refused, so ``'|x0|'`` is invalid. This method reports on what the reader
        reads rather than keeping a second, partial infix grammar of its own.

        Parameters
        ----------
        prefix_expression : str or list[str] or tuple[str, ...] or np.ndarray
            The expression, in any of the three forms.
        verbose : bool, optional
            If True, prints the reason for invalidity. Defaults to False.

        Returns
        -------
        bool
            True if the expression is valid, False otherwise.

        Raises
        ------
        TypeError
            If the input is neither an infix ``str`` nor a token sequence.
        """
        tokens = self._validation_tokens(prefix_expression)
        if tokens is None:
            # The infix reader refused the string outright: nothing to verify, and
            # nothing the diagnostic printer could say about tokens that never were.
            if verbose:
                print(f'Invalid expression {prefix_expression!r}: not a readable '
                      f'infix expression')
            return False
        if _TAGGED_DIALECT_TOKENS.intersection(tokens):
            # TAGGED: the shared AC parser is the arbiter for this dialect (the arity
            # oracle has no reading of the bag delimiters and would reject every one
            # of them). A projection with no pass and no search parses and serializes
            # WITHOUT running the rewrite loop, so this is a pure well-formedness
            # question; malformed input fails the parse and the FFI raises, which is the
            # False verdict.
            try:
                self._core.ac_simplify_in_mode(tokens, 0, 'tagged', 0, rule_mode='default')
            except ValueError:
                if verbose:
                    self._explain_invalid(tokens, 'tagged')
                return False
            return True
        # EXPLICIT (and everything else): the verdict ALWAYS comes from the compiled core
        # (single implementation); the pure-Python loop below only survives as a DIAGNOSTIC
        # printer for the (rare) verbose call, and only runs when the core has already
        # rejected the expression. This is the pre-existing path, unchanged, so every call
        # that could be written before this method grew its other two forms answers exactly
        # as it did.
        valid = self._core.is_valid(tokens)
        if verbose and not valid:
            self._explain_invalid(tokens, 'explicit')
        if valid:
            # The arity oracle reads the tree only; every reader also runs the token grammar
            # (no reserved numeric spelling such as `nan` or `1/0`, no empty or spaced token) and
            # the 4,096-token cap. An expression every reader refuses is not valid.
            try:
                self._core.check_form(tokens)
            except ValueError as error:
                if verbose:
                    print(f'Invalid expression {prefix_expression!r}: {error}')
                return False
        return valid

    def _validation_tokens(
            self, expression: 'str | list[str] | tuple[str, ...] | np.ndarray') -> 'list[str] | None':
        """Read :meth:`is_valid`'s input into tokens, mirroring the conversion API's
        detection. Returns ``None`` when an infix ``str`` does not read at all --
        an invalid expression, not an error. Refuses a non-expression TYPE loudly."""
        if isinstance(expression, str):
            try:
                return self._core.parse(expression, True, False)
            except ValueError:
                return None
        if isinstance(expression, np.ndarray):
            if expression.ndim != 1 or expression.dtype.kind not in {'U', 'S', 'O'}:
                raise TypeError(
                    'is_valid expects an infix str or a token sequence '
                    '(list, tuple or 1-D ndarray of string-like tokens)')
            if expression.dtype.kind == 'S':
                # a byte string is the token it spells, as the conversion API reads it
                expression = expression.astype(str)
            return [str(t) for t in expression.tolist()]
        if isinstance(expression, (list, tuple)):
            return list(expression)
        raise TypeError(
            f'is_valid expects an infix str or a token sequence '
            f'(list, tuple or 1-D ndarray), not {type(expression).__name__}')

    def _explain_invalid(self, prefix_expression: list[str], form: str) -> None:
        """Print WHY an expression in ``form`` failed :meth:`is_valid` (diagnostics only,
        no verdict)."""
        stack: list[str] = []
        if form == 'tagged':
            # Without this branch the walk below misreads the bag delimiters as variables
            # and prints 'Variable must be leaf node' -- actively misleading. is_valid
            # READS this dialect, so reaching here means the sequence is malformed AS
            # tagged, not that the dialect was refused.
            tagged = _TAGGED_DIALECT_TOKENS.intersection(prefix_expression)
            print(f'Invalid expression {prefix_expression}: malformed '
                  f'tagged-serialization sequence (carries {sorted(tagged)}). Every '
                  f'section must open and close in order and every operator must get '
                  f"its operands; re-project a valid expression with "
                  f"to_prefix(expr) or to_tagged(expr).")
            return
        arity = self._effective_arity
        if len(prefix_expression) > 1 and prefix_expression[0] not in arity:
            print(f'Invalid expression {prefix_expression}: Variable must be leaf node')
            return
        for token in reversed(prefix_expression):
            if token != '<constant>' and is_numeric_string(token):
                try:
                    float(token)
                except ValueError:
                    print(f'Invalid token {token} in expression {prefix_expression}')
                    return
            if token in arity:
                if len(stack) < arity[token]:
                    print(f'Not enough operands for operator {token} in expression {prefix_expression}')
                    return
                for _ in range(arity[token]):
                    stack.pop()
            stack.append(token)
        if len(stack) != 1:
            print(f'Stack is not empty after parsing the expression {prefix_expression}')

    def prefix_to_infix(self, tokens: list[str], power: Literal['func', '**'] = 'func', realization: bool = False) -> str:
        """Converts a prefix expression to an infix string with minimal parentheses.

        Parameters
        ----------
        tokens : list[str]
            The prefix expression to render.
        power : {'func', '**'}, optional
            Controls how power operators are emitted. ``'func'`` keeps the engine's
            function spelling, ``pow(x, 3)``, while ``'**'`` renders Python-style
            exponentiation, ``x ** 3``.
        realization : bool, optional
            If True, operator tokens are replaced with their runtime
            realizations (for example, ``'sin'`` becomes ``'np.sin'``), so the
            output can be compiled directly.

        Returns
        -------
        str
            The formatted infix expression.

        Raises
        ------
        ValueError
            If the provided tokens do not form a well-formed prefix expression --
            including the TAGGED serialization, which this converter does not read:
            convert it with :meth:`to_infix`, or to the binary prefix with
            :meth:`to_prefix`.
        """
        tagged = _TAGGED_DIALECT_TOKENS.intersection(tokens)
        if tagged:
            raise ValueError(
                f"prefix_to_infix reads the explicit binary-prefix dialect, but the input "
                f"carries tagged-serialization tokens {sorted(tagged)}. Convert it with "
                f"to_infix(expr), or to the binary prefix with to_prefix(expr).")
        return self._core.prefix_to_infix(list(tokens), power, realization)

    def infix_to_prefix(self, infix_expression: str) -> list[str]:
        """Converts an infix expression string to prefix notation.

        This method uses a standard algorithm (related to Shunting-yard) to
        parse the infix string, respecting operator precedence and parentheses.
        Whitespace separates tokens; see :meth:`read_infix` for how a function without
        parentheses and implicit products are read.

        Parameters
        ----------
        infix_expression : str
            The mathematical expression in infix notation.

        Returns
        -------
        list[str]
            A list of tokens representing the expression in prefix notation.

        Raises
        ------
        ValueError
            On a character the reader has no token for (``|x0|``; see :meth:`read_infix`),
            when two operands are separated only by whitespace (``x0 x1``, ``2 3``,
            ``zeta x0`` for an undeclared ``zeta``), or when an implicit product falls inside
            the argument of a function without parentheses (``sin 2x0``).
        """
        # Regex to tokenize expression properly (handles floating-point numbers and scientific notation)
        return self._core.infix_to_prefix(infix_expression)

    def convert_expression(self, prefix_expr: list[str]) -> list[str]:
        """Normalizes an expression into the engine's standard internal format.

        It resolves operator aliases, spells ``**`` as the engine's ``pow``
        (``** x0 2`` becomes ``pow x0 2``), and applies a unary negation of a numeric
        literal to the literal (``neg 3`` becomes ``-3``).

        Parameters
        ----------
        prefix_expr : list[str]
            The prefix expression to convert.

        Returns
        -------
        list[str]
            The normalized prefix expression.
        """
        return self._core.convert_expression(list(prefix_expr))

    def read_infix(
            self,
            infix_expression: str,
            convert_expression: bool = True,
            mask_numbers: bool = False) -> list[str]:
        """Read an infix string into the explicit binary-prefix dialect, TOLERANTLY
        and PRESERVING ITS SPELLING.

        The name states the contract because the contract is a CAPABILITY that no
        other public entry offers (it was called ``parse`` before 0.14.0):

        * **TOLERANT of unknown vocabulary.** A function this engine's config neither
          declares nor defines survives as a bare leaf --
          ``read_infix('zeta(x0)') == ['zeta', 'x0']`` -- so a corpus written in a
          wider vocabulary can be read first and desugared afterwards. Every
          conversion entry (:meth:`to_prefix`, :meth:`to_infix`, :meth:`to_tagged`)
          and :meth:`simplify` REFUSE such input: they read only the engine's
          operators and its defined functions (a config's ``defined_functions:``,
          such as ``sqrt``, whose head every reader keeps as written).
        * **SPELLING-PRESERVING: it does NOT canonicalise.** This is a reader, not a
          simplification -- it never enters the canonical state, so like terms are not
          collected and constants are not folded::

              read_infix('x0+x0')            ==  ['+', 'x0', 'x0']   # as written
              to_prefix('x0+x0')             ==  ['+', 'x0', 'x0']   # notation only
              simplify(read_infix('x0+x0'))  ==  ['*', '2', 'x0']    # canonical state

          Since the conversion/simplification split the CONVERSIONS preserve the
          spelling too -- they are notation, not content -- so the contrast that
          remains is with :meth:`simplify`, the one entry that canonicalises.
          ``read_infix``'s exclusive capability is the vocabulary TOLERANCE above;
          :meth:`to_prefix` refuses what this reader accepts.

        Implicit multiplication is read after a number or a closing parenthesis, with
        the precedence of ``*``: ``2x0`` is ``2*x0``, ``2(x0 + 1)`` is ``2*(x0 + 1)``,
        ``(a)(b)`` is ``(a)*(b)`` and ``1/2x0`` is ``(1/2)*x0``. A name before a
        parenthesis is always a call, so ``zeta(x0)`` stays ``['zeta', 'x0']``. Token
        lists are expected well formed and get no such reading.

        Whitespace separates tokens, and an implicit product forms across it as across
        touching tokens (``2 x0`` is ``2*x0``). A declared one-argument function without
        parentheses applies to the operand after it, taking powers and signs but not
        products, quotients or sums: ``sin x0^2`` is ``sin(x0^2)`` and ``log x0 / 2`` is
        ``log(x0)/2``. ``x0 * * 2`` is ``x0**2`` and ``1 e-5`` is ``1e-5``. ``ValueError``
        is raised for two operands with only whitespace between them and no product
        (``x0 x1``, ``2 3``, ``zeta x0`` for an undeclared ``zeta``) and for an implicit
        product inside such an argument (``sin 2x0``: ``sin(2*x0)`` or ``sin(2)*x0``?).

        A character the reader has no token for (anything but ASCII letters, digits, ``_``,
        ``.``, ``+ - * / ^ ( ) ,``, whitespace and the spellings ``<constant>``,
        ``<constant_NAME>`` and ``float("inf")``) raises ``ValueError`` naming the character and
        its position: ``'|x0|'``, ``'x0!'``, ``'?0 + x1'`` and a fullwidth ``'ｘ０'`` are
        refused. (Before 0.15 such a character was dropped, and the text around it read as if
        it were absent: ``'|x0|'`` read as ``['x0']``.) A hexadecimal spelling is no numeral
        but an implicit product: ``'0x1F'`` reads as ``['*', '0', 'x1F']``. Every infix reader
        shares this tokenizer, :meth:`simplify` and the conversions included.

        Mechanically it is :meth:`infix_to_prefix` plus optional
        :meth:`convert_expression` normalization.

        Parameters
        ----------
        infix_expression : str
            The mathematical expression in infix notation.
        convert_expression : bool, optional
            If True, the expression is normalized using `convert_expression`
            (``**`` becomes the engine's ``pow``). Defaults to True.
        mask_numbers : bool, optional
            If True, all numerical literals in the expression are replaced
            with a generic '<constant>' token. Defaults to False.

        Returns
        -------
        list[str]
            The processed prefix expression after conversion and masking (if
            enabled).
        """

        return self._core.parse(infix_expression, convert_expression, mask_numbers)

    def _read_input(
            self, expression: str | list[str] | tuple[str, ...] | np.ndarray,
            expand: bool = False,
    ) -> tuple[str, list[str]]:
        """``(form, tokens)``: THE reading of an expression argument, shared by
        :meth:`simplify`, :meth:`complexity`, :meth:`prettify`, the conversion API and
        :mod:`simplipy.normalization`.

        A ``str`` is INFIX and is read strictly into explicit prefix tokens. A list, tuple
        or 1-D ``ndarray`` of string-like tokens is a token sequence: TAGGED when it
        carries a bag delimiter or an inverse-section marker, EXPLICIT binary prefix
        otherwise. A sequence written in REALIZATION notation (each operator spelled as
        the callable it runs, ``simplipy.operators.neg`` for ``neg``) is read back to
        operator tokens first: it is a notation of the same expression. Only
        realizations that differ from their operator token identify it, and those are
        dotted names no operator vocabulary contains, so it is detected, never guessed.
        Reading it back needs an injective realization map, so a config where two
        operators share a realization refuses it (:meth:`_assert_realizations_are_invertible`).

        Refuses loudly: an ``ndarray`` that is not 1-D or not string-like raises
        ``ValueError``, any other type ``TypeError``. The reading does not validate;
        each consumer validates through the core.

        With ``expand`` (the entries into the canonical state) every DEFINED FUNCTION is
        expanded into its definition (:meth:`_expand_defined`); without it (the conversions)
        the tokens keep the head as written.
        """
        if isinstance(expression, str):
            tokens = self._core.parse(expression, strict=True)
            return 'infix', self._expand_defined(tokens) if expand else tokens
        if isinstance(expression, np.ndarray):
            if expression.ndim != 1:
                raise ValueError('expected a one-dimensional numpy array of tokens, '
                                 f'got {expression.ndim} dimensions')
            if expression.dtype.kind not in {'U', 'S', 'O'}:
                raise ValueError('expected a numpy array of string-like tokens, '
                                 f'got dtype {expression.dtype}')
            if expression.dtype.kind == 'O':
                tokens = [_token_of(t) for t in expression.tolist()]
            else:
                tokens = [str(t) for t in expression.astype(str).tolist()]
        elif isinstance(expression, (list, tuple)):
            tokens = [_token_of(t) for t in expression]
        else:
            raise TypeError(
                f'expected an infix str or a token sequence '
                f'(list, tuple or 1-D ndarray), not {type(expression).__name__}')
        if self._realization_only_tokens().intersection(tokens):
            self._assert_realizations_are_invertible()
            tokens = self.realizations_to_operators(tokens)
        if expand:
            tokens = self._expand_defined(tokens)
        return ('tagged' if _TAGGED_DIALECT_TOKENS.intersection(tokens) else 'explicit'), tokens

    def to_infix(self, expression: str | list[str] | tuple[str, ...] | np.ndarray) -> str:
        """Convert an expression to the INFIX form -- NOTATION ONLY.

        One of the three PURE conversions: a syntactic re-notation, never a
        simplification. No canonical state is built, no rule fires, nothing is
        collected, folded, reordered or re-spelled, and the answer does not depend on
        the engine ARTIFACT. Use :meth:`simplify` for content; compose them as
        ``simplify(to_infix(x))``.

        A ``str`` input is already in the target notation and comes back VERBATIM
        (validated). Infix erases one distinction the token dialects carry: it has no
        ``inv`` glyph, so ``inv X`` renders ``1/X`` and reads back as ``/ 1 X``.

        Raises ``ValueError`` on a malformed expression or undeclared vocabulary
        (conversions are strict; :meth:`read_infix` stays the tolerant reader),
        ``TypeError`` on a non-expression type.
        """
        form, tokens = self._read_input(expression)
        if form == 'infix':
            if tokens:
                self._core.check_form(tokens)
            return cast(str, expression)
        if not tokens:
            return ''
        if form == 'tagged':
            tokens = self._core.to_prefix_syntactic(tokens)
        else:
            self._core.check_form(tokens)
        return self._core.prefix_to_infix(tokens, 'func', False)

    def to_prefix(self, expression: str | list[str] | tuple[str, ...] | np.ndarray) -> list[str]:
        """Convert an expression to the explicit binary PREFIX form -- NOTATION ONLY.

        See :meth:`to_infix` for the shared contract. A prefix token sequence is
        already in the target form and comes back BYTE-IDENTICAL (validated only):
        this conversion expands bags and touches nothing else.

        Expanding a bag has to pick an association, and the choice is stated: the
        positive part is a RIGHT-nested chain (``<add> a b c </add>`` becomes
        ``+ a + b c``, the association the canonical explicit emitter already
        produces), while an inverse SECTION becomes a LEFT-nested chain
        (``<mul> a <div> c d </mul>`` becomes ``/ / a c d``) because a section
        inverts each member individually -- only a left chain spells that without
        inventing the product ``c*d``.
        """
        form, tokens = self._read_input(expression)
        if not tokens:
            return []
        if form == 'tagged':
            return cast(list, self._core.to_prefix_syntactic(tokens))
        self._core.check_form(tokens)
        return tokens

    def to_tagged(self, expression: str | list[str] | tuple[str, ...] | np.ndarray) -> list[str]:
        """Convert an expression to the TAGGED form -- NOTATION ONLY.

        See :meth:`to_infix` for the shared contract. Bag members keep their
        source ORDER (sorting is canonicalisation, which is :meth:`simplify`'s job)
        and literals keep their spelling (a fraction token aside, below), so a tagged
        sequence already in normal form comes back byte-identical.

        The regrouping is CONSERVATIVE: a chain flattens only across nodes of the same
        kind at the same POLARITY. The right operand of ``-``/``/`` and the operand of
        ``neg``/``inv`` become ONE section member, never a flattened run, because
        ``1/(a*b)`` and ``(1/a)*(1/b)`` are different expressions -- they part company
        at ``0`` and ``inf``, which is why the AC core gates that distribution behind a
        nonzero certificate. A tagged sequence declares its dialect by its bags, so every
        quotient carries one: a fraction token is its quotient (``1/2`` is
        ``<mul> 1 <div> 2 </mul>``), and a lone reciprocal, having no one-member bag, is
        bagged with the unit (``inv x0`` is ``<mul> 1 <div> x0 </mul>``). A lone negation
        keeps its unary spelling ``neg x0``, which both dialects read alike.
        """
        form, tokens = self._read_input(expression)
        if not tokens:
            return []
        return cast(list, self._core.to_tagged_syntactic(tokens))

    def _realization_only_tokens(self) -> set[str]:
        """Realization spellings that are NOT also operator tokens -- the detectable half: the
        config's, and the core's own for the core operators the config does not declare
        (:meth:`_core_realizations`), which the realization rendering writes too.

        Cached on the instance: the operator table is fixed at construction.
        """
        cached = getattr(self, '_realization_only_cache', None)
        if cached is None:
            ops = set(self._operators_config)
            cached = {spec['realization'] for spec in self._operators_config.values()
                      if spec['realization'] not in ops} | set(self._core_realizations().values())
            self._realization_only_cache = cached
        return cached

    def _assert_realizations_are_invertible(self) -> None:
        """Reading realization notation back REQUIRES an injective realization map.

        Two operators may legally share a realization -- nothing in the config format
        forbids `abs` and `absolute` both running `np.abs` -- and then the spelling
        `np.abs` names both, so no reader can recover which was meant. Injectivity is a
        property of a CONFIG, not a guarantee of the format, so it is checked when it
        matters and refused loudly rather than resolved by picking one.
        """
        seen: dict[str, str] = {}
        for op, spec in self._operators_config.items():
            r = spec['realization']
            if r in seen:
                raise ValueError(
                    f"this engine's realization map is not invertible: operators "
                    f"{seen[r]!r} and {op!r} both realize as {r!r}, so realization "
                    f"notation cannot be read back. Convert from operator notation "
                    f"instead, or give the two operators distinct realizations.")
            seen[r] = op

    def operators_to_realizations(self, prefix_expression: list[str] | tuple[str, ...]) -> list[str]:
        """Convert canonical operator names to their runtime realizations (e.g. ``'sin'`` -> ``'np.sin'``)."""
        return self._core.operators_to_realizations(list(prefix_expression))

    def realizations_to_operators(self, prefix_expression: list[str]) -> list[str]:
        """Convert realization tokens (e.g. ``'np.sin'``) back to canonical operator names."""
        return self._core.realizations_to_operators(list(prefix_expression))
