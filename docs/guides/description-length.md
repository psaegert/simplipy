# Description length

SimpliPy simplifies symbolic expressions: formulas such as `x1*exp(-x2/x3)`, built from
variables, numbers, operators and functions. A formula may hold slots, each written
`<constant>`, for numbers that are not known yet. Such a formula is a template. A fit is a
numerical search for the numbers that make a formula match data, and it fills the slots. When
several candidate formulas fit the same data, a common rule ranks them by fit and length
together: a short formula wins unless a longer one fits much better. SimpliPy measures that
length as the formula's description length. `complexity()` returns it in thousandths of a bit,
and [srbf](https://srbf.readthedocs.io/), a benchmark for methods that find formulas in data,
reports it in bits as MDL, named after the minimum description length principle.

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("acj-5-4-llm", install=True)
engine.complexity('pi*x1')   # -> 13000
```

An engine is SimpliPy's vocabulary of operators and functions together with a rule set, the
rewrites that `simplify` may apply; `install=True` downloads `acj-5-4-llm`, SimpliPy's default
engine. The description length does not depend on the rule set: an engine without rules gives
each formula the description length that `acj-5-4-llm` gives it. It depends on the mode only
through [reading](#simplipy-measures-the-form-it-reads). Every description length on this page
is in the default mode, `f64`, and srbf and Flash-ANSR call `complexity()` in it too.

## A description length is modelled on the length of a message

Picture sending a formula, as a string of bits, to someone who knows SimpliPy's vocabulary but
not the formula. The description length is modelled on the length of that message. The encoding
is fixed in advance and gives each symbol a fixed number of bits: a sum or a product costs 3
bits, `sin` 6 bits, and `asin` 8 bits. A number's length grows roughly with its significant
digits: `2` costs 2.585 bits, `3.14` costs 10.884 bits, and `3.141592653589793` costs 56.480
bits. Lengths are in bits and need not be whole: choosing among 3 equally likely options takes
\(\log_2 3 \approx 1.585\) bits. Unlike a real message, a few parts cost nothing; the
[formula](#the-description-length-is-a-sum-over-the-parts) lists them.

Under this measure, a shorter description length means a simpler formula. `pi*x1` costs 13 bits
and `3.141592653589793*x1` costs 65.480 bits, although their values agree to 16 digits.
[Simplifying a formula never raises its description length](#simplification-never-raises-the-description-length).

## SimpliPy measures the form it reads

Before it measures a formula, SimpliPy reads it into its canonical form, a standard spelling,
and every description length on this page is that of the read form. In the default mode,
reading:

- removes the grouping of sums and products and puts their parts in a fixed order;
- collects like terms and like factors;
- drops a factor 1, a term 0 and an exponent 1, and reads `x1^0` as `1` and `0*x1` as `0`;
- merges a power of a power into one power when both exponents are whole numbers, as
  `(x3^2)^-1` becomes `x3^-2`;
- holds each number exactly, as a fraction of two 128-bit integers, whose numerator and
  denominator are each below \(2^{127} \approx 1.7 \cdot 10^{38}\) in size, and computes with
  these numbers exactly; a number outside this range, such as `1e-300`, whose denominator has
  301 digits, is kept as written, and so is a computation whose result would fall outside it,
  such as `2^200`;
- replaces a function applied to a number, such as `sin(0)`, by its value when the value costs
  fewer bits than the function; the value is computed in float64, 64-bit floating point, and
  written as the shortest decimal that float64 reads back as the same value;
- writes `exp(1)` as `e`; `pi` and `e` are the two named constants SimpliPy knows;
- writes a power with the exponent \(1/2\), such as `x1^(1/2)`, as `rootn(x1, 2)`, where
  `rootn(x, n)` is the real \(n\)-th root of \(x\);
- drops a number that multiplies a `<constant>` or is added to it;
- places minus signs, as the [next section](#a-minus-sign-costs-a-sign-bit-a-product-or-nothing)
  describes;
- applies a few more identities that the mode accepts, such as `rootn(x1^3, 3)` read as `x1`;
  the [formal specification](../formal.md#3-the-normalization-function) describes when it
  accepts one.

<!-- price-table -->
| typed | read as | bits of the read form |
|---|---|---|
| `x2 + x1` | `x1 + x2` | 15 |
| `(x1 + x2) + x3` | `x1 + x2 + x3` | 21 |
| `x1 + x1` | `2*x1` | 11.585 |
| `x1*x1` | `x1^2` | 11.585 |
| `x1/x1` | `1` | 2 |
| `1*x1 + 0` | `x1` | 6 |
| `(x3^2)^-1` | `x3^-2` | 12.585 |
| `2*3` | `6` | 3.807 |
| `0.1 + 0.2` | `3/10`, printed `0.3` | 4 |
| `sin(0)` | `0` | 2 |
| `sin(2)` | `sin(2)` | 8.585 |
| `exp(1)` | `e` | 4 |
| `x1^(1/2)` | `rootn(x1, 2)` | 14.585 |
| `x1/9.81` | `100/981*x1` | 26.598 |
| `2*<constant>*x1` | `<constant>*x1` | 76 |
| `acos(-1)` | `acos(-1)` | 11 |

- `0.1 + 0.2` is read as \(3/10\): SimpliPy adds \(1/10\) and \(1/5\) exactly.
- `sin(2)` stays, because its float64 value, 0.9092974268256817, would cost more.
- `x1/9.81` is read as `100/981*x1`: numbers are exact, so dividing by \(9.81\) multiplies by
  its exact reciprocal, \(100/981\).
- `x1^(1/2)` costs 14.585 bits because it is read as `rootn(x1, 2)`; the typed spelling is never
  measured on its own.

Reading gives each formula one fixed form, so that formulas that differ only in the ways listed
above get the same description length. The fixed form can cost more than the typed one, as
`x1^(1/2)` shows. Finding a cheaper formula is the job of [simplification](simplify.md):
`acos(-1)` is read as written, at 11 bits, and simplification turns it into `pi`, at 4.

The other modes accept more identities while reading. In `real` and `permissive`, reading also
cancels some functions applied to their inverses, identities that hold for every real number:
`atanh(tanh(x1))`, `log(exp(x1))`, `asinh(sinh(x1))` and `sinh(asinh(x1))` are read as `x1`,
and `acosh(cosh(x1))` as `abs(x1)`. `permissive`, which accepts identities that fail for some
inputs, also reads `rootn(x1^2, 2)` as `x1`. An engine without rules reads these the same way,
so this is reading, not a rewrite. The argument `mode` picks the mode, and the result is again
in thousandths of a bit:

```python
engine.complexity('atanh(tanh(x1))')                # -> 22000
engine.complexity('atanh(tanh(x1))', mode='real')   # -> 6000
```

The [simplification guide](simplify.md) describes the modes.

## A minus sign costs a sign bit, a product, or nothing

A minus sign on a number held exactly is its sign bit: `-2` costs 3.585 bits, `2` costs 2.585.
A minus sign on a product that has a number held exactly merges into that number, as in
`-2*x1`. In a product without such a number, the sign is a factor \(-1\), which costs nothing:
`-x1*x2` costs what `x1*x2` costs. A minus sign on anything else, such as a variable, a function
or a number kept as written, makes a product of \(-1\) and that thing, and the product costs its
3 bits: `-x1` costs 9 bits, and `-1e-300` costs 3 bits more than `1e-300`. On `inf`, a minus
sign gives the symbol `-inf`.

Reading also moves minus signs. In two cases it compares two placements, and in both it puts the
sign in front of a sum or a product only when that is strictly cheaper:

- A sum, wherever it stands, is held with its terms' signs as typed, or as \(-1\) times the sum
  with every sign flipped. `-x1 - x2 - x3` is read as `-(x1 + x2 + x3)`, since one factor
  \(-1\) costs less than three negated terms. Inside a function's argument the same holds:
  `sin(-x2 - x3)` is read as `sin(-(x2 + x3))`, and the rule for odd functions below then
  decides whether that minus sign moves out of `sin`. `x1 - x2 - x3` and `-(-x1 + x2 + x3)` cost
  the same either way, and both read as `x1 - x2 - x3`.
- An odd function is one with \(f(-a) = -f(a)\): `sin`, `tan`, `asin`, `atan`, `sinh`, `tanh`,
  `asinh` and `atanh`. When its argument is a number or a sum with terms of both signs, a minus
  sign can sit inside the argument or in front of the function. `x1*sin(-2)` is read as
  `-x1*sin(2)`, since the factor \(-1\) is free and the sign of \(-2\) costs a bit. `-5*sin(2)`
  and `5*sin(-2)` cost the same, and both read as `5*sin(-2)`. `-tanh(x1 - x2)` is read as
  `tanh(x2 - x1)`, which saves the 3 bits of the product with \(-1\).

For any other argument of an odd function, such as `-x2`, `-2*x2` or `-(x2 + x3)`, the sign
stays inside: `x1*sin(-x2)` costs 24 bits, although `-x1*sin(x2)` costs 21, and
`x1*sin(-x2 - x3)` costs 33, although `-x1*sin(x2 + x3)` costs 30.

Two more moves do not depend on cost. An odd power takes a minus sign out of its base, unless
the base is a sum with terms of both signs, which the sum rule orients: `(-x1)^3` is read as
`-x1^3`, at the same 15 bits. An even power drops a minus sign on its base, and `cos`, `cosh`
and `abs` drop one on their argument: `(-x1)^2` is read as `x1^2`, and `cos(-x1)` as
`cos(x1)`. The printed result shows each term of a sum with its own sign, so
`-(x1 + x2 + x3)` prints as `-x1 - x2 - x3`. In the table, `(-1)*` marks a product with the
factor \(-1\).

<!-- price-table -->
| typed | read as | bits of the read form |
|---|---|---|
| `-2` | `-2` | 3.585 |
| `-2*x1` | `-2*x1` | 12.585 |
| `-x1*x2` | `(-1)*x1*x2` | 15 |
| `-x1` | `(-1)*x1` | 9 |
| `-1e-300` | `(-1)*1e-300` | 13.234 |
| `x1 - x2` | `x1 + (-1)*x2` | 18 |
| `x1 - 1` | `x1 + (-1)` | 12 |
| `1 - x1` | `1 + (-1)*x1` | 14 |
| `-x1 - x2 - x3` | `(-1)*(x1 + x2 + x3)` | 24 |
| `x1 - x2 - x3` | `x1 + (-1)*x2 + (-1)*x3` | 27 |
| `x1*sin(-2)` | `(-1)*x1*sin(2)` | 17.585 |
| `-5*sin(2)` | `5*sin(-2)` | 16.170 |
| `-tanh(x1 - x2)` | `tanh(x2 + (-1)*x1)` | 26 |
| `x1*sin(-x2)` | `x1*sin((-1)*x2)` | 24 |
| `(-x1)^3` | `(-1)*x1^3` | 15 |
| `-inf` | `-inf` | 8 |

## The description length is a sum over the parts

The description length of a read form is a sum over its parts:

$$
\text{description length} \;=\; \sum_{\text{symbols } s} c(s) \;+\; \sum_{\text{numbers } v} L(v)
$$

A symbol \(s\) is an operator, a function, a variable, `pi`, `e`, `<constant>`, `inf`, `-inf`
or `nan`, and \(c(s)\) is its entry in the [symbol table](#every-symbol-costs-a-fixed-number-of-bits).
A sum, a product and a power each count once, however many terms or factors they have. Two
kinds of \(-1\) are not numbers \(v\) and cost nothing: the factor \(-1\) of a product that has no
number held exactly, and the exponent \(-1\) of a power, as in `x1/x2` or `x1^-1`. A number
\(v\) costs

$$
L(v) \;=\; 1 \;+\; [\,v<0\,] \;+\; \min\!\big(\,\ell(p) + [\,q\neq 1\,]\,\ell(q),\;\; \ell(m) + \ell(k)\,\big),
\qquad \ell(x) \;=\; \max\!\big(1,\; \log_2(1+|x|)\big).
$$

- The first 1 is the selector bit, which says which of two codes follows. A number with only one
  code pays it too, since the receiver reads it before it knows the number.
- \([\,v<0\,]\) is 1 for a negative number and 0 otherwise: the sign bit.
- The fraction code writes \(v\) as \(\pm p/q\) in lowest terms. \([\,q\neq 1\,]\) is 1 when
  \(v\) is not a whole number, so a whole number pays only for \(p\).
- The decimal code writes \(v\) as \(\pm m \cdot 10^{k}\), with \(m\) a whole number that does
  not end in 0 and \(k \neq 0\): \(0.25\) is \(25 \cdot 10^{-2}\), and \(1000\) is
  \(1 \cdot 10^{3}\). The minimum takes it only where it exists. \(1/3\) has none, and neither has
  \(7\), whose \(k\) would be 0. SimpliPy computes \(m\) in the same 128-bit integers that hold
  \(p\) and \(q\), so the decimal code also needs \(m\) below \(2^{127}\):
  \(34359738369/(2^{60} \cdot 5^{20})\) has none, because its \(m\) is about
  \(3.1 \cdot 10^{38}\).
- \(\ell(x)\) is the cost of writing one whole number, about the number of binary digits it
  needs: \(\ell(1) = 1\), \(\ell(2) = 1.585\), \(\ell(3) = 2\). It ignores the sign of \(x\), and
  it is never below 1 bit, which matters only for the number 0.

`complexity()` rounds each logarithm to three decimal places before the sum. A number kept as
written is measured from its written digits by the same two codes. These parts cost nothing: the
two kinds of \(-1\) above, the sign of \(k\), whether a \(q\) follows, and the number of
terms or factors of a sum or product. So `1000` and `0.001` both cost 4 bits, and `x1*x2` and
`-x1*x2` both cost 15. The [formal specification](../formal.md#5-the-reduction-ordering) states
this formula, and its proof that simplification always ends rests on it.

## Every symbol costs a fixed number of bits

The table is set by hand. For sums, products, powers, `rootn` and the inverse and hyperbolic
functions, its bits are close to what their frequencies give in a corpus of 400 templates from
symbolic regression, the search for formulas that fit data: a kind of part, such as a sum or
`tanh`, that makes up a fraction \(f\) of all parts in the corpus would cost
\(\log_2(1/f)\) bits. A variable costs more than its frequency gives, because it also names
which variable it is: 6 bits, whatever its name and however many variables a formula has. `pi`
and `e` occur nowhere in the corpus; they cost less than the cheapest formula for either value,
so that writing the name lowers the description length. `<constant>` costs 67 bits: the largest
\(L(v)\) of a float64 number written in its shortest decimal spelling is 66.649 bits, and 67 is
that, rounded up. A negative number kept as written can still cost more than a slot, since its
minus sign makes a 3-bit product instead of a sign bit: `-8.9002954340287245e-308` costs 68.649
bits. The table is part of SimpliPy and the same for every engine; an engine's vocabulary only
decides which symbols exist.

<!-- symbol-table -->
| symbol | bits |
|---|---|
| a variable, such as `x1` or `x12` | 6 |
| a sum, a product, a power | 3 |
| `pi`, `e` | 4 |
| `exp`, `log`, `abs`, `sin`, `cos`, `tan`, `rootn` | 6 |
| `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, any other function, `inf`, `-inf`, `nan` | 8 |
| `<constant>` | 67 |

<!-- price-table -->
| typed | bits of the read form | parts |
|---|---|---|
| `x1 + x2` | 15 | sum 3 + `x1` 6 + `x2` 6 |
| `x1 + x2 + x3` | 21 | sum 3 + `x1` 6 + `x2` 6 + `x3` 6 |
| `x1*x2` | 15 | product 3 + `x1` 6 + `x2` 6 |
| `x1/x2` | 18 | product 3 + `x1` 6 + power 3 + `x2` 6 |
| `1/x1` | 9 | power 3 + `x1` 6 |
| `x1^2` | 11.585 | power 3 + `x1` 6 + `2` 2.585 |
| `sin(x1)` | 12 | `sin` 6 + `x1` 6 |
| `asin(x1)` | 14 | `asin` 8 + `x1` 6 |
| `rootn(x1, 3)` | 15 | `rootn` 6 + `x1` 6 + `3` 3 |
| `pi*x1` | 13 | product 3 + `pi` 4 + `x1` 6 |

- `x1 + x2 + x3` costs one sum, not two: SimpliPy holds a sum of any length as one sum.
- `x1/x2` costs 3 bits more than `x1*x2`. SimpliPy reads a division as a product with a power,
  `x1*x2^-1`, and pays for the power; its exponent \(-1\) is free.
- `1/x1` is the power `x1^-1` alone, 3 bits more than `x1`.

## A number costs the shorter of its codes

For a number held exactly, the value is measured, not the way it is typed: `0.5` and `1/2` cost
the same.

<!-- price-table -->
| number | bits | code |
|---|---|---|
| `0` | 2 | fraction, the only code |
| `2` | 2.585 | fraction, the only code |
| `7` | 4 | fraction, the only code |
| `10` | 3 | decimal \(1 \cdot 10^{1}\), the cheaper code |
| `1000` | 4 | decimal \(1 \cdot 10^{3}\), the cheaper code |
| `1000000` | 4.807 | decimal \(1 \cdot 10^{6}\), the cheaper code |
| `-2` | 3.585 | fraction and the sign bit |
| `0.5` | 3.585 | fraction \(1/2\), the cheaper code |
| `0.1` | 3 | decimal \(1 \cdot 10^{-1}\), the cheaper code |
| `0.001` | 4 | decimal \(1 \cdot 10^{-3}\), the cheaper code |
| `1/3` | 4 | fraction, the only code |
| `22/7` | 8.524 | fraction, the only code |
| `3.14` | 10.884 | decimal \(314 \cdot 10^{-2}\), the cheaper code |
| `3.141592653589793` | 56.480 | decimal, the cheaper code |
| `9.81` | 12.525 | decimal \(981 \cdot 10^{-2}\), the cheaper code |
| `100/981` | 17.598 | fraction, the only code |
| `6.674e-11` | 17.612 | decimal \(6674 \cdot 10^{-14}\), the cheaper code |
| `1e-300` | 10.234 | decimal \(1 \cdot 10^{-300}\), kept as written |

- `10` costs less than `7`. Ten is the selector bit and two whole numbers of 1 bit each, 3 bits;
  seven is the selector bit and 3 bits for 7, 4 bits.
- `1000000` costs only 4.807 bits: it is \(1 \cdot 10^{6}\), and writing the 6 costs 2.807 bits.
- `0.5` costs more than `0.1`. One half is cheapest as the fraction \(1/2\), while one tenth is
  \(1 \cdot 10^{-1}\), as cheap as `10`.
- `0.001` costs the same as `1000`: both write the whole numbers 1 and 3, and the sign of \(k\)
  is free.

## A formula is measured with the numbers it has

A template is measured with its slots, at 67 bits each, and a fitted formula with its fitted
numbers in place. Flash-ANSR, a neural network that proposes formulas for data, and srbf both
measure a fitted formula this way. The table shows a template and one fit of it.

<!-- price-table -->
| typed | bits of the read form | parts |
|---|---|---|
| `<constant>*x1 + <constant>` | 146 | sum 3 + product 3 + `<constant>` 67 + `x1` 6 + `<constant>` 67 |
| `1.9871384520976653*x1 + 0.4140937651248972` | 128.195 | sum 3 + product 3 + `1.9871384520976653` 59.229 + `x1` 6 + `0.4140937651248972` 56.966 |

## Worked examples, from a product to a law

Each part costs its entry in the tables above, or \(L(v)\) for a number not in them, and the
parts add up to the total.

<!-- price-table -->
| typed | bits of the read form | parts |
|---|---|---|
| `0.5*x1*x2^2` | 24.170 | product 3 + `1/2` 3.585 + `x1` 6 + power 3 + `x2` 6 + `2` 2.585 |
| `6.674e-11*x1*x2/x3^2` | 45.197 | product 3 + `6.674e-11` 17.612 + `x1` 6 + `x2` 6 + power 3 + `x3` 6 + `-2` 3.585 |
| `x1*exp(-x2/x3)` | 33 | product 3 + `x1` 6 + `exp` 6 + product 3 + `x2` 6 + power 3 + `x3` 6 |
| `(x1 + x2)^2` | 20.585 | power 3 + sum 3 + `x1` 6 + `x2` 6 + `2` 2.585 |
| `x1^2 + 2*x1*x2 + x2^2` | 43.755 | sum 3 + power 3 + `x1` 6 + `2` 2.585 + product 3 + `2` 2.585 + `x1` 6 + `x2` 6 + power 3 + `x2` 6 + `2` 2.585 |
| `2*pi*rootn(x1/9.81, 2)` | 44.768 | product 3 + `2` 2.585 + `pi` 4 + `rootn` 6 + product 3 + `100/981` 17.598 + `x1` 6 + `2` 2.585 |
| `6.283185307179586*rootn(x1/9.81, 2)` | 95.663 | product 3 + `6.283185307179586` 57.480 + `rootn` 6 + product 3 + `100/981` 17.598 + `x1` 6 + `2` 2.585 |

- `6.674e-11*x1*x2/x3^2` is Newton's law of gravitation. Its division by `x3^2` is the power
  `(x3^2)^-1`, which reading merges into `x3^-2`; the exponent \(-2\) costs 3.585 bits.
- `x1^2 + 2*x1*x2 + x2^2` costs more than twice `(x1 + x2)^2`. Both have the same value, and the
  factored form writes each variable once.
- A pendulum's period is \(2\pi\sqrt{x_1/9.81}\). Typed with the decimal `6.283185307179586`
  for \(2\pi\), it costs 95.663 bits instead of 44.768. The decimal costs 57.480 bits, where `2`
  and `pi` cost 2.585 and 4.

## Calling complexity()

`complexity()` takes a formula as an infix string, such as `'pi*x1'`, or as a prefix token list,
which writes each operator before its operands: `['*', 'np.pi', 'x1']`. In tokens, `pi` and `e`
are written with NumPy's names, `np.pi` and `np.e`; a token `pi` costs 6 bits, as a variable
does. A slot is the token `'<constant>'`, and a number is its decimal string, such as `'2.5'`.
`complexity()` returns an integer.

```python
engine.complexity('pi*x1')                       # -> 13000
engine.complexity(['*', 'np.pi', 'x1'])          # -> 13000
engine.complexity(['*', 'pi', 'x1'])             # -> 15000
engine.complexity(['*', '<constant>', 'x1'])     # -> 76000
engine.complexity(['*', '2.5', 'x1'])            # -> 14170
```

## Simplification never raises the description length

`engine.complexity(engine.simplify(expr))` is never more than `engine.complexity(expr)`, and it
may be the same.

```python
expr = 'x2*(x2 + (x1 + 1)/x2)'

engine.complexity(expr)                     # -> 41000
engine.simplify(expr)                       # -> 'x1 + x2^2 + 1'
engine.complexity(engine.simplify(expr))    # -> 22585
```

## What the description length does not measure

- How well a formula fits data.
- How fast the formula evaluates.
- How many tokens it has: `sin(x1)` costs 12 bits and `asin(x1)` costs 14, with the same number
  of tokens in the same shape.
- Simplicity in any absolute sense: description lengths compare formulas under this one
  encoding.

## Flash-ANSR ranks formulas by it, and srbf reports it

- [Flash-ANSR](https://flash-ansr.readthedocs.io/en/stable/getting_started/) ranks its candidates
  by a two-part code that adds a term for how well a formula fits the data to this description
  length; its documentation explains the data term.
- [srbf](https://srbf.readthedocs.io/en/stable/metrics/#predicted_mdl-ground_truth_mdl-mdl_ratio)
  reports the description length of each predicted and each true formula, and their ratio.
- [Flash-ANSR with PySR](https://flash-ansr.readthedocs.io/en/stable/hybrid/) adds the candidates
  of PySR, an evolutionary search for formulas, and measures them the same way as Flash-ANSR's
  own.
