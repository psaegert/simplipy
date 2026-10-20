# Displaying expressions

`simplify` returns the canonical form, and its numbers are exact: a literal keeps every digit
it was given, and a coefficient that combines several literals is their exact product.
`prettify` is the display of that form. It decides how each number is written for a reader and
leaves the expression's structure as it is: it simplifies nothing.

```python
import simplipy as sp

engine = sp.SimpliPyEngine.load("acj-5-4-llm", install=True)

answer = engine.simplify("2.14583142772210003669120562399025*x1")
answer
# -> '2.14583142772210003669120562399025*x1'
engine.prettify(answer)
# -> '2.1458314277221002*x1'
```

Nothing inside SimpliPy calls `prettify` or reads what it returns, so the answer of `simplify`
is the same whether or not it is displayed. Call it where people read expressions: tables,
plots, logs and reports.

## The rules

The digit budget `digits` (default 3) counts every digit a spelling writes, a numerator of 1
included: `1/12` counts 3.

A **number that fits the budget is written exactly, as `simplify` writes it**. The budget
decides only whether a number is written exactly or as its float; how an exact number is
written is `simplify`'s own rule: the decimal where it is the cheaper spelling (`0.3`), else a
fraction, with the reciprocal as the divisor where that is shorter (`5/21` beside a factor is
`x1/4.2`).

```python
engine.prettify(engine.simplify("x1*3/10"))
# -> '0.3*x1'
engine.prettify(engine.simplify("x1*5/21"))
# -> 'x1/4.2'
engine.prettify(engine.simplify("22/7*x1"))
# -> '22*x1/7'
engine.prettify(engine.simplify("x2 + 0.5"))
# -> 'x2 + 1/2'
```

**Otherwise the value is one float**: the float64 nearest to it, in its shortest form, as a
multiplier. A value a hair from a short number is written as that number.

```python
engine.prettify(engine.simplify("x1/1024"))
# -> '9.765625e-4*x1'
engine.prettify(engine.simplify("x1/0.225147640565301"))
# -> '4.441529999999994*x1'
engine.prettify(engine.simplify("1.000000000000000000003*x1"))
# -> 'x1'
```

**Precision.** `precision=k` shows a number written as a float with `k` significant digits
instead of its shortest float64 form; a number written exactly is not affected. The shortened
number is written as every number is, so the display reads back as itself: `0.175` is 7/40,
written `7/40`, and pi is factored out only at full precision. The default, `precision=None`,
writes floats in full.

```python
engine.prettify(answer, precision=3)
# -> '2.15*x1'
engine.prettify(engine.simplify("x1*5/21"), precision=1)
# -> 'x1/4.2'
engine.prettify(engine.simplify("x0/(2*3.141592653589793)"), precision=3)
# -> '0.159*x0'
```

**pi** is the literal `3.141592653589793`. A value that is a number written exactly times pi,
or over it, is written with the literal factored out.

```python
engine.prettify(engine.simplify("x0/(2*3.141592653589793)"))
# -> 'x0/(2*3.141592653589793)'
engine.prettify(engine.simplify("3*x0^2/(20*3.141592653589793*x1*x2)"))
# -> '0.15*x0^2/(3.141592653589793*x1*x2)'
```

**Divisors.** Every divisor of a product joins one denominator, functions and sums included:
one slash. The display is of the product reading it back builds, so it is the same whether the
canonical form holds a reciprocal per factor or one of their product. A float whose reciprocal
float64 holds as written stays a divisor where the product has another divisor; a bare
`x1/0.225147640565301` becomes a multiplier (above).

```python
engine.prettify(engine.simplify("x1/x2/log(x3)"))
# -> 'x1/(x2*log(x3))'
engine.prettify(engine.simplify("x1/sin(x3)/(x2 + 1)"))
# -> 'x1/(sin(x3)*(x2 + 1))'
engine.prettify(engine.simplify("x1/243/x2"))
# -> 'x1/(243*x2)'
```

Tagged tokens write a section member per reciprocal, as the tagged form reads them, so there a
reciprocal of a product holding a sum or a function stays one bag in the section.

**A function of floats is one float.** A function of numbers that would all be written as
floats is written as its float64, computed as SimpliPy's evaluator computes it and shown at the
`precision`. A function of a number written exactly stays a function, and so does one whose
float would be 0, infinite or subnormal. No numbers combine to inf, 0 or a subnormal.

```python
engine.prettify(engine.simplify("x1*exp(-381/2)"))
# -> '1.8488479557720672e-83*x1'
engine.prettify(engine.simplify("x1*exp(-381/2)"), precision=3)
# -> '1.85e-83*x1'
engine.prettify(engine.simplify("sin(sin(9463/890))"))
# -> '-0.8045036621463277'
engine.prettify(engine.simplify("x1 + sin(2)"))
# -> 'x1 + sin(2)'
```

**Exponents** follow the same rules, and `x^(1/n)` and `x^(-1/n)` are written `rootn(x, n)`
and `1/rootn(x, n)` for even `n`. Odd roots stay powers: `rootn(x, 3)` and `x^(1/3)` differ for
negative `x`.

```python
engine.prettify(engine.simplify("x1^(-1/2)"))
# -> '1/rootn(x1, 2)'
engine.prettify(engine.simplify("(x2 + 0.5)^(500000000000000/4427)"))
# -> '(x2 + 1/2)^1.1294330246216399e11'
```

**Notation**: scientific from `1e6` and below `1e-3`, with a lowercase `e`.

## Forms and modes

`prettify` answers in the dialect it was given, as `simplify` does: infix text for a `str`,
tokens for tokens. `form=` chooses one: `'infix'`, `'explicit'` or `'tagged'`.

```python
tokens = engine.simplify(["/", "1", "*", "2", "3.141592653589793"])
engine.prettify(tokens)
# -> ['/', '1', '*', '2', '3.141592653589793']
engine.prettify(tokens, form="infix")
# -> '1/(2*3.141592653589793)'
engine.prettify(tokens, form="tagged")
# -> ['<mul>', '1', '<div>', '2', '3.141592653589793', '</mul>']
```

At the default `precision=None` a number is written exactly or as the float64 nearest to it,
so the display reads back to the same float64 values: it evaluates as the expression does, up to
the order of its operations. A display at a `precision` does not read back to them.
Displaying a display gives it back:

```python
shown = engine.prettify(answer)
engine.prettify(shown) == shown
# -> True
```

The exception is a display in which writing numbers as their floats makes parts alike that the
expression keeps apart. `prettify` does not decide that they are alike, and it does not simplify
what it wrote, so the display of `sin(1.00000000000000000001*x1) - sin(x1)` is
`sin(x1) - sin(x1)` (whose own display is `0`), and an exponent a hair above 3 is written `3`
without multiplying the power out. `mode` (default `f64`) is the mode whose number domain reads
the numbers; the display rules are designed for `f64`.
