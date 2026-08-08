"""One-time boundary conversion of the legacy-vocabulary corpus to the new vocabulary.

Pure RESPELLING, no simplification: the corpus must keep every raw redex so the gates
exercise the engine on the original distribution. Legacy hyper-operators map to their
exact literal spellings (`mult3 t -> * 3 t`, `pow5 t -> pow t 5`, `pow1_3 t -> rootn t 3`);
everything else passes through unchanged. This is the strictly-clean doctrine: legacy data
converts ONCE at the boundary; the engine itself has no legacy reading.
"""
import json
import os

ARITY1 = {"abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "cos", "cosh",
          "exp", "inv", "log", "neg", "sin", "sinh", "tan", "tanh",
          "div2", "div3", "div4", "div5", "mult2", "mult3", "mult4", "mult5",
          "pow1_2", "pow1_3", "pow1_4", "pow1_5", "pow2", "pow3", "pow4", "pow5"}
ARITY2 = {"*", "+", "-", "/", "pow"}

MULT = {f"mult{k}": [str(k)] for k in (2, 3, 4, 5)}
# IN-VOCABULARY spellings (2026-08-06). These maps used to emit `0.5`, `1/3`, `0.25`, `0.2`
# and `pow t 0.5` / `pow t 0.25`, which are the ONLY reason this corpus carried literals
# outside the v24.0 numeric vocabulary (-10..10 + e + pi): 38.90% of its literals, in 56.2%
# of its rows, all four of them introduced HERE rather than present in the legacy data
# (`0.25` x113, `0.5` x99, `0.2` x68, `1/3` x65 -- exactly this table).
#
# The replacements are token-level only and PARSE IDENTICALLY: `/ 1 2` and `0.5` are the same
# `Ex::Num` leaf (complexity is spelling-independent, see contract Sec 10.10(1)), and an even
# `pow t 1/n` is normalized to `rootn t n` by the constructor anyway, so `rootn` is simply the
# canonical spelling written down. Verified row by row: `simplify` output is unchanged on all
# 400 expressions.
DIV = {f"div{k}": ["/", "1", str(k)] for k in (2, 3, 4, 5)}
POWK = {f"pow{k}": [str(k)] for k in (2, 3, 4, 5)}
EVEN_ROOT = {"pow1_2": "2", "pow1_4": "4"}
ODD_ROOT = {"pow1_3": "3", "pow1_5": "5"}


def respell(tokens: list[str], i: int = 0) -> tuple[list[str], int]:
    t = tokens[i]
    arity = 2 if t in ARITY2 else (1 if t in ARITY1 else 0)
    parts = []
    j = i + 1
    for _ in range(arity):
        p, j = respell(tokens, j)
        parts.append(p)
    if t in MULT:
        return ["*", *MULT[t], *parts[0]], j
    if t in DIV:
        return ["*", *DIV[t], *parts[0]], j
    if t in POWK:
        return ["pow", *parts[0], *POWK[t]], j
    if t in EVEN_ROOT:  # even root: `rootn` IS the canonical spelling of `pow t 1/n`
        return ["rootn", *parts[0], EVEN_ROOT[t]], j
    if t in ODD_ROOT:
        return ["rootn", *parts[0], ODD_ROOT[t]], j
    out = [t]
    for p in parts:
        out.extend(p)
    return out, j


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    corpus = json.load(open(os.path.join(here, "raw_skeletons.json")))
    out = []
    for expr in corpus:
        r, j = respell(list(expr))
        assert j == len(expr), expr
        out.append(r)
    json.dump(out, open(os.path.join(here, "raw_skeletons_nv.json"), "w"))
    legacy = set(MULT) | set(DIV) | set(POWK) | set(EVEN_ROOT) | set(ODD_ROOT)
    assert not any(legacy & set(e) for e in out)
    print(f"{len(out)} expressions respelled; tokens {sum(map(len, corpus))} -> "
          f"{sum(map(len, out))}; zero legacy operators remain")


if __name__ == "__main__":
    main()
