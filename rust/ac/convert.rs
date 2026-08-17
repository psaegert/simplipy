//! The boundary between the AC core and the prefix token language.
//!
//! [`from_prefix`] DESUGARS the full legacy grammar -- `-`, `/`, `neg`, `inv`, and the
//! DELETED hyper-operator family `mult{k}`/`div{k}`/`pow{k}`/`pow1_{k}` -- plus every
//! numeric literal form into the core. Legacy tokens are accepted
//! on INPUT ONLY, so old artifacts and corpora stay readable; `pow1_3`/`pow1_5` desugar to
//! `rootn` (the general signed root -- a real odd root differs from the rational power on
//! `x < 0`).
//!
//! [`to_prefix`] serializes into binary chains with literal coefficients and exponents
//! (`* 7 x`, `pow x 3`) and the `rootn` built-in emitted natively; negatives/inverses use
//! `-` / `/` / `neg` / `inv`. No hyper-operator is ever emitted.
//!
//! Exactness at the boundary: numeric tokens parse as DECIMALS into exact rationals (`"0.2"`
//! means one fifth, exactly). Serialization picks the ARGMIN of `mu_rat`'s two codes -- the
//! decimal token when the decimal code is cheaper (`1/5`, `6/5`, every power of ten), the
//! fraction otherwise (`1/2`, `5/8`, and everything non-terminating) -- so the print follows
//! the cost. The TAGGED form additionally spells an in-vocabulary fraction structurally
//! (`<mul> p <div> q </mul>`), since it is the model-facing dialect and carries no atomic
//! fraction or decimal token. String-level round-trip
//! `to_prefix . from_prefix . to_prefix == to_prefix` holds by construction.

use std::sync::OnceLock;

use crate::tokens::{Tok, TokenView};

use super::expr::{add, canon, fun, mul, pow, Cx, Ex};
use super::rat::Rat;

/// Parse a prefix token sequence into a canonical AC expression. `None` on malformed input
/// (arity underflow / trailing tokens) -- callers pass the input through unchanged, exactly as
/// the shipped engine treats malformed expressions.
pub fn from_prefix(tokens: &[Tok], cx: &Cx) -> Option<Ex> {
    let (e, next) = parse_one(tokens, 0, cx)?;
    if next != tokens.len() {
        return None;
    }
    Some(e)
}

fn parse_one(tokens: &[Tok], idx: usize, cx: &Cx) -> Option<(Ex, usize)> {
    let view = cx.view;
    let t = *tokens.get(idx)?;
    let s = view.resolve_owned(t);
    // The TAGGED bags of the strict prefix form: `<add> ... </add>` / `<mul> ... </mul>`.
    // Children are read until the matching closer; a bag needs >= 2 children and a closer
    // (anything else is malformed and fails the parse). The two languages share one parser:
    // tag tokens are not operators of any config, so there is no ambiguity.
    if s == "<add>" || s == "<mul>" {
        let is_add = s == "<add>";
        let (closer, section) = if is_add {
            ("</add>", "<sub>")
        } else {
            ("</mul>", "<div>")
        };
        let mut parts = Vec::new();
        let mut inverted = false; // past the <sub>/<div> INVERSE SECTION marker?
        let mut at = idx + 1;
        loop {
            let nt = *tokens.get(at)?;
            let ns = view.resolve_owned(nt);
            if ns == closer {
                at += 1;
                break;
            }
            if ns == section {
                if inverted {
                    return None; // at most one inverse section per bag
                }
                inverted = true;
                at += 1;
                continue;
            }
            let (p, next) = parse_one(tokens, at, cx)?;
            // The inverse section applies the bag's group inverse to each of its members:
            // subtracted terms after `<sub>`, denominator factors after `<div>`.
            let p = if !inverted {
                p
            } else if is_add {
                mul(vec![Ex::Num(Rat::NEG_ONE), p], cx)
            } else {
                pow(p, Ex::Num(Rat::NEG_ONE), cx)
            };
            parts.push(p);
            at = next;
        }
        // An empty numerator/positive part is allowed (`<mul> <div> x y </mul>` = 1/(x*y));
        // the CANONICAL result still needs at least 2 members overall, and the constructors
        // handle singleton collapse (`<mul> <div> x </mul>` would be `inv x`'s job -- reject
        // a bag that parses to fewer than 2 raw members to keep the form canonical-ish while
        // staying liberal about sections).
        if parts.len() < 2 {
            return None;
        }
        let e = if is_add {
            add(parts, cx)
        } else {
            mul(parts, cx)
        };
        return Some((e, at));
    }
    if s == "</add>" || s == "</mul>" || s == "<sub>" || s == "<div>" {
        return None; // a closer or section marker with no enclosing bag
    }
    // The general signed root `rootn x n` (IEEE 754's rootn): binary, config-independent
    // (like the tags, it belongs to the AC language, not the operator config).
    if s == "rootn" {
        let (base, at) = parse_one(tokens, idx + 1, cx)?;
        let (index, at) = parse_one(tokens, at, cx)?;
        return Some((fun(t, vec![base, index], cx), at));
    }
    let arity = view.arity(t);
    match arity {
        None => Some((parse_leaf_token(t, &s, view), idx + 1)),
        Some(n) => {
            let mut args = Vec::with_capacity(n as usize);
            let mut at = idx + 1;
            for _ in 0..n {
                let (a, next) = parse_one(tokens, at, cx)?;
                args.push(a);
                at = next;
            }
            Some((desugar(&s, t, args, cx), at))
        }
    }
}

/// Classify a leaf token: exact-decimal numerics, the named literals, `<constant>`, else an
/// opaque leaf (variable or wildcard). Public within the crate: the engine's numeric-fold glue
/// parses evaluator result tokens back through the same classification.
pub(crate) fn parse_leaf_token(t: Tok, s: &str, view: &TokenView) -> Ex {
    match s {
        "np.pi" => return Ex::Pi,
        "np.e" => return Ex::E,
        "float(\"inf\")" => return Ex::PosInf,
        "float(\"-inf\")" => return Ex::NegInf,
        "float(\"nan\")" => return Ex::NaN,
        "<constant>" => return Ex::Const,
        _ => {}
    }
    if let Some(r) = Rat::parse_decimal(s) {
        return Ex::Num(r);
    }
    // The tagged form's exact-fraction leaves: "1/3", "-7/4" (integer '/' integer, one slash).
    if let Some((p, q)) = s.split_once('/') {
        if let (Ok(p), Ok(q)) = (p.parse::<i128>(), q.parse::<i128>()) {
            if let Some(r) = Rat::new(p, q) {
                return Ex::Num(r);
            }
        }
    }
    // Parenthesized literal forms: "(-1)", "(-0.5)", "(-1/3)" -- the wrapping composes
    // with BOTH numeric grammars (hardening H-012, 2026-08-03: this branch knew only
    // decimals, so "(-1/3)" silently became an OPAQUE leaf -- it even sorted as a
    // SYMBOL while "(-1)" and bare "1/3" were numbers).
    if let Some(inner) = s.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
        if let Some(r) = Rat::parse_decimal(inner) {
            return Ex::Num(r);
        }
        if let Some((p, q)) = inner.split_once('/') {
            if let (Ok(p), Ok(q)) = (p.parse::<i128>(), q.parse::<i128>()) {
                if let Some(r) = Rat::new(p, q) {
                    return Ex::Num(r);
                }
            }
        }
    }
    // H-048 (2026-08-05): a SIGNED numeric spelling never becomes a leaf -- the sign
    // is STRUCTURE. Beyond-i128 spellings ("-1e309") fall through every exact parse
    // above and used to keep the sign inside the opaque leaf token, while the same
    // value built structurally kept a POSITIVE leaf under a subtracted section: two
    // canonical states, one function, conflated exactly at the infix boundary (the
    // infix renderer fuses a section sign into the literal and parse read it back as
    // a signed leaf -- extreme-lane fuzz row 61). The sign splits into an ordinary
    // -1 coefficient (the enclosing constructors normalize it), and the unsigned
    // remainder stays the opaque leaf. Scope: the bare '-' prefix on a
    // grammar-numeric spelling -- reserved spellings ("-inf") are not numeric and
    // stay refused; paren-wrapped signed beyond-i128 user spellings remain verbatim
    // leaves (unmintable by any engine path; they round-trip as themselves).
    if crate::utils::is_numeric_string(s) {
        if let Some(unsigned) = s.strip_prefix('-') {
            return Ex::Mul(vec![Ex::Num(Rat::NEG_ONE), Ex::Leaf(view.intern(unsigned))]);
        }
    }
    Ex::Leaf(t)
}

/// Desugar one LEGACY operator application into the core (input-side compatibility only --
/// the vocabulary itself is deleted): `mult{k} x = k * x`, `div{k} x = (1/k) * x`,
/// `pow{k} x = x^k`, `pow1_2/pow1_4 x = x^(1/2 | 1/4)` (even roots agree with rational
/// powers: NaN on negatives either way), and `pow1_3`/`pow1_5 x = rootn x (3|5)` (the real
/// odd root is a DIFFERENT function from the rational power on `x < 0`, so it desugars to
/// the `rootn` built-in, never to `pow`).
fn desugar(name: &str, op: Tok, mut args: Vec<Ex>, cx: &Cx) -> Ex {
    match (name, args.len()) {
        ("+", 2) => add(args, cx),
        ("*", 2) => mul(args, cx),
        ("-", 2) => {
            let b = args.pop().unwrap();
            let a = args.pop().unwrap();
            let neg_b = mul(vec![Ex::int(-1), b], cx);
            add(vec![a, neg_b], cx)
        }
        ("/", 2) => {
            let b = args.pop().unwrap();
            let a = args.pop().unwrap();
            let inv_b = pow(b, Ex::int(-1), cx);
            mul(vec![a, inv_b], cx)
        }
        ("neg", 1) => mul(vec![Ex::int(-1), args.pop().unwrap()], cx),
        ("inv", 1) => pow(args.pop().unwrap(), Ex::int(-1), cx),
        ("pow", 2) => {
            let e = args.pop().unwrap();
            let b = args.pop().unwrap();
            pow(b, e, cx)
        }
        ("pow1_3", 1) | ("pow1_5", 1) => {
            let k = if name == "pow1_3" { 3 } else { 5 };
            let rootn_tok = cx.view.intern("rootn");
            fun(rootn_tok, vec![args.pop().unwrap(), Ex::int(k)], cx)
        }
        _ => {
            // The legacy hyper-operator family is EXACTLY k in 2..=5: an unbounded
            // prefix match would give `mult7`/`pow9` parser meanings no evaluator has,
            // silently overriding a config's declared realization. A name outside the
            // historical family falls through to the config's own semantics.
            let legacy_k = |rest: &str| -> Option<i128> {
                rest.parse::<i128>().ok().filter(|k| (2..=5).contains(k))
            };
            if let Some(k) = name.strip_prefix("mult").and_then(&legacy_k) {
                if args.len() == 1 {
                    return mul(vec![Ex::int(k), args.pop().unwrap()], cx);
                }
            }
            if let Some(k) = name.strip_prefix("div").and_then(&legacy_k) {
                if args.len() == 1 {
                    if let Some(r) = Rat::new(1, k) {
                        return mul(vec![Ex::Num(r), args.pop().unwrap()], cx);
                    }
                }
            }
            if let Some(k) = name.strip_prefix("pow1_").and_then(&legacy_k) {
                if args.len() == 1 {
                    let rootn_tok = cx.view.intern("rootn");
                    return fun(rootn_tok, vec![args.pop().unwrap(), Ex::int(k)], cx);
                }
            }
            if let Some(k) = name.strip_prefix("pow").and_then(&legacy_k) {
                if args.len() == 1 {
                    return pow(args.pop().unwrap(), Ex::int(k), cx);
                }
            }
            fun(op, args, cx)
        }
    }
}

/// Serialize a canonical AC expression back to prefix tokens in the old token language --
/// the EXPLICIT form (literal coefficients, no hyper-operators).
pub fn to_prefix(e: &Ex, cx: &Cx) -> Vec<Tok> {
    let mut out = Vec::new();
    emit(e, cx, &mut out);
    out
}

/// Split an Add bag's terms into (positive-form terms, negated-form absolute terms): a term
/// whose rational coefficient is negative contributes its |coefficient| form to the second
/// list. The serializer renders `pos - n1 - n2 ...` (or `neg (...)` when no positive part).
fn add_sign_split(v: &[Ex], cx: &Cx) -> (Vec<Ex>, Vec<Ex>) {
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    for t in v {
        match t {
            Ex::Num(r) if r.is_negative() => neg.push(Ex::Num(r.checked_neg().unwrap())),
            Ex::Mul(f) => match f.first() {
                Some(Ex::Num(r)) if r.is_negative() => {
                    let mut rest: Vec<Ex> = f[1..].to_vec();
                    let a = r.checked_neg().unwrap();
                    if !a.is_one() {
                        rest.insert(0, Ex::Num(a));
                    }
                    neg.push(if rest.len() == 1 {
                        rest.pop().unwrap()
                    } else {
                        Ex::Mul(rest)
                    });
                }
                _ => pos.push(t.clone()),
            },
            Ex::NegInf => neg.push(Ex::PosInf),
            _ => pos.push(t.clone()),
        }
    }
    let _ = cx;
    (pos, neg)
}

/// DIVISOR-SIDE COEFFICIENT SPELLING (owner-ratified 2026-08-01;
/// design/UNIFIED_SIMPLICITY_MEASURE.md §8) -- user-facing aesthetics, EMISSION ONLY.
/// Every f64-born literal is a dyadic rational with a finite decimal spelling, but its
/// exact reciprocal generally is not: canon of `x0 / 0.3333333333333333` holds the exact
/// coefficient 10^16/3333333333333333, whose only exact one-token spelling is that
/// monster fraction. The emitters therefore spell a Mul coefficient on whichever side
/// has the SHORTER exact spelling: a coefficient with no exact decimal whose reciprocal
/// spells strictly shorter as one token moves behind the divide as the reciprocal's
/// token (`x0 / 0.3333333333333333`; `(1/3)*x0` prints `x0/3`). Same state, same value.
///
/// Guardrails: the choice NEVER enters the measure or any mint/skip decision
/// (spelling-dependence is the diagnosed Kruskal-skip disease -- every consumer of
/// serialized output re-parses to the identical state); ties and both-fraction
/// coefficients stay on the coefficient side (`22/7` beats `<div> 7/22`, and a fraction
/// reciprocal can never win strictly -- same digits, same length -- so a firing
/// reciprocal is always an integer or an exact decimal, safe as one den token in every
/// dialect); exact decimals never move (`0.5 x0`); `is_integer` covers the `-1`
/// display special-cases before they are consulted.
fn divisor_side(r: &Rat) -> Option<Rat> {
    // Integers never move: their reciprocal is the fraction, not the shorter spelling.
    //
    // The old second clause -- "nor anything with an exact decimal" -- was STALE. It
    // existed because `num_token` used to spell such values as decimals, so the
    // reciprocal could not be shorter. Now that `num_token` follows mu's argmin, `1/2`
    // spells `1/2` and its reciprocal `2` IS shorter, and keeping the clause made the
    // same shape render two ways: `1/2 * x0` beside `x0 / 3`. The length comparison
    // below, on argmin spellings, is the whole decision.
    if r.is_integer() {
        return None;
    }
    // A NEGATIVE coefficient never moves divisor-side: a signed literal inside a
    // den/`<div>` group re-parses the sign INTO the divisor bag, where the H-020
    // sign-fold clause may absorb it into a Const-bearing sum -- a different (family-
    // equal) state than the outer-coefficient parking the chain holds (caught by the
    // state-stability battery: `-1/3 * .. * inv(x17 + C)` spelled `-3` in the den,
    // re-parsed folded). The sign stays on the numerator side (the p/q fallback or the
    // caller's own sign pull), where re-parse restores the outer coefficient exactly.
    // The infix emitter negates before consulting and is unaffected.
    if r.is_negative() {
        return None;
    }
    let inv = r.checked_inv()?;
    let inv_token = num_token(&inv);
    // The den member is joined UNPARENTHESIZED, one slash per member, so a reciprocal that
    // itself spells as a fraction CHANGES THE VALUE: 3/400 reciprocates to 400/3 and renders
    // `x0/400/3`, which re-reads as x0/1200 -- a 9x error. The doc invariant above ("a firing
    // reciprocal is always an integer or an exact decimal, safe as one den token") is exactly
    // what makes the join sound, so it has to be TESTED rather than assumed. It used to hold
    // for free because `num_token` spelled every exact-decimal value as a decimal; under the
    // argmin spelling it does not.
    if inv_token.contains('/') {
        return None;
    }
    if inv_token.len() < num_token(r).len() {
        Some(inv)
    } else {
        None
    }
}

/// Divisor-side needs a PLAIN numerator factor to remain: a factor that is neither the
/// rational coefficient nor a negative-rational-exponent power (those move behind the
/// divide themselves). A bag of only coefficient + inverse factors keeps its coefficient
/// in the numerator (`(1/3)/x0` stays `<mul> 1/3 <div> x0 </mul>`), so no emitter ever
/// produces an empty numerator it did not produce before.
fn has_plain_mul_factor(v: &[Ex]) -> bool {
    v.iter().any(|f| match f {
        Ex::Num(_) => false,
        Ex::Pow(b, ex) => !matches!(&**ex, Ex::Num(r) if r.is_negative()
                && !matches!(&**b, Ex::Num(z) if z.is_zero())),
        _ => true,
    })
}

/// Split a Mul bag's factors into (numerator factors, denominator factors): a factor
/// `Pow(b, e)` with a negative rational exponent contributes `Pow(b, -e)` (or `b` when the
/// negated exponent is 1) to the denominator. A non-integer rational coefficient contributes
/// its numerator and denominator to the respective sides -- unless the divisor-side rule
/// (`divisor_side`) spells the whole coefficient as ONE reciprocal token in the denominator.
/// F73: does this bag carry an i128-overflow PARTITION -- its exact rational content
/// split across two or more `Num` members because folding them would overflow? The
/// atoms of a partition are load-bearing: any gathering (p/q splits into shared
/// num/den chains, divisor-side respells, section pooling) destroys the boundaries
/// the arithmetic was forced to keep, and the re-parse pools the pieces and re-CUTS
/// them -- one value, a different partition per dialect (the extreme lane's dominant
/// hard class; `* a/3 a/3` with a = 2^127-1 rendered `/ (a*a) (3*3)` and re-parsed to
/// `{a, a, 1/9}`). Partition members therefore render as SELF-CONTAINED spellings
/// (one atom token, or a locally re-foldable division), never pooled -- the sorted
/// accumulation is cut-stable on preserved atoms, so the round-trip is the identity.
/// Bags with at most one rational member are unaffected: a single fraction re-forms
/// uniquely from any of its spellings.
fn is_partition_bag(v: &[Ex]) -> bool {
    v.iter().filter(|f| matches!(f, Ex::Num(_))).count() >= 2
}

fn mul_div_split(v: &[Ex], cx: &Cx) -> (Vec<Ex>, Vec<Ex>) {
    let mut num = Vec::new();
    // (member, original factor): the split into a GROUPED denominator is decided after
    // collection -- see below.
    let mut cand: Vec<(Ex, &Ex)> = Vec::new();
    let mut den = Vec::new();
    let partition = is_partition_bag(v);
    for f in v {
        match f {
            // F73: partition atoms stay whole members -- `emit_num` prints each as one
            // token or a LOCAL `/ p q`, either of which re-folds to exactly this atom
            // before the bag pools.
            Ex::Num(r) if partition => num.push(Ex::Num(*r)),
            Ex::Num(r) => {
                // H-020 amendment: the SIGN never enters the den group (a signed den
                // literal re-parses the sign into the divisor bag, where the sign-fold
                // clause may absorb it into a Const-bearing sum -- a different parking
                // than the chain state's outer coefficient). The divisor-side rule is
                // consulted on the magnitude; a negative coefficient contributes the
                // pure sign to the numerator (`-1`, spelled `neg`) beside the positive
                // reciprocal den token.
                let mag = if r.is_negative() {
                    r.checked_neg()
                } else {
                    Some(*r)
                };
                // Split only when the FRACTION is the argmin. A value whose DECIMAL code
                // wins spells as one atomic token, and splitting it produces the fraction
                // mu prices higher: `0.12345 * x0` became `2469*x0/20000`. Removing this
                // test outright (rather than re-basing it on the argmin) also broke the
                // setup of the long-literal respell guard, so its assertion stopped
                // executing -- a silently unreached safety check.
                if r.is_integer() || crate::ac::expr::decimal_spelling_wins(r) {
                    num.push(Ex::Num(*r));
                } else if let Some(inv) = mag
                    .and_then(|m| divisor_side(&m))
                    .filter(|_| has_plain_mul_factor(v))
                {
                    // One reciprocal token beats the p-and-q integer split; re-parses to
                    // the exact original coefficient (division by a literal reciprocates
                    // exactly). Contributes one den entry, same as the split it replaces.
                    if r.is_negative() {
                        num.push(Ex::Num(Rat::NEG_ONE));
                    }
                    den.push(Ex::Num(inv));
                } else {
                    // p/q with no exact decimal: p joins the numerator (skipped when it is the
                    // multiplicative identity), q the denominator.
                    if r.num() != 1 {
                        num.push(Ex::Num(Rat::int(r.num())));
                    }
                    den.push(Ex::Num(Rat::int(r.den())));
                }
            }
            Ex::Pow(b, e) => match &**e {
                Ex::Num(r) if r.is_negative() && !matches!(&**b, Ex::Num(z) if z.is_zero()) => {
                    let flipped = r.checked_neg().unwrap();
                    let member = if flipped.is_one() {
                        (**b).clone()
                    } else {
                        Ex::Pow(b.clone(), Box::new(Ex::Num(flipped)))
                    };
                    cand.push((member, f));
                }
                // A LITERAL-ZERO base never joins a denominator chain: `1/(0 * x)` re-parses
                // with the zero collapsing the product, so that spelling is unstable; the
                // `inv 0` pole stays a numerator factor instead.
                _ => num.push(f.clone()),
            },
            _ => num.push(f.clone()),
        }
    }
    // A MULTI-member grouped denominator re-parses as `pow(product, -1)`, which only
    // re-distributes into the separate inverse factors under the zero-set licence
    // (`ac::expr::pow`) -- grouping an unlicensed member would re-parse into the JOINED
    // form, a structurally different (and at fat zeros semantically different) tree.
    // Unlicensed members therefore stay numerator factors and emit as explicit `inv`,
    // which re-parses per-factor. A SINGLE-member denominator has no bag to mis-split
    // and is always faithful, licence-free.
    if cand.len() + den.len() == 1 {
        den.extend(cand.into_iter().map(|(m, _)| m));
    } else {
        for (m, f) in cand {
            // A bare `Const` member groups only ALONE (handled above): in a multi-member
            // group the re-parsed PRODUCT runs the constructor's Const-coefficient
            // absorption (`3 * C -> C`) before the inversion, collapsing members the
            // internal form keeps distinct. No other member class interacts under
            // `mul` (a canonical flat Mul cannot yield two like-based members).
            if cx.nz_ae_licensed(&m) && !matches!(m, Ex::Const) {
                den.push(m);
            } else {
                num.push(f.clone());
            }
        }
    }
    (num, den)
}

/// Emit a right-leaning binary chain `op a (op b (op c d))` over `parts` (len >= 1).
fn emit_chain(op: Tok, parts: &[Ex], cx: &Cx, out: &mut Vec<Tok>) {
    for p in &parts[..parts.len() - 1] {
        out.push(op);
        emit(p, cx, out);
    }
    emit(&parts[parts.len() - 1], cx, out);
}

fn emit(e: &Ex, cx: &Cx, out: &mut Vec<Tok>) {
    let view = cx.view;
    match e {
        Ex::Num(r) => emit_num(r, cx, out),
        Ex::Pi => out.push(view.intern("np.pi")),
        Ex::E => out.push(view.intern("np.e")),
        Ex::PosInf => out.push(view.intern("float(\"inf\")")),
        Ex::NegInf => out.push(view.intern("float(\"-inf\")")),
        Ex::NaN => out.push(view.intern("float(\"nan\")")),
        Ex::Const => out.push(view.intern("<constant>")),
        Ex::Leaf(t) => out.push(*t),
        Ex::Fun(f, args) => {
            // `rootn` emits NATIVELY (the pow1_k spellings are
            // deleted from the vocabulary). The interval certificates and every evaluator
            // treat `rootn` as an engine built-in, so the serialization stays readable
            // under any pairing without legacy resugar.
            out.push(*f);
            for a in args {
                emit(a, cx, out);
            }
        }
        Ex::Pow(b, ex) => match &**ex {
            // pow(b, -1) -> inv b; pow(b, -n) -> inv pow b n (the shorter spellings).
            // Bases and expression exponents use the STRUCTURAL emission: a
            // licence-refused `Pow(Mul[-1, A], n)` survives canon (see `ac::expr::pow`),
            // and the flip display of its base does not round-trip.
            Ex::Num(r) if r.is_negative() => {
                out.push(view.intern("inv"));
                let flipped = r.checked_neg().unwrap();
                if flipped.is_one() {
                    emit_bin_structural(b, cx, out);
                } else {
                    out.push(view.intern("pow"));
                    emit_bin_structural(b, cx, out);
                    emit_num(&flipped, cx, out);
                }
            }
            Ex::Num(r) => {
                out.push(view.intern("pow"));
                emit_bin_structural(b, cx, out);
                emit_num(r, cx, out);
            }
            _ => {
                out.push(view.intern("pow"));
                emit_bin_structural(b, cx, out);
                emit_bin_structural(ex, cx, out);
            }
        },
        Ex::Add(v) => {
            let (pos, neg) = add_sign_split(v, cx);
            if neg.is_empty() {
                emit_chain(view.intern("+"), &pos, cx, out);
            } else if pos.is_empty() {
                // -(a + b + ...): neg then the positive chain.
                out.push(view.intern("neg"));
                emit_chain(view.intern("+"), &neg, cx, out);
            } else {
                // (pos-chain) - n1 - n2 ...: left-nested `-`, so each subtraction wraps the
                // previous: `- - P n1 n2` reads ((P - n1) - n2).
                emit_sub_chain(&pos, &neg, cx, out);
            }
        }
        Ex::Mul(v) => {
            // DISPLAY sign redistribution: the canonical `Mul[-1, Add]` (a primitive sum's
            // extracted sign) renders as the sign-flipped sum -- `-(x0 - x1)` prints as
            // `- x1 x0`, not `neg - x0 x1`. Parse + canon re-extract, so this is a pure
            // projection choice; the internal form stays unique.
            if v.len() == 2 {
                if let (Ex::Num(r), Ex::Add(terms)) = (&v[0], &v[1]) {
                    if *r == Rat::NEG_ONE {
                        if let Some(flipped) = flip_terms(terms, cx) {
                            emit(&Ex::Add(flipped), cx, out);
                            return;
                        }
                    }
                }
            }
            let (num, den) = mul_div_split(v, cx);
            if den.is_empty() {
                emit_mul_chain(&num, cx, out);
            } else if num.is_empty() {
                out.push(view.intern("inv"));
                emit_mul_chain(&den, cx, out);
            } else {
                out.push(view.intern("/"));
                emit_mul_chain(&num, cx, out);
                emit_mul_chain(&den, cx, out);
            }
        }
    }
}

/// `((pos) - n1) - n2 - ...` in prefix: `- (- (pos) n1) n2` -- built outermost-first, which in
/// prefix means the LAST subtrahend's `-` comes first.
fn emit_sub_chain(pos: &[Ex], neg: &[Ex], cx: &Cx, out: &mut Vec<Tok>) {
    let minus = cx.view.intern("-");
    // Emit `-` for the outermost subtraction (the last subtrahend), recurse on the rest.
    out.push(minus);
    if neg.len() == 1 {
        emit_chain(cx.view.intern("+"), pos, cx, out);
    } else {
        emit_sub_chain(pos, &neg[..neg.len() - 1], cx, out);
    }
    emit(&neg[neg.len() - 1], cx, out);
}

/// A `*`-chain over factors. Sign doctrine (shared with the tagged form): a sign lives in
/// the numeric coefficient literal whenever one is emitted (`-5 * x` -> `* -5 x`,
/// `-0.5 * x` -> `* -0.5 x`); `neg` wraps only the PURE sign, where no literal exists to
/// carry it (`-1 * x` -> `neg x`, `-np.pi * x` -> `neg * np.pi x`).
/// Elements route through the STRUCTURAL emission: a denominator member can be a
/// licence-refused `Mul[-1, Add]` base (see `mul_div_split`), whose flip display does
/// not round-trip.
fn emit_mul_chain(parts: &[Ex], cx: &Cx, out: &mut Vec<Tok>) {
    let view = cx.view;
    match parts {
        [] => out.push(view.intern("1")),
        [one] => emit_bin_structural(one, cx, out),
        _ => {
            // The pure sign has no literal to live in: `neg` wraps the remaining chain.
            // Any other negative coefficient flows into the plain chain below, where
            // `emit_num` prints it as one signed token.
            if matches!(&parts[0], Ex::Num(r) if *r == Rat::NEG_ONE) {
                out.push(view.intern("neg"));
                emit_mul_chain(&parts[1..], cx, out);
                return;
            }
            for p in &parts[..parts.len() - 1] {
                out.push(view.intern("*"));
                emit_bin_structural(p, cx, out);
            }
            emit_bin_structural(&parts[parts.len() - 1], cx, out);
        }
    }
}

/// Structure-faithful emission for Pow bases/exponents and denominator members (the
/// binary twin of the tagged `emit_structural`): the `Mul[-1, Add]` flip display is
/// value-exact but not INJECTIVE once masked-constant sign absorption re-normalizes the
/// re-parsed terms -- a licence-refused `Pow(Mul[-1, A], n)` round-tripped into a
/// structurally different tree. The explicit `neg` + faithful sum re-parses identically.
fn emit_bin_structural(e: &Ex, cx: &Cx, out: &mut Vec<Tok>) {
    if let Ex::Mul(v) = e {
        if v.len() == 2
            && matches!(&v[0], Ex::Num(r) if *r == Rat::NEG_ONE)
            && matches!(&v[1], Ex::Add(_))
        {
            out.push(cx.view.intern("neg"));
            emit(&v[1], cx, out);
            return;
        }
    }
    emit(e, cx, out);
}

/// Emit a rational literal in the ARGMIN spelling of `mu_rat`'s two codes: integers bare,
/// a decimal token when the decimal code is cheaper (`1/5 -> 0.2`, `6/5 -> 1.2`, every
/// power of ten), else the structural division `/ p q` (`1/2`, `5/8`, `11/2`, and every
/// non-terminating value). The old rule -- decimal whenever one exists -- printed `0.5`
/// and `0.625`, the spelling mu prices HIGHER.
fn emit_num(r: &Rat, cx: &Cx, out: &mut Vec<Tok>) {
    let view = cx.view;
    if r.is_integer() {
        out.push(view.intern(&r.num().to_string()));
        return;
    }
    if crate::ac::expr::decimal_spelling_wins(r) {
        if let Some(s) = r.exact_decimal() {
            out.push(view.intern(&s));
            return;
        }
    }
    // The structural division needs `/` to BE an operator of this config. A degenerate
    // vocabulary without it cannot render or re-parse the emission: the infix layer gives
    // an unknown binary token a fallback precedence, so `pow(1/2, x)` printed `1/2^(x)`,
    // which re-reads as `1/(2^x)` -- a closure breach on a different function. (Latent
    // before: only non-decimal rationals like `1/3` reached this path, so the shapes that
    // exposed it were rare.) Without `/`, fall back to the one-token argmin spelling, which
    // is a terminal and always round-trips.
    let slash = view.intern("/");
    if view.arity(slash).is_some() {
        out.push(slash);
        out.push(view.intern(&r.num().to_string()));
        out.push(view.intern(&r.den().to_string()));
    } else {
        out.push(view.intern(&num_token(r)));
    }
}

/// The STRICT TAGGED prefix form -- the AC engine's native serialization. n-ary bags are
/// delimited (`<add> ... </add>`, `<mul> ... </mul>`) and carry their group inverse as a
/// SECTION: terms after `<sub>` are subtracted, factors after `<div>` divide --
/// `(2*x1)/(x2*x3)` is `<mul> 2 x1 <div> x2 x3 </mul>`. `pow` and the unary functions stay
/// plain prefix (fixed arity needs no delimiters). `neg`/`inv` exist ONLY as the standalone
/// unary spellings (function arguments, lone terms: `tan neg x0`, `inv x0`); inside bags the
/// sections own all inverses, so bags contain no negative literals and no inverse operators.
/// Rational literals spell as one token: integers bare (`7`), exact decimals as decimals
/// (`0.2`), everything else as a fraction (`1/3`).
///
/// Faithful: `from_prefix` parses this form back to the same canonical expression (both
/// languages share the parser, which is LIBERAL -- it also accepts `neg`/`inv`/negative
/// literals inside bags and maps every spelling to the same canonical state). The emission is
/// a pure function of the canonical expression and is an OUTPUT PROJECTION of it: the
/// internal canonical form is UNIQUE (stripped bag orders + primitive sums, see `ac::expr`),
/// and every serialization -- this one included -- is a bijective-on-classes projection of it,
/// verified per chain state by a debug assertion in the simplify loop.
pub fn to_prefix_tagged(e: &Ex, cx: &Cx) -> Vec<Tok> {
    let mut out = Vec::new();
    emit_tagged(e, cx, &mut out);
    out
}

/// Tagged emission of a rational: integers bare, every NON-integer as the structural
/// fraction `<mul> p <div> q </mul>`.
///
/// The tagged dialect is the MODEL-FACING form (the infix form is what users read), and a
/// tokenized consumer's numeric vocabulary is a finite set of integers -- an atomic `2/3`
/// or `0.5` token is not in it, and a per-fraction token class would either explode the
/// vocabulary or admit sequences that denote nothing. Spelling the fraction with the bag
/// markers the dialect already has costs no new token, and every arrangement of integers
/// across the bag and its `<div>` section re-parses to the same canonical value, so there
/// is no malformed spelling for a model to emit.
///
/// This is a SERIALIZATION projection only: the internal value stays `Ex::Num(Rat)`, so
/// `mu`, `cmp_ex` and the reduction ordering are untouched.
fn emit_rational_tagged(r: &Rat, cx: &Cx, out: &mut Vec<Tok>) {
    let view = cx.view;
    if !fraction_spells_structurally(r) {
        // Integer, or a fraction outside the consumer's vocabulary: one argmin token.
        out.push(view.intern(&num_token(r)));
        return;
    }
    // `Rat` normalizes the sign onto the numerator (`den > 0`), so the sign rides `p` and
    // the `<div>` member stays positive -- the H-020 invariant that a signed literal never
    // enters a divisor group.
    out.push(view.intern("<mul>"));
    out.push(view.intern(&r.num().to_string()));
    out.push(view.intern("<div>"));
    out.push(view.intern(&r.den().to_string()));
    out.push(view.intern("</mul>"));
}

fn emit_tagged(e: &Ex, cx: &Cx, out: &mut Vec<Tok>) {
    let view = cx.view;
    match e {
        Ex::Num(r) => emit_rational_tagged(r, cx, out),
        Ex::Pi => out.push(view.intern("np.pi")),
        Ex::E => out.push(view.intern("np.e")),
        Ex::PosInf => out.push(view.intern("float(\"inf\")")),
        Ex::NegInf => out.push(view.intern("float(\"-inf\")")),
        Ex::NaN => out.push(view.intern("float(\"nan\")")),
        Ex::Const => out.push(view.intern("<constant>")),
        Ex::Leaf(t) => out.push(*t),
        Ex::Fun(f, args) => {
            out.push(*f);
            for a in args {
                emit_tagged(a, cx, out);
            }
        }
        Ex::Pow(b, ex) => {
            // The standalone reciprocal: `inv x` (2 tokens) beats `pow x -1` (3). Deeper
            // negative exponents keep the literal (`pow x -3` beats `inv pow x 3`).
            // Base and exponent use the STRUCTURAL emission: a licence-refused
            // `Pow(Mul[-1, A], n)` survives canon (see `ac::expr::pow`), and the flip
            // display of its base does not round-trip (see `emit_structural`).
            if matches!(&**ex, Ex::Num(r) if *r == Rat::NEG_ONE) {
                out.push(view.intern("inv"));
                emit_structural(b, cx, out);
                return;
            }
            out.push(view.intern("pow"));
            emit_structural(b, cx, out);
            emit_structural(ex, cx, out);
        }
        Ex::Add(v) => {
            // Sign section: positive-coefficient terms, then `<sub>` + the magnitudes of the
            // negative-coefficient terms. Bags therefore never contain negative literals.
            let mut pos: Vec<Ex> = Vec::new();
            let mut neg: Vec<Ex> = Vec::new();
            for t in v {
                let (is_neg, mag) = split_sign(t);
                if is_neg {
                    neg.push(mag);
                } else {
                    pos.push(mag);
                }
            }
            out.push(view.intern("<add>"));
            for p in &pos {
                emit_tagged(p, cx, out);
            }
            if !neg.is_empty() {
                out.push(view.intern("<sub>"));
                for n in &neg {
                    emit_tagged(n, cx, out);
                }
            }
            out.push(view.intern("</add>"));
        }
        Ex::Mul(v) => {
            // DISPLAY sign redistribution for the primitive form: `Mul[-1, Add]` renders as
            // the sign-flipped sum (its terms land in the `<sub>` section); the standalone
            // negation of anything else collapses to `neg factor` (`tan neg x0`). With more
            // factors the sign rides the coefficient literal for free.
            if v.len() == 2 {
                if let Ex::Num(r) = &v[0] {
                    if *r == Rat::NEG_ONE {
                        if let Ex::Add(terms) = &v[1] {
                            if let Some(flipped) = flip_terms(terms, cx) {
                                emit_tagged(&Ex::Add(flipped), cx, out);
                                return;
                            }
                        }
                        out.push(view.intern("neg"));
                        emit_tagged(&v[1], cx, out);
                        return;
                    }
                }
            }
            // Inverse section: factors with a negative RATIONAL exponent move behind `<div>`
            // with the exponent flipped (`x^-1` -> bare `x`, `x^-2` -> `pow x 2`). The
            // coefficient literal stays in the numerator with its own sign and its own
            // denominator (`1/3` is one token; `<div> 3` would cost more) -- EXCEPT when
            // the divisor-side rule (`divisor_side`) spells its reciprocal strictly
            // shorter: then the one reciprocal token leads the `<div>` members
            // (`<mul> x0 <div> 0.3333333333333333 </mul>`, `<mul> x0 <div> 3 </mul>`).
            let mut den: Vec<Ex> = Vec::new();
            let mut num: Vec<&Ex> = Vec::new();
            let mut coeff_num: Option<Ex> = None; // the fraction's numerator, when split
            let mut neg_one = false; // H-020: the sign never enters `<div>` -- split it out
            let mut unit_coeff_split = false; // F80: a p=1 coefficient split happened
            let partition = is_partition_bag(v);
            for f in v {
                match f {
                    // F73: partition atoms render verbatim (`emit_rational_tagged`: one
                    // token beyond the vocabulary bound, the nested structural fraction
                    // inside it -- both re-fold locally to the atom before the bag
                    // pools). The split and divisor-side arms below would pool their
                    // pieces into the shared `<div>` section, where member-kind flips
                    // across the round-trip reordered the section (the 30 residual
                    // P1 rows of the post-F72 census).
                    Ex::Num(_) if partition => num.push(f),
                    Ex::Num(r) => {
                        let mag = if r.is_negative() {
                            r.checked_neg()
                        } else {
                            Some(*r)
                        };
                        // A non-integer coefficient p/q SPLITS: the numerator stays in the
                        // bag, the denominator joins `<div>`. This replaces the old
                        // token-length heuristic (`divisor_side`, which moved only the
                        // reciprocals that spelled shorter), so the tagged form now carries
                        // no atomic fraction or decimal token at all -- see
                        // `emit_rational_tagged`. A magnitude-1 numerator is the bare sign
                        // and is omitted; the sign itself is split out below.
                        // F80 (owner-ruled spelling law, 2026-08-10): the split fires
                        // UNCONDITIONALLY for in-bound fractions -- previously it demanded
                        // a plain factor (`has_plain_mul_factor`), and the div-only case
                        // fell through to the leaf renderer, whose structural spelling is
                        // a mul bag: the same-type nesting the census measured at 8,918
                        // events/1M. The grammar's >=1-numerator-member floor is kept by
                        // the post-loop `1` (below); partition bags stay verbatim (F73).
                        match mag.filter(|m| fraction_spells_structurally(m)) {
                            Some(m) => {
                                if let Some(d) = Rat::new(m.den(), 1) {
                                    den.push(Ex::Num(d));
                                }
                                if m.num() == 1 {
                                    // Bare sign: no numerator token to carry it.
                                    neg_one = r.is_negative();
                                    unit_coeff_split = true;
                                } else {
                                    // The sign rides the numerator (`-2` beats `-1 2`).
                                    let signed = if r.is_negative() { -m.num() } else { m.num() };
                                    coeff_num = Rat::new(signed, 1).map(Ex::Num);
                                }
                            }
                            None => {
                                // H-059: §8 DIVISOR-SIDE, restored for the OUT-OF-BOUND
                                // case. The split above is what replaced `divisor_side`
                                // in the tagged form, and its whole point is that the
                                // dialect "carries no atomic fraction or decimal token
                                // at all". Outside the vocabulary bound that split is
                                // unavailable and `emit_rational_tagged` emits an ATOMIC
                                // token anyway -- so the invariant is ALREADY lost, and
                                // discarding §8 as well left both rules unsatisfied at
                                // once (`x0 / 0.3333333333333333` printing the 34-char
                                // `10000000000000000/3333333333333333`, which is
                                // verbatim the output §8 was ratified to fix). When an
                                // atom must be emitted regardless, the ratified rule
                                // applies again: spell the coefficient on whichever side
                                // has the shorter exact token. In-bound behaviour is
                                // untouched. H-020 holds -- the sign never enters
                                // `<div>`; it splits out as a `-1` bag literal.
                                match mag
                                    .and_then(|m| divisor_side(&m))
                                    .filter(|_| has_plain_mul_factor(v))
                                {
                                    Some(inv) => {
                                        neg_one = r.is_negative();
                                        den.push(Ex::Num(inv));
                                    }
                                    None => num.push(f),
                                }
                            }
                        }
                    }
                    Ex::Pow(b, ex) => match &**ex {
                        Ex::Num(r)
                            if r.is_negative() && !matches!(&**b, Ex::Num(z) if z.is_zero()) =>
                        {
                            let flipped = r.checked_neg().unwrap();
                            if flipped.is_one() {
                                den.push((**b).clone());
                            } else {
                                den.push(Ex::Pow(b.clone(), Box::new(Ex::Num(flipped))));
                            }
                        }
                        _ => num.push(f),
                    },
                    _ => num.push(f),
                }
            }
            // F80: when the COEFFICIENT SPLIT emptied the numerator side (unit
            // fraction, no plain factor, no sign to carry -- `(1/3)/x0`), the `1`
            // is the numerator member the owner-ratified table pins:
            // `<mul> 1 <div> 3 x0 </mul>`. With a sign, the split-out `-1` below
            // already fills the slot (`(-1/3)/x0` -> `<mul> -1 <div> 3 x0 </mul>`).
            // Coefficient-FREE den-only bags keep their established empty-numerator
            // spelling (`inv(acos(x0)*atan(x1))` -> `<mul> <div> acos x0 atan x1
            // </mul>`, pinned by TestLossyReciprocalRejoinProjection) -- that
            // dialect predates F80 and the ruling did not touch it.
            if unit_coeff_split && !neg_one && coeff_num.is_none() && num.is_empty() {
                coeff_num = Some(Ex::Num(Rat::ONE));
            }
            out.push(view.intern("<mul>"));
            if neg_one {
                // The split-out pure sign: a `-1` bag literal (parse folds it into the
                // coefficient exactly; the den token stays positive).
                emit_tagged(&Ex::Num(Rat::NEG_ONE), cx, out);
            }
            if let Some(n) = &coeff_num {
                emit_tagged(n, cx, out);
            }
            for f in num {
                emit_tagged(f, cx, out);
            }
            if !den.is_empty() {
                out.push(view.intern("<div>"));
                for f in &den {
                    // Den members are POW BASES (the exponent was flipped away), so they
                    // re-parse straight into a Pow -- structural emission for the same
                    // reason as the Pow arm.
                    emit_structural(f, cx, out);
                }
            }
            out.push(view.intern("</mul>"));
        }
    }
}

/// Structure-faithful emission for RE-PARSED positions (Pow bases/exponents and `<div>`
/// members): the `Mul[-1, Add]` flip display (see the Mul arm) is value-exact but not
/// INJECTIVE once masked-constant sign absorption re-normalizes the re-parsed terms --
/// a licence-refused `Pow(Mul[-1, A], n)` (the odd-negative distribution refusal in
/// `ac::expr::pow`) round-tripped into a structurally DIFFERENT tree, breaking
/// serialization stability (observed: idempotence loss on a Const-bearing refused
/// denominator, 64k corpus). The explicit `neg <add ..>` spelling re-parses to the
/// identical `Mul[-1, Add]`; every other shape already round-trips through
/// `emit_tagged`.
fn emit_structural(e: &Ex, cx: &Cx, out: &mut Vec<Tok>) {
    if let Ex::Mul(v) = e {
        if v.len() == 2
            && matches!(&v[0], Ex::Num(r) if *r == Rat::NEG_ONE)
            && matches!(&v[1], Ex::Add(_))
        {
            out.push(cx.view.intern("neg"));
            emit_tagged(&v[1], cx, out);
            return;
        }
    }
    emit_tagged(e, cx, out);
}

/// One-token spelling of an exact rational: integer, exact decimal, or `p/q` fraction.
///
/// The choice is the ARGMIN of `mu_rat`'s two codes, so the print follows the cost rather
/// than a separate heuristic: `1/2`, `1/4`, `5/8` keep the fraction (a power-of-two
/// denominator always spells shorter as a fraction), while `1/5 -> 0.2`, `6/5 -> 1.2` and
/// every power of ten take the decimal. The previous rule -- "exact decimal whenever one
/// exists" -- printed `0.5` and `0.625`, i.e. the spelling mu prices HIGHER.
fn num_token(r: &Rat) -> String {
    if r.den() == 1 {
        return r.num().to_string();
    }
    if crate::ac::expr::decimal_spelling_wins(r) {
        if let Some(s) = r.exact_decimal() {
            return s;
        }
    }
    format!("{}/{}", r.num(), r.den())
}

/// Largest |numerator| and denominator the TAGGED form spells structurally.
///
/// The tagged dialect is model-facing, and a tokenized consumer's numeric vocabulary is a
/// finite integer range; a fraction whose components fall inside it is spellable as
/// `<mul> p <div> q </mul>` with no new token, while one that does not is unspellable
/// either way and is better emitted as the compact argmin token than as a long structure
/// the consumer must discard anyway. Overridable so a consumer with a wider vocabulary can
/// widen the structural range to match it.
fn tagged_fraction_bound() -> i128 {
    static BOUND: OnceLock<i128> = OnceLock::new();
    *BOUND.get_or_init(|| {
        std::env::var("SIMPLIPY_TAGGED_FRACTION_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10)
    })
}

/// Does `r` spell structurally in the tagged form (both components inside the bound)?
fn fraction_spells_structurally(r: &Rat) -> bool {
    let bound = tagged_fraction_bound();
    !r.is_integer() && r.num().unsigned_abs() <= bound.unsigned_abs() && r.den() <= bound
}

/// The PRETTY INFIX form -- a human-readable rendering of the canonical expression:
/// `x8 + 1.2*x3`, `-x0/3`, `(x0 + 1)^2`, `sin(x0)`, `1/x0`, `pi`, `inf`. Negative-
/// coefficient addends render with `-`; negative-exponent factors move into a
/// denominator; the fitted constant keeps its `<constant>` spelling. ROUND-TRIPS through
/// the infix parser (`convert::infix_to_prefix` + `convert_expression`): the core symbols
/// carry built-in precedences under any config, and the bare constant names `pi`/`e`/
/// `inf`/`nan` are reserved spellings the parser reads back as the constants.
pub fn to_infix_pretty(e: &Ex, cx: &Cx) -> String {
    render(e, cx, 0)
}

/// Precedence levels: 1 = additive, 2 = multiplicative, 3 = power, 4 = atom.
/// `ctx_prec` is the surrounding level; a lower-precedence rendering gets parenthesized.
fn render(e: &Ex, cx: &Cx, ctx_prec: u8) -> String {
    let (s, prec) = render_prec(e, cx);
    if prec < ctx_prec {
        format!("({s})")
    } else {
        s
    }
}

fn render_prec(e: &Ex, cx: &Cx) -> (String, u8) {
    let view = cx.view;
    match e {
        Ex::Num(r) => {
            let s = num_token(r);
            // A fraction or negative literal is not an atom (1/3, -2): parenthesize in
            // tighter contexts via precedence 2 / 1.
            //
            // Precedence follows the SPELLING, not the availability of a decimal. The old
            // `exact_decimal().is_some() -> 4` arm assumed such a value always PRINTS as a
            // decimal atom; `num_token` now picks mu's argmin, so `1/2` prints as `1/2`,
            // and calling that an atom dropped the parentheses under `pow`: `1/2^(x)`
            // re-reads as `1/(2^x)`, a different function (caught by the degenerate-config
            // infix closure gate).
            let prec = if r.is_negative() {
                1
            } else if s.contains('/') {
                2
            } else {
                4
            };
            (s, prec)
        }
        Ex::Pi => ("pi".to_string(), 4),
        Ex::E => ("e".to_string(), 4),
        Ex::PosInf => ("inf".to_string(), 4),
        Ex::NegInf => ("-inf".to_string(), 1),
        Ex::NaN => ("nan".to_string(), 4),
        Ex::Const => ("<constant>".to_string(), 4),
        Ex::Leaf(t) => (view.resolve_owned(*t), 4),
        Ex::Fun(f, args) => {
            let inner: Vec<String> = args.iter().map(|a| render(a, cx, 0)).collect();
            (
                format!("{}({})", view.resolve_owned(*f), inner.join(", ")),
                4,
            )
        }
        Ex::Pow(b, ex) => {
            if let Ex::Num(r) = &**ex {
                if r.is_negative() {
                    // Standalone negative power: 1/b^|e|.
                    let flipped = r.checked_neg().unwrap();
                    let denom = if flipped.is_one() {
                        render(b, cx, 3)
                    } else {
                        format!("{}^{}", render(b, cx, 4), render_exponent(&flipped))
                    };
                    return (format!("1/{denom}"), 2);
                }
            }
            let base = render(b, cx, 4);
            let expo = match &**ex {
                Ex::Num(r) => render_exponent(r),
                other => format!("({})", render(other, cx, 0)),
            };
            (format!("{base}^{expo}"), 3)
        }
        Ex::Add(v) => {
            // Display rules (owner ruling 2026-08-08), DISPLAY-ONLY -- the bag order is
            // the canonical one and stays untouched (tagged/explicit emit it verbatim),
            // and the infix parse is permutation-invariant, so round-trip is unaffected:
            //  1. POSITIVE terms print first, then negative ones, relative order
            //     preserved within each group -- `2 - x0` must never print as `-x0 + 2`;
            //  2. an ODD function's literal-argument sign hoists to the joiner -- a sum
            //     never prints `+ sin(-2)` or `- sin(-2)`, always `+ sin(2)` / `- sin(2)`
            //     (a standalone `sin(-2)` outside a sum keeps its canonical spelling).
            let mut parts: Vec<(bool, Ex)> = v
                .iter()
                .map(|t| {
                    let (neg, mag) = split_sign(t);
                    odd_literal_hoist(neg, mag, cx)
                })
                .collect();
            parts.sort_by_key(|(neg, _)| *neg); // stable: positives first, order kept
            let mut s = String::new();
            for (i, (neg, mag)) in parts.iter().enumerate() {
                if i == 0 {
                    if *neg {
                        s.push('-');
                    }
                } else {
                    s.push_str(if *neg { " - " } else { " + " });
                }
                s.push_str(&render(mag, cx, 2));
            }
            (s, 1)
        }
        Ex::Mul(v) => {
            // Display sign redistribution (see the prefix emitters): -(a - b) prints b - a.
            if v.len() == 2 {
                if let (Ex::Num(r), Ex::Add(terms)) = (&v[0], &v[1]) {
                    if *r == Rat::NEG_ONE {
                        if let Some(flipped) = flip_terms(terms, cx) {
                            return render_prec(&Ex::Add(flipped), cx);
                        }
                    }
                }
            }
            // Coefficient sign out front; negative-exponent factors into the denominator.
            let mut neg = false;
            let mut num_parts: Vec<String> = Vec::new();
            let mut den_parts: Vec<String> = Vec::new();
            let partition = is_partition_bag(v);
            for f in v {
                match f {
                    // F73: partition atoms render as self-contained factors, PARENTHESIZED
                    // when fractional -- `(a/3)*(a/3)` re-parses each atom locally, while
                    // the bare chain `a/3*a/3` re-associates textually into a flat pool
                    // that re-cuts. The display sign still hoists to the front: sign
                    // hosting is value-determined since F72, so the re-parse re-hosts it
                    // identically.
                    Ex::Num(r) if partition => {
                        let mut r = *r;
                        if r.is_negative() {
                            neg = !neg;
                            r = r.checked_neg().unwrap();
                        }
                        let s = num_token(&r);
                        num_parts.push(if s.contains('/') { format!("({s})") } else { s });
                    }
                    Ex::Num(r) => {
                        let mut r = *r;
                        if r.is_negative() {
                            neg = !neg;
                            r = r.checked_neg().unwrap();
                        }
                        if r.den() != 1 && !crate::ac::expr::decimal_spelling_wins(&r) {
                            // Divisor-side spelling (see `divisor_side`): the firing
                            // reciprocal is an integer or exact decimal -- an atom, safe
                            // unparenthesized in the denominator join.
                            //
                            // The old `&& r.exact_decimal().is_none()` gate was the same
                            // staleness as in `divisor_side` itself: it sent every value
                            // with a decimal straight to the numerator on the assumption it
                            // printed as a decimal atom. Under argmin spelling `1/2` prints
                            // `1/2`, so that path emitted `1/2 * x0` beside `x0 / 3` -- two
                            // shapes for one thing. Every non-integer now takes the same
                            // route, matching the explicit dialect.
                            match divisor_side(&r).filter(|_| has_plain_mul_factor(v)) {
                                Some(inv) => den_parts.insert(0, num_token(&inv)),
                                None => {
                                    if r.num() != 1 {
                                        num_parts.insert(0, r.num().to_string());
                                    }
                                    den_parts.insert(0, r.den().to_string());
                                }
                            }
                        } else if !r.is_one() {
                            num_parts.insert(0, num_token(&r));
                        }
                    }
                    Ex::Pow(b, ex) => match &**ex {
                        Ex::Num(r)
                            if r.is_negative() && !matches!(&**b, Ex::Num(z) if z.is_zero()) =>
                        {
                            let flipped = r.checked_neg().unwrap();
                            if flipped.is_one() {
                                den_parts.push(render(b, cx, 3));
                            } else {
                                den_parts.push(format!(
                                    "{}^{}",
                                    render(b, cx, 4),
                                    render_exponent(&flipped)
                                ));
                            }
                        }
                        _ => num_parts.push(render(f, cx, 2)),
                    },
                    _ => num_parts.push(render(f, cx, 2)),
                }
            }
            let num_s = if num_parts.is_empty() {
                "1".to_string()
            } else {
                num_parts.join("*")
            };
            let s = if den_parts.is_empty() {
                num_s
            } else {
                // ONE SLASH PER DIVISOR MEMBER (hardening H-013, 2026-08-03):
                // `a/(b*c)` re-parses as a NESTED product in divisor position, which
                // only the licence-gated odd-negative-power distribution could
                // re-flatten -- when a factor's zero-set licence refuses (`cos` has
                // zeros), the round-trip lands on a DIFFERENT sound fixpoint and the
                // renderer breaks its own round-trip contract. `a/b/c` re-parses
                // member-by-member into the same flat section, no licence involved.
                // (Members that are themselves products/sums arrive parenthesized
                // from their prec-3 rendering, so a nested-bag member stays one
                // member.)
                let mut s = num_s;
                for d in &den_parts {
                    s.push('/');
                    s.push_str(d);
                }
                s
            };
            let s = if neg { format!("-{s}") } else { s };
            (s, if neg { 1 } else { 2 })
        }
    }
}

/// Exponent rendering: atoms bare (`x^2`), fractions/negatives parenthesized (`x^(1/2)`).
fn render_exponent(r: &Rat) -> String {
    let s = num_token(r);
    if r.is_integer() && !r.is_negative() {
        s
    } else {
        format!("({s})")
    }
}

/// Negate every term of a sum for DISPLAY sign redistribution (`None` on the i128 edge, in
/// which case the caller falls back to the literal `-1` rendering).
fn flip_terms(terms: &[Ex], cx: &Cx) -> Option<Vec<Ex>> {
    terms
        .iter()
        .map(|t| super::expr::negate_term(t, cx))
        .collect()
}

/// Display-only odd-literal sign hoist (owner ruling 2026-08-08): inside a printed sum,
/// a term whose magnitude is an odd function of a NEGATIVE literal moves that sign out to
/// the joiner (`- sin(-2)` becomes `+ sin(2)`), so no printed sum ever carries a double
/// negation. Literal arguments only: a non-literal argument's sign is structure, not a
/// spelling choice.
fn odd_literal_hoist(neg: bool, mag: Ex, cx: &Cx) -> (bool, Ex) {
    if let Ex::Fun(f, args) = &mag {
        if args.len() == 1 && cx.is_odd_fun(*f) {
            if let Ex::Num(r) = &args[0] {
                if r.is_negative() {
                    if let Some(p) = r.checked_neg() {
                        return (!neg, Ex::Fun(*f, vec![Ex::Num(p)]));
                    }
                }
            }
        }
    }
    (neg, mag)
}

/// Split an addend into (is-negative, magnitude form) for the `a - b` joiner.
fn split_sign(t: &Ex) -> (bool, Ex) {
    match t {
        Ex::Num(r) if r.is_negative() => (true, Ex::Num(r.checked_neg().unwrap())),
        Ex::NegInf => (true, Ex::PosInf),
        Ex::Mul(f) => match f.first() {
            Some(Ex::Num(r)) if r.is_negative() => {
                let mut rest: Vec<Ex> = f[1..].to_vec();
                let a = r.checked_neg().unwrap();
                if !a.is_one() {
                    rest.insert(0, Ex::Num(a));
                }
                let mag = if rest.len() == 1 {
                    rest.pop().unwrap()
                } else {
                    Ex::Mul(rest)
                };
                (true, mag)
            }
            _ => (false, t.clone()),
        },
        _ => (false, t.clone()),
    }
}

/// Convenience: parse, canonicalize, serialize -- the identity on already-canonical
/// serializations, and the canonicalization map on everything else.
#[cfg_attr(not(test), allow(dead_code))] // exercised by the round-trip tests
pub fn canonical_tokens(tokens: &[Tok], cx: &Cx) -> Option<Vec<Tok>> {
    let e = from_prefix(tokens, cx)?;
    let e = canon(e, cx);
    Some(to_prefix(&e, cx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::Operators;
    use crate::tokens::{TokenOverlay, TokenTable, TokenView};
    use std::cell::RefCell;

    /// Build the legacy-vocabulary operator universe from the IN-REPO fixture so desugaring
    /// sees the true arities (mult4, pow1_3, ...). The fixture ships with the repo (audit
    /// Tier-1 #3: no test depends on the dev_7-3 HF asset), so this never skips.
    fn full_ops() -> Option<(Vec<String>, Operators)> {
        let cfg = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/legacy_vocab_config.yaml"
        ))
        .expect("legacy fixture present in-repo");
        let parsed: serde_yaml_ng::Value = serde_yaml_ng::from_str(&cfg).unwrap();
        let mapping = parsed
            .get("operators")
            .and_then(|v| v.as_mapping())
            .unwrap();
        let mut order = Vec::new();
        let mut specs: rustc_hash::FxHashMap<String, crate::operators::OperatorSpec> =
            Default::default();
        for (k, v) in mapping {
            let name = k.as_str().unwrap().to_string();
            let spec: crate::operators::OperatorSpec =
                serde_yaml_ng::from_value(v.clone()).unwrap();
            order.push(name.clone());
            specs.insert(name, spec);
        }
        let ops = Operators::from_specs(order.clone(), specs);
        Some((order, ops))
    }

    /// Runs `f` against the legacy-vocabulary token view (in-repo fixture; None only kept
    /// for signature stability -- the fixture is always present).
    fn with_view<R>(f: impl FnOnce(&TokenView) -> R) -> Option<R> {
        let (order, ops) = full_ops()?;
        let table = TokenTable::build(&order, &ops);
        let overlay = RefCell::new(TokenOverlay::new(table.len()));
        let view = TokenView::new(&table, &overlay);
        Some(f(&view))
    }

    fn toks(view: &TokenView, s: &[&str]) -> Vec<Tok> {
        s.iter().map(|t| view.intern(t)).collect()
    }

    fn strs(view: &TokenView, t: &[Tok]) -> Vec<String> {
        t.iter().map(|&x| view.resolve_owned(x)).collect()
    }

    /// end-to-end: parse -> canon -> serialize, returning strings.
    fn roundtrip(view: &TokenView, input: &[&str]) -> Vec<String> {
        let cx = Cx::bare(view);
        let out = canonical_tokens(&toks(view, input), &cx).expect("parse");
        strs(view, &out)
    }

    #[test]
    fn desugars_hyperops_to_explicit_constants() {
        with_view(|view| {
            // mult4 x -> * 4 x
            assert_eq!(roundtrip(view, &["mult4", "x0"]), ["*", "4", "x0"]);
            // div5 x -> * 0.2 x: in the EXPLICIT dialect the coefficient takes its argmin
            // spelling, and for a 5-carrying denominator the decimal code wins (2585
            // milli-bits against the fraction's 3585) -- §10.10(1), owner-ratified
            // 2026-08-07. A 2^a denominator goes the other way (`1/2` stays `/ 1 2`).
            assert_eq!(roundtrip(view, &["div5", "x0"]), ["*", "0.2", "x0"]);
            // pow3 x -> pow x 3
            assert_eq!(roundtrip(view, &["pow3", "x0"]), ["pow", "x0", "3"]);
            // pow1_2 x -> rootn x 2: the even root KEEPS the root spelling (mu prefers it).
            assert_eq!(roundtrip(view, &["pow1_2", "x0"]), ["rootn", "x0", "2"]);
            // pow1_3 desugars to the rootn built-in and emits natively (the pow1_k
            // spellings are deleted from the vocabulary; input-side compat only)
            assert_eq!(roundtrip(view, &["pow1_3", "x0"]), ["rootn", "x0", "3"]);
            // neg x -> neg x (canonical Mul[-1, x] sugars back)
            assert_eq!(roundtrip(view, &["neg", "x0"]), ["neg", "x0"]);
            // inv x -> inv x
            assert_eq!(roundtrip(view, &["inv", "x0"]), ["inv", "x0"]);
        });
    }

    #[test]
    fn coefficient_arithmetic_is_computation() {
        with_view(|view| {
            // mult2(mult3(x)) -> * 6 x: the old coefficient-rule family as arithmetic.
            assert_eq!(roundtrip(view, &["mult2", "mult3", "x0"]), ["*", "6", "x0"]);
            // div2(x) * div2(y) -> x*y/4
            assert_eq!(
                roundtrip(view, &["*", "div2", "x0", "div2", "x1"]),
                ["/", "*", "x0", "x1", "4"]
            );
            // mult4(div2(x)) -> 2x
            assert_eq!(roundtrip(view, &["mult4", "div2", "x0"]), ["*", "2", "x0"]);
            // 7x via the old workaround `6x + x`:
            assert_eq!(
                roundtrip(view, &["+", "mult2", "mult3", "x0", "x0"]),
                ["*", "7", "x0"]
            );
        });
    }

    #[test]
    fn order_and_adjacency_invariance() {
        with_view(|view| {
            // ORDER: both operand spellings canonicalize identically.
            let a = roundtrip(view, &["*", "x1", "x0"]);
            let b = roundtrip(view, &["*", "x0", "x1"]);
            assert_eq!(a, b);
            // ADJACENCY: + x3 (+ x8 (div5 x3)) -- the campaign's proof case. The two x3 spellings
            // meet in one bag and collect: x3 + x3/5 = (6/5) x3.
            // Stripped term order: the 1.2*x3 term (key x3) sorts before x8.
            // The 6/5 coefficient spells as the decimal in the EXPLICIT dialect (argmin;
            // §10.10(1)). The COLLECTION is what this case proves, and it is unchanged.
            let out = roundtrip(view, &["+", "x3", "+", "x8", "div5", "x3"]);
            assert_eq!(out, ["+", "*", "1.2", "x3", "x8"]);
        });
    }

    #[test]
    fn sugar_round_trips() {
        with_view(|view| {
            let cases: &[&[&str]] = &[
                &["-", "x0", "x1"],
                &["/", "x0", "x1"],
                &["neg", "x0"],
                &["inv", "x0"],
                &["+", "x0", "*", "x1", "x2"],
                &["*", "7", "x0"],
                &["pow", "x0", "3"],
                &["-", "-", "x0", "x1", "x2"],
                &["/", "1", "+", "x0", "1"],
                &["sin", "+", "x0", "np.pi"],
                &["+", "<constant>", "x0"],
                &["*", "<constant>", "x0"],
                &["float(\"nan\")"],
            ];
            for c in cases {
                let once = roundtrip(view, c);
                let owned: Vec<&str> = once.iter().map(String::as_str).collect();
                let twice = roundtrip(view, &owned);
                assert_eq!(once, twice, "serialize/parse not a fixpoint for {c:?}");
            }
        });
    }

    #[test]
    fn rational_literals_spell_exactly() {
        with_view(|view| {
            // Every unit-fraction coefficient renders the SAME way -- as the division.
            // (`1/2` used to print `* 0.5 x0` because it had an exact decimal; that test
            // is gone, so the shape no longer depends on the denominator's factors.)
            assert_eq!(roundtrip(view, &["div3", "x0"]), ["/", "x0", "3"]);
            assert_eq!(roundtrip(view, &["div2", "x0"]), ["/", "x0", "2"]);
        });
    }

    #[test]
    fn signs_live_in_coefficient_literals() {
        // The sign doctrine, shared with the tagged form (which already absorbs: a bag
        // spells `<mul> -2 x0 </mul>`): a sign lives in the numeric coefficient literal
        // whenever one is emitted; `neg` wraps only the PURE sign, where no literal
        // exists to carry it. The parser is liberal (both spellings map to one state),
        // so this is purely an emission contract.
        with_view(|view| {
            // Negative coefficients: the sign rides the literal, no `neg`.
            assert_eq!(roundtrip(view, &["*", "(-2)", "x0"]), ["*", "-2", "x0"]);
            assert_eq!(
                roundtrip(view, &["*", "(-0.5)", "x0"]),
                ["/", "neg", "x0", "2"]
            );
            // A negative NON-decimal rational routes through the division split with the
            // sign on the numerator's coefficient literal.
            assert_eq!(
                roundtrip(view, &["*", "(-2)", "div3", "x0"]),
                ["/", "*", "-2", "x0", "3"]
            );
            // The pure sign has no literal to live in: `neg` stays.
            assert_eq!(roundtrip(view, &["neg", "x0"]), ["neg", "x0"]);
            assert_eq!(
                roundtrip(view, &["*", "(-1)", "*", "np.pi", "x0"]),
                ["neg", "*", "np.pi", "x0"]
            );
            // -1/3: divisor-side spelling (2026-08-01, design §8) is AMENDED by H-020
            // (2026-08-04): a sign never rides a DEN literal. A signed literal inside a
            // divisor group re-parses the sign into the divisor BAG, where the sign-fold
            // clause may absorb it into a Const-bearing sum -- a different (family-equal)
            // state than the outer-coefficient parking the chain holds. The reciprocal
            // spelling therefore yields to the pure-sign `neg` on the numerator side.
            assert_eq!(
                roundtrip(view, &["*", "(-1)", "div3", "x0"]),
                ["/", "neg", "x0", "3"]
            );
            // Absorbed spellings are fixpoints: parse(emit(s)) == s.
            for c in [
                vec!["*", "-2", "x0"],
                vec!["/", "neg", "x0", "2"],
                vec!["/", "*", "-2", "x0", "3"],
                vec!["/", "neg", "x0", "3"],
            ] {
                assert_eq!(roundtrip(view, &c), c, "not a fixpoint: {c:?}");
            }
        });
    }

    #[test]
    fn old_engine_can_parse_every_output() {
        with_view(|view| {
            // Everything we emit must be valid old-grammar prefix: check arity balance.
            let (_, ops) = full_ops().expect("closure only runs when the asset loaded");
            let cases: &[&[&str]] = &[
                &["+", "x3", "+", "x8", "div5", "x3"],
                &["*", "div2", "x0", "div2", "x1"],
                &["-", "x0", "+", "x1", "mult3", "x0"],
                &["pow", "x0", "pow", "x1", "x2"],
                &["neg", "div4", "x0"],
            ];
            for c in cases {
                let out = roundtrip(view, c);
                // Arity-balance check (the is_valid scan, inlined).
                let mut depth: i64 = 0;
                for t in out.iter().rev() {
                    if let Some(a) = ops.arity_of(t) {
                        depth -= a as i64;
                        assert!(depth >= 0, "arity underflow in {out:?}");
                    }
                    depth += 1;
                }
                assert_eq!(depth, 1, "unbalanced output {out:?}");
            }
        });
    }

    /// F73: an i128-overflow PARTITION's atoms survive every dialect's round-trip.
    /// The gathering emitters pooled the atoms into shared num/den chains, and the
    /// re-parse re-CUT them (`* a/3 a/3` -> `/ (a*a) (3*3)` -> `{a, a, 1/9}` -- one
    /// value, a different partition per dialect; 884 of the extreme lane's 900
    /// post-F72 hard rows). Partition members now render self-contained.
    #[test]
    fn f73_partition_atoms_survive_every_dialect() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let a3 = "170141183460469231731687303715884105727/3";
            // the canonical partition state, built directly
            let state = crate::ac::expr::mul(
                vec![
                    Ex::Num(Rat::new(170141183460469231731687303715884105727, 3).unwrap()),
                    Ex::Num(Rat::new(170141183460469231731687303715884105727, 3).unwrap()),
                ],
                &cx,
            );
            assert!(
                matches!(&state, Ex::Mul(v) if is_partition_bag(v)),
                "the probe pair must stay partitioned: {state:?}"
            );

            // TAGGED and EXPLICIT: parse(emit(state)) == state, member-exact.
            let tagged = to_prefix_tagged(&state, &cx);
            assert_eq!(
                from_prefix(&tagged, &cx).expect("tagged parses"),
                state,
                "tagged round-trip re-cut the partition"
            );
            let mut explicit = Vec::new();
            emit(&state, &cx, &mut explicit);
            assert_eq!(
                from_prefix(&explicit, &cx).expect("explicit parses"),
                state,
                "explicit round-trip re-cut the partition"
            );
            // The explicit spelling carries each atom as a LOCAL division, never a
            // pooled chain.
            let spelled: Vec<String> = strs(view, &explicit);
            assert_eq!(
                spelled,
                vec![
                    "*",
                    "/",
                    "170141183460469231731687303715884105727",
                    "3",
                    "/",
                    "170141183460469231731687303715884105727",
                    "3"
                ],
                "explicit gathering resurfaced"
            );

            // INFIX: atoms render parenthesized so the textual chain cannot re-associate.
            let infix = to_infix_pretty(&state, &cx);
            assert_eq!(infix, format!("({a3})*({a3})"));

            // The SIGNED partition keeps the sign in an atom (tagged/explicit) or
            // hoisted (infix) -- and still round-trips member-exact.
            let signed = crate::ac::expr::mul(
                vec![
                    Ex::Num(Rat::new(-170141183460469231731687303715884105727, 3).unwrap()),
                    Ex::Num(Rat::new(170141183460469231731687303715884105727, 3).unwrap()),
                ],
                &cx,
            );
            let tagged = to_prefix_tagged(&signed, &cx);
            assert_eq!(from_prefix(&tagged, &cx).expect("tagged parses"), signed);
        });
    }
}
