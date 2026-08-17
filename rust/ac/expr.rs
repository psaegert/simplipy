//! The AC expression type and its CANONICAL CONSTRUCTORS.
//!
//! Canonical invariants (established by [`add`], [`mul`], [`pow`], [`fun`]; every consumer may
//! rely on them):
//! * `Add(v)`: `v.len() >= 2`; no element is an `Add`; at most one `Num` (never 0); at most one
//!   `Const`; like terms collected as far as the soundness gates allow; sorted by [`cmp_ex`].
//! * `Mul(v)`: `v.len() >= 2`; no element is a `Mul`; at most one `Num` (never 1; 0 only in the
//!   uncertified `0 * t` case documented at the zero-coefficient gate); at most one bare `Const`;
//!   like bases collected under the gates; sorted by [`cmp_ex`].
//! * `Pow(b, e)`: `e` is never `Num(0)` or `Num(1)`; `b` is never a `Mul` when `e` is a
//!   non-negative or even integer (integer powers distribute) -- for ODD NEGATIVE `e` the
//!   distribution carries a zero-set licence (see `pow`), so a refused `Pow(Mul[..], e)`
//!   is a legal canonical form; rational^rational is folded when exact.
//! * No element anywhere is a nested bag of the same operator (Flat), and bags carry no order
//!   information beyond the canonical sort (Orderless).
//!
//! Soundness: every transform here is licensed either as TOTAL (equal as extended-real
//! evaluations everywhere), or as almost-everywhere equal with the null set certified by the
//! caller-provided certificates (the same two the shipped engine uses: `!` = finite a.e., `$` =
//! finite-and-nonzero a.e.), or as a `forall c_s exists c_t` fitted-constant absorption per the
//! contract. Each gate cites its counterexample. Refusing to transform is always sound; the
//! constructors prefer refusal over any uncertified step.

use std::cmp::Ordering;

use crate::tokens::{Tok, TokenView};

use super::rat::Rat;

/// An AC-core expression. Leaves carry interned [`Tok`] ids (variables `x0..`, wildcards
/// `_0`/`?0`/`!0`/`$0`); everything numeric is EXACT ([`Rat`], or the symbolic
/// transcendentals/extended values as their own variants).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ex {
    /// An exact rational literal.
    Num(Rat),
    /// pi -- a transcendental FACTOR, never absorbed into the rational coefficient.
    Pi,
    /// e, likewise.
    E,
    PosInf,
    NegInf,
    NaN,
    /// `<constant>` -- a fitted constant, existentially quantified over R. EVERY occurrence is an
    /// INDEPENDENT constant: two `Const`s are structurally equal but semantically distinct, which
    /// is why `Const`-containing keys never merge in collection.
    Const,
    /// A variable or wildcard leaf (sigil-classified by the token, exactly as the old matcher).
    Leaf(Tok),
    /// n-ary flat sorted addition bag.
    Add(Vec<Ex>),
    /// n-ary flat sorted multiplication bag.
    Mul(Vec<Ex>),
    Pow(Box<Ex>, Box<Ex>),
    /// A named function application (`sin`, `cos`, `log`, `abs`, ..., and the real odd roots
    /// `pow1_3`/`pow1_5` which deliberately do NOT desugar to `Pow`).
    Fun(Tok, Vec<Ex>),
}

impl Ex {
    pub fn int(n: i128) -> Ex {
        Ex::Num(Rat::int(n))
    }

    /// Does this expression contain a `<constant>` anywhere? (The independence guard: such
    /// subtrees never participate in key-equality collection, and the matcher refuses to rebind
    /// a placeholder to one -- both carried over from the shipped engine.)
    pub fn contains_const(&self) -> bool {
        match self {
            Ex::Const => true,
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Leaf(_) => false,
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(Ex::contains_const),
            Ex::Pow(b, e) => b.contains_const() || e.contains_const(),
        }
    }

    /// Does this expression contain a VARIABLE leaf or a `<constant>` -- i.e., does it have a
    /// measure space for "almost everywhere" to quantify over? The a.e. certificates are
    /// statements about null sets in (variable, constant)-space; a fully ground expression
    /// (`inv 0`) has a SINGLE value, so a.e. tolerance degenerates to exactness. The interval
    /// certificates handle this correctly themselves (`finite_ae("inv 0") == false`,
    /// verified); the guard makes the licences independent of that edge behavior and hands
    /// ground arithmetic to the numeric fold, which owns it exactly. (The shipped engine's
    /// `- inv 0 inv 0 -> 0` gap is its certificate-FREE `_`-sort rule tier, not the
    /// certificates.)
    pub fn has_measure_space(&self, view: &TokenView) -> bool {
        match self {
            // A sigil-free leaf is a variable UNLESS it merely SPELLS a number the exact
            // parser could not represent (beyond-i128 mantissas like `1e39`, minted by the
            // engine's own fold re-parse). Such a leaf denotes ONE real value: it is
            // ground, and treating it as a variable would hand the a.e. licences a
            // measure space that does not exist.
            Ex::Leaf(t) => {
                view.sigil(*t) == 0 && !crate::utils::is_numeric_string(&view.resolve_owned(*t))
            }
            Ex::Const => true,
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN => false,
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(|e| e.has_measure_space(view)),
            Ex::Pow(b, e) => b.has_measure_space(view) || e.has_measure_space(view),
        }
    }

    /// Does this expression contain any wildcard leaf? (Pattern detection.)
    pub fn contains_wildcard(&self, view: &TokenView) -> bool {
        match self {
            Ex::Leaf(t) => view.sigil(*t) != 0,
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Const => false,
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(|e| e.contains_wildcard(view)),
            Ex::Pow(b, e) => b.contains_wildcard(view) || e.contains_wildcard(view),
        }
    }
}

/// The COMPLEXITY functional IS the unified simplicity measure mu (stage 2, ratified in
/// design/UNIFIED_SIMPLICITY_MEASURE.md): description length under an exactness-respecting
/// cost model, measured on the CANONICAL internal form and never on a serialization.
/// "Simplify is lossless compression" -- pi is one symbol with zero degrees of freedom,
/// `23.14069263277927` is a ~52-bit object, `<constant>` is a full real degree of freedom.
/// Costs are scaled integers in units of 1/8 (one grammar symbol = 8), so all arithmetic
/// stays exact and the order is discrete -- no float comparisons anywhere in the ordering.
///
/// The weight table (each row a ratified decision, owner 2026-07-31/08-01):
///
/// | construct                          | cost (1/8 units) | rationale                     |
/// |------------------------------------|------------------|-------------------------------|
/// | structural node (bag, Pow, Fun)    | 8                | one grammar symbol            |
/// | variable leaf, pi, e, inf, nan     | 8                | one vocabulary symbol         |
/// | numeric literal p/q (exact VALUE)  | max(2, bits(p)+bits(q)-1) | its description      |
/// |   (spelling-free: `0.5` == `1/2`)  |                  | length; implicit /1 is free   |
/// | bag coefficient slot, magnitude!=1 | the literal cost | the slot PAYS (removes the    |
/// |   (magnitude 1: a bare sign, 0)    |                  | coefficient-rides-free        |
/// |                                    |                  | asymmetry that minted the     |
/// |                                    |                  | materialization fork)         |
/// | rational `Pow` exponent            | as coefficient   | `x^-1` = division = structure |
/// | `<constant>`                       | c_free = 128     | a free real dof; the priciest |
/// |                                    |                  | atom (owner floor: >= special)|
///
/// SIGN IS FREE throughout (`|p|`; a magnitude-1 coefficient or exponent costs 0):
/// `x - y` is not more complex than `x + y` -- standing doctrine, not revoked by mu.
/// The literal formula `max(2, bits(|p|) + bits(q) - 1)` is the unique reading
/// consistent with the ratified doc's worked examples: cost(2)=2 (T4: mu(2x)=18),
/// cost(5/2)=4 (T5: mu(2.5*pi)=20), a 52-bit dyadic ~105 (T1/T2), mu(0)=mu(1)=2 (T3).
///
/// Worked examples: `x + y` = 24, `x*y` = 24, `2x` = 18, `x^2` = 18, `1/x` = 16,
/// `sin(x)` = 16, `sin(pi)` = 16, `E*x` = 24 < `2.718281828459045*x` = ~117,
/// `<constant>*x` = 144, `exp(pi*x)` = 32 < `pow(23.14..., x)` = ~121.
/// One grammar symbol, in bit-units. 8 by default (the ratified 1/8 unit: one
/// symbol counts as 8 bits of description); `SIMPLIPY_MU_SYM` overrides it, read
/// ONCE per process -- the pre-registered P-R3 sensitivity axis is this
/// symbol-vs-bits RATIO (literal costs are absolute bits and do not scale).
/// Production never sets the variable; it exists for the sensitivity mine only.
// ARTIFACT-AFFECTING switch (and mu_free below): listed in
// engine.py::ARTIFACT_ENV_SWITCHES (H-042).
/// Returned in MILLI-BITS, but the environment variable is still read in BITS -- its
/// ratified meaning ("one grammar symbol counts as 8 bits of description") is unchanged,
/// only the unit the measure is carried in.
pub fn mu_sym() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SIMPLIPY_MU_SYM")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(8)
            * MU_MILLI
    })
}

/// The free constant's price, DERIVED rather than chosen (contract §10.10(5), H-054).
///
/// `<constant>` denotes an unknown value that a FIT will supply, i.e. an f64, and the ordering
/// requires it to dominate every literal it could be instantiated to. The former default of
/// `16 * mu_sym()` = 128 bits -- sixteen grammar symbols, a count with no derivation behind it --
/// did NOT: `mu(1e308) = 1024.154` even before §10.10(1), so the priciest atom was beaten
/// eightfold by an ordinary f64 extreme, and after §10.10(1) an everyday fitted constant beats it
/// too (`1.2345678901234567e-17` prices 163.078). So the price is the SUPREMUM instead:
///
/// ```text
///   max over f64 round-trip spellings of  L(significand) + |decimal scale| * log2(10)
///     = 1131.931 bits, attained at 5.5605781537525765e-308
///       (17 significant digits, decimal scale -324)
///   + 1 sign bit                    = 1132.931 bits
///   rounded up                      = 1133 bits
/// ```
///
/// Swept over 210,389 candidates: every binary exponent -1074..=1024 at five mantissas each,
/// the range boundaries, and 200,000 uniformly random bit patterns.
///
/// SCOPED TO THE f64 RANGE ON PURPOSE. `mu_numeric_str` prices a WRITTEN token beyond that range
/// higher still (`1e-400` costs 1330.8 bits under §10.10(1)), and no finite bound dominates every
/// writable spelling, because the exponent is unbounded. A beyond-f64 literal is not a value a fit
/// can produce, so it is not a value `<constant>` stands for; the deliberate consequence is that
/// such a literal outprices the free constant.
///
/// Pinned at the POST-§10.10(1) bound, not today's 1024.154, so that landing the print-only
/// decimal code does not have to move it a second time.
pub const MU_FREE_WORST_CASE_F64: u64 = 1_133 * MU_MILLI;

/// `<constant>`, the priciest atom. `SIMPLIPY_MU_FREE` (read in BITS) overrides explicitly, for
/// deployments whose fitted constants live in a narrower range than the whole f64 line.
pub fn mu_free() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("SIMPLIPY_MU_FREE")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(|bits| bits * MU_MILLI) // read in BITS, like `SIMPLIPY_MU_SYM`
            .unwrap_or(MU_FREE_WORST_CASE_F64)
    })
}

/// Milli-bits per bit. The whole measure is carried in MILLI-BITS so that a literal's
/// cost can be the real quantity `log2(1 + |n|)` rather than a bit COUNT, while the
/// ordering stays integer -- no float ever enters a comparison. Bit lengths quantise
/// brutally at exactly the scale that matters: `bits(100) = bits(127) = 7`, so the old
/// measure could not tell 100 from 1000 once a scale was involved, and priced two
/// spellings of one value up to a bit apart for no reason but rounding.
pub const MU_MILLI: u64 = 1000;

/// `1000 * log2(1 + n)`, exact to the milli-bit, in pure integer arithmetic.
///
/// Deterministic and identical on every platform, which `f64::log2` is NOT guaranteed to
/// be across libm versions -- and the reduction ordering must be byte-reproducible or the
/// mine's three-run identity gate is meaningless.
///
/// Method: write `1 + n = 2^(b-1) * x` with `x` in [1, 2). The integer part of the
/// logarithm is `b - 1`; the fraction comes from the classical bit-by-bit extraction,
/// squaring `x` and halving it whenever it passes 2, each squaring yielding one bit.
/// `x` is held in fixed point with `FRAC` fractional bits, so `x^2` needs `2*FRAC + 4`
/// bits and stays inside `u128`.
pub fn l_millibits(n: u128) -> u64 {
    let m = n.saturating_add(1);
    if m <= 1 {
        return 0; // log2(1) = 0, i.e. n = 0
    }
    let b = 128 - m.leading_zeros() as u64; // m has b bits: 2^(b-1) <= m < 2^b
    let int_part = b - 1;
    const FRAC: u32 = 48;
    // x = m / 2^(b-1), in [1, 2), as a FRAC-bit fixed-point value.
    let mut x: u128 = if (b - 1) as u32 >= FRAC {
        m >> ((b - 1) as u32 - FRAC)
    } else {
        m << (FRAC - (b - 1) as u32)
    };
    let one: u128 = 1u128 << FRAC;
    let two: u128 = one << 1;
    // Extract the fraction one bit at a time, accumulating in the same fixed point.
    let mut frac: u128 = 0;
    let mut weight: u128 = one >> 1; // 0.5, then 0.25, ...
    for _ in 0..FRAC {
        if weight == 0 {
            break;
        }
        x = (x * x) >> FRAC;
        if x >= two {
            x >>= 1;
            frac += weight;
        }
        weight >>= 1;
    }
    // Round to the nearest milli-bit.
    let frac_milli = ((frac * MU_MILLI as u128) + (one >> 1)) >> FRAC;
    int_part * MU_MILLI + frac_milli as u64
}

/// The DECIMAL code's cost components for a value that terminates in base ten
/// (`q = 2^a * 5^b`): the cost of the scaled mantissa `m = |p| * 10^k / q` and of the
/// scale `k = max(a, b)`, each priced by `l_millibits` -- the scale is just another
/// integer the spelling has to write down. `None` when the value does not terminate, when
/// it is an integer (the fraction code already prices those exactly, and offering a
/// decimal code there is the base-ten roundness question, deliberately left shut), or when
/// the scaled mantissa leaves `i128`, in which case the decimal spelling is not available
/// in range and the fraction code stands alone.
fn decimal_code(r: &Rat) -> Option<(u64, u64)> {
    if r.den() == 1 {
        return None;
    }
    let (mut a, mut b) = (0u32, 0u32);
    let mut rest = r.den();
    while rest % 2 == 0 {
        rest /= 2;
        a += 1;
    }
    while rest % 5 == 0 {
        rest /= 5;
        b += 1;
    }
    if rest != 1 {
        return None;
    }
    let k = a.max(b);
    let mut m = r.num().checked_abs()?;
    for _ in 0..(k - a) {
        m = m.checked_mul(2)?;
    }
    for _ in 0..(k - b) {
        m = m.checked_mul(5)?;
    }
    Some((l_millibits(m.unsigned_abs()), l_millibits(u128::from(k))))
}

/// Description length of the exact VALUE p/q (lowest terms), in MILLI-BITS and
/// spelling-free. The implicit denominator 1 is free (an integer writes no denominator);
/// the floor of 2 bits makes every literal a real object.
///
/// ONE RULE FOR EVERY LITERAL (owner ruling 2026-08-06). A literal is written as a small
/// number of integers, and each integer costs `L(n) = log2(1 + |n|)`:
///
/// | spelling            | cost                                   |
/// |---------------------|----------------------------------------|
/// | integer `n`         | `L(n)`                                 |
/// | fraction `p/q`      | `L(p) + L(q)`                          |
/// | decimal `m * 10^-k` | `L(m) + L(k)` -- the scale is an integer too |
/// | negative            | `+ 1 bit`, once                        |
///
/// mu is the MINIMUM over the codes the grammar offers. The crossover then falls out
/// instead of being legislated: powers of two keep the fraction (`1/2`, `5/8`), anything
/// carrying a factor of five prints decimal (`2/5` -> `0.4`, `6/5` -> `1.2`), and a power
/// of ten can never survive in a denominator.
///
/// WHY `L` AND NOT A BIT COUNT. `bits()` quantises exactly where it hurts: it cannot
/// separate 100 from 1000 once a scale is involved (both scales cost 2 bits), it priced
/// two spellings of one value up to a bit apart for no reason but rounding, and it made
/// mu non-monotone in magnitude at the `i128` boundary. `L` is the continuous lower
/// envelope of `bits` -- equal at 1, 3, 7, 127, never more than a bit below -- so this
/// DE-QUANTISES the ratified measure rather than repricing it.
///
/// WHY THE SIGN COSTS A BIT. Without it `mu(2) == mu(-2)`: the measure could not tell a
/// number from its negation, which a description length may not do. `Rat` normalises to
/// `q > 0` with the sign on the numerator, so the bit is unambiguous and charged once.
/// (This REVOKES the former "sign is free" doctrine for literals; `x - y` vs `x + y` is
/// unaffected, since a subtraction is structural and carries no signed literal.)
/// **The decimal code is NOT consulted here (contract §10.10(1), H-055).** mu is VALUE
/// complexity, not description length, so a rational costs its two written integers and the
/// `m * 10^-k` spelling earns it no discount. Taking the minimum over the two codes made mu
/// asymmetric about 1 by an order of operation -- `log2(value)` above, `log2(log(1/value))`
/// below, so `mu(1000) = 9.967` against `mu(0.001) = 3.000` -- because `decimal_code` refuses
/// integers and only the sub-unit side had an escape. With the min gone, `mu(1/n) = mu(n) + 1`
/// exactly for every `n >= 2`, `L(1) = log2(2)` being exactly one bit: the multiplicative
/// inversion bit is EMERGENT from the notation (an integer writes no denominator; a unit
/// fraction must write its numerator), the exact analogue of the additive sign bit, and it must
/// not be charged a second time explicitly.
///
/// `decimal_code` survives for [`decimal_spelling_wins`] alone, which is now a PRINT rule with
/// no bearing on cost -- a printed `/ 1 5` and a printed `0.2` re-parse to the SAME `Ex::Num`
/// leaf, so complexity is spelling-independent either way.
pub fn mu_rat(r: &Rat) -> u64 {
    let (fraction, _decimal) = mu_rat_codes(r);
    let signed = fraction + if r.num() < 0 { MU_MILLI } else { 0 };
    signed.max(2 * MU_MILLI)
}

/// The two codes' costs separately, in milli-bits and WITHOUT the sign bit or the floor
/// (the caller applies both once): `(fraction, decimal_if_terminating)`.
///
/// An INTEGER pays only its own cost -- its denominator is implicit, not written. A genuine
/// fraction pays BOTH components. The former blanket `- 1` made the implicit denominator
/// free for integers (correct) but also made `1/q` cost exactly what `q` costs, so `p/q`
/// and `q/p` were indistinguishable and mu could not tell a number from its reciprocal.
fn mu_rat_codes(r: &Rat) -> (u64, Option<u64>) {
    let pb = l_millibits(r.num().unsigned_abs());
    let qb = l_millibits(r.den() as u128);
    let fraction = if r.den() == 1 { pb } else { pb + qb };
    let decimal = decimal_code(r).map(|(m, k)| m + k);
    (fraction, decimal)
}

/// Whether the DECIMAL spelling is the canonical PRINT for this value. Ties go to the
/// fraction, which is what keeps the powers of two (`1/2`, `1/4`, `5/8`) spelled as fractions
/// while anything carrying a factor of five (`1/5` -> `0.2`, `6/5` -> `1.2`) prints as a
/// decimal.
///
/// **PRINT-ONLY since §10.10(1) (H-055).** It still compares the two codes, but `mu_rat` no
/// longer does, so this no longer follows the cost -- it is a rendering choice and nothing
/// more. That is sound because the choice is not observable in the measure: `/ 1 5` and `0.2`
/// re-parse to the SAME `Ex::Num` leaf (`complexity(["/","1","5"]) == complexity(["0.2"]) ==
/// 2.585`), so complexity is spelling-independent. Keeping the existing rule is deliberate --
/// it holds the emitted dialect FIXED across the measure change, so the re-pins below have
/// exactly one cause.
pub fn decimal_spelling_wins(r: &Rat) -> bool {
    let (fraction, decimal) = mu_rat_codes(r);
    decimal.is_some_and(|d| d < fraction)
}

/// `L(n)` in MILLI-BITS for an integer written as a decimal DIGIT STRING of any length:
/// the leading digits carry the mantissa and each remaining digit adds `log2(10)`.
/// Exact for anything `l_millibits` can hold outright, and within a milli-bit beyond it.
fn l_millibits_digits(digits: &str) -> u64 {
    let d = digits.trim_start_matches('0');
    if d.is_empty() {
        return 0;
    }
    let head_len = d.len().min(19);
    let head: u128 = d[..head_len].parse().unwrap_or(0);
    let tail = (d.len() - head_len) as u64;
    l_millibits(head).saturating_add(tail.saturating_mul(L10_MILLI))
}

/// `log2(10)` and `log2(5)` in milli-bits: the cost of one decimal digit, and of the
/// factor five that turns a power of two into a power of ten.
const L10_MILLI: u64 = 3322;

/// The ASTRONOMIC knee (audit B22). The exact linear schedule `|scale| * L10_MILLI`
/// exhausts u64 at |scale| ~ 5.553e15; saturating there collided every larger scale --
/// and every beyond-i64 exponent -- on the single price `u64::MAX`, and that poisoned
/// leaf then overflowed the tree sums in `complexity()` (a debug panic; a release WRAP
/// pricing a composite below its parts, the direction that licenses false mu-descents).
/// Beyond this knee the schedule switches to `KNEE_COST + L(|scale|)`: still monotone
/// in the scale in both regimes and across the seam (the first astronomic price is
/// `KNEE_COST + ~32` bits > the linear ceiling `KNEE_COST`), bounded at ~1.4e13 plus
/// ~3.3 bits per exponent DIGIT (u64::MAX needs a petabyte spelling), and priced from
/// the exponent's own digits, so absurd scales stay ORDERED instead of colliding.
/// The knee sits nine orders of magnitude past every f64-derived literal (|scale| <=
/// 324), so every honest literal prices on the exact linear schedule, unchanged.
const MU_SCALE_KNEE: u64 = 1 << 32;
const MU_SCALE_KNEE_COST: u64 = MU_SCALE_KNEE * L10_MILLI;

/// Description length of a BEYOND-`Rat` numeric literal, from its canonical print, under
/// exactly the rule `mu_rat` applies in range: every integer the spelling writes down
/// costs `L(n) = log2(1 + |n|)`, a fraction pays both components, a decimal pays its
/// mantissa and its SCALE, the sign costs a bit and the floor is two bits.
///
/// Literals whose exact p/q exceeds 128 bits (`4.159653437657682e-35` has q = 10^50) live
/// as numeric-string LEAVES, invisible to `mu_rat` -- pricing them at one vocabulary symbol
/// would (a) invert the ordering at the `Rat` boundary (a ~220-bit object undercutting
/// `0.001`), (b) reopen the literal-respell hole the old `lit_size` tier closed in exactly
/// this range, and (c) let the mu-governed fold materialize deep-magnitude roundings
/// (`exp(-80) -> 1.80485e-35`) that it refuses in-range.
///
/// An approximation of the true reduced p/q by design (the 2/5 gcd is ignored; the
/// boundary this exists for cannot turn on a milli-bit), deterministic, and monotone in
/// both the significand and the scale.
pub fn mu_numeric_str(s: &str) -> u64 {
    let negative = s.starts_with('-');
    let body = s.strip_prefix('-').unwrap_or(s);
    let sign = if negative { MU_MILLI } else { 0 };
    let finish = |cost: u64| cost.saturating_add(sign).max(2 * MU_MILLI);

    // Fraction-shaped beyond-`Rat` literals (`p/q` with a component past i128 -- in-range
    // fractions parse to `Num` and never reach the leaf pricing).
    if let Some((p, q)) = body.split_once('/') {
        let ps = p.trim_start_matches('0');
        let qs = q.trim_start_matches('0');
        if ps.is_empty() {
            return finish(0); // zero numerator: the value is 0
        }
        let lp = l_millibits_digits(ps);
        if qs == "1" {
            return finish(lp); // an implicit denominator is free, as for any integer
        }
        // BOTH components, and no decimal code: §10.10(1) revoked the minimum over codes, so
        // a `p/q` token pays for the integers it actually writes. The arm that used to offer
        // the decimal code here is gone with it -- it existed to stop one value pricing ~14x
        // apart by spelling alone (`1e-40` against `1/10^40`), and that agreement is now
        // reached from the other side: both spellings pay the whole 10^40 denominator, 133.880
        // bits, and the test pins BOTH so the boundary cannot drift apart again.
        return finish(lp.saturating_add(l_millibits_digits(qs)));
    }

    // A decimal/exponent spelling. An exponent too large for `i64` keeps its DIGIT STRING,
    // so the scale's own cost stays exact however absurd it is; falling back to scale 0
    // once priced `1e99999999999999999999999` below a variable, and saturating alone made
    // two different absurd scales collide on one price.
    let (mant, exp, exp_digits) = match body.split_once(['e', 'E']) {
        Some((m, e)) => match e.parse::<i64>() {
            Ok(v) => (m, v, None),
            Err(_) => {
                let negative_exp = e.trim_start().starts_with('-');
                let digits = e.trim_start_matches(['-', '+']);
                (
                    m,
                    if negative_exp { i64::MIN } else { i64::MAX },
                    Some(digits),
                )
            }
        },
        None => (body, 0, None),
    };
    let (int_part, frac_part) = match mant.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mant, ""),
    };
    // Significand digits with leading zeros stripped; net scale in decimal places.
    let digits: String = int_part.chars().chain(frac_part.chars()).collect();
    let sig = digits.trim_start_matches('0');
    let trailing = sig.len() - sig.trim_end_matches('0').len();
    let sig_core = &sig[..sig.len() - trailing];
    if sig_core.is_empty() {
        return finish(0); // the string denotes zero
    }
    let l_sig = l_millibits_digits(sig_core);
    let scale = exp
        .saturating_sub(frac_part.len() as i64)
        .saturating_add(trailing as i64);
    // `unsigned_abs`, not `-scale`: negating `i64::MIN` PANICS in a debug build, and a
    // saturated exponent reaches exactly that value.
    // The scale's OWN magnitude cost, continuous across the i64 boundary: `l_millibits`
    // of the value while the exponent parses, `l_millibits_digits` of its digit string
    // once it does not (same quantity, within a milli-bit -- see `l_millibits_digits`).
    let scale_digits_cost = match exp_digits {
        Some(d) => l_millibits_digits(d),
        None => l_millibits(u128::from(scale.unsigned_abs())),
    };
    // Two regimes, one knee (B22, doc at `MU_SCALE_KNEE`): the exact linear schedule up
    // to the knee -- every honest literal, unchanged -- then `KNEE_COST + L(|scale|)`,
    // monotone and far from u64::MAX where the old arms saturated and collided.
    let power_of_ten_cost = if exp_digits.is_none() && scale.unsigned_abs() <= MU_SCALE_KNEE {
        scale.unsigned_abs() * L10_MILLI // <= MU_SCALE_KNEE_COST: cannot overflow
    } else {
        MU_SCALE_KNEE_COST.saturating_add(scale_digits_cost)
    };
    if scale >= 0 {
        // An INTEGER: value = sig * 10^scale, and it is offered NO decimal code, exactly
        // as `decimal_code` refuses one in range. Both sides of the boundary must agree
        // or mu stops being monotone in magnitude -- offering it here once priced 10^38
        // at 127 and 10^39 at 10, a value ten times larger costing 12.7x LESS.
        return finish(l_sig.saturating_add(power_of_ten_cost));
    }
    // A non-integer. The fraction code writes the whole power of ten in the denominator; the
    // decimal code would write only the SCALE, and taking the minimum of the two is exactly
    // what §10.10(1) REVOKES -- mu is value complexity, not description length, so terminating
    // in base ten earns no discount. This is also what keeps the two pricers in step: `mu_rat`
    // takes the fraction code alone, and the `Rat` boundary must not be an ordering cliff.
    //
    // The deliberate consequence, named rather than implied: for a terminating decimal the
    // denominator is FORCED by the scale and carries no independent information, so this class
    // is charged its magnitude twice. `0.0293847502 = 146923751/5000000000` prices 59.350
    // against the decimal code's 31.590, roughly double, and so does a full-precision decimal
    // (`3.14159265358979`: 94.665 against 52.065). Owner-accepted as the consistent price of
    // refusing the roundness discount for 1000.
    finish(l_sig.saturating_add(power_of_ten_cost))
}

/// Structural equality MODULO literal content: two canonical trees are
/// skeleton-equal when they differ at most in the VALUES of numeric atoms
/// (`Num` nodes and beyond-`Rat` numeric-string leaves). The resolution guard's
/// predicate: a resolved target skeleton-equal to its mark differs only in
/// literal content -- either the same value (not strictly below, already
/// refused) or a VALUE CHANGE, i.e. the respell disease. Canon puts at most one
/// merged Num per bag and sorts deterministically, so positional comparison is
/// robust.
pub fn eq_mod_nums(a: &Ex, b: &Ex, view: &TokenView) -> bool {
    let numeric_leaf = |t: &crate::tokens::Tok| {
        view.with_str(*t, |s| {
            s.as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_digit() || *b == b'-' || *b == b'.' || *b == b'+')
                && crate::utils::is_numeric_string(s)
        })
    };
    let is_num = |e: &Ex| match e {
        Ex::Num(_) => true,
        Ex::Leaf(t) => numeric_leaf(t),
        _ => false,
    };
    if is_num(a) && is_num(b) {
        return true;
    }
    match (a, b) {
        (Ex::Add(x), Ex::Add(y)) | (Ex::Mul(x), Ex::Mul(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| eq_mod_nums(p, q, view))
        }
        (Ex::Pow(bx, ex), Ex::Pow(by, ey)) => {
            eq_mod_nums(bx, by, view) && eq_mod_nums(ex, ey, view)
        }
        (Ex::Fun(fx, ax), Ex::Fun(fy, ay)) => {
            fx == fy
                && ax.len() == ay.len()
                && ax.iter().zip(ay).all(|(p, q)| eq_mod_nums(p, q, view))
        }
        _ => a == b,
    }
}

// SATURATING accumulation throughout (audit B22): every per-node price is bounded far
// below u64::MAX (`MU_SCALE_KNEE` caps the literal schedule), but a large enough bag of
// astronomic leaves -- ~1.3M members, a ~16 MB input -- still overflows an unchecked
// sum. Overflow WRAPS in release and a wrapped total prices a composite below its own
// parts, exactly the direction that licenses false mu-descents (and it panics the debug
// build outright). Saturating at the TOP is sound: a rewrite fires only on a STRICT
// decrease, so two totals pinned at the ceiling can only ever REFUSE.
pub fn complexity(e: &Ex, view: &TokenView) -> u64 {
    let sym = mu_sym();
    match e {
        Ex::Num(r) => mu_rat(r),
        Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN => sym,
        // A vocabulary leaf is one symbol; a NUMERIC-STRING leaf is a beyond-`Rat`
        // literal and pays the description length of the value its print denotes
        // (see `mu_numeric_str` -- the boundary must not be an ordering cliff).
        // HOT PATH: no allocation (`with_str`), and a first-byte filter skips the
        // full numeric parse for ordinary vocabulary leaves (`x0`, function names),
        // which never start with a digit, dot, or sign.
        Ex::Leaf(t) => view.with_str(*t, |s| {
            if s.as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_digit() || *b == b'-' || *b == b'.' || *b == b'+')
                && crate::utils::is_numeric_string(s)
            {
                mu_numeric_str(s)
            } else {
                sym
            }
        }),
        Ex::Const => mu_free(),
        Ex::Add(v) => v
            .iter()
            .fold(sym, |t, x| t.saturating_add(complexity(x, view))),
        Ex::Mul(v) => {
            let mut total = sym;
            let mut members = 0u64;
            for f in v {
                match f {
                    // The rational coefficient slot: magnitude 1 is a bare sign (free);
                    // any other magnitude pays its description length -- THE stage-2
                    // decision (an `E` factor at 8 now undercuts a `2.718...` coefficient
                    // at ~109 instead of riding free below it).
                    Ex::Num(r) => {
                        if !(r.is_one() || *r == Rat::NEG_ONE) {
                            total = total.saturating_add(mu_rat(r));
                        }
                    }
                    _ => {
                        members += 1;
                        total = total.saturating_add(complexity(f, view));
                    }
                }
            }
            if members == 0 {
                // Degenerate all-coefficient bag: the literal (or bare Const) itself.
                return (total - sym).max(2);
            }
            total
        }
        Ex::Pow(b, ex) => {
            if let Ex::Num(r) = &**ex {
                // Rational exponent as a coefficient slot: `x^-1` is division (pure
                // structure, the sign is free), `x^2` pays cost(2), `x^-2` the same.
                let mag = if r.is_one() || *r == Rat::NEG_ONE {
                    0
                } else {
                    mu_rat(r)
                };
                return sym.saturating_add(complexity(b, view)).saturating_add(mag);
            }
            sym.saturating_add(complexity(b, view))
                .saturating_add(complexity(ex, view))
        }
        Ex::Fun(_, args) => args
            .iter()
            .fold(sym, |t, x| t.saturating_add(complexity(x, view))),
    }
}

/// Rank for the canonical total order (cross-kind comparisons decide on this alone).
fn rank(e: &Ex) -> u8 {
    match e {
        Ex::Num(_) => 0,
        Ex::Pi => 1,
        Ex::E => 2,
        Ex::PosInf => 3,
        Ex::NegInf => 4,
        Ex::NaN => 5,
        Ex::Const => 6,
        Ex::Leaf(_) => 7,
        Ex::Fun(..) => 8,
        Ex::Pow(..) => 9,
        Ex::Add(_) => 10,
        Ex::Mul(_) => 11,
    }
}

/// The canonical total order. View-dependent because `Leaf`/`Fun` tokens compare by their
/// STRINGS (stable across engines/interning orders), not by id. Order is a CANONICALIZATION
/// concern only -- bags denote the same function in any order -- so no soundness rests here,
/// but determinism does: this order is the whole fix for the ORDER axis of the invariance
/// defect, so it must be total and stable.
pub fn cmp_ex(a: &Ex, b: &Ex, view: &TokenView) -> Ordering {
    let (ra, rb) = (rank(a), rank(b));
    if ra != rb {
        return ra.cmp(&rb);
    }
    match (a, b) {
        (Ex::Num(x), Ex::Num(y)) => x.cmp_exact(y),
        (Ex::Leaf(x), Ex::Leaf(y)) => view.str_cmp(*x, *y),
        (Ex::Fun(fa, va), Ex::Fun(fb, vb)) => {
            view.str_cmp(*fa, *fb).then_with(|| cmp_vec(va, vb, view))
        }
        (Ex::Pow(ba, ea), Ex::Pow(bb, eb)) => {
            cmp_ex(ba, bb, view).then_with(|| cmp_ex(ea, eb, view))
        }
        (Ex::Add(va), Ex::Add(vb)) | (Ex::Mul(va), Ex::Mul(vb)) => cmp_vec(va, vb, view),
        _ => Ordering::Equal, // same rank, no payload (Pi, E, infinities, NaN, Const)
    }
}

fn cmp_vec(a: &[Ex], b: &[Ex], view: &TokenView) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match cmp_ex(x, y, view) {
            Ordering::Equal => continue,
            o => return o,
        }
    }
    a.len().cmp(&b.len())
}

/// The canonicalization context: the token view (leaf classification, string order) plus the two
/// OPTIONAL soundness certificates. With a certificate absent the corresponding gated collection
/// simply does not happen -- fail-closed, exactly like the old matcher's certifier plumbing.
pub struct Cx<'a> {
    pub view: &'a TokenView<'a>,
    /// The `!`-certificate: is this expression defined and finite almost everywhere? Gates
    /// additive SIGN-CANCELLING collection (`2t - t -> t` moves a `t = +-inf` point from NaN to
    /// `+-inf`, sound only when that set is null).
    pub cert_fin: Option<&'a dyn Fn(&Ex) -> bool>,
    /// The `$`-certificate: defined, finite AND nonzero a.e.? Gates multiplicative
    /// exponent-CANCELLING collection (`t^2 * t^-1 -> t` fills the NaN at `t = 0`).
    pub cert_finnz: Option<&'a dyn Fn(&Ex) -> bool>,
    /// The zero-set licence: NONZERO almost everywhere? Consumed by the odd-negative
    /// power distribution (see `pow`), whose identity breaks ONLY on fat zero sets --
    /// finiteness and NaN domains are irrelevant there (both spellings agree at every
    /// infinite/undefined configuration), so this is deliberately WEAKER than the
    /// `$`-certificate (`interval::zero_set_null` alone, not `finite_nonzero_ae`).
    pub cert_nzae: Option<&'a dyn Fn(&Ex) -> bool>,
    /// The nonconstant-entire certificate (`interval::nonconstant_entire`): licenses the
    /// SYMBOLIC-exponent occurrence merge (see `mul`'s like-base collection) on
    /// sign-indefinite bases by proving the exponent's level sets are null.
    pub cert_nce: Option<&'a dyn Fn(&Ex) -> bool>,
    /// LOSSY mode (the shipped `wildcard_all`): every gate in the constructors passes without
    /// certification. Training-corpus canonicalisation only, never an inference path.
    pub lossy: bool,
    /// SENTINEL EXPIRY (H-015 class (a), 2026-08-04): lossy parse deliberately keeps
    /// `inv(inf)`-class reciprocals unfolded so the mul bag can find their `$`-cancel
    /// partner (the mask-sentinel doctrine). That licence is for the SEARCH, not the
    /// answer: at the phase-1 fixpoint every surviving sentinel is unpartnered and IS
    /// the determined value 0 -- keeping it both inflates the endpoint and BLOCKS
    /// downstream rules (measured: 609/750 mode-ordering rows carried an inf/nan the
    /// sound endpoint lacked, e.g. a kept `C * inv(inf)` term whose sum never reached
    /// the shape a parity rule needed). The engine's phase-2 chain re-canonicalizes
    /// the fixpoint with this flag set -- `pow`'s determined fold fires again -- and
    /// keeps descending. Only meaningful with `lossy`; sound mode always folds.
    pub sentinels_expired: bool,
}

impl<'a> Cx<'a> {
    /// A certificate-free context (conversions, pattern handling, tests).
    pub fn bare(view: &'a TokenView<'a>) -> Self {
        Cx {
            view,
            cert_fin: None,
            cert_finnz: None,
            cert_nzae: None,
            cert_nce: None,
            lossy: false,
            sentinels_expired: false,
        }
    }

    /// Symbolic-exponent merge licence (see `mul`'s like-base collection): may n
    /// occurrences of `base^y` merge into `base^(n*y)`? The disagreement set is
    /// {base < 0} x {y in (1/n)Z \ Z} -- NaN on the unmerged side, defined on the
    /// merged one wherever n*y lands on an integer that y misses (the branch-cut
    /// question lives in the EXPONENT's level sets, not in the integer occurrence
    /// counts). Sound when the base never goes negative, or when y's level sets are
    /// certifiably NULL (nonconstant entire analyticity, identity theorem): a fat
    /// exponent pinned to 1/2 by an abs-construction, or an identically-1/2 unfolded
    /// analytic identity, is refused. Fail-closed. (Audit finding: the sibling of the
    /// pow-distribution bug, with the fat set in the exponent.)
    fn sym_merge_licensed(&self, base: &Ex, y: &Ex) -> bool {
        if self.lossy || self.certainly_nonneg(base) {
            return true;
        }
        match self.cert_nce {
            Some(f) => f(y),
            None => false,
        }
    }

    /// Is `e` CERTAINLY defined-and-finite (a.e.) everywhere, syntactically? Variables range
    /// over finite reals by domain; `Pi`/`E`/`Num` are finite literals; `Const` is a fitted REAL
    /// (finite). The certainty is CLOSED under finite-preserving structure: sums and products of
    /// finite reals are finite; non-negative integer powers preserve finiteness; the
    /// everywhere-finite unary functions preserve it. Anything else (wildcards, `log`, negative
    /// or fractional powers, ...) is never certain here -- it goes through `cert_fin`, the real
    /// interval machinery. A conservative subset by construction: syntactic certainty may say
    /// "no" where the certificate would say "yes", never the reverse.
    fn certainly_finite(&self, e: &Ex) -> bool {
        match e {
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::Const => true,
            Ex::Leaf(t) => self.view.sigil(*t) == 0,
            Ex::Add(v) | Ex::Mul(v) => v.iter().all(|x| self.certainly_finite(x)),
            Ex::Pow(b, ex) => {
                self.certainly_finite(b)
                    && matches!(&**ex, Ex::Num(r) if r.as_integer().map(|n| n >= 0).unwrap_or(false))
            }
            Ex::Fun(f, v) => {
                let s = self.view.resolve_owned(*f);
                // (D3: the retired `pow1_3`/`pow1_5` names were dropped from this list --
                // `from_prefix` maps those spellings to `rootn` unconditionally, so no
                // `Fun` ever carries them. `rootn` itself stays OUT: an even literal
                // index on a negative base is NaN, so it is not finiteness-preserving.)
                matches!(
                    s.as_str(),
                    "sin" | "cos" | "tanh" | "atan" | "abs" | "exp" | "sinh" | "cosh" | "asinh"
                ) && v.iter().all(|x| self.certainly_finite(x))
            }
            _ => false,
        }
    }

    /// `certainly_finite` for every element of a term key (a key is the non-coefficient factor
    /// list of an addend).
    fn fin_licensed(&self, e: &Ex) -> bool {
        if self.lossy || self.certainly_finite(e) {
            return true;
        }
        // Ground expressions have no measure space: a.e. tolerance degenerates to exactness,
        // and only `certainly_finite` (exact) may license them (see `has_measure_space`).
        if !e.has_measure_space(self.view) {
            return false;
        }
        match self.cert_fin {
            Some(f) => f(e),
            None => false,
        }
    }

    /// Finite-and-nonzero a.e. licence for a multiplicative base. Variables qualify (a variable
    /// is zero only on a null set); `Pi`/`E` are nonzero literals; nonzero `Num` likewise;
    /// `Const` does NOT qualify syntactically (`c = 0` is a reachable fitted value, and the
    /// licence's job is exactly to protect `c^1 * c^-1` at `c = 0`)... but the certificate may
    /// still pass it (the null set `{0}` in c-space is what `forall c` quantifies over -- we
    /// stay conservative and leave `Const` to the certificate).
    fn finnz_licensed(&self, e: &Ex) -> bool {
        if self.lossy {
            return true;
        }
        match e {
            Ex::Pi | Ex::E => true,
            Ex::Num(r) => !r.is_zero(),
            Ex::Leaf(t) => self.view.sigil(*t) == 0,
            _ => {
                if !e.has_measure_space(self.view) {
                    return false; // ground: no a.e. tolerance (see `has_measure_space`)
                }
                match self.cert_finnz {
                    Some(f) => f(e),
                    None => false,
                }
            }
        }
    }

    /// Zero-set-null licence for the odd-negative power distribution (see `pow`).
    /// `Const` QUALIFIES here ({c = 0} is a null set of the mask's own parameter space --
    /// unlike the SELFCANCEL licence, which deliberately protects the c = 0 atom);
    /// variables are zero on a null hyperplane; everything else goes to the structural
    /// zero-set certificate, fail-closed.
    pub(crate) fn nz_ae_licensed(&self, e: &Ex) -> bool {
        if self.lossy {
            return true;
        }
        self.nz_ae_certified(e)
    }

    /// The certificate body of [`nz_ae_licensed`] WITHOUT the lossy blanket: the answer
    /// the SOUND licence would give. Lossy-mode callers use it to ask "would sound-mode
    /// `pow` re-distribute this joined reciprocal?" (the `mul` rejoin arm, H-015) --
    /// inside lossy mode `nz_ae_licensed` is constant true and cannot pose that question.
    pub(crate) fn nz_ae_certified(&self, e: &Ex) -> bool {
        match e {
            Ex::Pi | Ex::E | Ex::Const => true,
            Ex::Num(r) => !r.is_zero(),
            Ex::Leaf(t) => self.view.sigil(*t) == 0,
            _ => match self.cert_nzae {
                Some(f) => f(e),
                None => false,
            },
        }
    }

    /// Is `e` CERTAINLY non-negative wherever it is defined, syntactically? Every clause is a
    /// theorem of the form "wherever this expression takes a value, that value is >= 0" --
    /// undefined (NaN-domain) points are outside the claim, so consumers must pair it with
    /// identities that agree on the undefined set. Conservative by construction: `false` means
    /// "not certain", never "certainly negative".
    ///
    /// Consumers: the branch-cut licence for merging NON-INTEGER rational exponents
    /// (`x^(1/2) * x^(1/2) -> x` is WRONG on `x < 0` -- NaN vs x, full measure -- but sound on
    /// a certainly-non-negative base), the pow licences, and `fun`'s constructor-level abs
    /// elimination (`abs t -> t`, a `_`-grade pointwise identity on such an operand).
    ///
    /// A useful induction the abs fold leans on: no clause below can EVALUATE to IEEE -0.0
    /// (even powers, exp/cosh, sums/products of such, ... all land on +0.0 when they vanish),
    /// so `abs t -> t` is exact even under signed-zero observation -- though the one-zero
    /// contract (SIMPLIFICATION_CONTRACT_v2 s9.2) already makes the zero sign non-normative.
    /// Can this subtree NEVER evaluate to an infinity? FAIL-CLOSED: leaves only.
    ///
    /// §9.1 quantifies variables, wildcard bindings and constants over the REALS, and §10.8's
    /// free-constant convention makes an infinity a LIMIT rather than an attained value, so no
    /// LEAF ever takes one as a value. Every COMPOUND is refused, and that refusal is the whole
    /// content of the predicate: `inv(x0)` is `+inf` at `x0 = 0` and `neg(inv(x0))` is `-inf`
    /// there, which is precisely the H-056 counterexample. `Ex::PosInf` / `Ex::NegInf` SPELL an
    /// infinity and are excluded for that reason; `Ex::NaN` is not a real value at all.
    ///
    /// Used by the power-of-power composition, whose odd-negative-exponent arm is sound exactly
    /// when the base cannot reach an infinity (see the licence comment in `pow`).
    fn never_infinite(&self, e: &Ex) -> bool {
        matches!(e, Ex::Leaf(_) | Ex::Const | Ex::Num(_) | Ex::Pi | Ex::E)
    }

    // (pub(crate) for the F80 E3 offline instrument `ac_odd_neg_carriers`, which
    // re-asks the deployed pow-distribution licence per factor; no semantic change.)
    /// Is `e` CERTAINLY without DEFINED zeros anywhere, syntactically? (F83, the
    /// odd-negative distribution's negative-coefficient guard: it needs
    /// zero-FREENESS, not zero-set-nullity -- the constructed exceptional point IS
    /// the null set the a.e. licence tolerates.) Zeros can also arrive through
    /// infinite arguments (`exp(-inf) = 0`; `pow(b, -q)` vanishes at b = +-inf),
    /// so clauses with that route demand `certainly_finite` of the argument.
    /// Conservative: `false` means "not certain", never "certainly vanishes".
    pub(crate) fn certainly_nonvanishing(&self, e: &Ex) -> bool {
        match e {
            Ex::Num(r) => !r.is_zero(),
            Ex::Pi | Ex::E => true,
            Ex::Mul(v) => v.iter().all(|f| self.certainly_nonvanishing(f)),
            Ex::Pow(b, q) => match &**q {
                Ex::Num(r) if r.as_integer().is_some_and(|n| n > 0) => {
                    self.certainly_nonvanishing(b)
                }
                Ex::Num(r) if r.as_integer().is_some_and(|n| n < 0) => {
                    // b^-n = 0 exactly where b = +-inf: demand finiteness too.
                    self.certainly_nonvanishing(b) && self.certainly_finite(b)
                }
                _ => false,
            },
            Ex::Fun(f, v) => {
                let s = self.view.resolve_owned(*f);
                match s.as_str() {
                    // cosh >= 1 on finite args, +inf at infinite ones: never 0.
                    "cosh" => true,
                    // exp(u) = 0 exactly at u = -inf: finite arguments only.
                    "exp" => v.iter().all(|x| self.certainly_finite(x)),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub(crate) fn certainly_nonneg(&self, e: &Ex) -> bool {
        match e {
            Ex::Num(r) => !r.is_negative(),
            Ex::Pi | Ex::E | Ex::PosInf => true,
            // Sums and products of certainly-non-negative parts: with no negative part
            // reachable there is no cancellation or sign flip; an infinite part drives the
            // bag to +inf or NaN, both within the claim.
            Ex::Add(v) | Ex::Mul(v) => v.iter().all(|t| self.certainly_nonneg(t)),
            // Even integer exponents force b^n >= 0 for ANY base; a certainly-non-negative
            // base stays non-negative under EVERY real exponent (0^negative = +inf under the
            // one-zero contract -- still non-negative).
            Ex::Pow(b, ex) => {
                matches!(&**ex, Ex::Num(r) if r.as_integer().map(|n| n % 2 == 0).unwrap_or(false))
                    || self.certainly_nonneg(b)
            }
            Ex::Fun(f, v) => {
                let s = self.view.resolve_owned(*f);
                match s.as_str() {
                    // Ranges within [0, +inf] for every argument: |.|, exp, cosh; acos
                    // ([0, pi]), acosh ([0, inf)).
                    "abs" | "exp" | "cosh" | "acos" | "acosh" => true,
                    // `rootn` was MISSING here, and the four `pow1_*` spellings that stood
                    // in its place were DEAD: `from_prefix` maps them to `rootn`
                    // unconditionally, so no `Fun` ever carries one. The gap is invisible
                    // on the shipped engine because the ruleset masks it for whatever
                    // shapes the mine happened to cover -- `abs(rootn(x,2))` folds by RULE,
                    // and a `rules=[]` engine keeps the `abs`.
                    "rootn" => {
                        v.len() == 2
                            && match &v[1] {
                                Ex::Num(r) => match r.as_integer() {
                                    // EVEN index: defined only on [0, inf) and its
                                    // principal value is non-negative there -- a range
                                    // fact, exactly like `acos`. Off-domain it is NaN,
                                    // which no licence reads as a sign. (A negative even
                                    // index is the reciprocal of one, still non-negative;
                                    // index 0 is excluded, it is not a root.)
                                    Some(n) if n != 0 && n % 2 == 0 => true,
                                    // ODD index: an increasing odd bijection on the whole
                                    // line, so it maps [0, inf) into [0, inf) -- the same
                                    // clause as `sinh`/`asinh` below.
                                    Some(_) => self.certainly_nonneg(&v[0]),
                                    None => false,
                                },
                                _ => false,
                            }
                    }
                    // GLOBALLY increasing odd functions map [0, inf) into [0, inf); asin is
                    // increasing on its whole domain. `tan` is deliberately absent: odd but
                    // periodic (tan 2 < 0).
                    "sinh" | "tanh" | "asinh" | "atan" | "asin" => {
                        v.len() == 1 && self.certainly_nonneg(&v[0])
                    }
                    // cos of a [-1, 1]-bounded argument: cos restricted to [-1, 1] is
                    // >= cos(1) > 0 (1 < pi/2).
                    "cos" => v.len() == 1 && self.certainly_in_unit(&v[0]),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Is `f` EVEN about the origin -- `f(-t) = f(t)` at every value, including the
    /// extension's? A CERTIFIED table, deliberately keyed by operator NAME and owned by the
    /// engine rather than declared in a config: this is a soundness licence, exactly like
    /// `certainly_nonneg` above, and a config author writing `parity: even` on an operator
    /// whose realization is not even would silently corrupt the canon.
    ///
    /// `abs`, `cos`, `cosh` -- and the extension agrees at both ends (`cos(-inf)` and
    /// `cos(inf)` are both NaN; `abs`/`cosh` both +inf).
    fn is_even_fun(&self, f: Tok) -> bool {
        self.view.tok_is(f, "abs") || self.view.tok_is(f, "cos") || self.view.tok_is(f, "cosh")
    }

    /// Is `f` ODD about the origin -- `f(-t) = -f(t)` at every value? The eight globally
    /// odd members of the vocabulary. Each holds on the extension: `sin`/`tan` propagate the
    /// sign and are NaN at both infinities; `sinh`/`tanh`/`asinh`/`atan` are odd bijections
    /// with matching signed limits; `asin`/`atanh` are odd on their domain and NaN off it on
    /// both sides.
    ///
    /// `inv` is DELIBERATELY ABSENT and its absence is load-bearing. It is odd on the reals
    /// minus the origin and NOT odd here: with one zero (contract §9.2) `neg(0) = 0`, so
    /// `inv(neg 0) = +inf` while `neg(inv 0) = -inf` (measured, not argued). That is §9.8.4's
    /// pole trilemma, which is a THEOREM: no total semantics with `+inf != -inf` keeps both
    /// pole symmetry and absorption, and this contract keeps absorption. Nothing is lost --
    /// the `Pow` arm of `drop_sign` takes POSITIVE odd exponents only, where `(-x)^n =
    /// -(x^n)` holds at 0 and at both infinities as well.
    pub(crate) fn is_odd_fun(&self, f: Tok) -> bool {
        odd_fun(self.view, f)
    }

    /// The representative of `e` in a SIGN-BLIND context -- a position whose consumer cares
    /// only about `|value|` -- or `None` when `e` is already it.
    ///
    /// Sign-blindness is a property of the CONTEXT, not of a node, and it PROPAGATES. An even
    /// function plants it (the arm in `fun` is the only place that does); from there:
    ///
    ///   * a CARRIED sign is dropped: a negative literal, `-inf`, a bag's negative
    ///     coefficient. `neg 3`, `-3` and `(-1)*x` are one case here and three token patterns
    ///     to a rule. That is the whole point -- this reads the canonical SIGN, never a `neg`
    ///     NODE, so the constructor's own fold of `neg(3)` into `-3` cannot blind it, which is
    ///     exactly how all thirty mined rules went blind at every numeric binding.
    ///   * an ODD function propagates sign-blindness INWARD: `f(-t) = -f(t)`, and the context
    ///     discards that sign anyway, so the recursion continues through it.
    ///   * an `abs` is a NO-OP and is stripped: its whole job was erasing a sign this context
    ///     erases too. `|tanh(|x|)| = |tanh(x)|` falls out of this line. The miner minted
    ///     exactly that rule while the arm stripped `abs` only at the TOP -- the gap was
    ///     found by the mine, not by inspection, and the fix was to make propagation the rule
    ///     rather than a special case.
    ///   * an INTEGER power propagates: an EVEN exponent erases the base's sign
    ///     (`(-u)^2 = u^2`), a POSITIVE ODD one carries it out. The negative odd half is
    ///     refused -- see `is_odd_fun` on the pole trilemma.
    ///
    /// Structural recursion, so it terminates; a projection, so it is idempotent.
    /// `-e` when `e` SYNTACTICALLY carries a negative sign, `None` otherwise. The one place
    /// that decides what "carries a sign" means, so the two consumers cannot drift: a
    /// sign-blind context drops it (`sign_blind_rep`) and a literal power base absorbs it
    /// (`pow`). The claim is about the SPELLING, not the value -- `Mul[-1, x0]` is positive
    /// at x0 < 0 -- which is exactly what both consumers need.
    fn carried_negative(&self, e: &Ex) -> Option<Ex> {
        match e {
            Ex::Num(r) if r.is_negative() => r.checked_neg().map(Ex::Num),
            Ex::NegInf => Some(Ex::PosInf),
            Ex::Mul(v) if v.iter().any(|m| matches!(m, Ex::Num(r) if r.is_negative())) => {
                let mut out = Vec::with_capacity(v.len());
                for m in v {
                    match m {
                        Ex::Num(r) if r.is_negative() => out.push(Ex::Num(r.checked_neg()?)),
                        other => out.push(other.clone()),
                    }
                }
                Some(mul(out, self))
            }
            _ => None,
        }
    }

    fn sign_blind_rep(&self, e: &Ex) -> Option<Ex> {
        if let Some(flipped) = self.carried_negative(e) {
            return Some(flipped);
        }
        match e {
            Ex::Fun(f, a) if a.len() == 1 && self.view.tok_is(*f, "abs") => {
                Some(self.sign_blind_rep(&a[0]).unwrap_or_else(|| a[0].clone()))
            }
            Ex::Fun(f, a) if a.len() == 1 && self.is_odd_fun(*f) => {
                self.sign_blind_rep(&a[0]).map(|s| fun(*f, vec![s], self))
            }
            Ex::Pow(b, ex) => {
                let ok = matches!(&**ex, Ex::Num(r)
                    if r.as_integer().is_some_and(|n| n % 2 == 0 || n > 0));
                if !ok {
                    return None;
                }
                self.sign_blind_rep(b).map(|s| pow(s, (**ex).clone(), self))
            }
            _ => None,
        }
    }

    /// Does `e` certainly take values in [-1, 1] wherever defined? Range facts only:
    /// sin/cos/tanh land there for EVERY argument. (Literal arguments never reach this
    /// predicate -- ground trig compounds fold numerically before licences run.)
    fn certainly_in_unit(&self, e: &Ex) -> bool {
        match e {
            Ex::Fun(f, _) => {
                let s = self.view.resolve_owned(*f);
                matches!(s.as_str(), "sin" | "cos" | "tanh")
            }
            _ => false,
        }
    }

    /// `outer(inner(t))` collapsed, when the pair is an INVERSE pair whose licence holds --
    /// `None` otherwise. The table and every licence in it was adjudicated by the contract
    /// judge (`verify._contract.judge_rule`) BEFORE this was written, and the judge's answers
    /// are what the three groups below are:
    ///
    ///   TOTAL, no licence (CERTIFIED unlicensed) -- the genuine bijections of the extended
    ///   line. `acosh o cosh` is in this group but is NOT the identity: `cosh` is even, so the
    ///   collapse is `|t|`, and the judge certifies exactly that shape.
    ///
    ///   LICENSED (KILL unlicensed, CERTIFIED with the licence) -- `exp o log` extends NaN to
    ///   a defined value on `t < 0`, and the three arcsin-side pairs do the same off `[-1,1]`.
    ///   Both licences already exist and are used verbatim: `certainly_nonneg` (which is also
    ///   why the artifact's `exp log` rules cover exactly `abs`/`acos`/`acosh`/`cosh` -- that
    ///   IS the lattice's true-set, enumerated by the mine one member at a time) and
    ///   `certainly_in_unit`.
    ///
    ///   REFUSED ENTIRELY. `asin o sin`, `acos o cos` and `atan o tan` are NOT identities in
    ///   any direction -- the judge convicts them under clause (a), a REAL CHANGE, because
    ///   they fold the argument into the inverse's range (`asin(sin 4) = pi - 4`). The
    ///   artifact's rules for these shapes always carry a range-bounded inner argument and
    ///   say something different; none of them belongs here. `tan o atan` is left out too: it
    ///   judges TOLERATED (the null-set pole at `atan(inf) = pi/2`), and a tolerated identity
    ///   is a RULE-level allowance -- the shipped `tan atan !0 -> !0` keeps it under a
    ///   finiteness certificate, which is where it should stay until an owner says otherwise.
    ///
    /// Every arm drops two operator nodes, so the descent is unconditional.
    /// Is `m` a pi member of an Add bag whose coefficient is EXACTLY +-1?
    ///
    /// The coefficient bound is the whole scope of the pi-shift arm. In a canonical bag like
    /// terms merge, so `t + pi + pi` is a COEFFICIENT-2 pi member, and folding that is
    /// 2pi-PERIODICITY -- genuine argument reduction modulo the period, the new concept the
    /// owner deferred. Keyed on +-1 the arm cannot reach it. There is no corpus pressure for
    /// the wider fact either: `sin + x0 * 2 np.pi` is six tokens, past the mine's source cap.
    fn is_unit_pi(m: &Ex) -> bool {
        match m {
            Ex::Pi => true,
            // `-pi` is a two-factor product and nothing else: a third factor (`-pi*x0`) is a
            // different member and must not strip.
            Ex::Mul(v) => {
                v.len() == 2
                    && v.iter().any(|x| matches!(x, Ex::Pi))
                    && v.iter()
                        .any(|x| matches!(x, Ex::Num(r) if *r == Rat::NEG_ONE))
            }
            _ => false,
        }
    }

    /// `u` with one coefficient-+-1 `pi` member removed, or `None`. Reads the CANONICAL bag,
    /// which is the entire point: at a negative literal the constructor factors the bag's
    /// common sign, so `(-2) - pi` is stored as `neg(2 + pi)` and the `-pi` member a pattern
    /// rule needs no longer exists. That is why `sin - _0 np.pi -> neg sin _0` fires at a
    /// variable and at `2` but goes blind at `(-2)` -- C1.19 family A's defect, one family over.
    ///
    /// The negated-bag case needs no parity argument: `u = -(v' + pi) = (-v') - pi`, so
    /// stripping under the negation and RE-NEGATING the remainder lands on the same shift law
    /// as the direct case. One rule for both.
    fn strip_unit_pi(&self, u: &Ex) -> Option<Ex> {
        let strip = |ms: &Vec<Ex>| -> Option<Ex> {
            let i = ms.iter().position(Self::is_unit_pi)?;
            let rest: Vec<Ex> = ms
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, m)| m.clone())
                .collect();
            Some(add(rest, self))
        };
        if let Ex::Add(ms) = u {
            return strip(ms);
        }
        if let Some(Ex::Add(ms)) = self.carried_negative(u) {
            let stripped = strip(&ms)?;
            return Some(mul(vec![Ex::Num(Rat::NEG_ONE), stripped], self));
        }
        None
    }

    fn inverse_pair_collapse(&self, outer: Tok, inner: Tok, t: &Ex) -> Option<Ex> {
        let o = self.view.resolve_owned(outer);
        let i = self.view.resolve_owned(inner);
        match (o.as_str(), i.as_str()) {
            ("log", "exp") | ("asinh", "sinh") | ("sinh", "asinh") | ("atanh", "tanh") => {
                Some(t.clone())
            }
            ("acosh", "cosh") => Some(fun(self.view.intern("abs"), vec![t.clone()], self)),
            ("exp", "log") if self.certainly_nonneg(t) => Some(t.clone()),
            ("cos", "acos") | ("sin", "asin") | ("tanh", "atanh") if self.certainly_in_unit(t) => {
                Some(t.clone())
            }
            _ => None,
        }
    }
}

/// STRIPPED comparators -- the bag orders of the canonical form.
///
/// Add terms compare by their COEFFICIENT-STRIPPED structural key (the `term_split` key),
/// with constant-like terms (pure literals, `Const`) LAST, ties broken by the coefficient.
/// Mul factors compare by their EXPONENT-STRIPPED base (the `factor_split` base), with the
/// rational coefficient FIRST, ties broken by the exponent.
///
/// Stripping is LOAD-BEARING for the normal form, not aesthetics: under the raw structural
/// order a term's position depends on its coefficient's sign (`Mul[-2, a]` sorts before
/// `Mul[3, b]` because -2 < 3), so "make the lead coefficient positive" ALTERNATES on bags
/// like `-2a + 3b` -- negating every term re-sorts a different term to the front, negative
/// again. With stripped keys the lead term of a bag and of its negation is the SAME term,
/// and the sign rule of the primitive form (below) is well-defined. Two side benefits: sign
/// or exponent edits no longer relocate members (edit locality), and sums read in the
/// classical polynomial order (`x + 1`, not `1 + x`).
fn is_constlike(e: &Ex) -> bool {
    matches!(
        e,
        Ex::Num(_) | Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN | Ex::Const
    )
}

pub fn add_term_cmp(a: &Ex, b: &Ex, view: &TokenView) -> Ordering {
    let (ca, ka) = term_split(a.clone(), view);
    let (cb, kb) = term_split(b.clone(), view);
    let ga = is_constlike(&ka) as u8;
    let gb = is_constlike(&kb) as u8;
    ga.cmp(&gb)
        .then_with(|| cmp_ex(&ka, &kb, view))
        .then_with(|| ca.cmp_exact(&cb))
}

pub fn mul_factor_cmp(a: &Ex, b: &Ex, view: &TokenView) -> Ordering {
    // The rational coefficient (a bare Num) leads the product; everything else by base.
    let ga = !matches!(a, Ex::Num(_)) as u8;
    let gb = !matches!(b, Ex::Num(_)) as u8;
    if ga != gb {
        return ga.cmp(&gb);
    }
    let (ba, ea) = factor_split_ref(a);
    let (bb, eb) = factor_split_ref(b);
    cmp_ex(ba, bb, view).then_with(|| cmp_ex(ea, eb, view))
}

/// Borrowing (base, exponent) view of a factor for the comparator (`x` is `x^1`).
fn factor_split_ref(e: &Ex) -> (&Ex, &Ex) {
    static ONE: Ex = Ex::Num(Rat::ONE);
    match e {
        Ex::Pow(b, ex) => (b, ex),
        other => (other, &ONE),
    }
}

/// Split an addend into `(rational coefficient, term key)`. The key is what like-term collection
/// compares: `Mul[3, x, y] -> (3, Mul[x, y])`, `Mul[x, y] -> (1, Mul[x, y])`, `sin(x) -> (1,
/// sin(x))`. A `Mul` whose coefficient is 0 (the uncertified `0 * t` form) is NOT split -- it
/// stays an opaque term so collection cannot manufacture a licence it does not have.
///
/// The key is SIGN-NORMALIZED (audit B19): an odd function's negative-literal sign
/// lifts out of the key into the coefficient (`sin(-2)` splits as `(-1, sin(2))`;
/// `Mul[5, sin(-2), x0]` as `(-5, Mul[sin(2), x0])`), so the two spellings of one
/// value share a collection key and `sin(-2) + sin(2)` cancels IN ONE PASS instead of
/// needing a render/re-parse cycle. `term_join` re-sinks the sign through the
/// trade-site route (`mul()` -> `sign_place`), so split/join still round-trips onto
/// the canonical spelling. Scope: top-level factors with a literal argument only -- a
/// sign under a `Pow` base is structure, not a spelling choice, and stays put.
fn term_split(e: Ex, view: &TokenView) -> (Rat, Ex) {
    // Flip `f(-r)` -> `f(r)` in place for odd f; true iff a sign was extracted. RAW
    // rebuild on purpose: keys are comparison objects, and every re-JOIN routes
    // through `fun()` (which folds boundary grounds); a fold here would change the
    // key's shape mid-collection.
    fn unsign(e: &mut Ex, view: &TokenView) -> bool {
        if let Ex::Fun(f, args) = e {
            if args.len() == 1 && odd_fun(view, *f) {
                if let Ex::Num(r) = &args[0] {
                    if r.is_negative() {
                        if let Some(p) = r.checked_neg() {
                            *e = Ex::Fun(*f, vec![Ex::Num(p)]);
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
    let (c, key) = match e {
        Ex::Mul(v) => match v.first() {
            Some(Ex::Num(r)) if !r.is_zero() => {
                let r = *r;
                let mut rest = v;
                rest.remove(0);
                let key = if rest.len() == 1 {
                    rest.pop().unwrap()
                } else {
                    Ex::Mul(rest) // still flat + sorted: removing the leading Num keeps both
                };
                (r, key)
            }
            _ => (Rat::ONE, Ex::Mul(v)),
        },
        other => (Rat::ONE, other),
    };
    // The coefficient must be able to host the sign before any factor flips (the
    // i128::MIN edge fails safe: unnormalized split, exactly the pre-B19 behavior).
    let Some(nc) = c.checked_neg() else {
        return (c, key);
    };
    let mut key = key;
    let mut flips = 0usize;
    match &mut key {
        Ex::Mul(v) => {
            for f in v.iter_mut() {
                if unsign(f, view) {
                    flips += 1;
                }
            }
            if flips > 0 {
                // A flip edits a factor's content; re-sort so the key stays canonical
                // for equality comparison against keys split from other spellings.
                v.sort_by(|a, b| mul_factor_cmp(a, b, view));
            }
        }
        k => {
            if unsign(k, view) {
                flips += 1;
            }
        }
    }
    (if flips % 2 == 1 { nc } else { c }, key)
}

/// Negate an addend exactly. Literals and infinities negate DIRECTLY; everything else
/// negates its coefficient through [`term_split`]/[`term_join`]. TOTAL: `(-1) * t` IS
/// coefficient negation in the bag representation. `None` only on the i128::MIN edge.
/// Used by the DISPLAY emitters for sign redistribution of `Mul[-1, Add]`.
pub(crate) fn negate_term(t: &Ex, cx: &Cx) -> Option<Ex> {
    match t {
        Ex::Num(r) => Some(Ex::Num(r.checked_neg()?)),
        Ex::PosInf => Some(Ex::NegInf),
        Ex::NegInf => Some(Ex::PosInf),
        Ex::NaN => Some(Ex::NaN),
        _ => {
            let (c, key) = term_split(t.clone(), cx.view);
            Some(term_join(c.checked_neg()?, key, cx))
        }
    }
}

/// Scale an addend by a rational, exactly (`None` on overflow). The distribution step of the
/// Add flatten (see the `Mul[Num, Add]` arm there).
///
/// The ZERO product routes through `mul()` (H-031): `term_join` is canonical-form-
/// preserving only for NONZERO coefficients -- `term_join(0, x0)` would mint `Mul[0, x0]`,
/// a product `mul()` itself collapses (x0 is certainly finite), i.e. a locally
/// non-canonical transient. Every exposure surface happened to heal such terms bottom-up
/// (entry `canon` re-walks the parse output, the pass rebuild re-walks every child), so
/// this was never a live defect -- but the canonicity of scaled terms should be local,
/// not a consequence of who re-walks them. `mul()` owns the kept-zero licence logic
/// (licensed factors collapse, unlicensed keep `Mul[0, t]`), so the zero case lands on
/// exactly the state the healing used to reach.
fn scale_term(t: &Ex, r: &Rat, cx: &Cx) -> Option<Ex> {
    match t {
        Ex::Num(v) => Some(Ex::Num(v.checked_mul(r)?)),
        Ex::Const => {
            if r.is_zero() {
                // 0 * c = 0 EXACTLY (masked constants are FINITE by doctrine): the
                // structural zero must not launder into the fittable class -- the
                // forall direction fails there (the output realizes c' + .. while the
                // input realizes only .. a.e.). The sibling zero guards (term_join's
                // kept-zero absorption) carry exactly this exemption; this arm was the
                // one path around them (audit finding, constructor-licence sweep).
                return Some(Ex::Num(Rat::ZERO));
            }
            Some(Ex::Const) // r * c refits to c' (forall-exists)
        }
        Ex::PosInf | Ex::NegInf => {
            if r.is_zero() {
                return None; // 0 * inf: leave to the caller's exact paths
            }
            let flip = r.is_negative();
            Some(match (t, flip) {
                (Ex::PosInf, false) | (Ex::NegInf, true) => Ex::PosInf,
                _ => Ex::NegInf,
            })
        }
        Ex::NaN => Some(Ex::NaN),
        _ => {
            let (c, key) = term_split(t.clone(), cx.view);
            let p = c.checked_mul(r)?;
            if p.is_zero() {
                // r == 0 (term_split never yields c == 0): the kept-zero licence
                // decision belongs to mul() -- see the doc above.
                return Some(mul(vec![Ex::Num(Rat::ZERO), key], cx));
            }
            Some(term_join(p, key, cx))
        }
    }
}

/// Rebuild an addend from `(coefficient, key)` -- the inverse of [`term_split`], preserving
/// canonical form without a full re-canon (the key is canonical and `Num` sorts first).
/// The odd-function table, shared by the constructors (`term_join`'s sign fusion), the
/// parity machinery and the infix renderer's sign hoist -- ONE table, or it is the
/// five-arity-tables disease again.
pub(crate) fn odd_fun(view: &TokenView, f: Tok) -> bool {
    const ODD: [&str; 8] = [
        "sin", "sinh", "tan", "tanh", "asin", "asinh", "atan", "atanh",
    ];
    ODD.iter().any(|s| view.tok_is(f, s))
}

fn term_join(c: Rat, key: Ex, cx: &Cx) -> Ex {
    // An ODD function of a LITERAL argument owns any adjacent SIGN (owner ruling
    // 2026-08-08; I4): `-1 * sin(2)` joins as `sin(-2)`, and `-5 * sin(2)` as
    // `5 * sin(-2)`. HOISTED (audit B5+B19): the fusion is no longer a private arm
    // here -- an odd function of a literal is a SIGN-TRADE SITE
    // (`sign_trade_flip`'s literal arm), so a negative coefficient joining such a
    // key routes through the full `mul()` assembly below (the trade-site
    // conditions), where the shared orbit owner prices every route onto ONE
    // spelling. The private arm fused only on THIS path, so `mul()`-built and
    // collector-built states disagreed (`-5 * sin(2)` vs `5 * sin(-2)`: two
    // fixpoints for one value, construction-history dependence).
    if c.is_one() {
        return key;
    }
    // A bare `Const` FACTOR absorbs any rational coefficient (`c * C` refits to `C'`, the
    // contract's forall-exists direction), so a Const-bearing product term canonically
    // carries NO coefficient -- and in particular no SIGN. Joining through here (collection
    // rebuilds, scaling, negation) must respect that, or raw `Mul[-1, Const, ...]` terms
    // leak into bags and the serialized sign flip-flops across parse (a term that prints
    // subtracted re-parses added once the constructor absorbs the -1).
    match key {
        // c * nan = nan for every rational c: total, fold outright.
        Ex::NaN => Ex::NaN,
        // A bare `Const` absorbs a NONZERO coefficient (the contract's forall-exists
        // refit) -- but an exact ZERO stays OUTSIDE, mirroring `mul()`: absorbing a
        // STRUCTURAL zero into the constant class would hand the fitter a fittable
        // constant where mathematics has an exact 0 (the masking-soundness principle).
        Ex::Const if !c.is_zero() => Ex::Const,
        Ex::Const => Ex::Mul(vec![Ex::Num(c), Ex::Const]),
        // An infinity absorbs the coefficient's magnitude and takes its sign, EXACTLY
        // (`c * inf = sign(c) * inf` for rational c != 0): joining a raw `Num` next to an
        // Inf factor would leave an unfolded product whose serialization is not
        // parse-stable (the constructors fold it on re-parse).
        Ex::PosInf | Ex::NegInf => {
            // 0 * (+-inf) = nan: the IEEE invalid operation, total -- no certificate
            // needed. The zero must never be absorbed sign-only.
            if c.is_zero() {
                return Ex::NaN;
            }
            let pos = matches!(key, Ex::PosInf);
            if c.is_negative() == pos {
                Ex::NegInf
            } else {
                Ex::PosInf
            }
        }
        Ex::Mul(v) => {
            // A KEPT-ZERO product (`Mul[0, t]`, the unlicensed zero-collapse) absorbs any
            // rational coefficient EXACTLY: `c * (0*t) = (c*0)*t = 0*t` pointwise, and the
            // contract has ONE zero, so there is no sign to keep. Threading the coefficient
            // through instead would mint a non-canonical `Mul[c, 0, t]` twin: `term_split`
            // reads a kept-zero term as coefficient +1 (its `!r.is_zero()` guard), so the
            // decorated twin reads `c` -- and with `c = -1` from `negate_term` the sum
            // orientation comparison loses flip-antisymmetry, leaving TWO stable sign
            // spellings around such terms (the 5 idempotence failures across the 64k/1M
            // gates: rows 29663/37873/59042 + 274133/514869).
            if matches!(v.first(), Some(Ex::Num(r)) if r.is_zero()) {
                return Ex::Mul(v);
            }
            if let Some(pos) = v.iter().position(|f| matches!(f, Ex::PosInf | Ex::NegInf)) {
                // 0 * (inf * t) = nan EVERYWHERE: t finite-nonzero gives 0 * (+-inf),
                // t = 0 gives 0 * nan, t = nan propagates. Pointwise-total, so the raw
                // path may fold it without a certificate; the zero must never be
                // absorbed sign-only.
                if c.is_zero() {
                    return Ex::NaN;
                }
                // Fold the coefficient into the Inf factor (sign only; magnitude absorbed).
                if c.is_negative() {
                    let mut v = v;
                    v[pos] = if matches!(v[pos], Ex::PosInf) {
                        Ex::NegInf
                    } else {
                        Ex::PosInf
                    };
                    return Ex::Mul(v);
                }
                return Ex::Mul(v);
            }
            if v.iter().any(|f| matches!(f, Ex::Const)) {
                // A NONZERO coefficient is absorbed by the constant (forall-exists);
                // an exact ZERO stays OUTSIDE, mirroring `mul()` (see the bare-Const
                // arm: a structural zero must never launder into the constant class).
                if c.is_zero() {
                    let mut v = v;
                    v.insert(0, Ex::Num(c));
                    return Ex::Mul(v);
                }
                return Ex::Mul(v);
            }
            // F63: a NEGATIVE coefficient joining a bag that holds a sign-trade site
            // must land in the canonical placement, or this join mints the spelling
            // the orientation machinery prices away -- negate_term/scale_term/
            // divide_terms all build through here, and a priced-vs-built gap here is
            // construction-history dependence one level down (corpus row 120). The
            // route is the FULL `mul()` assembly, not `sign_place` alone: the H-020/
            // H-030 absorption arms outrank the trade owner, and a join that skips
            // them rests the sign beside an absorbing sum that `mul()`-built states
            // fold it into (caught by the h030 confluence pin). Safe: `mul()` never
            // calls `term_join` (only `add()` does), so the recursion strictly
            // descends. Positive coefficients never trade (a trade would mint the
            // sign it sheds), so the raw join stays.
            //
            // F72: a key that itself CARRIES Num members (an i128 overflow partition
            // bag) also takes the full assembly, for BOTH coefficient signs: the raw
            // coefficient-first insert is canonical only under "Num sorts first",
            // and with several Nums in one bag their relative order and the sign
            // HOST are `mul()`'s sign-factored accumulation's to decide -- a raw
            // `Mul[-P, N]` beside the canonical `Mul[-N, P]` is exactly the P1
            // route-dependence one join away.
            if v.iter().any(|f| matches!(f, Ex::Num(_)))
                || (c.is_negative() && v.iter().any(|f| is_sign_trade_site(f, cx)))
            {
                let mut items = Vec::with_capacity(v.len() + 1);
                items.push(Ex::Num(c));
                items.extend(v);
                let placed = mul(items, cx);
                debug_assert!(
                    !matches!(placed, Ex::Add(_)),
                    "term_join produced a bare Add term: {placed:?}"
                );
                return placed;
            }
            let mut v = v;
            v.insert(0, Ex::Num(c));
            Ex::Mul(v)
        }
        k => {
            if matches!(k, Ex::Num(_)) || (c.is_negative() && is_sign_trade_site(&k, cx)) {
                // A bare-Num key (the other partition-bag shape, F72) folds or
                // re-hosts through `mul()` exactly like the Num-bearing Mul above.
                let placed = mul(vec![Ex::Num(c), k], cx);
                debug_assert!(
                    !matches!(placed, Ex::Add(_)),
                    "term_join produced a bare Add term: {placed:?}"
                );
                return placed;
            }
            Ex::Mul(vec![Ex::Num(c), k])
        }
    }
}

/// Split a factor into `(base, rational exponent | symbolic exponent occurrence)`.
enum FactorExp {
    Rat(Rat),
    /// A symbolic exponent, counted by occurrence: `x^y * x^y -> (x, Sym(y), count 2)`.
    Sym(Ex),
}

fn factor_split(e: Ex, cx: &Cx) -> (Ex, FactorExp) {
    // An EVEN `rootn(b, n)` reads as the rational power `b^(1/n)`: at an even index the
    // root IS the principal root, so the two agree pointwise (both NaN on negatives) and
    // exponent collection may treat it as a power -- this is what keeps
    // `sqrt(x) * x -> x^(3/2)` working now that even roots are no longer spelled as powers.
    // An ODD root must NEVER be read this way: `rootn(-8,3) = -2` where `(-8)^(1/3)` is
    // NaN, so they are different functions and collecting them would fabricate values.
    if let Ex::Fun(op, args) = &e {
        if args.len() == 2 && cx.view.tok_is(*op, "rootn") {
            if let Ex::Num(idx) = &args[1] {
                if let Some(n) = idx.as_integer() {
                    if n >= 2 && n % 2 == 0 {
                        if let Some(r) = Rat::new(1, n) {
                            let Ex::Fun(_, args) = e else { unreachable!() };
                            let mut it = args.into_iter();
                            return (it.next().unwrap(), FactorExp::Rat(r));
                        }
                    }
                }
            }
        }
    }
    match e {
        Ex::Pow(b, ex) => match *ex {
            Ex::Num(r) => (*b, FactorExp::Rat(r)),
            sym => (*b, FactorExp::Sym(sym)),
        },
        other => (other, FactorExp::Rat(Rat::ONE)),
    }
}

/// `(base, rational exponent)` when `e` reads faithfully as a rational power. Same rule as
/// `factor_split`: a `Pow` with a rational exponent always, an EVEN `rootn` as `b^(1/n)`,
/// an odd `rootn` never.
fn as_rational_power<'a>(e: &'a Ex, cx: &Cx) -> Option<(&'a Ex, Rat)> {
    match e {
        Ex::Pow(b, ex) => match &**ex {
            Ex::Num(r) => Some((&**b, *r)),
            _ => None,
        },
        Ex::Fun(op, args) if args.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            let Ex::Num(idx) = &args[1] else {
                return None;
            };
            let n = idx.as_integer()?;
            if n >= 2 && n % 2 == 0 {
                Rat::new(1, n).map(|r| (&args[0], r))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A POWER OF `exp` IS AN `exp`: `(e^a)^b = e^(a*b)`. Returns the composed `exp(a*b)` when
/// `base` reads faithfully as a power of Euler's number AND the product folds to a single
/// node; `None` otherwise. `exp(a)` IS `E^a` and the leaf `E` is `E^1`, so this is the same
/// "reads faithfully as a power" idea `as_rational_power` implements for `Pow` and `rootn`
/// -- it needs its own entry point only because an `exp` exponent is an arbitrary SUBTREE,
/// not a rational, so it does not fit that helper's signature.
///
/// Both power spellings call it: `pow(base, b)` passes `b`, and `rootn(base, n)` passes
/// `1/n`. That is what makes the reduction route-independent (H-057's lesson) and what
/// makes `rootn(e, n) -> exp(1/n)` -- shipped 2026-08-07 as a single-BASE arm -- one
/// INSTANCE of a principle rather than a patch for one shape.
///
/// THE FOLD CONDITION IS BOTH GATES AT ONCE, which is why the arm needs no licence:
///
///   * mu. With a symbolic `a` the product needs a `Mul` node and the composition stops
///     paying: `pow(exp x0, x1)` and `exp(x0*x1)` both price 32,000, and through the root
///     spelling the composition is an outright ASCENT -- `rootn(exp x0, 3)` is 26,000
///     against 27,000 for `exp(x0/3)`. A constructor arm that does not descend has no
///     business firing, so that case stays with the RULES: `pow exp (-1) _0 -> exp neg _0`
///     (a non-unit literal `a` against a symbolic `b`) is shipped and still needed. The
///     `E`-base rule `pow np.e _0 -> exp _0` is NOT -- `a = 1` needs no fold, so the arm
///     covers every exponent and the mine can no longer mint it.
///     Inside the fold condition the descent is universal, not merely measured: one whole
///     operator node disappears, and §10.10's `L(n) = log2(1+|n|)` is subadditive on the
///     numerator and the denominator alike, so the surviving literal never costs more than
///     the two it replaces.
///   * soundness. The one configuration where `(e^a)^b` and `e^(a*b)` disagree is `0 * inf`
///     in the exponent: at `a = -inf, b = 0` the source is `0^0 = 1` while the target is
///     `e^nan = nan` -- defined -> undefined, which §9.1's R2 forbids at ZERO measure
///     tolerance (the same trap the pow-of-pow arm above guards with `never_infinite`).
///     `Ex::Num` is finite by construction (`PosInf`/`NegInf` are their own variants) and
///     `a = 1` is finite, so inside the fold condition that configuration is unreachable.
///
/// TOTAL wherever it fires: `exp` is strictly positive on the whole real line, so the odd
/// signed root and the even principal root coincide with the power on this base, and every
/// non-finite argument agrees on both sides (`a = -inf` gives 0 = 0 for `b > 0` and
/// +inf = +inf for `b < 0` under the one-zero convention; NaN propagates on both).
///
/// GATED on the config declaring `exp`: the rules this subsumes were gated for free by the
/// config's own vocabulary, and a constructor arm is not.
fn compose_e_power(base: &Ex, exponent: &Ex, cx: &Cx) -> Option<Ex> {
    // `None` marks the bare leaf `E`, i.e. `a = 1`, where the product IS `exponent` for
    // ANY exponent -- no `Mul` node is created, so the fold condition holds unconditionally
    // and this is the arm that drops the `E` leaf outright (`pow(e, x0)` 24,000 -> 16,000).
    let a = match base {
        Ex::E => None,
        Ex::Fun(op, args) if args.len() == 1 && cx.view.tok_is(*op, "exp") => Some(&args[0]),
        _ => return None,
    };
    let composed = match a {
        None => exponent.clone(),
        // `exp(1)` denotes the same value as `E` and takes the same unconditional path.
        // UNREACHABLE while the write half holds -- `fun` folds `exp(1)` to `E`, so no
        // such node exists -- and kept anyway: this helper's whole contract is that the
        // two spellings behave identically, and an arm that states it cannot go stale
        // the way a comment can.
        Some(Ex::Num(ra)) if ra.is_one() => exponent.clone(),
        Some(Ex::Num(ra)) => match exponent {
            Ex::Num(rb) => Ex::Num(ra.checked_mul(rb)?),
            _ => return None,
        },
        Some(_) => return None,
    };
    let exp_tok = cx.view.intern("exp");
    if cx.view.arity(exp_tok) != Some(1) {
        return None;
    }
    // Through `fun`, so a composed exponent of exactly 1 comes back out as the leaf `E`
    // rather than as `exp(1)` -- the two halves of the invariant meeting.
    Some(fun(exp_tok, vec![composed], cx))
}

/// The canonical n-ary addition constructor. See the module doc for the invariants; the gates:
///
/// * NaN element -> NaN (TOTAL: `nan + t = nan` for every extended-real `t`).
/// * `+inf` and `-inf` both present -> NaN (TOTAL: the pair alone forces it pointwise).
/// * One infinity present: it absorbs the rational accumulator and every term with a
///   finite-a.e. licence; absorbing an UNLICENSED term is refused (`inf + log(x)` stays a bag:
///   at `x = 0` the truth is `inf + -inf = nan`, not `inf`).
/// * Like terms merge SAME-SIGN freely (TOTAL: same-sign coefficient addition commutes with the
///   `t = +-inf` points). OPPOSITE-SIGN merging cancels mass and needs the `!`-licence on the
///   key (`2t - t` at `t = +inf` is `nan`; `t` is `+inf`; sound only where that hole is null).
/// * `Const`-containing keys never merge with anything (independence), EXCEPT bare `r*Const`
///   terms, which all collapse into ONE `Const` absorbing the rational accumulator
///   (`c1 + c2 + 5 = c3`, the contract's forall-exists direction).
pub fn add(items: Vec<Ex>, cx: &Cx) -> Ex {
    // Flatten (Flat) + literal scan.
    let mut lits: Vec<Rat> = Vec::new();
    let mut acc_overflow: Vec<Ex> = Vec::new(); // Num partials an overflowed accumulator emitted
    let mut pos_inf = false;
    let mut neg_inf = false;
    let mut has_const = false;
    let mut terms: Vec<Ex> = Vec::new();
    let mut stack: Vec<Ex> = items;
    stack.reverse();
    while let Some(e) = stack.pop() {
        match e {
            Ex::NaN => return Ex::NaN,
            Ex::Add(v) => stack.extend(v.into_iter().rev()),
            // A FACTORED sum entering an Add context unfactors: `r * (a + b)` contributes the
            // terms `ra, rb` (TOTAL: distribution of a finite rational over extended-real
            // sums holds pointwise). Without this, extraction on an inner sub-sum would
            // produce a `Mul[r, Add]` term the enclosing bag cannot see through, and the
            // canonical form would depend on ASSOCIATION order (`(2a + 2b) + c` vs
            // `2a + (2b + c)`). Extraction decisions are thereby always made once, on the
            // complete bag. Scaled terms re-enter the flatten stack so every classification
            // (literals, Const, infinities, nested sums) reruns on them.
            Ex::Mul(v)
                if v.len() == 2 && matches!(&v[0], Ex::Num(_)) && matches!(&v[1], Ex::Add(_)) =>
            {
                let (r, inner) = match (&v[0], &v[1]) {
                    (Ex::Num(r), Ex::Add(inner)) => (*r, inner.clone()),
                    _ => unreachable!(),
                };
                let mut scaled: Vec<Ex> = Vec::with_capacity(inner.len());
                let mut ok = true;
                for t in &inner {
                    match scale_term(t, &r, cx) {
                        Some(s) => scaled.push(s),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    stack.extend(scaled.into_iter().rev());
                } else {
                    terms.push(Ex::Mul(v)); // overflow: keep the factored term opaque
                }
            }
            Ex::Num(r) => lits.push(r),
            Ex::PosInf => pos_inf = true,
            Ex::NegInf => neg_inf = true,
            Ex::Const => has_const = true,
            other => terms.push(other),
        }
    }
    if pos_inf && neg_inf {
        return Ex::NaN;
    }

    // DETERMINISTIC literal accumulation: same rationale (and same shape) as the mul()
    // coefficient fold above -- sorted-order summation makes the overflow partition a
    // function of the multiset, not of the input spelling.
    lits.sort_unstable_by(|a, b| a.cmp_exact(b));
    let mut acc = Rat::ZERO;
    for r in lits {
        match acc.checked_add(&r) {
            Some(t) => acc = t,
            None => {
                acc_overflow.push(Ex::Num(acc));
                acc = r;
            }
        }
    }

    // Like-term collection: (key, positive-coefficient sum, negative-coefficient sum), in
    // first-seen order for determinism (the final sort canonicalizes anyway).
    struct Bucket {
        key: Ex,
        /// ALL same-key coefficients, collected raw; the fold happens at rebuild in
        /// SORTED order (B3b: an arrival-order fold made the overflow partition depend
        /// on the input spelling).
        coeffs: Vec<Rat>,
    }
    let mut buckets: Vec<Bucket> = Vec::new();
    let mut const_terms = false; // any r*Const term seen (merges into the bare-Const pool)
    for t in terms {
        let (c, key) = term_split(t, cx.view);
        if key == Ex::Const {
            // r*Const: joins the bare-Const pool (c1*r1 + c2*r2 = c3, forall-exists).
            const_terms = true;
            continue;
        }
        if key.contains_const() {
            // Independent constants: never merge; keep verbatim (a one-element bucket).
            buckets.push(Bucket {
                key,
                coeffs: vec![c],
            });
            continue;
        }
        match buckets
            .iter_mut()
            .find(|b| !b.key.contains_const() && b.key == key)
        {
            Some(b) => b.coeffs.push(c),
            None => buckets.push(Bucket {
                key,
                coeffs: vec![c],
            }),
        }
    }
    if has_const || const_terms {
        has_const = true;
        acc = Rat::ZERO; // absorbed into the constant (c + 5 = c', forall-exists)
        acc_overflow.clear();
    }

    // Rebuild terms, applying the opposite-sign gate per bucket. A rebuilt term can itself be
    // an `Add`: when merged coefficients sum to exactly 1 on an Add-valued key
    // (`(1/4)A + (3/4)A -> A`), `term_join` returns the bare key -- SPLICE it, or the outer bag
    // would nest a same-kind bag and break the Flat invariant.
    struct RebuiltBucket {
        key: Ex,
        pos: Rat,
        neg: Rat,
        unmerged: Vec<(Rat, Ex)>,
    }
    let mut out: Vec<Ex> = Vec::new();
    let mut spliced = false;
    let push_term = |out: &mut Vec<Ex>, spliced: &mut bool, t: Ex| match t {
        Ex::Add(v) => {
            out.extend(v);
            *spliced = true;
        }
        // F72: `term_join` on a partition-bag key routes through `mul()`, which may
        // COMPLETE the fold (the regrouped product fits i128 after all) and hand back
        // a bare literal. That literal must re-enter literal accumulation beside
        // `acc` -- parking it as a term would leave two unfolded literal members in
        // one bag. Ride the splice re-run (same termination argument: the completed
        // fold strictly reduced node count).
        t @ Ex::Num(_) => {
            out.push(t);
            *spliced = true;
        }
        t => out.push(t),
    };
    for b in buckets {
        // DETERMINISTIC per-key fold: sort the collected coefficients, then sum each
        // sign pool in that order; an overflow emits the partial and restarts. The
        // partition is now a function of the coefficient MULTISET (B3b).
        let Bucket { key, mut coeffs } = b;
        coeffs.sort_unstable_by(|x, y| x.cmp_exact(y));
        let (mut bpos, mut bneg) = (Rat::ZERO, Rat::ZERO);
        let mut unmerged: Vec<(Rat, Ex)> = Vec::new();
        for c in coeffs {
            let slot = if c.is_negative() {
                &mut bneg
            } else {
                &mut bpos
            };
            match slot.checked_add(&c) {
                Some(t) => *slot = t,
                None => unmerged.push((c, key.clone())),
            }
        }
        let b = RebuiltBucket {
            key,
            pos: bpos,
            neg: bneg,
            unmerged,
        };
        let cancels = !b.pos.is_zero() && !b.neg.is_zero();
        if cancels && cx.fin_licensed(&b.key) {
            match b.pos.checked_add(&b.neg) {
                Some(c) if c.is_zero() => {} // fully cancelled; 0 * (finite-a.e. t) -> 0 licensed
                Some(c) => push_term(&mut out, &mut spliced, term_join(c, b.key, cx)),
                None => {
                    push_term(&mut out, &mut spliced, term_join(b.pos, b.key.clone(), cx));
                    push_term(&mut out, &mut spliced, term_join(b.neg, b.key, cx));
                }
            }
        } else {
            // Same-sign only (TOTAL), or the licence is absent: emit each sign separately.
            if !b.pos.is_zero() {
                push_term(&mut out, &mut spliced, term_join(b.pos, b.key.clone(), cx));
            }
            if !b.neg.is_zero() {
                push_term(&mut out, &mut spliced, term_join(b.neg, b.key, cx));
            }
        }
        for (c, key) in b.unmerged {
            push_term(&mut out, &mut spliced, term_join(c, key, cx));
        }
    }
    out.extend(acc_overflow);
    if spliced {
        // A spliced Add's terms must COLLECT against the outer bag's other terms (the splice
        // alone leaves e.g. a bag-resident `2x` and a spliced `x` unmerged, breaking canon
        // idempotence). Re-run the constructor over everything; terminates because each splice
        // removes one bag node.
        if pos_inf {
            out.push(Ex::PosInf);
        }
        if neg_inf {
            out.push(Ex::NegInf);
        }
        if has_const {
            out.push(Ex::Const);
        }
        if !acc.is_zero() {
            out.push(Ex::Num(acc));
        }
        return add(out, cx);
    }

    // H-020/H-030 congruence at the TERM level: `term_join` builds raw products (no
    // `mul()` pass -- that is its point), so a collected, scaled or negated term can
    // carry a negative coefficient beside an Add factor that `mul()`'s sign-fold
    // clauses ban from canonical states: Const-bearing (H-020; the stability battery
    // caught a chain term `-1 * (x15 + C*x3) * (..C..)` whose re-parse folded while
    // the term did not), bare-inf-bearing (H-014's arm; `-(P + inf)*(T)` shipped as a
    // subtracted section while re-parse routed the same value through `mul()` and
    // folded the sign INTO the inf sum -- 2/1M fuzz rows 212777/914946), or holding a
    // negation-ABSORBING member (H-030's arm, incl. its recursive nestings). The
    // predicate mirrors the UNION of the three `mul()` arms -- a re-settle that
    // matched fewer shapes would ship terms `mul()` itself refuses to build.
    // Re-settle exactly those terms through `mul()`, which owns the fold. Terminates
    // by structural descent: the nested settle works on strictly deeper subterms.
    let mut resettled_reshaped = false;
    for t in out.iter_mut() {
        if let Ex::Mul(v) = t {
            let neg_coeff = v.iter().any(|f| matches!(f, Ex::Num(r) if r.is_negative()));
            let banned_add = v.iter().any(|f| {
                matches!(f, Ex::Add(ts) if ts.iter().any(|m| {
                    m.contains_const()
                        || matches!(m, Ex::PosInf | Ex::NegInf)
                        || term_absorbs_negation(m)
                }))
            });
            if neg_coeff && banned_add {
                let v = std::mem::take(v);
                *t = mul(v, cx);
                resettled_reshaped |= !matches!(t, Ex::Mul(_));
            }
        }
    }
    // A re-settled term can leave the Mul shape only from the exact input
    // `Mul[-1, Add[..Const..]]` (any other coefficient magnitude or extra factor keeps
    // the Mul; the collapse is the bare flipped Add) -- believed UNREACHABLE here: the
    // flatten distributes every 2-element `Mul[Num, Add]` (its scale never overflows at
    // r = -1: no Rat holds i128::MIN), and Const-bearing bucket keys never merge to
    // mint the shape at rebuild (independence keeps them one-element). Handle it anyway
    // so the bag invariants (I1/I3) are local to this function rather than a
    // consequence of that argument: re-enter the constructor exactly like the splice
    // path (same pending-flag tail; terminates because the H-020 fold's winning
    // orientation does not re-fire).
    if resettled_reshaped {
        if pos_inf {
            out.push(Ex::PosInf);
        }
        if neg_inf {
            out.push(Ex::NegInf);
        }
        if has_const {
            out.push(Ex::Const);
        }
        if !acc.is_zero() {
            out.push(Ex::Num(acc));
        }
        return add(out, cx);
    }

    // One infinity: absorb what is licensed; keep the bag if anything is not.
    // `primitive_sum` is DELIBERATELY skipped for the kept bag, and loses nothing: the
    // infinity pins the magnitude multiset at 1 (it absorbs scalar magnitudes exactly),
    // so no unanimous content != 1 can exist, and the sign family closes without the
    // pass -- +-1 coefficients distribute totally (`scale_term` flips the infinity) and
    // per-term signs live in the sections, so every negation spelling lands in this same
    // flat bag (docs/formal.md, "infinity-bearing sums are stored flat"). Consequence:
    // bare infinities NEVER reach `primitive_sum`.
    if pos_inf || neg_inf {
        let inf = if pos_inf { Ex::PosInf } else { Ex::NegInf };
        // Rational accumulator and Const absorb totally (finite by construction).
        let unlicensed: Vec<Ex> = out.drain(..).filter(|t| !cx.fin_licensed(t)).collect();
        if unlicensed.is_empty() {
            return inf;
        }
        let mut v = unlicensed;
        v.push(inf);
        v.sort_by(|a, b| add_term_cmp(a, b, cx.view));
        return Ex::Add(v);
    }
    // INFINITE-TERM ABSORPTION (class C, owner-approved 2026-08-03): a term that is
    // an infinity-carrying product takes values only in {±inf, NaN} (a canonical Mul
    // keeps at most one ±inf factor; inf * 0 = NaN, inf * finite = ±inf, NaN
    // propagates), and a sum of such terms stays in {±inf, NaN}. Every co-term F
    // with the finite-a.e. licence is absorbed: F + I = I wherever F is finite (a
    // finite F vanishes against I = ±inf; I = NaN forces both sides NaN), so the
    // disagreement set lives inside {F not finite} -- null under the `!`-licence.
    // The rational accumulator and a fitted Const are finite by construction and
    // absorb totally. This generalizes the literal-infinity arm above to
    // variable-bearing infinite terms (`x + inf*x -> inf*x`); the kept bag falls
    // through to the ordinary tail (inf-carrying PRODUCTS, unlike bare infinities,
    // always went through `primitive_sum` -- their coefficient is pinned at 1
    // because `mul` folds any scalar into the infinity, so no unanimous content
    // can arise and the pass is a no-op on them).
    if out.iter().any(is_inf_carrying) {
        out.retain(|f| is_inf_carrying(f) || !cx.fin_licensed(f));
        acc = Rat::ZERO;
        has_const = false;
    }
    if has_const {
        out.push(Ex::Const);
    } else if !acc.is_zero() {
        out.push(Ex::Num(acc));
    }

    match out.len() {
        0 => Ex::Num(Rat::ZERO),
        1 => out.pop().unwrap(),
        _ => {
            out.sort_by(|a, b| add_term_cmp(a, b, cx.view));
            primitive_sum(out, cx)
        }
    }
}

/// The PRIMITIVE-SUM normal form: a canonical `Add` carries the ORIENTATION-MAXIMAL sign and
/// no UNANIMOUS common coefficient magnitude. The extracted unit/content wraps the sum in a
/// coefficient (`-a - b -> -1 * (a + b)`; `a/3 + b/3 -> 1/3 * (a + b)`; `-2a + 3b ->
/// -1 * (2a - 3b)`), which is exactly the classical content/primitive-part decomposition
/// restricted to the sign-and-unanimity case -- the restriction keeps extraction
/// length-neutral-or-shortening (`x8 + 1.2*x3` has no unanimous content and stays put).
///
/// This closes every sign/grouping spelling family in the CONSTRUCTORS, with no iteration:
/// the class {u * A : u rational != 0, A primitive} has exactly one representative
/// `Mul[u, A]` (or `A` itself when u == 1), and every constructor path lands on it because
/// the sign decision is a function of the orientation CLASS {A, -A} (see
/// `flipped_orientation_wins`), never of any single lead term. Soundness: dividing every
/// term by a finite nonzero rational and multiplying the sum by it is TOTAL over the
/// extended reals (`u*(x+y)` and `ux + uy` agree at every point, including infinities and
/// NaN); `Const` terms pass through unchanged (`c / u` re-fits to `c'`, the contract's
/// forall-exists direction).
fn primitive_sum(terms: Vec<Ex>, cx: &Cx) -> Ex {
    // Bare infinities never arrive here: the `add()` constructor's one-infinity exit
    // returns its flat bag before this call (deliberately -- see the closure argument
    // at that exit).
    debug_assert!(
        !terms.iter().any(|t| matches!(t, Ex::PosInf | Ex::NegInf)),
        "primitive_sum received a bare infinity: the add() inf-exit must precede it"
    );
    // A Const-bearing sum NEVER extracts content: absorption owns it (`mul`'s H-020
    // fold arm unwraps `-1 x Add[Const-bearing]` through per-term `negate_term`,
    // where Const-carrying terms absorb the sign by the forall-exists refit), so
    // wrapping one here would ping-pong with that arm (measured: stack overflow on
    // the degenerate-config battery when the exemption was gated on the FIRST term
    // being constlike -- a Const-bearing sum led by an ordinary term slipped through
    // to extraction and re-wrapped what the fold arm had just unwrapped). The flat
    // bag is already unique for the class: Const terms are sign-neutral at term
    // level, every other term carries its true sign as ordinary structure.
    // ATOM-bearing constlike sums (pi/e with numerals) do NOT fold -- exact folds
    // are mu-refused, the bag persists -- so they need the orientation/content
    // decision like every other sum (H-014: `-2 - e` shipped as the unoriented
    // all-negative bag while parse of its own display built the `Mul[-1, Add[2, e]]`
    // wrap: two states, one spelling).
    if terms.is_empty() || terms.iter().any(Ex::contains_const) {
        return Ex::Add(terms);
    }
    // Unanimity: every term's |coefficient| equal (Const counts as 1; NaN cannot appear --
    // it collapses the whole sum earlier).
    let mut magnitudes: Vec<Rat> = Vec::with_capacity(terms.len());
    for t in &terms {
        let c = match t {
            Ex::Const => Rat::ONE,
            Ex::Num(r) => *r,
            _ => term_split(t.clone(), cx.view).0,
        };
        magnitudes.push(if c.is_negative() {
            match c.checked_neg() {
                Some(m) => m,
                None => return Ex::Add(terms), // overflow: skip extraction
            }
        } else {
            c
        });
    }
    let unanimous = magnitudes.windows(2).all(|w| w[0] == w[1]);
    // H-030: a sum holding a negation-absorbing member (see `term_absorbs_negation`)
    // never takes a SIGN decision here -- absorption owns the sign, exactly like the
    // Const-bearing skip above (and with the same ping-pong risk: `mul()`'s H-030
    // sign arm folds a negative coefficient into this class unconditionally, so
    // wrapping one here would re-mint what that arm just unwrapped). Magnitude-only
    // content extraction stays: a positive u moves no signs, so the absorbing member's
    // canonical spelling is untouched.
    let g = if unanimous { magnitudes[0] } else { Rat::ONE };
    // (b), owner ruling 2026-08-08: the filed orientation is the mu-CHEAPER one ("the
    // mirrors score equal; the bigger expression scores bigger"); the historical
    // first-in-sort-positive lex rule survives only as the exact-tie breaker. For a
    // content-1 sum the flip's wrapper cost is one mu_sym (the `-1 x Add` wrapper exists
    // only in the flipped spelling); for a non-unit content the wrapper exists in BOTH
    // spellings and the flip pays only the coefficient's sign bit -- and the per-term
    // deltas must then be priced on the DIVIDED terms (the candidates actually stored),
    // not the raw ones, or g's sign bit is counted once per term instead of once.
    let sign_neg = if terms.iter().any(term_absorbs_negation) {
        false
    } else if g.is_one() {
        flipped_orientation_wins(&terms, mu_sym() as i128, Tie::Keep, cx)
    } else {
        let Some(ng) = g.checked_neg() else {
            return Ex::Add(terms);
        };
        match divide_terms(&terms, &g, cx) {
            Some(divided_pos) => {
                let wsd = mu_rat(&ng) as i128 - mu_rat(&g) as i128;
                flipped_orientation_wins(&divided_pos, wsd, Tie::Keep, cx)
            }
            None => return Ex::Add(terms),
        }
    };
    let u = if sign_neg {
        match g.checked_neg() {
            Some(n) => n,
            None => return Ex::Add(terms),
        }
    } else {
        g
    };
    if u.is_one() || u.is_zero() {
        return Ex::Add(terms);
    }
    let inv_u = match u.checked_inv() {
        Some(i) => i,
        None => return Ex::Add(terms),
    };
    // Divide every term by u (exact; total -- see the doc above).
    let mut divided: Vec<Ex> = Vec::with_capacity(terms.len());
    for t in terms.iter() {
        let d = match t {
            Ex::Const => Ex::Const,
            Ex::Num(r) => match r.checked_mul(&inv_u) {
                Some(v) => Ex::Num(v),
                None => return Ex::Add(terms),
            },
            _ => {
                let (c, key) = term_split(t.clone(), cx.view);
                match c.checked_mul(&inv_u) {
                    Some(v) => term_join(v, key, cx),
                    None => return Ex::Add(terms),
                }
            }
        };
        divided.push(d);
    }
    // Re-sort (coefficient ties may reorder) and wrap. `mul`'s sign-orientation arm
    // (H-014) re-enters `add` only when the FLIPPED orientation wins -- and `divided`
    // is the winning orientation by construction (sign_neg came from the same
    // decision), so this tail call cannot recurse back into `add`.
    divided.sort_by(|a, b| add_term_cmp(a, b, cx.view));
    // Defense-in-depth: with the class-level sign decision a self-reproducing division can
    // no longer be SELECTED (a fully flip-symmetric bag compares equal in both orientations
    // and keeps sign +, and `u == 1` returned above) -- but refuse it outright if one ever
    // slips through, since wrapping a bag in a coefficient that divides back to the same
    // bag is a non-terminating spelling. (A residual, documented non-uniqueness remains for
    // ENCLOSING coefficients of self-negative sums: `Mul[2, S]` and `Mul[-2, S]` denote the
    // same 0-or-NaN function and both persist; measure-zero corner, the certificates cancel
    // these bags wherever they are licensed.)
    if divided == terms {
        return Ex::Add(terms);
    }
    mul(vec![Ex::Num(u), Ex::Add(divided)], cx)
}

/// Divide every term by `u` exactly (`None` on any overflow) -- the same loop as
/// `primitive_sum`'s tail, factored so the orientation decision can price the DIVIDED
/// candidates for a non-unit content.
fn divide_terms(terms: &[Ex], u: &Rat, cx: &Cx) -> Option<Vec<Ex>> {
    let inv_u = u.checked_inv()?;
    let mut divided: Vec<Ex> = Vec::with_capacity(terms.len());
    for t in terms {
        let d = match t {
            Ex::Const => Ex::Const,
            Ex::Num(r) => Ex::Num(r.checked_mul(&inv_u)?),
            _ => {
                let (c, key) = term_split(t.clone(), cx.view);
                term_join(c.checked_mul(&inv_u)?, key, cx)
            }
        };
        divided.push(d);
    }
    Some(divided)
}

/// Which of the two sign orientations {A, -A} of a sum is canonical? Compare the SORTED
/// coefficient sequences of the two orientations lexicographically (exact rational order)
/// and prefer the LARGER -- a decision that is a function of the orientation class, so both
/// entry spellings land on the same representative. Positional comparison is well-defined
/// because the stripped-key multiset is negation-invariant: position i carries the same
/// stripped key in both sorted bags.
///
/// A positive-lead rule would be ill-defined on sums containing an uncertified
/// flip-symmetric pair `{t, -t}` (cert-refused cancellation): the pair maps to itself
/// under negation, so BOTH orientations lead with the negative twin and no lead-based
/// decision is flip-antisymmetric. The lex rule decides that class by the first
/// ASYMMETRIC position: `-a - b` flips to `-1 * (a + b)`, `-2a + 3b` flips to
/// `-1 * (2a - 3b)`, `a - b` stays.
///
/// `Const` terms are sign-free (negation re-fits, `term_join` absorbs the sign) and compare
/// equal at their positions; infinities carry their sign. `false` (keep the original
/// orientation, extraction-free) on any negation overflow.
/// What decides an EXACT mu tie between the two orientations (owner ruling A,
/// 2026-08-08): at a SIGN-TRADE site one spelling is distinguished -- bare / positive
/// coefficient -- and it wins the tie, so `Keep` (the current spelling is the
/// distinguished one) and `Flip` (the flipped spelling is) resolve structurally. `Lex`
/// survives ONLY for free-orientation sites (even carriers, wrapper delta 0), where
/// both spellings are bare and no structural member is distinguished -- there the
/// historical sorted-coefficient-sequence comparison remains the class-antisymmetric
/// order.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tie {
    Keep,
    /// The A-convention's mirror pole: no current call site prefers the flipped
    /// spelling on an exact tie (tie A gives every sign-trade site `Keep`), but the
    /// enum documents the complete decision space -- a future carrier whose bare
    /// spelling is the WRAPPED one would use it. Kept deliberately over deletion so
    /// the tie policy reads as the three-state choice it is.
    #[allow(dead_code)]
    Flip,
    Lex,
}

fn flipped_orientation_wins(terms: &[Ex], wrapper_sign_delta: i128, tie: Tie, cx: &Cx) -> bool {
    // (b), owner ruling 2026-08-08: the mu-CHEAPER orientation wins. `wrapper_sign_delta`
    // is what the flip costs at the wrapper itself, signed from the caller's side:
    // +mu_sym in `primitive_sum` for a content-1 sum (the wrapper exists only in the
    // flipped spelling), NEGATIVE mu_sym in `mul()`'s distribution arm (the wrapper
    // exists only in the CURRENT spelling), the coefficient sign-bit delta for a
    // non-unit content (wrapper in both spellings). The two callers' deltas are exact
    // mirrors, so their decisions cannot ping-pong.
    let mut delta = wrapper_sign_delta;
    let mut flipped: Vec<Ex> = Vec::with_capacity(terms.len());
    for t in terms {
        match negate_term(t, cx) {
            Some(n) => {
                delta += complexity(&n, cx.view) as i128 - complexity(t, cx.view) as i128;
                flipped.push(n);
            }
            None => return false,
        }
    }
    if delta != 0 {
        return delta < 0;
    }
    // EXACT mu tie -> the caller's structural policy (owner ruling A). Only `Lex` --
    // the free-orientation sites, where neither spelling is structurally
    // distinguished -- still runs the historical rule: compare the sorted coefficient
    // sequences lexicographically and prefer the larger. Total and a function of the
    // orientation CLASS (both entry spellings land on one representative), so the D9
    // flip-flop disease cannot return through the tie path.
    match tie {
        Tie::Keep => return false,
        Tie::Flip => return true,
        Tie::Lex => {}
    }
    flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
    for (a, b) in terms.iter().zip(flipped.iter()) {
        let ca = orientation_coeff(a, cx.view);
        let cb = orientation_coeff(b, cx.view);
        match ca.cmp_exact(&cb) {
            std::cmp::Ordering::Less => return true,
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal => {}
        }
    }
    false // fully flip-symmetric: both orientations are the same multiset
}

/// The signed rational coefficient a term contributes to the orientation comparison
/// (`Const` counts +1 -- sign-free; infinities carry their sign as +/-1).
/// Is this Add term an infinity-carrying product? Such a term is never finite: its
/// values are ±inf (infinity times a nonzero finite co-product) or NaN (times zero,
/// or a NaN co-factor). Bare `Ex::PosInf`/`Ex::NegInf` never reach the term list
/// (the flatten scan diverts them to the sign flags), so `Mul` is the only carrier.
fn is_inf_carrying(t: &Ex) -> bool {
    matches!(t, Ex::Mul(v) if v.iter().any(|f| matches!(f, Ex::PosInf | Ex::NegInf)))
}

/// H-030 (2026-08-05): does this Add member absorb a negation WITHOUT keeping a
/// coefficient sign? Carriers: a product with a bare infinity factor (`term_join`
/// folds the sign into the infinity, exactly), a product with an inf-bearing Add
/// factor (`mul()`'s H-014 arm folds the sign INTO that sum, for any negative
/// coefficient), and -- RECURSIVELY -- a product whose Add factor holds an absorbing
/// member (the sign chases the absorber through `mul()`'s H-030 arm at each level,
/// mirroring the depth `Ex::contains_const` gives the H-020 Const side). A sum
/// holding such a member can never park a class sign in a `-1 x Add` wrapper AT ANY
/// factor multiset: the display splits co-factors (divisor-side literals render as
/// `-(S)/den`), so re-parse associates the sign with the sum alone, routes it
/// through `mul()`, and the absorbing member eats it -- the wrapper spelling is
/// never re-parse-stable (the H-014 conflation, one level up; fuzz row 18943 pinned
/// the non-lone shape). `mul()`'s H-030 sign arm distributes unconditionally on this
/// class, and `primitive_sum` never makes a sign decision for it (the same division
/// of ownership as Const-bearing sums: absorption owns the sign).
fn term_absorbs_negation(t: &Ex) -> bool {
    matches!(t, Ex::Mul(v) if v.iter().any(|f| {
        matches!(f, Ex::PosInf | Ex::NegInf)
            || matches!(f, Ex::Add(ts) if ts.iter().any(|m| {
                matches!(m, Ex::PosInf | Ex::NegInf) || term_absorbs_negation(m)
            }))
    }))
}

fn orientation_coeff(t: &Ex, view: &TokenView) -> Rat {
    match t {
        Ex::Const => Rat::ONE,
        Ex::PosInf => Rat::ONE,
        Ex::NegInf => Rat::NEG_ONE,
        Ex::Num(r) => *r,
        // Sign-normalized split (B19): an odd-literal term's sign is SEMANTIC and the
        // orientation comparison must see it -- sin(-2) contributes -1, not +1.
        _ => term_split(t.clone(), view).0,
    }
}

/// The canonical n-ary multiplication constructor. Gates:
///
/// * NaN factor -> NaN (TOTAL: `nan * t = nan` for every extended-real `t`, including 0 and inf).
/// * Rational coefficients multiply exactly; a coefficient meeting an infinity folds exactly
///   (`r * inf` keeps the sign algebra; `0 * inf = nan`, TOTAL).
/// * Coefficient 0 (no infinity): the product IS 0 wherever every other factor is finite --
///   collapse licensed per-factor by the `!`-certificate; unlicensed factors keep `Mul[0, t]`
///   (`0 * (1/0)` is NaN on full measure, and the certified rule `* 0 !0 -> 0` still gets its
///   chance later).
/// * Like bases merge exponents: SAME-SIGN rational exponents freely -- with the branch-cut
///   licence when either is non-integer (`x^(1/2) * x^(1/2) -> x` is wrong on `x < 0`; requires
///   a certainly-non-negative base OR the merged sum staying non-integer). OPPOSITE-SIGN needs
///   the `$`-licence on the base (`t^2 * t^-1 -> t` fills the NaN at `t = 0`).
/// * `Const` independence as in `add`; bare `Const` factors collapse into ONE `Const`, absorbing
///   a NONZERO rational coefficient (`c * r = c'`, forall-exists; 0 stays outside).
pub fn mul(items: Vec<Ex>, cx: &Cx) -> Ex {
    let mut nums: Vec<Rat> = Vec::new();
    let mut coeff_overflow: Vec<Ex> = Vec::new();
    let mut inf_sign: Option<bool> = None; // Some(true) = +inf so far, Some(false) = -inf
    let mut has_const = false;
    let mut factors: Vec<Ex> = Vec::new();
    let mut stack: Vec<Ex> = items;
    stack.reverse();
    loop {
        while let Some(e) = stack.pop() {
            match e {
                Ex::NaN => return Ex::NaN,
                Ex::Mul(v) => stack.extend(v.into_iter().rev()),
                Ex::Num(r) => nums.push(r),
                Ex::PosInf => inf_sign = Some(inf_sign.unwrap_or(true)),
                Ex::NegInf => inf_sign = Some(!inf_sign.unwrap_or(true)),
                Ex::Const => has_const = true,
                other => factors.push(other),
            }
        }
        // ODD-ROOT JOIN (P2, owner-approved 2026-08-02): rootn factors sharing the
        // same odd literal index merge their arguments -- rootn(a,n) * rootn(b,n) =
        // rootn(a*b, n) is a TOTAL identity (the real odd root is completely
        // multiplicative over the extended reals, including +-inf and nan), so no
        // certificate is needed. This is the sound form of the exponent-collection
        // behavior rootn deliberately opted out of by not being a power (B1: the
        // principal rational power differs on negative bases); the joining rule is
        // unminable (7-token source > the length-4 mine). The joined `fun` result
        // re-enters the dispatch (it may collapse further: rootn(x^2,3)*rootn(x,3)
        // -> rootn(x^3,3) -> x, or fold exactly via the P1 arm); each join strictly
        // reduces total node count, so the loop terminates by well-foundedness.
        // One scan collects EVERY odd index's occurrence list (a single-latch scan
        // missed joins whose index differs from the first rootn seen); the loop
        // joins one group per round and re-scans, so the fixpoint -- every group
        // reduced to one factor -- is reached regardless of group order (products
        // are AC, so the per-group result is order-independent).
        let mut by_index: Vec<(i128, Vec<usize>)> = Vec::new();
        for (i, f) in factors.iter().enumerate() {
            if let Ex::Fun(op, fargs) = f {
                if fargs.len() == 2 && cx.view.tok_is(*op, "rootn") {
                    if let Ex::Num(r) = &fargs[1] {
                        if let Some(n) = r.as_integer() {
                            if n >= 3 && n % 2 == 1 {
                                match by_index.iter_mut().find(|(gn, _)| *gn == n) {
                                    Some((_, idxs)) => idxs.push(i),
                                    None => by_index.push((n, vec![i])),
                                }
                            }
                        }
                    }
                }
            }
        }
        match by_index.into_iter().find(|(_, idxs)| idxs.len() >= 2) {
            Some((n, idxs)) => {
                let mut args = Vec::with_capacity(idxs.len());
                let mut op_tok = None;
                for i in idxs.into_iter().rev() {
                    let Ex::Fun(op, mut fargs) = factors.remove(i) else {
                        unreachable!()
                    };
                    op_tok = Some(op);
                    args.push(fargs.swap_remove(0));
                }
                stack.push(fun(op_tok.unwrap(), vec![mul(args, cx), Ex::int(n)], cx));
            }
            _ => break,
        }
    }

    // DETERMINISTIC coefficient accumulation: folding rational
    // factors in ARRIVAL order made the overflow partition order-dependent -- which
    // pairs merged before an i128 overflow depended on the input spelling, yielding two
    // sound-but-different canonical states for one multiset. Sorting first (cmp_exact
    // is exact at every magnitude since B3) makes the merge sequence a function of the
    // MULTISET: permutation invariance by construction. On overflow the accumulated
    // value is emitted as a factor and accumulation restarts.
    //
    // SIGN-FACTORED (F72, 2026-08-09): the accumulation runs on ABSOLUTE values and the
    // net sign lands on the final coefficient, which `sign_place` owns. Accumulating
    // SIGNED rationals let the sign ride whichever partial the sorted fold left it in,
    // so with an overflow partition the sign HOST was a function of the arrival
    // spelling: `-N * P` hosted it on the `-N` partial while the re-parse of its own
    // rendering hosted it on the coefficient slot -- two canonical-looking states for
    // one value (extreme-lane P1 rows; the 7-token falsifier in test_ac_core). The
    // sign of a product IS a bag-level attribute: |a*b| = |a|*|b| and the partition of
    // the UNSIGNED multiset is spelling-invariant. (`checked_neg` on a Rat is total
    // outside i128::MIN, which `Rat::new`'s normalization never stores.)
    let mut net_neg = false;
    for r in nums.iter_mut() {
        if r.is_negative() {
            match r.checked_neg() {
                Some(a) => {
                    net_neg = !net_neg;
                    *r = a;
                }
                None => debug_assert!(false, "Rat holding i128::MIN reached mul()"),
            }
        }
    }
    nums.sort_unstable_by(|a, b| a.cmp_exact(b));
    let mut coeff = Rat::ONE;
    for r in nums {
        match coeff.checked_mul(&r) {
            Some(p) => coeff = p,
            None => {
                coeff_overflow.push(Ex::Num(coeff));
                coeff = r;
            }
        }
    }
    if net_neg {
        match coeff.checked_neg() {
            Some(nc) => coeff = nc,
            None => debug_assert!(false, "positive Rat negation overflowed"),
        }
    }

    // Coefficient x infinity folds exactly and totally.
    if let Some(sign) = inf_sign {
        if coeff.is_zero() {
            return Ex::NaN; // 0 * inf = nan, and nan * rest = nan: TOTAL
        }
        let sign = if coeff.is_negative() { !sign } else { sign };
        coeff = Rat::ONE;
        inf_sign = Some(sign);
        if has_const {
            // inf * c: c = 0 is reachable (0 * inf = nan), so the constant does NOT absorb the
            // infinity; keep both factors.
            factors.push(Ex::Const);
            has_const = false;
        }
    }

    // Like-base exponent collection. The BRANCH-CUT licence gates every single merge step,
    // including same-sign ones: merging exponents `a` and `b` into `a + b` is sound iff both
    // are integers (TOTAL up to null poles), OR the base is certainly non-negative, OR the sum
    // stays a nonzero non-integer (the `base < 0` region is NaN on both sides). Counterexample
    // that forces the same-sign gate: `x^(1/2) * x^(1/2) -> x` fabricates values on `x < 0`.
    struct FBucket {
        base: Ex,
        sym: Option<Ex>, // Some(e): occurrences of base^e; None: rational exponent pool
        /// ALL rational exponents for this (base, sym), raw; the licence-gated fold
        /// happens at rebuild in SORTED order (B3b determinism).
        exps: Vec<Rat>,
    }
    let mut buckets: Vec<FBucket> = Vec::new();
    let mut opaque: Vec<Ex> = Vec::new(); // Const-containing and merge-refused factors, verbatim
    let branch_ok = |cx: &Cx, base: &Ex, a: &Rat, b: &Rat| -> bool {
        if cx.lossy || (a.is_integer() && b.is_integer()) {
            return true;
        }
        if cx.certainly_nonneg(base) {
            return true;
        }
        a.checked_add(b)
            .is_some_and(|s| !s.is_integer() && !s.is_zero())
    };
    for f in factors {
        let (base, fe) = factor_split(f, cx);
        let (sym, r) = match fe {
            FactorExp::Rat(r) => (None, r),
            FactorExp::Sym(s) => (Some(s), Rat::ONE),
        };
        if base.contains_const() || sym.as_ref().is_some_and(Ex::contains_const) {
            // Independent constants: keep verbatim, never merged with anything.
            opaque.push(rebuild_factor(base, sym, r, cx));
            continue;
        }
        match buckets.iter_mut().find(|b| b.base == base && b.sym == sym) {
            Some(b) => b.exps.push(r),
            None => buckets.push(FBucket {
                base,
                sym,
                exps: vec![r],
            }),
        }
    }

    // A rebuilt factor can itself be a `Mul` (pow distributes integer exponents over Mul
    // bases; the Add sign-flip wraps odd powers in `Mul[-1, ...]`): SPLICE those and re-run the
    // constructor at the end, exactly as `add` does -- the spliced factors must collect against
    // the other buckets, and the Flat invariant must hold.
    let mut out: Vec<Ex> = opaque;
    let mut opaque2: Vec<Ex> = Vec::new(); // licence/overflow refusals from the sorted fold
    let mut spliced = false;
    let push_factor = |out: &mut Vec<Ex>, spliced: &mut bool, f: Ex| match f {
        Ex::Mul(v) => {
            out.extend(v);
            *spliced = true;
        }
        f => out.push(f),
    };
    for b in buckets {
        let FBucket {
            base,
            sym,
            mut exps,
        } = b;
        // DETERMINISTIC licence-gated fold (B3b): sort the exponents, then merge each
        // sign pool in that order, re-checking the branch-cut licence at every step
        // exactly as the arrival-order code did -- but the merge partition (and thus
        // WHICH merges the licence sees) is now a function of the exponent MULTISET.
        exps.sort_unstable_by(|x, y| x.cmp_exact(y));
        let (mut pos, mut neg) = (Rat::ZERO, Rat::ZERO);
        for r in exps {
            let slot = if r.is_negative() { &mut neg } else { &mut pos };
            let sym_ok = match &sym {
                None => true,
                Some(y) => cx.sym_merge_licensed(&base, y),
            };
            if slot.is_zero() {
                *slot = r; // first exponent of this sign: no merge happens yet
            } else if sym_ok && branch_ok(cx, &base, slot, &r) {
                match slot.checked_add(&r) {
                    Some(s) => *slot = s,
                    None => opaque2.push(rebuild_factor(base.clone(), sym.clone(), r, cx)),
                }
            } else {
                opaque2.push(rebuild_factor(base.clone(), sym.clone(), r, cx));
            }
        }
        let cancels = !pos.is_zero() && !neg.is_zero();
        let sym_cancel_ok = match &sym {
            None => true,
            Some(y) => cx.sym_merge_licensed(&base, y),
        };
        if cancels
            && sym_cancel_ok
            && (cx.lossy || cx.finnz_licensed(&base))
            && branch_ok(cx, &base, &pos, &neg)
        {
            match pos.checked_add(&neg) {
                Some(s) if s.is_zero() => {
                    // base^0 -> 1 on the licensed (finite-nonzero a.e.) base: factor drops.
                }
                Some(s) => {
                    let f = rebuild_factor(base, sym, s, cx);
                    push_factor(&mut out, &mut spliced, f);
                }
                None => {
                    let f1 = rebuild_factor(base.clone(), sym.clone(), pos, cx);
                    push_factor(&mut out, &mut spliced, f1);
                    let f2 = rebuild_factor(base, sym, neg, cx);
                    push_factor(&mut out, &mut spliced, f2);
                }
            }
        } else {
            // Keep the two sign pools separate (each pool is already exactly merged, every
            // step licensed at accumulation time).
            if !pos.is_zero() {
                let f = rebuild_factor(base.clone(), sym.clone(), pos, cx);
                push_factor(&mut out, &mut spliced, f);
            }
            if !neg.is_zero() {
                let f = rebuild_factor(base, sym, neg, cx);
                push_factor(&mut out, &mut spliced, f);
            }
        }
    }
    out.extend(opaque2);
    out.extend(coeff_overflow);
    if spliced {
        if !coeff.is_one() {
            out.push(Ex::Num(coeff));
        }
        if has_const {
            out.push(Ex::Const);
        }
        if let Some(sign) = inf_sign {
            out.push(if sign { Ex::PosInf } else { Ex::NegInf });
        }
        return mul(out, cx);
    }

    // Coefficient 0 (no infinity): collapse where licensed (see the gate doc above). A bare
    // Const vanishes here (0 * c = 0 -- Const is certainly finite), so `has_const` is simply
    // not consulted on this path.
    if inf_sign.is_none() && coeff.is_zero() {
        // Each LICENSED factor drops individually (`0 * f == 0` wherever `f` is finite --
        // the same finite-a.e. licence as the full collapse, applied per-factor). Keeping
        // licensed factors would make the kept-zero form depend on ASSOCIATION order
        // (`(0 * x) * (1/D)` collapses the inner pair, `0 * (x * (1/D))` would not) and
        // break serialization stability; only the unlicensed factors stay with the 0.
        let v: Vec<Ex> = out.into_iter().filter(|f| !cx.fin_licensed(f)).collect();
        if v.is_empty() {
            return Ex::Num(Rat::ZERO);
        }
        // F68 (fuzz rows 392777/647852, introduced by F63): assemble through the
        // sign-placement owner, not raw -- the zero is a sign-eating carrier (see
        // sign_place's carrier list), so the orbit files ONE orientation for every
        // trade-site factor where the raw assembly froze whichever mirror arrived
        // (direct build vs parse-of-own-rendering shipped two states; row-120 law).
        return sign_place(Rat::ZERO, v, cx);
    }

    if has_const {
        // Bare Const absorbs the (nonzero) rational coefficient: c * r = c'.
        coeff = Rat::ONE;
        out.push(Ex::Const);
    }
    if let Some(sign) = inf_sign {
        out.push(if sign { Ex::PosInf } else { Ex::NegInf });
    }
    // CANONICAL SIGN ORIENTATION for `Mul[-1, Add[S]]` (H-014 root cause, 2026-08-03):
    // the orientation of a sum class {S, -S} is owned by `flipped_orientation_wins`, but
    // two assembly paths could ship the LOSING orientation wrapped in a -1 coefficient:
    // (a) infinity-bearing sums bypass `primitive_sum` entirely (the one-infinity exit
    // in `add()`), so nothing ever re-orients them; (b) this constructor bagged
    // `Num(-1) x Add[S]` verbatim, whatever the orientation. The tagged renderer's
    // display redistribution (`emit_tagged`'s Mul arm) prints the factored form and the
    // flipped sum IDENTICALLY, so parse rebuilds the DISTRIBUTED state and the two
    // spellings were distinct derivation-reachable canonical states sharing one
    // serialization -- invisible to the stability assert (it compares spellings, which
    // agree). Observable: a rule with a `Mul[-1, F(..)]`-shaped LHS fires on the
    // distributed member but cannot see inside the factored node, so one-call fixpoints
    // differed from re-entry fixpoints (P1/P7, 15/200k fuzz rows, every one
    // infinity-bearing). The fix has two clauses, one per abdicating owner:
    //
    // * BARE-INFINITY sums (the class `primitive_sum` never sees -- `add`'s
    //   one-infinity exit returns its flat bag with no orientation enforcement): the
    //   sign lives IN the sum, for ANY negative coefficient. A negative rational
    //   coefficient never coexists with a bare-inf Add factor: `Mul[-c, Add[S]]` and
    //   `Mul[c, Add[flip S]]` are one value, and the projections conflate them (the
    //   infix renderer spells the coefficient sign as a leading `-(..)`, which
    //   re-parses as the sum flip -- fuzz row 10985). The distributed positive-coeff
    //   spelling is the one parse builds and the one whose members are visible rule
    //   sites (the sign-pull family fires on a `Mul[-1, F(..)]` MEMBER, never through
    //   a wrapper); `add(flipped)` re-enters the same inf-exit and returns flat, and
    //   the negated coefficient is positive, so no bounce is possible.
    // * Everything else, for the lone `-1 x Add` shape: consult the SAME orientation
    //   decision `primitive_sum` uses -- flip exactly when the flipped orientation
    //   wins. Antisymmetry gives termination (the flipped bag's own check is false),
    //   and `add`'s primitive-sum tail call always passes the WINNING orientation, so
    //   this arm is a no-op on it. General negative coefficients on non-inf sums stay:
    //   they ARE the content form (`u` extraction).
    //
    // A `flipped_orientation_wins`/`negate_term`/negation overflow refusal keeps the
    // factored form, whose `neg <add>` / literal-coefficient display is injective.
    if coeff.is_negative() {
        let inf_add = out.iter().position(
            |f| matches!(f, Ex::Add(v) if v.iter().any(|t| matches!(t, Ex::PosInf | Ex::NegInf))),
        );
        if let Some(pos) = inf_add {
            let Ex::Add(terms) = &out[pos] else {
                unreachable!()
            };
            let flipped: Option<Vec<Ex>> = terms.iter().map(|t| negate_term(t, cx)).collect();
            if let (Some(flipped), Some(nc)) = (flipped, coeff.checked_neg()) {
                let re = add(flipped, cx);
                // Provable (D3): merging/cancellation decisions ride STRIPPED keys,
                // which negation preserves (the coefficient sign lives outside the
                // key), so no licence fires on the flipped bag that did not fire on
                // the original; a canonical Add carries exactly ONE infinity term
                // (both-signs folds to NaN, same-sign merges, NaN absorbs -- all at
                // construction) and `negate_term` maps it 1:1 to the opposite sign.
                // The flip therefore re-enters `add`'s one-infinity exit with the
                // same term count >= 2: always a flat Add.
                debug_assert!(
                    matches!(re, Ex::Add(_)),
                    "flip of a bare-inf sum must stay a flat Add: {re:?}"
                );
                out[pos] = re;
                coeff = nc;
            }
        }
    }
    // H-020 (2026-08-04, owner-ruled): a NEGATIVE coefficient never coexists with a
    // Const-bearing Add factor -- the sign folds INTO the sum, wherever the sign sits.
    // `negate_term` is exact family algebra per term (a Const-carrying term absorbs the
    // sign through `term_join`'s forall-exists refit; every other term negates its
    // coefficient as ordinary structure), so `-c x Add[S]` IS `c x Add[flip S]` as a
    // fitted family, and the subtracted Const TERM already loses its sign at
    // construction (the `has_const` absorption above) -- this clause completes that
    // doctrine at the sum level. Two failed narrowings pinned the required scope:
    // keeping the wrapper handed the projections a sign no spelling can carry on a
    // Const term (`-<constant>` is not a token: the display redistribution silently ate
    // it under pow bases, fuzz rows 148693/166766, P7-infix), and folding only the LONE
    // `-1 x Add` shape was not a CONGRUENCE -- the same value routes the sign through a
    // negative divisor literal (`S/(-3.89)`, bag coefficient) or a wider bag
    // (`-(S)/rootn(..)`), while the infix display pulls it out front and re-parse folds
    // through the lone shape: two canonical states, one family (47 P7-infix + 20
    // P7-explicit at 200k under the narrow arm). Folding at the BAG level makes the
    // decision a function of the bag alone: every construction route converges, and
    // displays never face an unspellable Const sign because the coefficient beside a
    // Const-bearing sum is always positive. The first Const-bearing Add under the bag
    // order takes the sign (deterministic; bags rarely hold two). Termination: the
    // negated coefficient is positive, and `add` of the flipped terms re-enters
    // `primitive_sum`, whose Const-bearing skip returns the flat bag -- no re-wrap, no
    // bounce. A `negate_term`/negation overflow refusal (i128::MIN) keeps the negative
    // coefficient, whose literal display is injective.
    if coeff.is_negative() {
        let const_add = out
            .iter()
            .position(|f| matches!(f, Ex::Add(v) if v.iter().any(Ex::contains_const)));
        if let Some(pos) = const_add {
            let Ex::Add(terms) = &out[pos] else {
                unreachable!()
            };
            let flipped: Option<Vec<Ex>> = terms.iter().map(|t| negate_term(t, cx)).collect();
            if let (Some(flipped), Some(nc)) = (flipped, coeff.checked_neg()) {
                let re = add(flipped, cx);
                debug_assert!(
                    matches!(re, Ex::Add(_)),
                    "flip of a Const-bearing sum must stay a flat Add: {re:?}"
                );
                out[pos] = re;
                coeff = nc;
            }
        }
    }
    // H-030 (2026-08-05): a NEGATIVE coefficient never coexists with an Add factor
    // holding a negation-ABSORBING member (`term_absorbs_negation`: bare-inf products,
    // inf-Add-factor products, and their recursive nestings) -- the sign folds INTO
    // that sum, member-wise, for ANY coefficient and ANY co-factor multiset, exactly
    // like the H-014 bare-inf and H-020 Const arms above. The wrapper spelling is
    // never re-parse-stable for this class: the display splits co-factors (fuzz row
    // 18943: `-(S)/0.0795` re-parses as `(-1 x S) / den`, the lone unit distributes,
    // and the two routes diverged), and a rewrite deep inside a factor can mint the
    // absorbing member AFTER the wrapper was legitimately built (fuzz rows
    // 212777/914946: the spine rebuild kept a wrapper `flipped_orientation_wins`
    // judged by the non-absorbed member spelling). `add(flipped)` re-settles the
    // absorbing members through `mul()` (term-level congruence) and `primitive_sum`'s
    // H-030 skip never re-wraps: no bounce, and the negated coefficient is positive.
    // A `negate_term` overflow refusal keeps the factored form, whose display is
    // injective.
    if coeff.is_negative() {
        let abs_add = out
            .iter()
            .position(|f| matches!(f, Ex::Add(ts) if ts.iter().any(term_absorbs_negation)));
        if let Some(pos) = abs_add {
            let Ex::Add(terms) = &out[pos] else {
                unreachable!()
            };
            let flipped: Option<Vec<Ex>> = terms.iter().map(|t| negate_term(t, cx)).collect();
            if let (Some(flipped), Some(nc)) = (flipped, coeff.checked_neg()) {
                let re = add(flipped, cx);
                debug_assert!(
                    matches!(re, Ex::Add(_)),
                    "flip of an absorbing-member sum must stay a flat Add: {re:?}"
                );
                out[pos] = re;
                coeff = nc;
            }
        }
    }
    // F63 SIGN-PLACEMENT OWNER (owner-ruled 2026-08-08: FULL family scope, tie
    // convention A). A product's sign can trade between the coefficient and any
    // ODD-carrier factor -- a bare mixed-sign Add, Pow(Add, odd POSITIVE integer),
    // rootn(Add, odd index >= 3), or an odd function of an Add -- because
    // f(-S) = -f(S) is TOTAL on those carriers (negative odd exponents/indices are
    // NOT carriers: the one-zero pole trilemma, see is_odd_fun). With n trade sites
    // the value has 2^n spellings (each site independently flippable, the
    // coefficient's sign = entry sign times flip parity). The owner MATERIALIZES
    // every spelling, prices it with the real mu, and returns the argmin -- so the
    // priced spelling IS the built spelling (the row-120 law: any gap between the
    // two re-opens construction-history dependence), and the decision is a function
    // of the ORBIT, not of the entry spelling. Ties: the positive-coefficient
    // spelling wins (ruling A -- a leading minus is only ever minted when strictly
    // cheaper, and what the user typed survives whenever prices tie); residual
    // equal-mu same-sign ties fall to a fixed structural order. n > 6 refuses to
    // trade (2^n materializations; n is orbit-invariant, so the cap is a legal
    // class function -- and unreachable on real corpora). A negate_term overflow
    // refusal keeps the entry spelling, whose display is injective. This arm
    // SUBSUMES the former lone `-1 x Add` distribution arm (its case is n=1 with
    // out.len() == 1; the mu comparison and the A-tie give the identical decision).
    //
    // No bounce: the returned spelling is the orbit argmin under a total tie order;
    // re-running the owner on it re-selects it (idempotent), primitive_sum's re-file
    // of any flipped inner sum stays bare (the flip only wins here when its whole-
    // product mu is <= the wrap spelling's, which keeps the sum inside fow's
    // wrapper-protection band), and the H-020/H-030 absorption arms above never see
    // these bags (their sum classes are excluded from the site set).
    sign_place(coeff, out, cx)
}

/// F63: assemble a product from `(coefficient, factor bag)` in its CANONICAL sign
/// placement -- the shared owner behind `mul()`'s final assembly and `term_join`'s
/// negative-coefficient joins, so every site that mints a product builds the SAME
/// spelling the orientation machinery prices (the row-120 law). With no trade site
/// (or the n > 6 cap, or a negation overflow) this is exactly the plain assembly:
/// push the non-unit coefficient, sort, wrap.
fn sign_place(coeff: Rat, out: Vec<Ex>, cx: &Cx) -> Ex {
    let assemble = |factors: Vec<Ex>, c: &Rat| -> Ex {
        let mut v = factors;
        if !c.is_one() {
            v.push(Ex::Num(*c));
        }
        match v.len() {
            0 => Ex::Num(Rat::ONE),
            1 => v.pop().unwrap(),
            _ => {
                v.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
                Ex::Mul(v)
            }
        }
    };
    // The bag's SIGN CARRIER -- where a sign toggle legally lives (mirroring the
    // H-014/H-020/H-030 absorption arms above, which guarantee the ENTRY spelling is
    // carrier-normalized; the owner must return only carrier-normalized spellings
    // too, or the arms and the owner mint two states for one value -- measured live
    // as 9 idem failures at 64k / 50 at 1M when `assemble` parked a raw `-1` beside
    // a bare Const (row 1419) and a bare infinity (row 7320)):
    //   * a bare +-Inf factor: the infinity carries the sign itself (H-014);
    //   * any Const-carrier (bare Const factor or Const-bearing Add factor): the
    //     sign VANISHES into the forall-exists refit -- every site orientation is
    //     value-equal as a family, so orientations are chosen freely (H-020);
    //   * an Add factor with a negation-absorbing member: the sign folds into that
    //     sum member-wise (H-030);
    //   * a ZERO coefficient (the kept-zero bag `Mul[0, ..]`): the zero eats every
    //     sign -- `0 * X` and `0 * (-X)` are value-equal in EVERY case (finite -> 0,
    //     +-inf -> nan, nan -> nan), so the whole sign dimension collapses exactly
    //     like a Const-carrier's (F68; fuzz rows 392777/647852, introduced by F63:
    //     the kept-zero arm assembled RAW and froze the arrival mirror, so the
    //     direct build and the parse of its own rendering shipped two states);
    //   * otherwise: the rational coefficient.
    enum Carrier {
        Inf(usize),
        Free,
        Absorb(usize),
        Coeff,
    }
    {
        let sites: Vec<usize> = out
            .iter()
            .enumerate()
            .filter_map(|(i, f)| is_sign_trade_site(f, cx).then_some(i))
            .collect();
        if !sites.is_empty() && sites.len() <= 6 {
            // Priority: Free FIRST -- a Const-carrier eats EVERY sign (coefficient
            // and bare-infinity signs alike, by the forall-exists refit), so with one
            // present the whole sign dimension collapses and orientations are chosen
            // freely. Checking Inf first split the orbit into two components joined
            // only through the refit (fuzz row 39122: {PosInf, x4-x1} and
            // {PosInf, x1-x4} both stable, one value class). A zero coefficient is
            // Free by the same collapse (see the carrier list above).
            let carrier = if coeff.is_zero()
                || out.iter().any(|f| {
                    matches!(f, Ex::Const)
                        || matches!(f, Ex::Add(v) if v.iter().any(Ex::contains_const))
                }) {
                Carrier::Free
            } else if let Some(i) = out
                .iter()
                .position(|f| matches!(f, Ex::PosInf | Ex::NegInf))
            {
                Carrier::Inf(i)
            } else if let Some(i) = out
                .iter()
                .position(|f| matches!(f, Ex::Add(ts) if ts.iter().any(term_absorbs_negation)))
            {
                Carrier::Absorb(i)
            } else {
                Carrier::Coeff
            };
            if let Some(nc) = coeff.checked_neg() {
                let mut best: Option<(i128, bool, Ex)> = None;
                let mut all_ok = true;
                'subsets: for mask in 0u32..(1u32 << sites.len()) {
                    let mut factors = out.clone();
                    for (bit, &pos) in sites.iter().enumerate() {
                        if mask & (1 << bit) != 0 {
                            match sign_trade_flip(&factors[pos], cx) {
                                Some(nf) => factors[pos] = nf,
                                None => {
                                    all_ok = false;
                                    break 'subsets;
                                }
                            }
                        }
                    }
                    // A flip may mint a factor whose BASE collides with another's
                    // (`sin(-2)` flipped beside `sin(2)`, or a flipped sum landing on
                    // its own twin): `assemble` builds RAW, so the candidate would be
                    // an unmerged duplicate-base bag `mul()` itself never emits -- a
                    // non-canonical spelling that re-parses to a different state
                    // (idempotence loss, B5 hazard). Such masks are not candidates.
                    // Mask 0 never collides (the entry bag is `mul()`-merged), so
                    // `best` is always Some.
                    if mask != 0 {
                        let collides = (0..factors.len()).any(|i| {
                            (i + 1..factors.len()).any(|j| {
                                cmp_ex(
                                    factor_split_ref(&factors[i]).0,
                                    factor_split_ref(&factors[j]).0,
                                    cx.view,
                                ) == std::cmp::Ordering::Equal
                            })
                        });
                        if collides {
                            continue;
                        }
                    }
                    let toggled = mask.count_ones() % 2 == 1;
                    let c = match carrier {
                        Carrier::Coeff => {
                            if toggled {
                                nc
                            } else {
                                coeff
                            }
                        }
                        Carrier::Inf(i) => {
                            if toggled {
                                factors[i] = match &factors[i] {
                                    Ex::PosInf => Ex::NegInf,
                                    Ex::NegInf => Ex::PosInf,
                                    _ => unreachable!(),
                                };
                            }
                            coeff
                        }
                        // The refit eats the toggle: every mask is value-equal as a
                        // fitted family. Normalize the WHOLE sign dimension into the
                        // refit: coefficient magnitude only, bare infinities positive.
                        Carrier::Free => {
                            for f in factors.iter_mut() {
                                if matches!(f, Ex::NegInf) {
                                    *f = Ex::PosInf;
                                }
                            }
                            if coeff.is_negative() {
                                nc
                            } else {
                                coeff
                            }
                        }
                        Carrier::Absorb(i) => {
                            if toggled {
                                let Ex::Add(ts) = &factors[i] else {
                                    unreachable!()
                                };
                                match ts
                                    .iter()
                                    .map(|x| negate_term(x, cx))
                                    .collect::<Option<Vec<Ex>>>()
                                {
                                    Some(mut fl) => {
                                        fl.sort_by(|a, b| add_term_cmp(a, b, cx.view));
                                        factors[i] = Ex::Add(fl);
                                    }
                                    None => {
                                        all_ok = false;
                                        break 'subsets;
                                    }
                                }
                            }
                            coeff
                        }
                    };
                    let cand = assemble(factors, &c);
                    let mu = complexity(&cand, cx.view) as i128;
                    let better = match &best {
                        None => true,
                        Some((bmu, bneg, bex)) => {
                            mu < *bmu
                                || (mu == *bmu && !c.is_negative() && *bneg)
                                || (mu == *bmu
                                    && c.is_negative() == *bneg
                                    && cmp_ex(&cand, bex, cx.view) == std::cmp::Ordering::Less)
                        }
                    };
                    if better {
                        best = Some((mu, c.is_negative(), cand));
                    }
                }
                if all_ok {
                    if let Some((_, _, chosen)) = best {
                        return chosen;
                    }
                }
            }
        }
    }
    assemble(out, &coeff)
}

/// F63: cheap shape test for [`sign_trade_flip`] -- true iff the factor is an odd
/// carrier of a tradeable mixed-sign sum. Must stay the exact mirror of that
/// function's match (this one detects, that one materializes).
fn is_sign_trade_site(f: &Ex, cx: &Cx) -> bool {
    // Delegates to the materializing test: the FILES-BARE gate needs the flipped
    // terms, so a cheap shape-only mirror cannot exist without drifting (the
    // mixed-sign counting attempt regressed 5 corpus rows -- see sign_trade_flip).
    sign_trade_flip(f, cx).is_some()
}

/// F63: is `f` a sign-trade site -- an odd carrier of a mixed-sign sum -- and if so,
/// the factor with that sum FLIPPED (`None` = not a site, or refused: Const-bearing
/// and negation-absorbing sums have absorption owners (H-020/H-030), and negation
/// overflow fails safe). The odd carriers, each a TOTAL identity f(-S) = -f(S) on the
/// extended reals: the sum itself; Pow(S, odd n >= 3) (positive odd only -- the
/// negative-odd identity fails at the one-zero pole); rootn(S, odd m >= 3) (the
/// SIGNED total root is an odd bijection); the eight odd functions (`odd_fun`).
fn sign_trade_flip(f: &Ex, cx: &Cx) -> Option<Ex> {
    let flip_sum = |ts: &[Ex]| -> Option<Vec<Ex>> {
        // Exclusions = the absorption owners' classes: Const-bearing (H-020),
        // absorbing-member (H-030), and BARE-infinity terms (the inf carries the
        // sign itself, H-014 -- `term_absorbs_negation` only matches inf-carrying
        // PRODUCTS, so the bare member needs its own test; missing it let an
        // odd-fun trade flip an inf-bearing sum and inflate mu: fuzz row 131227,
        // P5, the one violation in 200k).
        if ts.iter().any(Ex::contains_const)
            || ts.iter().any(term_absorbs_negation)
            || ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
        {
            return None;
        }
        let mut out = ts
            .iter()
            .map(|t| negate_term(t, cx))
            .collect::<Option<Vec<Ex>>>()?;
        // FILES-BARE gate: the flipped multiset must be a spelling primitive_sum
        // itself would file bare -- if its own orientation decision would WRAP it
        // (`-a - b` rests as -1 x (a+b)), the "site" is not a genuine hiding place:
        // trading into it re-mints the wrapper (bounce), and the pow pre-fold would
        // freeze a non-canonical one-signed sum under the refused carrier (caught by
        // the H-027 completion pin). A sign-counting heuristic is NOT equivalent:
        // the odd-literal fusion parks signs inside function arguments where no
        // coefficient shows them (measured: 5 corpus rows + 4 rust pins regressed
        // under mixed-sign counting).
        if flipped_orientation_wins(&out, mu_sym() as i128, Tie::Keep, cx) {
            return None;
        }
        out.sort_by(|a, b| add_term_cmp(a, b, cx.view));
        Some(out)
    };
    let odd_int_ge3 = |e: &Ex| -> bool {
        matches!(e, Ex::Num(r) if r.as_integer().is_some_and(|n| n >= 3 && n % 2 == 1))
    };
    match f {
        Ex::Add(ts) => flip_sum(ts).map(Ex::Add),
        Ex::Pow(b, e) if odd_int_ge3(e) => {
            let Ex::Add(ts) = &**b else { return None };
            flip_sum(ts).map(|s| Ex::Pow(Box::new(Ex::Add(s)), e.clone()))
        }
        Ex::Fun(op, args)
            if args.len() == 2 && cx.view.tok_is(*op, "rootn") && odd_int_ge3(&args[1]) =>
        {
            let Ex::Add(ts) = &args[0] else { return None };
            flip_sum(ts).map(|s| Ex::Fun(*op, vec![Ex::Add(s), args[1].clone()]))
        }
        Ex::Fun(op, args) if args.len() == 1 && odd_fun(cx.view, *op) => match &args[0] {
            Ex::Add(ts) => flip_sum(ts).map(|s| Ex::Fun(*op, vec![Ex::Add(s)])),
            // B5+B19: an odd function of a LITERAL is a trade site too -- f(-r) = -f(r)
            // exactly, so the sign relocates between the coefficient slot and the
            // literal, and the orbit argmin (mu, then the NON-NEGATIVE-coefficient tie,
            // then cmp_ex) lands every route on I4's ratified direction: the literal
            // owns the sign (`-5 * sin(2)` files as `5 * sin(-2)`; `-1 * sin(2)` as
            // `sin(-2)`, the Mul node dissolving). This is the hoist of `term_join`'s
            // former private fusion arm into the shared owner: `mul()`'s direct
            // assembly and the collector's rebuild now price the SAME orbit. Rebuilt
            // through `fun()`, NEVER raw, so a boundary ground keeps folding
            // (`atanh(-1)` must reach `-inf`); a rebuild that FOLDS to a non-Fun is
            // refused as a site (fail-safe: the orbit stays type-stable, and such
            // grounds fold on their own paths).
            Ex::Num(r) => {
                let flipped = fun(*op, vec![Ex::Num(r.checked_neg()?)], cx);
                matches!(flipped, Ex::Fun(..)).then_some(flipped)
            }
            _ => None,
        },
        _ => None,
    }
}

// F72 (2026-08-09): the residual equal-mu same-coefficient-sign tie in the
// sign-placement owner breaks on `cmp_ex` -- the ONE canonical total order, which
// compares Leaf/Fun tokens by their STRINGS. The dedicated `ex_struct_cmp` this
// replaced compared them by raw token ID, i.e. by INTERNING ORDER, so on a fully
// mu-tied orbit (a -1 coefficient rides mu-free) the winner was a function of which
// literal the input stream interned first: `-1 * (a-b)*(c-d)` and its factor-swapped
// spelling picked OPPOSITE mirror pairings (extreme-lane P2 rows 829655/866090, the
// two 1M regressions of the F62..F71 window).

/// H-015 class (b) (2026-08-04): LOSSY simplification priced reciprocal products WORSE
/// than sound's on the sound-REFUSED set. The blanket zero-set licence makes the lossy
/// `pow` distribute `(a*b)^-1 -> a^-1 * b^-1` unconditionally -- the match-enabling
/// normal form the `$`-cancellation needs (`x * inv(x*y) -> inv y` only cancels through
/// the flat bag) -- but each distributed factor carries its own `^-1` wrapper, so where
/// NO partner cancels, the lossy endpoint priced strictly above sound's licence-refusal
/// form `(a*b)^-1` (registered row 1414: mu 80 vs 72). Neither mode consulted `mu` when
/// choosing the spelling; sound's smaller form was a fail-closed accident.
///
/// This is an OUTPUT-BOUNDARY projection, deliberately NOT a constructor arm. The first
/// build of this fix put the join into `mul`'s final assembly, making the joined form
/// the WORKING representation -- disqualified by the debug stability assert: `pow`'s
/// distribution arm settles its two parts in a nested vacuum `mul` that joined them
/// before the enclosing bag could splice, so construction order leaked into the state
/// (and joined mid-chain states also shift rule reachability and the descent gate's mu
/// ordering). The distributed form is the right working representation (match-enabling,
/// cancellation-visible); the joined form is only ever a better FINAL SPELLING. So the
/// rewrite chain runs entirely in the distributed canon, byte-identical to pre-fix, and
/// [`rejoin_projection`] applies this decision once, bottom-up, to the returned
/// endpoint (measured on the 200k battery: P6 mode-ordering rows 822 -> 780, 42 fixed,
/// 0 added).
///
/// The decision, per `Mul` node: offer the negative-integer-power members back as one
/// joined `(prod f^|n|)^-1`; the strictly `mu`-smaller of the two spellings wins (tie:
/// distributed, the incumbent). The `mu` arithmetic makes the win set exact: for k
/// joined factors the join saves (k-1) `Pow` wrappers and pays one inner `Mul` symbol,
/// so k = 2 INSIDE a wider bag is a tie (distributed stays) while the whole-bag pair
/// (the outer `Mul` unwraps: row 1414's shape) and every k >= 3 join strictly. Two
/// gates keep the decision honest:
///
/// * SOUND ALIGNMENT: the join is offered only where sound-mode `pow` would REFUSE to
///   re-distribute it (`nz_ae_certified` / `certainly_nonneg` per member -- the same
///   conjuncts as the distribution licence, minus the lossy blanket). Where sound
///   distributes, lossy stays distributed too: the two output forms agree there.
///   Idempotence is a funnel property: re-simplifying a joined output re-parses it
///   through `pow`'s blanket distribution back to a working endpoint whose
///   projection must reproduce the output. That endpoint is NOT in general the
///   state the join was priced against (the two construction routes -- positive
///   inner bag vs hidden `^-1` members -- orient trade-site mirrors differently),
///   so the decision is additionally gated on the candidate's RE-SETTLE IMAGE
///   (F68, see the comparison at the return).
/// * ELIGIBILITY excludes BARE-`Const` bases (joining `c^-1 * c'^-1` would let the
///   inner bag's `has_const` flag conflate INDEPENDENT constants -- the flag collapses
///   multiplicity) and literal-`Num` bases (only `0^neg` survives `pow`, and a literal
///   zero never joins a denominator -- the inner product would collapse; cf. the
///   renderer's den rule). DEEP Consts (inside function arguments, `atanh(C*x2)`) are
///   eligible: the inner bag keeps Const-bearing factors verbatim in the opaque bucket,
///   never merged -- the first build excluded them wholesale (`contains_const`) and
///   silently forfeited the join on exactly the sound-refused shapes the projection
///   exists for (measured: whole-bag pairs under `sin(..)` stayed distributed at mu +8
///   over sound's refusal form, H-015 class (a) census).
///
/// Termination: the joined node is built RAW (`Ex::Pow`), never through `pow` (whose
/// lossy blanket would immediately re-distribute it); the inner `mul` sees only
/// positive-power members, so no recursion re-enters this decision. The decision
/// itself is guarded by the F68 re-derivability comparison at the return (see there):
/// a join ships only when it strictly beats its own re-settle image, which is what
/// makes the shipped spelling a fixpoint under re-parse.
fn rejoin_reciprocals(settled: Ex, cx: &Cx) -> Ex {
    let Ex::Mul(members) = &settled else {
        return settled;
    };
    let eligible: Vec<usize> = members
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            let Ex::Pow(b, e) = f else { return None };
            // A literal-infinity base never joins: `inv(inf)` is a DETERMINED zero
            // (sound folds it at construction; lossy keeps it unfolded only as the
            // mask-sentinel cancellation partner). Joining it hides that determined
            // fold behind an opaque base -- measured: every net-regressed battery row
            // had `inf` as a direct member of the joined product. DEEP infinities
            // (inside function arguments, row 1414's `acos(inf * x4)`) stay eligible:
            // no fold sees through those either way.
            if matches!(**b, Ex::Num(_) | Ex::PosInf | Ex::NegInf | Ex::Const) {
                return None;
            }
            let Ex::Num(r) = &**e else { return None };
            match r.as_integer() {
                Some(n) if n < 0 && n.checked_neg().is_some() => Some(i),
                _ => None,
            }
        })
        .collect();
    if eligible.len() < 2 {
        return settled;
    }
    let inner_members: Vec<Ex> = eligible
        .iter()
        .map(|&i| {
            let Ex::Pow(b, e) = &members[i] else {
                unreachable!()
            };
            let Ex::Num(r) = &**e else { unreachable!() };
            let n = r.as_integer().unwrap();
            if n == -1 {
                (**b).clone()
            } else {
                pow((**b).clone(), Ex::int(-n), cx)
            }
        })
        .collect();
    let inner = mul(inner_members, cx);
    {
        let Ex::Mul(v) = &inner else {
            // The positive-power bag collapsed below two factors: no joined SHAPE exists.
            return settled;
        };
        let sound_distributes =
            v.iter().all(|f| cx.nz_ae_certified(f)) || v.iter().all(|f| cx.certainly_nonneg(f));
        if sound_distributes {
            return settled;
        }
    }
    let rest: Vec<Ex> = members
        .iter()
        .enumerate()
        .filter(|(i, _)| !eligible.contains(i))
        .map(|(_, f)| f.clone())
        .collect();
    let assemble = |mut cand: Vec<Ex>| match cand.len() {
        1 => cand.pop().unwrap(),
        _ => {
            cand.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
            Ex::Mul(cand)
        }
    };
    let joined = Ex::Pow(Box::new(inner.clone()), Box::new(Ex::Num(Rat::NEG_ONE)));
    let mut cand = rest.clone();
    cand.push(joined);
    let mut best = assemble(cand);
    // COEFFICIENT COMPLETION (H-015 class (a) residue, 2026-08-04): the bag's rational
    // coefficient can ride INSIDE the joined base, reciprocated -- `0.5 * (x4*S)^-1`
    // IS `(2*x4*S)^-1` -- and when the bag then collapses to the lone joined node, the
    // outer `Mul` unwraps (mu -8, and mu(1/c) vs mu(c) can differ too). Sound's
    // licence-refusal forms keep their coefficient inside the base exactly like this
    // (`inv(-1 * x0 * S)` from `inv(neg(..))` spellings), so without the completion
    // the projection stopped one symbol short of sound's spelling on every such row
    // (23/200k, all gap 8). The funnel stays closed: re-parse distributes the joined
    // base, `pow` folds the literal's reciprocal back into the coefficient, and this
    // decision re-derives.
    //
    // RE-DERIVABILITY GATE (H-027, 2026-08-05, B3+ P1-lossy oracle): the completion is
    // funnel-stable only if the reciprocated coefficient SURVIVES VERBATIM as a `Num`
    // member of the joined base. `mul()` inside the completion runs the full assembly,
    // and the H-020 sign fold may CONSUME a negative `cinv` into a Const-bearing Add
    // member -- a strict mu win whose source is the sign fold, not the join, and one
    // the funnel cannot re-derive: the re-parsed (absorbed) state has no Num left in
    // the bag, the completion cannot re-fire, the plain join ties, and
    // tie -> distributed -- so the joined spelling shipped exactly once and the next
    // lossy pass moved (P1-lossy-idempotence, 2/200k, fuzz rows 92892 + 115616).
    // When the coefficient survives (positive c always; negative c beside sums the
    // fold does not touch), re-parse re-extracts it through pow's distribution and
    // the completion re-derives -- including sound-aligned `inv(-1 * x0 * S)` joins
    // (blanket refusal of negative c donated a uniform mu gap of 8 on 8/200k rows).
    if let Some(ci) = rest.iter().position(|f| matches!(f, Ex::Num(_))) {
        let Ex::Num(c) = &rest[ci] else {
            unreachable!()
        };
        if let Some(cinv) = c.checked_inv() {
            let inner2 = mul(vec![Ex::Num(cinv), inner.clone()], cx);
            let survives = matches!(&inner2, Ex::Mul(v)
                if v.iter().any(|f| matches!(f, Ex::Num(n) if *n == cinv)));
            // F63 amendment to the H-027 gate: a NEGATIVE reciprocated coefficient
            // beside a sign-trade site is no longer funnel-stable -- pow()'s
            // odd-negative pre-fold now folds that sign into the site on re-parse
            // (the row-310 pole-orientation cure), so the joined spelling cannot
            // re-derive verbatim (measured: corpus row 33 lossy idempotence). The
            // completion skips exactly the pre-fold's firing condition.
            let funnel_stable = !cinv.is_negative()
                || !matches!(&inner2, Ex::Mul(v)
                    if v.iter().any(|f| is_sign_trade_site(f, cx)));
            if survives && funnel_stable {
                let joined2 = Ex::Pow(Box::new(inner2), Box::new(Ex::Num(Rat::NEG_ONE)));
                let mut cand2: Vec<Ex> = rest
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != ci)
                    .map(|(_, f)| f.clone())
                    .collect();
                cand2.push(joined2);
                let candidate2 = assemble(cand2);
                if complexity(&candidate2, cx.view) < complexity(&best, cx.view) {
                    best = candidate2;
                }
            }
        }
    }
    // F68 RE-DERIVABILITY COMPARISON (fuzz rows 303116/472156/603382 and, after the
    // first-cut fix, 59022/194208/308376/593418/713415/790992 -- all introduced or
    // exposed by F63): the join used to be judged against the WORKING state, but the
    // working state is not what a shipped join re-parses to. Parse builds the joined
    // base's inner bag POSITIVELY first, where bare trade sites get re-oriented (the
    // Coeff orbit's argmin, or the H-020 Free flip beside a Const-bearing member)
    // before the blanket re-distributes -- while the division route builds the same
    // value member-by-member with every site HIDDEN inside a `^-1` wrapper (not a
    // carrier: the one-zero pole trilemma), preserving arrival mirrors. Two routes,
    // two working states, and a join priced against one route re-derived into the
    // other: the strict win evaporated into a tie on the second pass and the lossy
    // endpoint moved (P1-lossy). The gate is the H-027 principle applied to the
    // WHOLE decision: a join ships only if it strictly beats BOTH the working state
    // (the improvement test, as before) AND its own RE-SETTLE IMAGE -- the state its
    // re-parse provably lands on (inner is `mul`-built == what parse rebuilds, `mul`
    // is idempotent on it, so the image is exact). A shipped join then re-picks
    // itself on every later pass (fixpoint by construction), and a refused join
    // ships the working state, whose rendering re-parses member-by-member through
    // the hidden-site route back to itself.
    let resettle = |cand: &Ex| -> Ex {
        let members: Vec<Ex> = match cand {
            Ex::Mul(v) => v.clone(),
            other => vec![other.clone()],
        };
        let mut parts: Vec<Ex> = Vec::with_capacity(members.len() + 2);
        for m in members {
            match m {
                Ex::Pow(b, e)
                    if matches!(&*e, Ex::Num(r) if *r == Rat::NEG_ONE)
                        && matches!(&*b, Ex::Mul(_)) =>
                {
                    let Ex::Mul(bm) = *b else { unreachable!() };
                    for f in bm {
                        parts.push(pow(f, Ex::Num(Rat::NEG_ONE), cx));
                    }
                }
                other => parts.push(other),
            }
        }
        mul(parts, cx)
    };
    let mu_best = complexity(&best, cx.view);
    if mu_best < complexity(&settled, cx.view) && mu_best < complexity(&resettle(&best), cx.view) {
        best
    } else {
        settled
    }
}

/// The output-boundary rejoin walk (see [`rejoin_reciprocals`]): apply the reciprocal
/// join decision bottom-up to every `Mul` node of the final lossy state. STRUCTURAL
/// rebuild on purpose -- routing through the constructors would re-distribute the joins
/// (`pow`'s lossy blanket). Bags whose members changed are re-sorted under their own
/// comparator so the projected tree spells deterministically.
pub fn rejoin_projection(e: Ex, cx: &Cx) -> Ex {
    match e {
        Ex::Add(v) => {
            let mut terms: Vec<Ex> = v.into_iter().map(|x| rejoin_projection(x, cx)).collect();
            terms.sort_by(|a, b| add_term_cmp(a, b, cx.view));
            Ex::Add(terms)
        }
        Ex::Mul(v) => {
            let mut m: Vec<Ex> = v.into_iter().map(|x| rejoin_projection(x, cx)).collect();
            m.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
            rejoin_reciprocals(Ex::Mul(m), cx)
        }
        Ex::Pow(b, ex) => Ex::Pow(
            Box::new(rejoin_projection(*b, cx)),
            Box::new(rejoin_projection(*ex, cx)),
        ),
        Ex::Fun(f, v) => Ex::Fun(f, v.into_iter().map(|x| rejoin_projection(x, cx)).collect()),
        leaf => leaf,
    }
}

/// Rebuild `base^exp` (rational or symbolic-with-count) through the [`pow`] constructor so
/// integer-power distribution and exact folds apply.
fn rebuild_factor(base: Ex, sym: Option<Ex>, r: Rat, cx: &Cx) -> Ex {
    match sym {
        None => pow(base, Ex::Num(r), cx),
        Some(e) => {
            if r.is_one() {
                pow(base, e, cx)
            } else {
                let scaled = mul(vec![Ex::Num(r), e], cx);
                pow(base, scaled, cx)
            }
        }
    }
}

/// The canonical power constructor. Folds and gates:
///
/// * `e == 1` -> base. `e == 0` -> 1 (TOTAL under the engine's C99-aligned pow: `pow(x, 0) = 1`
///   for EVERY x including NaN and the infinities).
/// * `base == 1` -> 1 (TOTAL: `pow(1, y) = 1` for every y including NaN).
/// * rational^integer folds exactly (checked; overflow keeps the symbolic form).
///   rational^(p/q) folds exactly when the q-th root is exact (`4^(1/2) -> 2`,
///   `(-8)^(1/3)` is NOT folded: the engine's rational pow is NaN on negative bases with
///   non-integer exponents -- the REAL odd root is the separate `pow1_3` operator).
///   `0^negative` and inexact roots stay symbolic (the rules-pass numeric fold owns IEEE edge
///   parity with the shipped engine).
/// * `(a^r)^s -> a^(r*s)` under the composition licence: sound iff (`r` integer AND (`s` integer
///   OR `r` odd)) OR (`r` non-integer AND `r*s` non-integer) OR the base is certainly
///   non-negative. Counterexamples pinned in tests: `(a^2)^(1/2) = |a| != a`;
///   `(a^(1/2))^2 = a` only on `a >= 0`.
/// * `(a*b)^n` for INTEGER n distributes over the factors (TOTAL as extended-real evaluations;
///   for non-integer exponents `(ab)^(1/2) != a^(1/2) b^(1/2)` on `a, b < 0`).
pub fn pow(base: Ex, exp: Ex, cx: &Cx) -> Ex {
    if let Ex::Num(e) = &exp {
        if e.is_zero() {
            return Ex::Num(Rat::ONE);
        }
        if e.is_one() {
            return base;
        }
    }
    if matches!(&base, Ex::Num(b) if b.is_one()) {
        return Ex::Num(Rat::ONE);
    }
    // rational^rational exact folds.
    if let (Ex::Num(b), Ex::Num(e)) = (&base, &exp) {
        // DETERMINED POLE (H1 finding, 2026-08-02): a literal-zero base with a
        // negative literal exponent is exact non-finite arithmetic under the
        // one-zero convention (0^n = +0 for n > 0, so 0^{-n} = 1/+0 = +inf; the
        // same convention the signed-inf licence settled for `inv 0`). Without
        // this arm, exponent collection could BUILD `Pow(0, -2)` (from
        // `inv 0 * inv 0`) -- a state whose serialization `inv pow 0 2` re-reads
        // to `inv 0` because the inner `pow 0 2` folds on the way in: a genuine
        // violation of the canonical-form uniqueness contract, caught by the
        // debug `stable()` assertion (release outputs were value-correct; the
        // contract was not).
        if b.is_zero() && e.is_negative() {
            return Ex::PosInf;
        }
        if let Some(n) = e.as_integer() {
            if let Some(r) = b.checked_pow_int(n) {
                return Ex::Num(r);
            }
        } else if !b.is_negative() {
            // b^(p/q): exact iff the q-th root of b is rational.
            if let Some(root) = b.checked_root(e.den()) {
                if let Some(r) = root.checked_pow_int(e.num()) {
                    return Ex::Num(r);
                }
            }
        }
    }
    // DETERMINED RECIPROCAL OF INFINITY (dual of the H1 pole arm, 2026-08-02): an
    // infinite literal base under a negative literal exponent is the same
    // reciprocal-convention arithmetic as the pole arm above -- inf^q = 1/inf^|q| = 0
    // for every rational q < 0, and (-inf)^{-n} = 1/(+-inf) = 0 for every integer
    // n > 0 (unsigned zero). It is the unique determined case where an infinite
    // ground yields a FINITE value, so the interval classification arm (which folds
    // only non-finite classes) cannot carry it; without it `inv float("inf")`
    // survives as a composite and the mine re-learns 1/inf as data. (-inf)^q with
    // q < 0 NON-integer stays unfolded here: the principal real power of a negative
    // base is NaN territory, which interval classification certifies instead.
    // Infinite EXPONENTS are untouched: b^inf is a limit notion, not arithmetic --
    // that family (exp(-inf) -> 0 etc.) arrives as certified rules by doctrine.
    // LOSSY mode keeps the composite: folding `inv inf -> 0` at construction
    // destroys the structural x * x^-1 pair the relaxed $-certificate cancels
    // (`(inf/inf) * x0 -> x0`, the mask-sentinel doctrine) -- the reciprocal
    // must still EXIST when the mul bag looks for its cancellation partner. The
    // keep EXPIRES at the phase-1 fixpoint (`sentinels_expired`, H-015): a
    // surviving reciprocal is unpartnered, IS the value 0, and blocks the rules
    // its unfolded shape hides from.
    if !cx.lossy || cx.sentinels_expired {
        if let Ex::Num(e) = &exp {
            if e.is_negative() {
                match &base {
                    Ex::PosInf => return Ex::Num(Rat::ZERO),
                    Ex::NegInf if e.is_integer() => return Ex::Num(Rat::ZERO),
                    _ => {}
                }
            }
        }
    }
    // Integer powers distribute over products -- TOTAL for non-negative and for EVEN
    // integer exponents (at a zero of the product both spellings give 0 resp. +inf; at
    // infinities they agree for every integer). For ODD NEGATIVE exponents the identity
    // (a*b)^n = a^n * b^n FAILS under the one-zero contract wherever the product VANISHES
    // with a negative co-factor: (-1 * (x-|x|))^-1 is +inf on the whole half-line x >= 0
    // (1/0 = +inf) while -1 * (x-|x|)^-1 is -inf there -- a positive-measure
    // infinity-sign change (surfaced by the 1M corpus gate, 2 ppm). The failure set
    // lives entirely on the factors' ZERO SETS (every infinite/NaN configuration agrees
    // on both sides), so the licence is per-factor NONZERO-A.E. (`nz_ae_licensed`:
    // variables and Const qualify -- null hyperplane resp. null {c=0} in c-space --
    // abs-style factors with half-line zero sets are refused by the zero-set
    // certificate) or an all-certainly-nonneg bag (at a zero every co-factor is >= 0,
    // so both spellings agree on +inf). Refusal keeps Pow(Mul[..], n): on the refused
    // set the two spellings are genuinely DIFFERENT functions, so canonicalization
    // must not identify them.
    if let (Ex::Mul(v), Ex::Num(e)) = (&base, &exp) {
        if e.is_integer() {
            let licensed = !e.is_negative()
                || e.as_integer().is_some_and(|n| n % 2 == 0)
                || v.iter().all(|f| cx.nz_ae_licensed(f))
                || v.iter().all(|f| cx.certainly_nonneg(f));
            if licensed {
                // F63: for a NEGATIVE ODD exponent, fold a negative bag coefficient
                // into the first sign-trade site BEFORE distributing (exact at the
                // finite bag level: -c * S == c * (-S)). After distribution the sum
                // sits under a REFUSED odd-negative carrier -- (S)^-n and -((-S))^-n
                // are a.e.-equal but POLE-DIFFERENT, so no owner may reconcile them
                // later; whether the sign had already met the bag decided which
                // rendering the a.e. licence produced (corpus row 310: chain held
                // (1-x4)^-1 while parse of its own serialization built
                // -(x4-1)^-1). The pre-fold is UNCONDITIONAL (not mu-judged):
                // post-distribute there is no legal trade, so entry-independence
                // requires one fixed pre-state.
                let vv: Vec<Ex>;
                let bag: &[Ex] = if e.is_negative() && e.as_integer().is_some_and(|n| n % 2 != 0) {
                    let ci = v
                        .iter()
                        .position(|f| matches!(f, Ex::Num(r) if r.is_negative()));
                    let si = v.iter().position(|f| is_sign_trade_site(f, cx));
                    match (ci, si) {
                        (Some(ci), Some(si)) => {
                            let nc = match &v[ci] {
                                Ex::Num(r) => r.checked_neg().map(Ex::Num),
                                _ => None,
                            };
                            match (sign_trade_flip(&v[si], cx), nc) {
                                (Some(nf), Some(nc)) => {
                                    let mut w = v.clone();
                                    w[si] = nf;
                                    w[ci] = nc;
                                    if matches!(&w[ci], Ex::Num(r) if r.is_one()) {
                                        w.remove(ci);
                                    }
                                    vv = w;
                                    &vv
                                }
                                _ => v,
                            }
                        }
                        _ => v,
                    }
                } else {
                    v
                };
                // F83 (owner-ruled F75 group D, 2026-08-11; extreme row 508487): a
                // NEGATIVE rational coefficient that SURVIVED the F63 pre-fold (no
                // sign-trade site in the bag) flips the pole sign at every zero of a
                // co-factor: the source folds the product to THE unsigned zero first
                // (`inv(0) = +inf`, contract §9.2/§9.8), while the distributed
                // spelling ships `sign(c)·inf` -- a REAL value change at a constructed
                // rational exceptional point, the class the licence registry (F66)
                // refuses for mined rules, arrived at constructor level
                // (`rootn(x0/(-3), -1) -> -3/x0`: exp of it turned 0.0 into +inf).
                // Refuse the distribution unless every non-coefficient factor
                // certainly never vanishes -- nonzero-a.e. is NOT enough here, the
                // exceptional point IS the null set. Positive coefficients agree at
                // the atom and stay licensed; traded signs (F63) stay licensed.
                // SOUND MODE ONLY: lossy canonicalisation keeps its training
                // semantics; the phase-2 sound re-canon applies the guard to every
                // endpoint. Boundary note (owner-visible, audit F83): a negative
                // VARIABLE co-factor at another factor's zero flips the same way,
                // but that is not a CONSTRUCTED-rational exceptional point -- it
                // stays under the standing a.e. doctrine.
                // VARIANT MEASUREMENT (owner boundary question, 2026-08-11): the
                // `*r != Rat::NEG_ONE` clause is the NARROW line -- pure-sign
                // coefficients keep the standing §9.8.4-priced compression; only
                // magnitude-carrying negative literals refuse. The BROAD line
                // (every negative coefficient) is the same condition without it.
                let neg_coeff_stuck = !cx.lossy
                    && e.is_negative()
                    && e.as_integer().is_some_and(|n| n % 2 != 0)
                    && bag
                        .iter()
                        .any(|f| matches!(f, Ex::Num(r) if r.is_negative() && *r != Rat::NEG_ONE))
                    && bag
                        .iter()
                        .any(|f| !matches!(f, Ex::Num(_)) && !cx.certainly_nonvanishing(f));
                if !neg_coeff_stuck {
                    let parts: Vec<Ex> = bag
                        .iter()
                        .cloned()
                        .map(|f| pow(f, Ex::Num(*e), cx))
                        .collect();
                    return mul(parts, cx);
                }
                // refused: fall through to the kept carrier below.
            }
            // REFUSED: normalize the kept form to the exponent -1 shape so refusal is
            // CONFLUENT. The same object reaches this point structured two ways -- the
            // exponent composed before meeting the bag (pow-of-pow: Pow(Mul[-1,A], -3))
            // or after (parse of the emitted spelling: Pow(Mul[-1, A^3], -1)) -- and a
            // form-dependent refusal oscillates between them across passes (observed:
            // idempotence broke on 3/400 corpus rows). (prod f)^n == (prod f^|n|)^-1 is
            // TOTAL for odd n < 0: x^n == (x^|n|)^-1 pointwise (at 0 both +inf, at
            // +-inf both 0, finite exact) and the inner |n| is a NON-NEGATIVE integer
            // distribution (total, licensed above).
            if let Some(n) = e.as_integer() {
                if n < -1 {
                    let parts: Vec<Ex> =
                        v.iter().cloned().map(|f| pow(f, Ex::int(-n), cx)).collect();
                    return pow(mul(parts, cx), Ex::int(-1), cx);
                }
            }
        }
    }
    // Power-of-power composition under the licence. (x^r)^s -> x^(r*s) is a THEOREM
    // only where both sides agree on the negative half-line; the three sound cases:
    //   * s integer: an integer power of a total function, exact everywhere;
    //   * r odd AND POSITIVE AND r*s NON-integer: on x < 0 the source is NaN (x^r < 0
    //     under a non-integer outer exponent) and the target is NaN too (r*s
    //     non-integer). The non-integer-product conjunct is LOAD-BEARING: with r*s
    //     integer the target is FINITE on the whole negative half-line while the source
    //     is NaN ((x^5)^0.2 -> x would fabricate values). Oddness alone licenses nothing
    //     about the outer principal root.
    //
    //     For a NEGATIVE odd r the base must additionally be unable to reach an infinity
    //     (`never_infinite`), and the absence of that conjunct was H-056. The "x^r < 0 on
    //     x < 0" step is false at x = -inf when r is NEGATIVE: there x^r is ZERO (one
    //     zero, contract §9.2 -- an f64 zero's sign is measurement rendering, never a
    //     value), so the source is DEFINED (0^s) while the target x^(r*s) is NaN on a
    //     negative base with a non-integer exponent. That is defined -> undefined, which
    //     §9.1's R2 forbids at ZERO measure tolerance: R3's null-set licence covers
    //     undefined -> defined only, "the one asymmetry". Reachable at a REAL valuation,
    //     not merely at the point -inf, because a slot binds an arbitrary SUBTREE and
    //     `neg(inv(x0))` is -inf at x0 = 0 -- the Dirac on a null set that contract v2's
    //     own repair #2 warns about. A LEAF base keeps its fold: §9.1 quantifies variables
    //     over the REALS and §10.8 makes a free constant's infinity a limit, never an
    //     attained value, so only a COMPOUND can reach one. Counterexample:
    //         x1 - rootn(inv(neg(inv(x0))), 4)  ->  x1 - inv(pow(-1/x0, 1/4))
    //         at x0 = 0:  source 1.0,  target nan.
    //     `cert_fin` cannot rescue the negative case: it certifies finiteness ALMOST
    //     EVERYWHERE, and the whole defect lives on the null set.
    //   * a certainly-non-negative base, where every spelling agrees.
    // (r even with r*s non-integer must stay refused: (x^2)^(1/4) = |x|^(1/2) is
    // DEFINED on x < 0 while x^(1/2) is NaN -- the dual failure direction.)
    if let (Some((inner_base, r)), Ex::Num(s)) = (as_rational_power(&base, cx), &exp) {
        {
            let r = &r;
            let licence = if cx.lossy {
                true
            } else if let Some(ri) = r.as_integer() {
                s.is_integer()
                    || (ri % 2 != 0
                        && (ri > 0 || cx.never_infinite(inner_base))
                        && r.checked_mul(s).is_some_and(|p| !p.is_integer()))
                    || cx.certainly_nonneg(inner_base)
            } else {
                let rs = r.checked_mul(s);
                rs.is_some_and(|p| !p.is_integer()) || cx.certainly_nonneg(inner_base)
            };
            if licence {
                if let Some(p) = r.checked_mul(s) {
                    return pow(inner_base.clone(), Ex::Num(p), cx);
                }
            }
        }
    }
    // The SAME composition where the power is spelled as a FUNCTION: `pow(exp a, b)` and
    // `pow(e, b)` collapse into one `exp`. See `compose_e_power` -- the base is positive
    // everywhere, so no licence is needed, and the fold condition is what keeps the arm
    // descending. Placement is load-bearing: it runs BEFORE the unit-fraction -> `rootn`
    // conversion at the bottom of this function, the rewrite that blinded both
    // `pow np.e _0 -> exp _0` and `pow exp (-1) _0 -> exp neg _0` at a unit-fraction
    // exponent (measured before this arm: `pow(exp(-1), 1/2)` shipped as
    // `rootn(exp(-1), 2)` at mu 20,000 where `exp(-1/2)` is 11,585).
    if let Some(composed) = compose_e_power(&base, &exp, cx) {
        return composed;
    }
    // ROOT/POWER INVERSE under licence: `pow(rootn(b, n), n) -> b`. TOTAL for odd n (the
    // signed root is a bijection on the extended reals: `rootn(-8,3)^3 = -8`); for even n
    // it needs a certainly-non-negative base, because `rootn(b,2)^2` is NaN on `b < 0`
    // where `b` itself is defined -- unlicensed, that would be a positive-measure
    // extension. This restores the licence the pow-of-pow arm used to supply back when
    // even roots were spelled as powers; without it `rootn(abs(x),2)^2` stopped folding
    // to `abs(x)`.
    //
    // The NEGATIVE mirror `pow(rootn(b, n), -n) -> 1/b` carries the same licence by the same
    // argument: reciprocating a total identity is total (`1/nan = nan`, `1/0 = +inf`,
    // `1/+inf = 0`), and the even case needs the same non-negative base. Without it the
    // reduction was ROUTE-DEPENDENT -- `pow(rootn(x,3), -3)` stayed put while its own
    // explicit spelling `inv pow rootn x 3 3` reduced to `inv x`, so the explicit projection
    // was not a fixpoint of the parser (F1/F2, AUDIT 2026-08-06).
    if let (Ex::Fun(rop, rargs), Ex::Num(e)) = (&base, &exp) {
        if rargs.len() == 2 && cx.view.tok_is(*rop, "rootn") {
            if let (Ex::Num(idx), Some(k)) = (&rargs[1], e.as_integer()) {
                let n = k.abs();
                if n >= 2
                    && idx.as_integer() == Some(n)
                    && (n % 2 == 1 || cx.certainly_nonneg(&rargs[0]))
                {
                    if k >= 2 {
                        return rargs[0].clone();
                    }
                    return pow(rargs[0].clone(), Ex::int(-1), cx);
                }
            }
        }
    }
    // ODD-ROOT COMPOSITION under licence: `pow(rootn(b, m), s) -> pow(b, s/m)` for an ODD
    // index `m >= 3` and a NON-INTEGER rational `s`.
    //
    // An odd `rootn` is deliberately NOT readable as `b^(1/m)` (`factor_split`), because the
    // signed total root and the principal power are different functions on `b < 0`, so the
    // generic composition licence cannot reach this shape. It is nonetheless TOTAL exactly
    // when `s` is not an integer, and the case split is short: on `b < 0` the left side is a
    // negative real raised to a non-integer power, i.e. NaN, and the right side is `b^(s/m)`
    // whose exponent is also non-integer -- `s = p/q` in lowest terms with `q >= 2` gives
    // `s/m = p/(qm)` with reduced denominator `qm/gcd(p,m) >= q >= 2` -- hence NaN too; on
    // `b > 0` both are `b^(s/m)`; on `b == 0` both are `0^(s/m)`, agreeing at the pole by the
    // sign of the exponent; `+-inf` and NaN follow the same two lines.
    //
    // The INTEGER `s` case is genuinely unsound and stays excluded: `rootn(-8,3)^2 = 4` while
    // `(-8)^(2/3)` is NaN, a positive-measure disagreement.
    if let (Ex::Fun(rop, rargs), Ex::Num(s)) = (&base, &exp) {
        if rargs.len() == 2 && cx.view.tok_is(*rop, "rootn") && !s.is_integer() {
            if let Ex::Num(idx) = &rargs[1] {
                if let Some(m) = idx.as_integer() {
                    if m >= 3 && m % 2 == 1 {
                        if let Some(t) = Rat::new(1, m).and_then(|inv| s.checked_mul(&inv)) {
                            return pow(rargs[0].clone(), Ex::Num(t), cx);
                        }
                    }
                }
            }
        }
    }
    // A LITERAL BASE AT AN INFINITE EXPONENT IS DETERMINED: `b^(+-inf)` is a STEP in |b|,
    // and with `b` a literal the step is decided, so this is exact arithmetic and not a
    // limit notion (the spike-flattener refusal is about a SYMBOLIC base, where the step can
    // be straddled). Judge: CERTIFIED at every case below.
    //
    // The engine already folded the halves that go to +inf (`2^inf`, `(1/2)^(-inf)`) through
    // the determined-reciprocal arm; the halves that go to ZERO were MISSING, and it was the
    // MINER that said so -- the reciprocal-base arm below made `inv pow 2 _0 -> pow 0.5 _0`
    // native, and the re-mine promptly minted eighteen ground rules
    // (`inv pow k float("inf") -> 0`, k in -10..10) to cover what the wildcard rule had been
    // reaching through. An incomplete arm shows up as rules: the same signal that found
    // B1(d)'s missing sign half, F49's write half and the parity arm's abs propagation.
    //
    // |b| = 1 and b = 0 stay OUT: `(-1)^inf` is 1 and `0^inf` is 0, both certified, but
    // neither is reached by the reciprocal arm and neither was asked for here.
    if matches!(&exp, Ex::PosInf | Ex::NegInf) {
        if let Ex::Num(bv) = &base {
            if !bv.is_zero() {
                // |b| against 1, exactly: `Rat` is normalized with a positive denominator,
                // so comparing |num| with den decides it without any division.
                let mag = bv.num().unsigned_abs().cmp(&bv.den().unsigned_abs());
                let to_zero = matches!(
                    (&exp, mag),
                    (Ex::PosInf, std::cmp::Ordering::Less)
                        | (Ex::NegInf, std::cmp::Ordering::Greater)
                );
                if to_zero {
                    return Ex::Num(Rat::ZERO);
                }
            }
        }
    }
    // A RECIPROCAL BASE AT AN INFINITE EXPONENT MOVES INTO THE EXPONENT'S SIGN:
    // `(1/t)^(+-inf) -> t^(-+inf)`. Judge: CERTIFIED (the general symbolic-base version
    // `pow(1/t, s) -> pow(t, -s)` is only TOLERATED, so this is deliberately the infinite
    // case ALONE, where the pole configurations collapse). The descent is unconditional --
    // the `inv` node disappears and the two infinities cost the same, 32,000 -> 24,000.
    if matches!(&exp, Ex::PosInf | Ex::NegInf) {
        if let Ex::Pow(inner, k) = &base {
            if matches!(&**k, Ex::Num(r) if *r == Rat::NEG_ONE) {
                let flipped = if matches!(&exp, Ex::PosInf) {
                    Ex::NegInf
                } else {
                    Ex::PosInf
                };
                return pow((**inner).clone(), flipped, cx);
            }
        }
    }
    // A LITERAL BASE ABSORBS THE EXPONENT'S SIGN: `b^(-t) -> (1/b)^t` for a literal `b`.
    // Judge: CERTIFIED at every literal, including the two the one-zero convention decides
    // (`0^(-t) -> inf^t`, `inf^(-t) -> 0^t`) and negative bases.
    //
    // GATED ON THE FLIP REMOVING A NODE, which is a structural test and NOT a mu guard --
    // the direction is genuinely context-dependent and this is where it turns. Measured:
    //   exponent `Mul[-1, x0]`        26,000 -> 18,585   the Mul wrapper collapses     TAKE
    //   exponent `Mul[-1, sin x0]`    34,000 -> 26,585   same                          TAKE
    //   exponent `Num(-1/2)`          13,585 -> 12,585   sign is free, rootn pays      TAKE
    //   exponent `Num(-1/3)`          14,000 -> 13,585   sign is free                  TAKE
    //   exponent `Mul[-1, x0, x1]`    34,000 -> 34,585   the Mul SURVIVES              REFUSE
    // The reciprocal costs the base exactly one bit (§10.10: `mu(1/n) = mu(n) + 1`), so the
    // flip pays iff it buys back a whole node. With two or more members the bag survives the
    // flip, nothing is bought, and the arm would ASCEND -- which is why the shipped rules
    // could never be a blanket constructor arm and why this is not one either.
    // The literal's reciprocal, under the one-zero convention the judge priced these against
    // (`1/0 = +inf`). RATIONAL LITERALS ONLY, and both exclusions were paid for:
    //
    //   * `-inf` is REFUSED because the judge KILLs it, bc-positive-measure: `(-inf)^(-t)` is
    //     NaN at every non-integer `t` while `0^t` is 0 there, so the flip would extend
    //     undefined to defined on positive measure. I had written `PosInf | NegInf` from
    //     symmetry WITHOUT asking, and the h045 wall caught it -- the judge-first rule exists
    //     precisely because symmetry is not an argument.
    //   * `+inf` is judge-CERTIFIED but still refused HERE: `Pow(inf, negative)` is already
    //     owned by the determined-reciprocal arm above, which carries a LOSSY gate this arm
    //     does not (`inv inf` must survive as a composite in lossy mode so the $-certificate
    //     can find its cancellation partner -- the mask-sentinel doctrine). Taking it here
    //     preempted that gate and broke `(inf/inf)*x0 -> x0`. The rules keep the +inf case.
    let literal_reciprocal = |e: &Ex| -> Option<Ex> {
        match e {
            Ex::Num(b) if b.is_zero() => Some(Ex::PosInf),
            Ex::Num(b) => b.checked_inv().map(Ex::Num),
            _ => None,
        }
    };
    if let Some(positive) = cx.carried_negative(&exp) {
        let single_member = match &exp {
            Ex::Num(_) => true,
            Ex::Mul(v) => v.iter().filter(|m| !matches!(m, Ex::Num(_))).count() == 1,
            _ => false,
        };
        if single_member {
            if let Some(r) = literal_reciprocal(&base) {
                return pow(r, positive, cx);
            }
        }
    }
    // The `inv` SPELLING of the same fact: `1/(b^t) -> (1/b)^t` at a literal `b`. It arrives
    // as `Pow(Pow(b, t), -1)`, which the pow-of-pow composition above cannot touch -- that
    // arm needs a RATIONAL inner exponent and here `t` is symbolic. No structural gate is
    // needed: the exponent is carried across untouched, so the outer `Pow` node disappears
    // against the base's one bit and the descent is unconditional (26,000 -> 18,585).
    // Judge: CERTIFIED at a literal base (`inv pow 3 _0 -> pow (1/3) _0`), TOLERATED at a
    // symbolic one -- literal only, exactly as above.
    if matches!(&exp, Ex::Num(r) if *r == Rat::NEG_ONE) {
        if let Ex::Pow(ib, ie) = &base {
            if let Some(r) = literal_reciprocal(ib) {
                return pow(r, (**ie).clone(), cx);
            }
        }
    }
    // EVEN UNIT-FRACTION EXPONENT -> `rootn` (owner 2026-08-06). For an even index the two
    // spellings are the SAME function -- both are the principal root, NaN on negatives --
    // so the orientation is free, and mu now settles it: `rootn x 2` prices 18 against
    // `pow x 1/2` at 19, because a genuine fraction pays for its denominator while the
    // integer index does not. It is also 3 tokens against 7 in the tagged form, and the
    // index stays inside a consumer's integer vocabulary.
    //
    // ODD n must NOT convert, in either direction: `rootn(-8,3) = -2` where `pow(-8,1/3)`
    // is NaN, so those are different functions. Only UNIT fractions convert: `pow(x, 3/2)`
    // is not `rootn(x,2)`, and rewriting it as `pow(rootn(x,2),3)` is a separate question.
    //
    // Runs LAST, so every exact fold, pole, infinity and composition arm above has already
    // had its chance -- `pow(4, 1/2)` still folds to `2` and never reaches here.
    if let Ex::Num(e) = &exp {
        let n = e.den();
        if e.num() == 1 && n >= 2 && n % 2 == 0 {
            return fun(cx.view.intern("rootn"), vec![base, Ex::int(n)], cx);
        }
    }
    // F63 even-carrier orientation (owner-ruled 2026-08-08, full family): an EVEN
    // integer exponent erases the base's sign -- (-S)^2k == S^2k EXACTLY, at the
    // poles too (negative even: at S = 0 both are +inf, at +-inf both 0) -- so a
    // mixed-sign Add base's orientation is FREE and both orientations denote one
    // value. Canonical form picks the fow winner; the exact tie (pure mirrors) is
    // the one place the historical lex rule still decides, because neither spelling
    // is structurally distinguished (Tie::Lex). Const-bearing and absorbing sums
    // keep their absorption owners. Runs at the very end: base and exponent are in
    // final form, and the flip re-enters `pow` exactly once (fow on the flipped
    // orientation answers keep -- class-antisymmetry).
    if let (Ex::Add(ts), Ex::Num(r)) = (&base, &exp) {
        if r.as_integer().is_some_and(|n| n != 0 && n % 2 == 0)
            && !ts.iter().any(Ex::contains_const)
            && !ts.iter().any(term_absorbs_negation)
            && !ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
            && flipped_orientation_wins(ts, 0, Tie::Lex, cx)
        {
            if let Some(mut flipped) = ts
                .iter()
                .map(|t| negate_term(t, cx))
                .collect::<Option<Vec<Ex>>>()
            {
                // FILES-BARE gate (same rule as sign_trade_flip): only adopt an
                // orientation primitive_sum itself would file bare.
                if !flipped_orientation_wins(&flipped, mu_sym() as i128, Tie::Keep, cx) {
                    flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
                    return pow(Ex::Add(flipped), exp, cx);
                }
            }
        }
    }
    Ex::Pow(Box::new(base), Box::new(exp))
}

/// The canonical function-application constructor. No folding here on purpose: literal
/// evaluation (`sin(1) -> 0.841...`) and `<constant>`-collapse (`sin(C) -> C`) are RULES-PASS
/// FALLBACKS, exactly as in the shipped engine -- running them at construction would destroy
/// the exact rules' chance to fire first (`sin(pi) -> 0` must beat `sin(3.14159...) ->
/// 1.22e-16`).
pub fn fun(op: Tok, args: Vec<Ex>, cx: &Cx) -> Ex {
    // `exp(1) -> E`. The two spellings denote the SAME constant exactly -- this is a
    // choice of symbol, not an evaluation, so the rules-pass doctrine above does not
    // cover it (nothing is rounded and no exact rule can be preempted; `exp(0) -> 1`
    // IS a literal evaluation and stays with the rules, where it already lives).
    //
    // It is the WRITE half of the invariant `compose_e_power` READS: that helper takes
    // the leaf `E` to mean `E^1`, so the canon must have exactly one spelling for the
    // value or the composition can land on the dearer one. Measured, and not
    // hypothetically: with the composition in and this fold out, `inv(inv(e))` and
    // `x0/exp(-1)` both settled on `exp 1` at mu 10,000 where `np.e` is 8,000, and the
    // re-mine dropped 42 rules that had been carrying exactly that respelling in
    // context (`* exp 1 _0 -> * np.e _0`, `- _0 exp 1 -> - _0 np.e`, ...) -- the mine
    // papering over a missing constructor fold, one context at a time.
    //
    // Unconditional: `np.e` is a universal leaf, parsed and emitted with no config
    // declaration behind it (`convert.rs` maps the token both ways for every table),
    // unlike `exp` itself, which is why the arms that BUILD an `exp` are gated and this
    // one is not.
    if args.len() == 1 && cx.view.tok_is(op, "exp") && matches!(&args[0], Ex::Num(r) if r.is_one())
    {
        return Ex::E;
    }
    // `abs` of a rational literal is EXACT arithmetic (|p/q| = |p|/q, no rounding),
    // so it folds in the constructor -- the fold unification (2026-08-02) deleted
    // the f64 evaluation path, and abs is the one vocabulary function that is exact
    // on rationals (everything else is transcendental and arrives as mined,
    // symbolically certified rules).
    if args.len() == 1 && cx.view.tok_is(op, "abs") {
        if let Ex::Num(r) = &args[0] {
            if r.is_negative() {
                if let Some(a) = r.checked_neg() {
                    return Ex::Num(a);
                }
            } else {
                return Ex::Num(*r);
            }
        }
    }
    // AN EVEN FUNCTION EATS THE SIGN OF ITS ARGUMENT: `g(-t) -> g(t)`.
    //
    // ONE arm, replacing thirty mined rules -- the entire
    // `{abs, cos, cosh} x {sin, sinh, tan, tanh, asin, asinh, atan, atanh}` cross product
    // plus `cos neg _0` and `cosh neg _0`. Those thirty are not thirty facts: they are TWO
    // (an odd function propagates a sign, an even one absorbs it) multiplied by fifteen
    // pairs, and the mine enumerated the product because the engine had no name for either
    // factor. `drop_sign` is the missing name.
    //
    // It fixes what no rule can. Three separate holes close at once:
    //   * LITERAL arguments. The rules name a `neg` NODE; `neg(3)` folds to the literal
    //     `-3` before any matcher runs, so every one of the thirty is blind at every
    //     numeric binding (`abs asinh neg 3` shipped as `abs asinh -3`, mu 19,000, where
    //     `asinh 3` is 10,000). This arm reads the canonical SIGN instead, so `neg 3`,
    //     `-3`, `(-1)*x` and `neg(sin x)` are one case.
    //   * DEPTH. The mine caps source patterns at length 4, so `abs sin tanh neg _0` (five
    //     tokens) has no rule at all -- it did not reduce even at a VARIABLE. A recursion
    //     has no depth limit.
    //   * The bare case the rules never covered: `cos (-2)` shipped as `cos -2` at 10,585
    //     against 10,000 for `cos 2`.
    //
    // It composes with `certainly_nonneg` without either knowing about the other:
    // `abs asinh -3` --(here)--> `abs asinh 3` --(nonneg licence)--> `asinh 3`.
    //
    // TOTAL, no licence needed: evenness is a pointwise identity on the whole extension,
    // and `is_even_fun` is a certified table, not a config field.
    if args.len() == 1 && cx.is_even_fun(op) {
        if let Some(sign_free) = cx.sign_blind_rep(&args[0]) {
            return fun(op, vec![sign_free], cx);
        }
        // F63 even-carrier orientation (owner-ruled 2026-08-08, full family): beyond
        // the CARRIED sign `sign_blind_rep` drops, an even function also erases its
        // argument's ORIENTATION -- `cos(x - y)` and `cos(y - x)` are one value --
        // and bag orientation is invisible to `sign_blind_rep` (a mixed-sign sum
        // carries no top-level sign to drop). Same doctrine as the even-power base:
        // fow picks, exact ties (pure mirrors) to the lex rule, Const-bearing and
        // absorbing sums keep their absorption owners, and the flip re-enters `fun`
        // exactly once (class-antisymmetry answers keep on the flipped orientation).
        if let Ex::Add(ts) = &args[0] {
            if !ts.iter().any(Ex::contains_const)
                && !ts.iter().any(term_absorbs_negation)
                && !ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
                && flipped_orientation_wins(ts, 0, Tie::Lex, cx)
            {
                if let Some(mut flipped) = ts
                    .iter()
                    .map(|t| negate_term(t, cx))
                    .collect::<Option<Vec<Ex>>>()
                {
                    // FILES-BARE gate (same rule as sign_trade_flip).
                    if !flipped_orientation_wins(&flipped, mu_sym() as i128, Tie::Keep, cx) {
                        flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
                        return fun(op, vec![Ex::Add(flipped)], cx);
                    }
                }
            }
        }
    }
    // A HALF-PERIOD SHIFT DROPS OUT: `f(t +- pi)` for f in {sin, cos, tan}. C1.19 family C.
    //
    // Eleven pattern rules say this (`sin|cos|tan` x `+|-` x both argument orders, plus
    // `f(pi - t)`), and every one of them is blind at a LITERAL argument, for TWO different
    // reasons that no rule can fix:
    //   * sin/cos sit at the `_` sort and go blind at a NEGATIVE literal, because the
    //     constructor factors the bag's common sign: `(-2) - pi` is stored as `neg(2 + pi)`,
    //     and the `-pi` member the pattern needs is gone. Family A's mechanism exactly.
    //   * tan sits at `!`, and `bind_ok` REFUSES ground subjects outright -- a documented,
    //     accepted recall loss on certificate-carrying slots, not a defect. So `tan + 2 np.pi`
    //     cannot bind `2` at all, and tan was blind at EVERY literal, both signs.
    // Measured before the arm: of the 18 `{sin,cos,tan} x {+,-} x {x0, 2, (-2)}` states, cos
    // reduced 6/6, sin 5/6, tan only the 2 variable cases.
    //
    // Each function's law is DERIVED, not taken by symmetry (F53's lesson), and the derivation
    // is what picks the output spelling:
    //   * sin is ODD:  sin(t +- pi) = -sin(t) = sin(-t), so the sign may go either side, and
    //     WHICH side is a structural question the first build got wrong. Pushing it into the
    //     argument always looked right from the literal case (`sin -2` prices 10,585 where
    //     `neg sin 2` is 18,000, and the wrapped form would be a WORSE answer than the rule it
    //     replaces). At a VARIABLE the two spellings are mu-EQUAL, and there the choice is
    //     decided by what they COMPOSE with: `test_function_rooted_rules_mint_no_twin` failed
    //     on `neg(sin(x0 + pi))`, which the wrapped form collapses to `sin x0` and the pushed-in
    //     form leaves at `neg sin neg x0` -- an outer `neg` cancels a wrapped sign and cannot
    //     reach one buried in the argument. The condition is therefore whether the sign has
    //     something to CANCEL AGAINST: push in iff the remainder already carries a negative
    //     (`carried_negative`), wrap otherwise. STRUCTURAL, not a mu guard -- F41 showed mu
    //     guards in constructor arms break idempotence. The first build tested `rest is a Num`
    //     instead, which is the same answer on literals but left `sin(pi - t)` at the
    //     uncollapsible `neg sin neg t` and kept a rule alive to say so.
    //   * cos is EVEN: cos(t +- pi) = -cos(t), and evenness gives no argument trick, so the
    //     `neg` stays outside -- which is what the existing rule already emits.
    //   * tan:         tan(t +- pi) = tan(t), no sign at all.
    // The sign of the pi member is irrelevant to all three, so `+pi` and `-pi` are one case.
    //
    // The judge was asked first, at the plain sort and at literals: all of
    // `sin|cos|tan {+,-} _0 np.pi` and the ground `tan +- 2 np.pi`, `sin + (-2) np.pi` come
    // back CERTIFIED. tan's `!` is a miner-side sort choice, not a judge requirement.
    if args.len() == 1 {
        let shift = match cx.view.resolve_owned(op).as_str() {
            "sin" | "cos" | "tan" => Some(cx.view.tok_is(op, "cos")),
            _ => None,
        };
        if let Some(negate_outside) = shift {
            if let Some(rest) = cx.strip_unit_pi(&args[0]) {
                return if cx.view.tok_is(op, "tan") {
                    fun(op, vec![rest], cx)
                } else if !negate_outside {
                    // sin: push the sign INTO the argument exactly when the argument ABSORBS
                    // it without adding a node. Two ways that happens, and both are needed --
                    // testing only one of them was wrong twice:
                    //   * a LITERAL of either sign absorbs it into its own sign
                    //     (`sin(-2)` prices 10,585 where `neg sin 2` is 18,000);
                    //   * a subtree already CARRYING a negative annihilates it
                    //     (`sin(pi - t)` lands on `sin t`, not the uncollapsible
                    //     `neg sin neg t` that kept a rule alive to say so).
                    // A bare symbolic argument absorbs nothing, so it WRAPS -- mu-equal there,
                    // and only the wrapped sign can be cancelled by an outer `neg`.
                    if let Some(positive) = cx.carried_negative(&rest) {
                        fun(op, vec![positive], cx)
                    } else if matches!(rest, Ex::Num(_)) {
                        fun(op, vec![mul(vec![Ex::Num(Rat::NEG_ONE), rest], cx)], cx)
                    } else {
                        mul(vec![Ex::Num(Rat::NEG_ONE), fun(op, vec![rest], cx)], cx)
                    }
                } else {
                    mul(vec![Ex::Num(Rat::NEG_ONE), fun(op, vec![rest], cx)], cx)
                };
            }
        }
    }
    // AN INVERSE PAIR COLLAPSES: `f(g(t)) -> t` where the licence holds. See
    // `inverse_pair_collapse` -- the table and its three licence groups are the contract
    // judge's own verdicts, taken before the arm was written.
    //
    // C1.19 family E asked for ONE of these (`exp log abs _0 -> abs _0`). The artifact was
    // carrying **101** identity-shaped inverse-pair rules, which is the same picture family A
    // showed: the mine enumerating a fact the engine has no name for, one member and one
    // LITERAL at a time (`asinh sinh 1`, `asinh sinh 2`, ... `log exp 10`, `acosh cosh np.pi`).
    // A pair is a pair at every argument, so a table costs what one arm costs.
    if args.len() == 1 {
        if let Ex::Fun(inner, iargs) = &args[0] {
            if iargs.len() == 1 {
                if let Some(collapsed) = cx.inverse_pair_collapse(op, *inner, &iargs[0]) {
                    return collapsed;
                }
            }
        }
    }
    // `rootn` index normalization lives HERE, in the constructor, so parse-time and
    // mid-pass rebuilds agree: an index subtree that a numeric fold collapses mid-pass
    // (`cos 0 -> 1`) must normalize in the SAME call, or idempotence breaks. Under the
    // IEEE `rootn` semantics every arm is a sound identity:
    //   rootn(x, 1) = x;  rootn(x, even) = pow(x, 1/even) (both principal, NaN on
    //   negatives);  rootn(x, -n) = pow(rootn(x, n), -1);  rootn(x, 0) and a
    //   provably-non-integer index = NaN everywhere (invalid operation, total).
    // Only binary Fun heads are inspected (rootn is the sole binary function), so the
    // name check stays off the unary hot path. `Const` and compound indices stay
    // symbolic: the evaluators fail closed on them.
    if args.len() == 2 && cx.view.resolve_owned(op) == "rootn" {
        let index = &args[1];
        match index {
            Ex::Num(r) => {
                if let Some(n) = r.as_integer() {
                    if n == 0 {
                        return Ex::NaN;
                    }
                    if n == 1 {
                        return args.into_iter().next().unwrap();
                    }
                    // A ROOT of `exp` is an `exp`: `rootn(exp a, n) -> exp(a/n)`, which is
                    // `compose_e_power` with the exponent `1/n` -- the SAME call `pow` makes
                    // with `b`, so the two spellings of a power reach the identical
                    // reduction (H-057's route-independence lesson) and the shape is
                    // unreachable by a mined rule either way: `rootn np.e 2 -> exp 0.5` has
                    // a var-free target, which `candidate_fold_filter` drops at mine time.
                    //
                    // `n` is a nonzero integer here (`n == 0` and `n == 1` returned above),
                    // so `1/n` is a finite nonzero rational and the sign follows through the
                    // reciprocal for free: `rootn(e, -m) = 1/e^(1/m) = exp(-1/m)`. The
                    // 2026-08-07 predecessor of this arm handled the LEAF `e` only, and the
                    // census that came out of it (`remine/probe_rule_reachability.py`) found
                    // its sibling still blind on the very next base: `pow(exp(-1), 1/2)` was
                    // shipping as `rootn(exp(-1), 2)` at mu 20,000 where `exp(-1/2)` is
                    // 11,585. One base is a patch; the family is the principle.
                    if let Some(composed) =
                        Rat::new(1, n).and_then(|inv| compose_e_power(&args[0], &Ex::Num(inv), cx))
                    {
                        return composed;
                    }
                    // n >= 2 EVEN: kept as `rootn`. The conversion to `pow(x, 1/n)` used
                    // to live here; it now runs the OTHER way (see `pow`), because mu
                    // prefers the root spelling once a genuine fraction pays for its
                    // denominator -- `rootn x 2` prices 18 against `pow x 1/2` at 19.
                    if n <= -1 {
                        // checked_neg, not `-n`: a MIN index cannot be negated (release
                        // wraps silently -> the still-negative index would recurse here
                        // forever). Unreachable while the no-MIN Rat invariant holds;
                        // refusing to fold (stay symbolic) is sound either way.
                        if let Some(pos) = n.checked_neg() {
                            let base = args.into_iter().next().unwrap();
                            let inner = fun(op, vec![base, Ex::int(pos)], cx);
                            return pow(inner, Ex::int(-1), cx);
                        }
                    }
                    // ROOT COMPOSITION: rootn(rootn(x, m), n) -> rootn(x, m*n). Sound at
                    // every parity. m even: the inner root is NaN on x < 0 and m*n is even
                    // too, so both sides are NaN there. m odd with n even: the inner keeps
                    // the sign and the outer even root is NaN on it, and m*n is even.
                    // Both odd: signs pass through on both sides. This composition used to
                    // arrive for free through the pow-of-pow arm (even roots WERE pows);
                    // now that `rootn` survives, it has to be explicit or nested roots stop
                    // composing (`rootn(rootn(x,2),2)` measured 28 against 19 for the
                    // fourth root).
                    if let Ex::Fun(inner_op, inner_args) = &args[0] {
                        if inner_args.len() == 2 && cx.view.tok_is(*inner_op, "rootn") {
                            if let Ex::Num(m) = &inner_args[1] {
                                if let Some(mi) = m.as_integer() {
                                    if mi >= 2 {
                                        if let Some(prod) = mi.checked_mul(n) {
                                            let inner_base = inner_args[0].clone();
                                            return fun(op, vec![inner_base, Ex::int(prod)], cx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // POWER-THROUGH-ROOT at an EVEN index: `rootn(A^k, n) -> A^(k/n)` under the
                    // SAME licence as the pow-of-pow composition in `pow`. An even `rootn` IS the
                    // principal power `.^(1/n)`, so this IS that composition, merely reached
                    // through the root spelling; sharing the licence is what makes the two routes
                    // agree. Without it the constructor and the rejoin path disagreed:
                    // `rootn(1/cosh(t), 4)` stayed put at construction but reduced to
                    // `cosh(t)^(-1/4)` after a projection round-trip, so the explicit form was not
                    // a fixpoint (64k row 51607).
                    //
                    // The licence is also what keeps H-056 CLOSED. A NEGATIVE odd `k` is refused
                    // unless the base is certainly non-negative, because at `A = -inf` the inner
                    // `A^k` is ZERO and the source is then DEFINED while `A^(k/n)` is NaN --
                    // defined -> undefined, which §9.1's R2 forbids at zero measure tolerance.
                    // `cosh` clears it on the certainly-non-negative disjunct, a bare variable
                    // does not. `s = 1/n` is never an integer here, so that disjunct is vacuous.
                    if n % 2 == 0 {
                        let composed = if let Ex::Pow(inner, ex) = &args[0] {
                            match &**ex {
                                Ex::Num(k) => {
                                    Rat::new(1, n).and_then(|s| k.checked_mul(&s)).filter(|p| {
                                        cx.lossy
                                            || cx.certainly_nonneg(inner)
                                            || match k.as_integer() {
                                                Some(ki) => {
                                                    ki % 2 != 0
                                                        && (ki > 0 || cx.never_infinite(inner))
                                                        && !p.is_integer()
                                                }
                                                None => !p.is_integer(),
                                            }
                                    })
                                }
                                _ => None,
                            }
                        } else {
                            None
                        };
                        if let Some(p) = composed {
                            let Ex::Pow(inner, _) = args.into_iter().next().unwrap() else {
                                unreachable!()
                            };
                            return pow(*inner, Ex::Num(p), cx);
                        }
                    }
                    // n >= 3 odd: the genuine signed root. The contract-blessed
                    // inverse composition folds FIRST (SIMPLIFICATION_CONTRACT_v2 s5
                    // Corollaries: pow1_3(pow3 t) = t at EVERY value including nan and
                    // +-inf, hence _-sound): rootn(t^n, n) -> t for the SAME odd n.
                    //
                    // The ODD guard is load-bearing and newly EXPLICIT: at even n the
                    // identity is false -- rootn(t^2, 2) is |t|, not t. It was previously
                    // unreachable at even n only because even indices converted to `pow`
                    // above; with them surviving, the parity has to be tested here.
                    if n % 2 != 0 {
                        if let Ex::Pow(b, ex) = &args[0] {
                            let _ = b;
                            if let Ex::Num(rr) = &**ex {
                                if rr.as_integer() == Some(n) {
                                    let Ex::Pow(inner, _) = args.into_iter().next().unwrap() else {
                                        unreachable!()
                                    };
                                    return *inner;
                                }
                            }
                        }
                    }
                    // EXACT RATIONAL ODD ROOT (P1, owner-approved 2026-08-02): a
                    // rational base that is a perfect n-th power folds exactly --
                    // rootn(8,3) = 2, rootn(1,k) = 1, rootn(-8,3) = -2 (checked_root
                    // carries the sign through odd k). This is rational ARITHMETIC,
                    // not transcendental evaluation, so fold unification mandates it
                    // in the constructor (the deleted f64 arm had been carrying it,
                    // same story as abs; the mine cannot cover it -- index/base
                    // literals beyond the alphabet). Inexact bases keep the Fun.
                    if let Ex::Num(r) = &args[0] {
                        if let Some(root) = r.checked_root(n) {
                            return Ex::Num(root);
                        }
                    }
                    // Otherwise keep the Fun.
                } else {
                    return Ex::NaN; // exact non-integer index: invalid, NaN everywhere
                }
            }
            Ex::Pi | Ex::E | Ex::PosInf | Ex::NegInf | Ex::NaN => return Ex::NaN,
            _ => {}
        }
    }
    // Constructor-level abs elimination: `abs t -> t` on a certainly-non-negative operand.
    // A `_`-grade pointwise identity: both sides share t's domain exactly (abs(nan) = nan,
    // abs(+inf) = +inf), t's value is >= 0 wherever defined, and the zero sign is
    // non-normative (one-zero contract) -- and in fact unobservable, since certainly_nonneg
    // witnesses never evaluate to -0.0 (see its doc). Living HERE rather than in the mined
    // ruleset makes the fold reachable from every entry path and structurally immune to
    // serve-order shadowing: the 7-31 row-diff found the F5-era `inv abs t -> abs inv t`
    // orientation respelling |x^2| into |x^-2| before the exponent-literal abs rules could
    // match, sticking a dead abs in 9/65,536 (64k) + 25/1M outputs. The abs rules this
    // subsumes self-retire at the next mine (mint-vs-mark: the mark now carries the fold).
    // `tok_is` keeps the head dispatch allocation-free on the unary hot path.
    if args.len() == 1 && cx.view.tok_is(op, "abs") && cx.certainly_nonneg(&args[0]) {
        return args.into_iter().next().unwrap();
    }
    Ex::Fun(op, args)
}

/// Full recursive re-canonicalization: rebuild `e` bottom-up through the canonical
/// constructors. Used on freshly substituted rule RHS instances and at the parse boundary.
pub fn canon(e: Ex, cx: &Cx) -> Ex {
    match e {
        Ex::Add(v) => {
            let parts = v.into_iter().map(|x| canon(x, cx)).collect();
            add(parts, cx)
        }
        Ex::Mul(v) => {
            let parts = v.into_iter().map(|x| canon(x, cx)).collect();
            mul(parts, cx)
        }
        Ex::Pow(b, ex) => {
            let b = canon(*b, cx);
            let ex = canon(*ex, cx);
            pow(b, ex, cx)
        }
        Ex::Fun(f, v) => {
            let parts = v.into_iter().map(|x| canon(x, cx)).collect();
            fun(f, parts, cx)
        }
        leaf => leaf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{OperatorSpec, Operators};
    use crate::tokens::{TokenOverlay, TokenTable, TokenView};
    use std::cell::RefCell;

    fn test_ops() -> (Vec<String>, Operators) {
        let unary = ["sin", "cos", "abs", "exp", "log", "cosh"];
        let mut order = Vec::new();
        let mut specs: rustc_hash::FxHashMap<String, OperatorSpec> = Default::default();
        for n in ["+", "-", "*", "/", "pow"] {
            order.push(n.to_string());
            specs.insert(
                n.to_string(),
                OperatorSpec {
                    realization: String::new(),
                    alias: vec![],
                    inverse: None,
                    arity: 2,
                    precedence: None,
                    commutative: n == "+" || n == "*",
                },
            );
        }
        for n in unary {
            order.push(n.to_string());
            specs.insert(
                n.to_string(),
                OperatorSpec {
                    realization: String::new(),
                    alias: vec![],
                    inverse: None,
                    arity: 1,
                    precedence: None,
                    commutative: false,
                },
            );
        }
        let ops = Operators::from_specs(order.clone(), specs);
        (order, ops)
    }

    /// Run `f` with a fresh view over a minimal operator universe.
    fn with_view<R>(f: impl FnOnce(&TokenView) -> R) -> R {
        let (order, ops) = test_ops();
        let table = TokenTable::build(&order, &ops);
        let overlay = RefCell::new(TokenOverlay::new(table.len()));
        let view = TokenView::new(&table, &overlay);
        f(&view)
    }

    fn x(view: &TokenView) -> Ex {
        Ex::Leaf(view.intern("x0"))
    }

    fn y(view: &TokenView) -> Ex {
        Ex::Leaf(view.intern("x1"))
    }

    #[test]
    fn add_collects_and_orders() {
        with_view(|view| {
            let cx = Cx::bare(view);
            // x + x -> 2x (same-sign, no licence needed)
            let e = add(vec![x(view), x(view)], &cx);
            assert_eq!(e, Ex::Mul(vec![Ex::int(2), x(view)]));
            // 2x + 3x -> 5x
            let e = add(
                vec![
                    mul(vec![Ex::int(2), x(view)], &cx),
                    mul(vec![Ex::int(3), x(view)], &cx),
                ],
                &cx,
            );
            assert_eq!(e, Ex::Mul(vec![Ex::int(5), x(view)]));
            // Order invariance: y + x == x + y
            let a = add(vec![y(view), x(view)], &cx);
            let b = add(vec![x(view), y(view)], &cx);
            assert_eq!(a, b);
            // Flattening: (x + y) + x -> 2x + y
            let inner = add(vec![x(view), y(view)], &cx);
            let e = add(vec![inner, x(view)], &cx);
            // Stripped term order: the 2x term (key x0) sorts before y (key x1).
            assert_eq!(
                e,
                Ex::Add(vec![Ex::Mul(vec![Ex::int(2), x(view)]), y(view)])
            );
            // Literals fold exactly: 1/2 + 1/3 = 5/6
            let e = add(
                vec![
                    Ex::Num(Rat::new(1, 2).unwrap()),
                    Ex::Num(Rat::new(1, 3).unwrap()),
                ],
                &cx,
            );
            assert_eq!(e, Ex::Num(Rat::new(5, 6).unwrap()));
        });
    }

    #[test]
    fn add_sign_cancel_gate() {
        with_view(|view| {
            let cx = Cx::bare(view);
            // x - x -> 0: variables are certainly finite, licence-free.
            let neg_x = mul(vec![Ex::int(-1), x(view)], &cx);
            assert_eq!(add(vec![x(view), neg_x], &cx), Ex::int(0));
            // log(x) - log(x): NO certificate in a bare context -> refuses to cancel.
            let lg = fun(view.intern("log"), vec![x(view)], &cx);
            let neg_lg = mul(vec![Ex::int(-1), lg.clone()], &cx);
            let e = add(vec![lg.clone(), neg_lg.clone()], &cx);
            assert!(
                matches!(e, Ex::Add(_)),
                "uncertified cancel must refuse: {e:?}"
            );
            // With the certificate it cancels.
            let yes = |_: &Ex| true;
            let cx2 = Cx {
                view,
                cert_fin: Some(&yes),
                cert_finnz: None,
                cert_nzae: None,
                cert_nce: None,
                lossy: false,
                sentinels_expired: false,
            };
            assert_eq!(add(vec![lg.clone(), neg_lg], &cx2), Ex::int(0));
            // Same-sign merging never needs the licence: log(x) + log(x) -> 2 log(x).
            let e = add(vec![lg.clone(), lg.clone()], &cx);
            assert_eq!(e, Ex::Mul(vec![Ex::int(2), lg]));
        });
    }

    #[test]
    fn add_extended_values() {
        with_view(|view| {
            let cx = Cx::bare(view);
            assert_eq!(add(vec![Ex::NaN, x(view)], &cx), Ex::NaN);
            assert_eq!(add(vec![Ex::PosInf, Ex::NegInf], &cx), Ex::NaN);
            assert_eq!(add(vec![Ex::PosInf, Ex::int(5)], &cx), Ex::PosInf);
            assert_eq!(add(vec![Ex::PosInf, x(view)], &cx), Ex::PosInf); // variable: finite
                                                                         // inf + log(x): log(x) can be -inf (x=0) -- refuses without certificate.
            let lg = fun(view.intern("log"), vec![x(view)], &cx);
            let e = add(vec![Ex::PosInf, lg], &cx);
            assert!(matches!(e, Ex::Add(_)));
        });
    }

    #[test]
    fn add_const_independence_and_absorption() {
        with_view(|view| {
            let cx = Cx::bare(view);
            // C + C -> C (one constant absorbs, forall-exists).
            assert_eq!(add(vec![Ex::Const, Ex::Const], &cx), Ex::Const);
            // C + 5 -> C.
            assert_eq!(add(vec![Ex::Const, Ex::int(5)], &cx), Ex::Const);
            // 2C + x + 3 -> C + x.
            let e = add(
                vec![mul(vec![Ex::int(2), Ex::Const], &cx), x(view), Ex::int(3)],
                &cx,
            );
            // Constant-like terms sort LAST under the stripped order (x + C, not C + x).
            assert_eq!(e, Ex::Add(vec![x(view), Ex::Const]));
            // C*x + C*x must NOT merge (independent constants).
            let cx_term = || Ex::Mul(vec![Ex::Const, x(view)]);
            let e = add(vec![cx_term(), cx_term()], &cx);
            assert_eq!(e, Ex::Add(vec![cx_term(), cx_term()]));
        });
    }

    #[test]
    fn mul_collects_exponents() {
        with_view(|view| {
            let cx = Cx::bare(view);
            // x * x -> x^2
            let e = mul(vec![x(view), x(view)], &cx);
            assert_eq!(e, Ex::Pow(Box::new(x(view)), Box::new(Ex::int(2))));
            // x^2 * x^-1 -> x (variable base: finite-nonzero a.e. licence is syntactic)
            let x2 = pow(x(view), Ex::int(2), &cx);
            let xm1 = pow(x(view), Ex::int(-1), &cx);
            assert_eq!(mul(vec![x2, xm1], &cx), x(view));
            // x * 1/x -> 1 (full cancel on a variable)
            let xm1 = pow(x(view), Ex::int(-1), &cx);
            assert_eq!(mul(vec![x(view), xm1], &cx), Ex::int(1));
            // log(x)^1 * log(x)^-1: composite base, no certificate -> refuses.
            let lg = fun(view.intern("log"), vec![x(view)], &cx);
            let lgm1 = pow(lg.clone(), Ex::int(-1), &cx);
            let e = mul(vec![lg.clone(), lgm1], &cx);
            assert!(
                matches!(e, Ex::Mul(_)),
                "uncertified exponent cancel: {e:?}"
            );
            // 2 * 3 * x -> 6x; coefficient exact.
            let e = mul(vec![Ex::int(2), Ex::int(3), x(view)], &cx);
            assert_eq!(e, Ex::Mul(vec![Ex::int(6), x(view)]));
            // Zero: 0 * x -> 0 (variable licensed); 0 * log(x) stays.
            assert_eq!(mul(vec![Ex::int(0), x(view)], &cx), Ex::int(0));
            let e = mul(vec![Ex::int(0), lg.clone()], &cx);
            assert_eq!(e, Ex::Mul(vec![Ex::int(0), lg]));
        });
    }

    #[test]
    fn mul_branch_cut_gate() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let half = Ex::Num(Rat::new(1, 2).unwrap());
            // x^(1/2) * x^(1/2) must NOT become x (wrong on x < 0). The PROPERTY is what
            // matters, not the shape: with even roots spelled `rootn`, equal factors now
            // collect into `pow(rootn(x,2), 2)`, which is value-exact (NaN on x < 0 both
            // sides) and judge-OK. Asserting "still a Mul" was a proxy that only held while
            // the sound merge did not exist.
            let s = pow(x(view), half.clone(), &cx);
            let e = mul(vec![s.clone(), s.clone()], &cx);
            assert_ne!(e, x(view), "branch-cut merge must refuse: {e:?}");
            // abs(x)^(1/2) * abs(x)^(1/2) -> abs(x): certainly non-negative base.
            let ax = fun(view.intern("abs"), vec![x(view)], &cx);
            let s = pow(ax.clone(), half.clone(), &cx);
            assert_eq!(mul(vec![s.clone(), s], &cx), ax);
            // x^(1/2) * x^1 -> x^(3/2): sum stays non-integer, sound. `factor_split` reads
            // the even root as `x^(1/2)`, so the root spelling does not cost the merge.
            let s = pow(x(view), half, &cx);
            let e = mul(vec![s, x(view)], &cx);
            assert_eq!(
                e,
                Ex::Pow(
                    Box::new(x(view)),
                    Box::new(Ex::Num(Rat::new(3, 2).unwrap()))
                )
            );
        });
    }

    #[test]
    fn abs_elimination_on_certainly_nonneg() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let abs_t = view.intern("abs");
            // |x^2| -> x^2 and |x^-2| -> x^-2: even exponents of EITHER sign (the
            // F5-era serve-order shadow lived exactly at the -2 spelling).
            for n in [2, -2] {
                let p = pow(x(view), Ex::int(n), &cx);
                assert_eq!(fun(abs_t, vec![p.clone()], &cx), p);
            }
            // |1/|x|| -> 1/|x|: certainly-non-negative base under ANY exponent.
            let ax = fun(abs_t, vec![x(view)], &cx);
            let inv_ax = pow(ax.clone(), Ex::int(-1), &cx);
            assert_eq!(fun(abs_t, vec![inv_ax.clone()], &cx), inv_ax);
            // |cos(cos x)| -> cos(cos x): range positivity via the unit-bounded argument.
            let cos_t = view.intern("cos");
            let cc = fun(cos_t, vec![fun(cos_t, vec![x(view)], &cx)], &cx);
            assert_eq!(fun(abs_t, vec![cc.clone()], &cx), cc);
            // `|sinh |x||` settles at `|sinh x|`, NOT at `sinh |x|` (2026-08-07). Both are
            // the same function and both price 24,000 -- the abs elimination this test
            // guards is not what changed. What changed is which representative wins: the
            // parity arm reaches the argument FIRST and strips the inner `abs` (an `abs`
            // inside a sign-blind context is a no-op), leaving `sinh x`, whose sign is
            // indefinite, so the OUTER abs correctly stays. Before, the inner `abs` survived
            // and made the argument certainly-non-negative, so the outer one dropped
            // instead. Same node count, same mu, one canonical form rather than two.
            let sh_inner = fun(view.intern("sinh"), vec![ax.clone()], &cx);
            let sh_outer = fun(
                abs_t,
                vec![fun(view.intern("sinh"), vec![x(view)], &cx)],
                &cx,
            );
            assert_eq!(fun(abs_t, vec![sh_inner], &cx), sh_outer);
            // Bag closure: a product of non-negatives sheds the outer abs.
            let prod = mul(vec![pow(x(view), Ex::int(2), &cx), ax.clone()], &cx);
            assert_eq!(fun(abs_t, vec![prod.clone()], &cx), prod);
            // Controls: sign-indefinite operands KEEP the abs. (`abs Const` folding is the
            // Const channel's absorption in the pipeline, deliberately NOT this fold --
            // certainly_nonneg(Const) is false because a fitted value can be negative.)
            for keep in [
                x(view),
                fun(view.intern("sin"), vec![x(view)], &cx),
                fun(cos_t, vec![x(view)], &cx), // cos of an UNBOUNDED argument
                pow(x(view), Ex::int(3), &cx),
                // periodic, not monotone -- so `certainly_nonneg` refuses and the abs
                // stays. The argument is a BARE variable since 2026-08-07: it used to be
                // `|x|`, and the parity arm now strips that inner `abs` (a no-op inside a
                // sign-blind context), which was testing the parity arm rather than this
                // control's subject.
                fun(view.intern("tan"), vec![x(view)], &cx),
                Ex::Const,
            ] {
                let e = fun(abs_t, vec![keep.clone()], &cx);
                assert_eq!(e, Ex::Fun(abs_t, vec![keep]), "abs must stay");
            }
        });
    }

    #[test]
    fn mul_extended_and_const() {
        with_view(|view| {
            let cx = Cx::bare(view);
            assert_eq!(mul(vec![Ex::NaN, x(view)], &cx), Ex::NaN);
            assert_eq!(mul(vec![Ex::int(0), Ex::PosInf], &cx), Ex::NaN);
            assert_eq!(mul(vec![Ex::int(-2), Ex::PosInf], &cx), Ex::NegInf);
            assert_eq!(mul(vec![Ex::NegInf, Ex::NegInf], &cx), Ex::PosInf);
            // C * C -> C; C * 5 -> C; C * 0 -> 0.
            assert_eq!(mul(vec![Ex::Const, Ex::Const], &cx), Ex::Const);
            assert_eq!(mul(vec![Ex::Const, Ex::int(5)], &cx), Ex::Const);
            assert_eq!(mul(vec![Ex::Const, Ex::int(0)], &cx), Ex::int(0));
            // inf * C: C = 0 reachable -> keep both factors.
            let e = mul(vec![Ex::PosInf, Ex::Const], &cx);
            assert_eq!(e, Ex::Mul(vec![Ex::PosInf, Ex::Const]));
        });
    }

    #[test]
    fn pow_folds_and_licences() {
        with_view(|view| {
            let cx = Cx::bare(view);
            assert_eq!(pow(x(view), Ex::int(1), &cx), x(view));
            assert_eq!(pow(x(view), Ex::int(0), &cx), Ex::int(1));
            assert_eq!(pow(Ex::int(1), x(view), &cx), Ex::int(1));
            assert_eq!(pow(Ex::int(2), Ex::int(10), &cx), Ex::int(1024));
            assert_eq!(
                pow(Ex::int(4), Ex::Num(Rat::new(1, 2).unwrap()), &cx),
                Ex::int(2)
            );
            assert_eq!(
                pow(Ex::int(2), Ex::int(-2), &cx),
                Ex::Num(Rat::new(1, 4).unwrap())
            );
            // Inexact EVEN root stays symbolic -- as a `rootn`, which is now the canonical
            // spelling for an even index.
            let e = pow(Ex::int(2), Ex::Num(Rat::new(1, 2).unwrap()), &cx);
            assert!(matches!(e, Ex::Fun(..)), "expected rootn(2,2), got {e:?}");
            // (-8)^(1/3) is NOT the real root in Pow semantics -> stays symbolic.
            let e = pow(Ex::int(-8), Ex::Num(Rat::new(1, 3).unwrap()), &cx);
            assert!(matches!(e, Ex::Pow(..)));
            // (x^2)^3 -> x^6 (both integers).
            let e = pow(pow(x(view), Ex::int(2), &cx), Ex::int(3), &cx);
            assert_eq!(e, Ex::Pow(Box::new(x(view)), Box::new(Ex::int(6))));
            // (x^2)^(1/2) must NOT become x (|x| != x).
            let e = pow(
                pow(x(view), Ex::int(2), &cx),
                Ex::Num(Rat::new(1, 2).unwrap()),
                &cx,
            );
            // Refusal is the PROPERTY: (x^2)^(1/2) must not become x. It is now spelled
            // `rootn(x^2, 2)` (= |x|), which is exactly the refusal, so assert the property
            // rather than the old Pow-of-Pow shape.
            assert_ne!(e, x(view), "sign-erasing composition must refuse: {e:?}");
            // (x^3)^(1/3) must NOT become x in POW semantics: on x < 0 the source is
            // NaN (principal non-integer power of a negative value) while x is finite
            // -- a value fabrication; oddness of the inner exponent licenses
            // nothing about the OUTER principal root. The sound spelling of the
            // inverse composition is rootn (contract s5), tested below.
            let e = pow(
                pow(x(view), Ex::int(3), &cx),
                Ex::Num(Rat::new(1, 3).unwrap()),
                &cx,
            );
            assert!(
                matches!(&e, Ex::Pow(b, _) if matches!(&**b, Ex::Pow(..))),
                "integer-product composition must refuse: {e:?}"
            );
            // (x^5)^0.2 must refuse too.
            let e = pow(
                pow(x(view), Ex::int(5), &cx),
                Ex::Num(Rat::new(1, 5).unwrap()),
                &cx,
            );
            assert!(matches!(&e, Ex::Pow(b, _) if matches!(&**b, Ex::Pow(..))));
            // (x^3)^(1/5) -> x^(3/5): odd inner AND non-integer product -- both sides
            // are NaN on the whole negative half-line, so the fold is exact.
            let e = pow(
                pow(x(view), Ex::int(3), &cx),
                Ex::Num(Rat::new(1, 5).unwrap()),
                &cx,
            );
            assert_eq!(
                e,
                Ex::Pow(
                    Box::new(x(view)),
                    Box::new(Ex::Num(Rat::new(3, 5).unwrap()))
                )
            );
            // rootn(x^n, n) -> x for the SAME odd n: total at every value including
            // nan and +-inf (SIMPLIFICATION_CONTRACT_v2 s5 Corollaries), so the
            // CONSTRUCTOR folds it -- the composition pow refuses lives here soundly.
            let rootn_tok = view.intern("rootn");
            let e = fun(
                rootn_tok,
                vec![pow(x(view), Ex::int(3), &cx), Ex::int(3)],
                &cx,
            );
            assert_eq!(e, x(view));
            let e = fun(
                rootn_tok,
                vec![pow(x(view), Ex::int(5), &cx), Ex::int(5)],
                &cx,
            );
            assert_eq!(e, x(view));
            // ...but only for the MATCHING index: rootn(x^3, 5) keeps the Fun.
            let e = fun(
                rootn_tok,
                vec![pow(x(view), Ex::int(3), &cx), Ex::int(5)],
                &cx,
            );
            assert!(matches!(&e, Ex::Fun(..)), "{e:?}");
            // (x^(1/2))^2 -> must NOT become x (would fabricate a value on x < 0). The
            // inner is spelled `rootn(x,2)` now, so assert the PROPERTY: the licence arm
            // refuses without a non-negativity certificate.
            let e = pow(
                pow(x(view), Ex::Num(Rat::new(1, 2).unwrap()), &cx),
                Ex::int(2),
                &cx,
            );
            assert_ne!(
                e,
                x(view),
                "unlicensed root/power inverse must refuse: {e:?}"
            );
            // ...but (x^(1/2))^3 -> x^(3/2) (still non-integer: NaN region preserved). The
            // composition arm reads the even root as a rational power via
            // `as_rational_power`, so the licence logic applies unchanged.
            let e = pow(
                pow(x(view), Ex::Num(Rat::new(1, 2).unwrap()), &cx),
                Ex::int(3),
                &cx,
            );
            assert_eq!(
                e,
                Ex::Pow(
                    Box::new(x(view)),
                    Box::new(Ex::Num(Rat::new(3, 2).unwrap()))
                )
            );
            // (2x)^2 -> 4x^2 (integer power distributes).
            let e = pow(mul(vec![Ex::int(2), x(view)], &cx), Ex::int(2), &cx);
            assert_eq!(
                e,
                Ex::Mul(vec![
                    Ex::int(4),
                    Ex::Pow(Box::new(x(view)), Box::new(Ex::int(2)))
                ])
            );
        });
    }

    /// F1/F2 (AUDIT 2026-08-06): a NEGATIVE exponent over an ODD `rootn` had no arm at all,
    /// so the reduction was route-dependent -- the explicit projection re-spelled it and the
    /// re-parse then reduced, leaving the projection outside its own fixpoint (and tripping
    /// the serialization-stability `debug_assert` in `engine/ac.rs`).
    #[test]
    fn odd_rootn_negative_exponent_composes() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let rootn_tok = view.intern("rootn");
            let root = |b: Ex, n: i128| fun(rootn_tok, vec![b, Ex::int(n)], &cx);
            let r = |p: i128, q: i128| Ex::Num(Rat::new(p, q).unwrap());

            // ODD-ROOT COMPOSITION: sound for every NON-INTEGER exponent, because both sides
            // are NaN on a negative base. `rootn(x,3)^(-1/2) -> x^(-1/6)`, matching what the
            // reciprocal spelling already produced.
            assert_eq!(
                pow(root(x(view), 3), r(-1, 2), &cx),
                Ex::Pow(Box::new(x(view)), Box::new(r(-1, 6)))
            );
            assert_eq!(
                pow(root(x(view), 5), r(-1, 2), &cx),
                Ex::Pow(Box::new(x(view)), Box::new(r(-1, 10)))
            );
            assert_eq!(
                pow(root(x(view), 3), r(-3, 2), &cx),
                Ex::Pow(Box::new(x(view)), Box::new(r(-1, 2)))
            );
            // The positive direction is unchanged and still lands on the root spelling.
            assert_eq!(pow(root(x(view), 3), r(1, 2), &cx), root(x(view), 6));

            // INTEGER exponents stay excluded -- `rootn(-8,3)^2 = 4` while `(-8)^(2/3)` is
            // NaN, a positive-measure disagreement. The `rootn` must survive.
            assert!(matches!(
                pow(root(x(view), 3), Ex::int(2), &cx),
                Ex::Pow(..)
            ));
            assert!(matches!(
                pow(root(x(view), 3), Ex::int(-2), &cx),
                Ex::Pow(..)
            ));

            // ROOT/POWER INVERSE, negative mirror: `rootn(x,n)^(-n) -> 1/x` for odd n.
            assert_eq!(
                pow(root(x(view), 3), Ex::int(-3), &cx),
                Ex::Pow(Box::new(x(view)), Box::new(Ex::int(-1)))
            );
            assert_eq!(
                pow(root(x(view), 5), Ex::int(-5), &cx),
                Ex::Pow(Box::new(x(view)), Box::new(Ex::int(-1)))
            );
            // ...positive mirror unchanged, and only for the MATCHING index.
            assert_eq!(pow(root(x(view), 3), Ex::int(3), &cx), x(view));
            assert!(matches!(
                pow(root(x(view), 3), Ex::int(-5), &cx),
                Ex::Pow(..)
            ));
            // EVEN index needs a certainly-non-negative base, in both directions.
            assert!(matches!(
                pow(root(x(view), 2), Ex::int(-2), &cx),
                Ex::Pow(..)
            ));
        });
    }

    #[test]
    fn complexity_weight_table() {
        // The mu weight table (stage 2), in MILLI-BITS: structural nodes and vocabulary
        // symbols 8 bits, literals pay L(n) = log2(1 + |n|) per written integer with a
        // bit for a negative sign and a two-bit floor, magnitude-1 coefficient and
        // exponent slots are free (a bare sign), <constant> = MU_FREE_WORST_CASE_F64
        // = 1133 bits (contract Sec 10.10(5): the supremum of mu over f64 round-trip
        // spellings plus the sign bit, rounded up -- DERIVED, not chosen).
        with_view(|view| {
            let cx = Cx::bare(view);
            let c = |e: &Ex| super::complexity(e, view);
            // x - y = 32: Add(8) + x(8) + the sign-wrapper Mul (8, a real structural
            // node of the canonical tree; the -1 slot itself is free) + y(8).
            // x*y = 24; x/y = 32 (the Pow(y, -1) wrapper, exponent -1 free).
            let x_ = x(view);
            let y_ = y(view);
            let x_minus_y = add(
                vec![x_.clone(), mul(vec![Ex::int(-1), y_.clone()], &cx)],
                &cx,
            );
            assert_eq!(c(&x_minus_y), 32_000);
            assert_eq!(c(&mul(vec![x_.clone(), y_.clone()], &cx)), 24_000);
            let x_over_y = mul(vec![x_.clone(), pow(y_.clone(), Ex::int(-1), &cx)], &cx);
            assert_eq!(c(&x_over_y), 32_000);
            // 2x = 18 (the coefficient PAYS its bits: 8 + 2 + 8), x^2 = 18, 1/x = 16,
            // sin(x) = 16.
            assert_eq!(c(&mul(vec![Ex::int(2), x_.clone()], &cx)), 18_000);
            assert_eq!(c(&pow(x_.clone(), Ex::int(2), &cx)), 18_000);
            assert_eq!(c(&pow(x_.clone(), Ex::int(-1), &cx)), 16_000);
            assert_eq!(c(&fun(view.intern("sin"), vec![x_.clone()], &cx)), 16_000);
            // x^-2 costs as x^2 (exponent sign free); C*x = 144 (the priciest atom);
            // -5 costs its magnitude's bits.
            // The exponent -2 is a real literal now: L(2) + 1 sign bit = 2.585 bits.
            assert_eq!(c(&pow(x_.clone(), Ex::int(-2), &cx)), 18_585);
            // Mul(8) + <constant>(1_133_000) + x(8) = 1_149_000.
            assert_eq!(
                c(&mul(vec![Ex::Const, x_.clone()], &cx)),
                2 * mu_sym() + MU_FREE_WORST_CASE_F64
            );
            assert_eq!(mu_free(), 1_133 * MU_MILLI);
            assert_eq!(c(&Ex::int(-5)), 3_585); // L(5) = 2.585, + 1 for the sign
                                                // x8 + 1.2*x3 = 8 + 8 + (8 + cost(6/5) + 8).
                                                // cost(6/5) = L(6) + L(5) = 2.807 + 2.585 = 5.392 under Sec 10.10(1): the decimal
                                                // code (L(12) + L(1) = 4.700) is no longer consulted, mu being value complexity.
            let t = mul(vec![Ex::Num(Rat::new(6, 5).unwrap()), y_.clone()], &cx);
            assert_eq!(c(&add(vec![x_.clone(), t], &cx)), 37_392);
            // The stage-2 point, in one line: an E factor undercuts its f64 image.
            let sym = c(&mul(vec![Ex::E, x_.clone()], &cx));
            let mat = c(&mul(
                vec![
                    Ex::Num(Rat::new(2718281828459045, 1000000000000000).unwrap()),
                    x_.clone(),
                ],
                &cx,
            ));
            assert!(sym == 24_000 && sym < mat, "{sym} vs {mat}");
        });
    }

    #[test]
    fn canon_is_idempotent_on_samples() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let lg = fun(view.intern("log"), vec![x(view)], &cx);
            let samples = vec![
                add(vec![x(view), y(view), x(view)], &cx),
                mul(vec![Ex::int(2), x(view), lg.clone(), x(view)], &cx),
                pow(add(vec![x(view), Ex::int(1)], &cx), Ex::int(2), &cx),
                add(vec![Ex::PosInf, lg], &cx),
                mul(
                    vec![Ex::int(0), fun(view.intern("log"), vec![y(view)], &cx)],
                    &cx,
                ),
            ];
            for s in samples {
                assert_eq!(canon(s.clone(), &cx), s, "canon not idempotent on {s:?}");
            }
        });
    }

    /// PIN (C1 read, 2026-08-05): `mu_numeric_str` reads fraction-shaped beyond-`Rat`
    /// spellings through the component model (L(p) + L(q) from the digit strings),
    /// not through the decimal path with the slash counted as a significand digit.
    /// The boundary properties that matter: far above every in-range literal, monotone
    /// in either component's digit count, and the decimal path untouched.
    #[test]
    fn mu_numeric_str_fraction_component_model() {
        let big41 = "1".repeat(41); // beyond i128 (39 digits max)
        let frac = format!("{big41}/7");
        let c = mu_numeric_str(&frac);
        // Component model, both components PAID (the implicit-denominator discount applies
        // only when there is no genuine denominator): L(111...1) + L(7) = 136.031 bits.
        assert_eq!(c, 136_031);
        // Far above any in-range literal; the boundary that matters is the ~105-BIT
        // 52-bit dyadics.
        assert!(c > 105 * MU_MILLI);
        // Monotone in each component.
        assert!(mu_numeric_str(&format!("{big41}1/7")) > c);
        assert!(mu_numeric_str(&format!("{big41}/71")) > c);
        // The sign costs exactly one bit here, as it does in range (owner 2026-08-06);
        // a zero numerator floors at two bits.
        assert_eq!(mu_numeric_str(&format!("-{big41}/7")), c + MU_MILLI);
        assert_eq!(mu_numeric_str("0/7"), 2 * MU_MILLI);
        // The decimal path is unchanged by the new arm.
        assert_eq!(
            mu_numeric_str("4.159653437657682e-35"),
            mu_numeric_str("4.159653437657682E-35")
        );
        // Sec 10.10(1) REVOKED the minimum over codes: `1e-40` pays the whole 10^40
        // denominator again, L(1) + 40*log2(10) = 133.880 bits, not the 6.358 the decimal
        // code used to buy. mu is VALUE complexity, so a short DESCRIPTION earns no
        // discount -- 1e-40 is a genuinely extreme value. The boundary property this test
        // exists for is unchanged and is the reason both spellings are asserted: the token
        // pricer agrees with what `mu_rat` charges in range, and with the SAME value
        // spelled as a fraction.
        //
        // The two spellings sit ONE MILLI-BIT apart, and that residue is pinned rather than
        // papered over: the e-notation path multiplies by the ROUNDED `L10_MILLI = 3322`
        // while the `p/q` path takes `l_millibits` of the digit string exactly. The min over
        // codes used to hide it (both spellings took the decimal code). 0.001 bits is not an
        // ordering cliff -- the property this test exists for -- but it IS spelling-dependent
        // pricing, which §10.10 says should not happen, so the BOUND is asserted too and will
        // fail loudly if the seam ever widens.
        let e_notation = mu_numeric_str("1e-40");
        let as_fraction = mu_numeric_str(&format!("1/1{}", "0".repeat(40)));
        assert_eq!(e_notation, 133_880);
        assert_eq!(as_fraction, 133_879);
        assert!(
            e_notation.abs_diff(as_fraction) <= 1,
            "the two spellings of 1e-40 drifted apart: {e_notation} vs {as_fraction}"
        );
    }

    /// B22, both halves. (1) PRICING: the linear schedule `|scale| * log2(10)` exhausts
    /// u64 at |scale| ~ 5.553e15, and saturating there collided every larger scale --
    /// and every beyond-i64 exponent -- on the single price `u64::MAX`, so mu could not
    /// tell `1e5553000000000000` from `1e10^30`. The plan's acceptance line is asserted
    /// verbatim, then a strict ladder across the whole astronomic range in both
    /// exponent signs. (2) ACCUMULATION: a `u64::MAX`-priced leaf made the tree sums in
    /// `complexity()` overflow -- a debug-build panic and a release-build WRAP, and a
    /// wrapped sum prices a composite below its own parts, which is the direction that
    /// licenses false mu-descents. Raw `Ex` shells (no canon) exercise the sums
    /// directly.
    #[test]
    fn astronomic_literals_stay_ordered_and_sums_never_wrap() {
        // (1) the plan's acceptance, verbatim
        assert_ne!(mu_numeric_str("1e5553000000000000"), u64::MAX);

        // strictly ordered across the old ceiling, the i64 boundary, and into
        // digit-string-only exponents; both signs ride the same schedule
        let huge = format!("1e1{}", "0".repeat(30));
        let huge_neg = format!("1e-1{}", "0".repeat(30));
        let ladder = [
            "1e308", // f64's edge: the exact linear regime
            "1e4294967296",
            "1e5553000000000000", // the old saturating_mul ceiling
            "1e9223372036854775807",
            "1e99999999999999999999", // exponent past i64: digit string only
            huge.as_str(),
        ];
        for w in ladder.windows(2) {
            assert!(
                mu_numeric_str(w[0]) < mu_numeric_str(w[1]),
                "mu not strictly increasing: mu({}) = {} !< mu({}) = {}",
                w[0],
                mu_numeric_str(w[0]),
                w[1],
                mu_numeric_str(w[1])
            );
        }
        let neg_ladder = [
            "1e-308",
            "1e-4294967296",
            "1e-5553000000000000",
            "1e-99999999999999999999",
            huge_neg.as_str(),
        ];
        for w in neg_ladder.windows(2) {
            assert!(
                mu_numeric_str(w[0]) < mu_numeric_str(w[1]),
                "mu not strictly increasing on the denominator side: {} vs {}",
                w[0],
                w[1]
            );
        }

        // (2) no wrap: a composite always prices at or above every part, and never
        // below a bare variable, whatever its leaves cost
        with_view(|view| {
            let a = Ex::Leaf(view.intern("1e5553000000000000"));
            let b = Ex::Leaf(view.intern("1e9223372036854775807"));
            let (ca, cb) = (complexity(&a, view), complexity(&b, view));
            let bare = complexity(&x(view), view);
            let shells = [
                Ex::Add(vec![a.clone(), b.clone()]),
                Ex::Mul(vec![a.clone(), b.clone()]),
                Ex::Pow(Box::new(a.clone()), Box::new(b.clone())),
                Ex::Fun(view.intern("sin"), vec![a.clone(), b.clone()]),
            ];
            for s in &shells {
                let cs = complexity(s, view);
                assert!(
                    cs >= ca.max(cb),
                    "composite priced below a part: {cs} < max({ca}, {cb}) on {s:?}"
                );
                assert!(cs >= bare, "composite priced below a bare variable: {cs}");
            }
            // The saturation point itself: enough astronomic members that an UNCHECKED
            // sum exceeds u64::MAX (with the pricing fixed, one leaf is ~1.4e13, so
            // this is ~1.3M members -- a ~16 MB hostile input, well within a caller's
            // reach). The property is stated so no wrap can hide: the total must be
            // NON-DECREASING in member count across the overflow threshold. A wrapped
            // sum shows as a decrease at the crossing pair wherever the modulus lands
            // (asserting only `total >= part` misses wraps that alias into [part, MAX)),
            // while a SATURATING sum only ever holds flat at the ceiling -- refusal is
            // sound, a composite under its parts is not.
            let q = usize::try_from(u64::MAX / ca).unwrap();
            let mut prev = 0u64;
            for n in [q - 1, q, q + 1, q + 2] {
                let cs = complexity(&Ex::Add(vec![a.clone(); n]), view);
                assert!(
                    cs >= prev,
                    "the Add total WRAPPED crossing {n} members: {prev} -> {cs}"
                );
                assert!(cs >= ca && cs >= bare, "an Add bag priced below a part");
                prev = cs;
            }
            // The Mul and Fun arms accumulate through the same repaired path; one
            // past-threshold bag each pins them at the saturated ceiling.
            for bag in [
                Ex::Mul(vec![a.clone(); q + 1]),
                Ex::Fun(view.intern("sin"), vec![a.clone(); q + 1]),
            ] {
                let cs = complexity(&bag, view);
                assert!(
                    cs >= ca && cs >= bare,
                    "an overflowing bag wrapped below its parts: {cs}"
                );
            }
        });
    }

    /// B5+B19: ONE owner for the odd-function literal sign, on every construction
    /// route. Before the hoist, `term_join` (the Add-collector's rebuild) fused
    /// `-5 * sin(2)` to `5 * sin(-2)` while `mul()`'s direct assembly did not -- two
    /// fixpoints for one value, differing by construction history (the invariance
    /// defect class). And the collector could not SPLIT the sign back out, so
    /// `sin(-2) + sin(2)` needed a render/re-parse cycle to cancel.
    #[test]
    fn odd_literal_sign_is_route_independent() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let sin = view.intern("sin");
            let sin2 = fun(sin, vec![Ex::int(2)], &cx);
            let sin_neg2 = fun(sin, vec![Ex::int(-2)], &cx);

            // The two routes must agree, and on the ratified I4 direction (the
            // literal owns the sign): -5 * sin(2) spells 5 * sin(-2).
            let via_mul = mul(vec![Ex::int(-5), sin2.clone()], &cx);
            let via_join = term_join(Rat::int(-5), sin2.clone(), &cx);
            assert_eq!(
                via_mul, via_join,
                "mul() and term_join disagree on the sign"
            );
            assert_eq!(
                via_mul,
                mul(vec![Ex::int(5), sin_neg2.clone()], &cx),
                "the fused spelling is not the canonical one"
            );
            // I4's other example: -1 * sin(2) files as sin(-2) (the Mul node itself
            // dissolves).
            assert_eq!(mul(vec![Ex::int(-1), sin2.clone()], &cx), sin_neg2);

            // B19 / adv-1: the collector conflates the pair IN ONE PASS -- sin(-2)
            // and sin(2) share a key with opposite coefficients and cancel exactly.
            assert_eq!(
                add(vec![sin_neg2.clone(), sin2.clone()], &cx),
                Ex::int(0),
                "sin(-2) + sin(2) must cancel at collection, not via re-parse"
            );

            // The collision guard: flipping sin(-2) beside sin(2) would mint an
            // unmerged duplicate-base bag; the orbit must skip such masks and keep
            // the entry spelling stable (idempotent).
            let prod = mul(vec![sin_neg2.clone(), sin2.clone()], &cx);
            assert_eq!(canon(prod.clone(), &cx), prod, "ground product not stable");
        });
    }

    /// Resolved-string rendering for CROSS-VIEW comparisons: two views may intern the
    /// same names to different ids, so `Ex` equality cannot compare their outputs --
    /// the resolved shape can.
    fn ex_str(e: &Ex, view: &TokenView) -> String {
        match e {
            Ex::Num(r) => format!("{r:?}"),
            Ex::Pi => "pi".into(),
            Ex::E => "e".into(),
            Ex::PosInf => "inf".into(),
            Ex::NegInf => "-inf".into(),
            Ex::NaN => "nan".into(),
            Ex::Const => "C".into(),
            Ex::Leaf(t) => view.resolve_owned(*t),
            Ex::Fun(f, args) => {
                let a: Vec<String> = args.iter().map(|x| ex_str(x, view)).collect();
                format!("{}({})", view.resolve_owned(*f), a.join(","))
            }
            Ex::Pow(b, x) => format!("pow({},{})", ex_str(b, view), ex_str(x, view)),
            Ex::Add(v) => {
                let a: Vec<String> = v.iter().map(|x| ex_str(x, view)).collect();
                format!("add[{}]", a.join(","))
            }
            Ex::Mul(v) => {
                let a: Vec<String> = v.iter().map(|x| ex_str(x, view)).collect();
                format!("mul[{}]", a.join(","))
            }
        }
    }

    /// F72: the sign-placement tie-break is a function of CONTENT, never of token
    /// interning order. The mu-tied orbit of `-1 * (a-b) * (c-d)` (a -1 coefficient
    /// rides mu-free, so every mask ties) must land on the same resolved spelling
    /// when the same four opaque literals are interned in OPPOSITE orders and the
    /// factors arrive swapped -- the pre-F72 `ex_struct_cmp` compared `Leaf`/`Fun`
    /// tokens by raw id and flipped the mirror decision with the interning (extreme
    /// fuzz rows 829655/866090, the two P2 regressions of the F62..F71 window).
    #[test]
    fn f72_tiebreak_is_interning_independent() {
        fn build(intern_first: &[&str], swap_factors: bool) -> String {
            with_view(|view| {
                for s in intern_first {
                    view.intern(s);
                }
                let cx = Cx::bare(view);
                let leaf = |s: &str| Ex::Leaf(view.intern(s));
                let neg = |e: Ex, cx: &Cx| mul(vec![Ex::int(-1), e], cx);
                let ab = add(vec![leaf("lit_a"), neg(leaf("lit_b"), &cx)], &cx);
                let cd = add(vec![leaf("lit_c"), neg(leaf("lit_d"), &cx)], &cx);
                let items = if swap_factors {
                    vec![Ex::int(-1), cd, ab]
                } else {
                    vec![Ex::int(-1), ab, cd]
                };
                ex_str(&mul(items, &cx), view)
            })
        }
        let base = build(&["lit_a", "lit_b", "lit_c", "lit_d"], false);
        assert_eq!(base, build(&["lit_d", "lit_c", "lit_b", "lit_a"], false));
        assert_eq!(base, build(&["lit_c", "lit_d", "lit_a", "lit_b"], true));
        assert_eq!(base, build(&["lit_b", "lit_a", "lit_d", "lit_c"], true));
    }

    /// F72: with the rational content PARTITIONED across several `Num` members (the
    /// product overflows i128), the sign has exactly ONE canonical host, a function
    /// of the value -- not of which member carried it on arrival. Before the
    /// sign-factored accumulation, `-N * P` kept the sign on the `-N` partial while
    /// the negation of `N * P` hosted it on the coefficient slot: two stable states
    /// for one value (the extreme lane's P1-idempotence family, 88 -> 30 rows).
    #[test]
    fn f72_partition_sign_host_is_route_invariant() {
        with_view(|view| {
            let cx = Cx::bare(view);
            // 0.9999999999999999 and 1/(2^127 - 1): the pair's product overflows the
            // i128 denominator, so the bag keeps BOTH as members.
            let n = Rat::new(9999999999999999, 10000000000000000).unwrap();
            let p = Rat::new(1, 170141183460469231731687303715884105727).unwrap();
            let route_signed_member = mul(vec![Ex::Num(n.checked_neg().unwrap()), Ex::Num(p)], &cx);
            let route_negated_bag =
                negate_term(&mul(vec![Ex::Num(n), Ex::Num(p)], &cx), &cx).unwrap();
            let route_other_host = term_join(p.checked_neg().unwrap(), Ex::Num(n), &cx);
            let route_swapped = mul(vec![Ex::Num(p), Ex::Num(n.checked_neg().unwrap())], &cx);
            assert_eq!(route_signed_member, route_negated_bag);
            assert_eq!(route_signed_member, route_other_host);
            assert_eq!(route_signed_member, route_swapped);
            // The chosen state is a fixpoint of its own constructor.
            let Ex::Mul(v) = route_signed_member.clone() else {
                panic!("partition bag expected, got {route_signed_member:?}");
            };
            assert_eq!(mul(v, &cx), route_signed_member);
        });
    }
}
