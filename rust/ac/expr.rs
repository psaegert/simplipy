//! The AC expression type and its CANONICAL CONSTRUCTORS.
//!
//! Canonical invariants (established by [`add`], [`mul`], [`pow`], [`fun`]; every consumer may
//! rely on them):
//! * `Add(v)`: `v.len() >= 2`; no element is an `Add`; at most one `Num` (never 0); at most one
//!   `Const`; like terms collected as far as the soundness gates allow; sorted by [`cmp_ex`].
//! * `Mul(v)`: `v.len() >= 2`; no element is a `Mul`; at most one `Num` (never 1; 0 only in the
//!   uncertified `0 * t` case documented at the zero-coefficient gate); at most one bare `Const`;
//!   like bases collected under the gates, and at most one power of e (`exp(a) exp(b)` is
//!   `exp(a + b)`); sorted by [`cmp_ex`].
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

use crate::engine::RuleMode;
use std::cmp::Ordering;

use crate::tokens::{Tok, TokenView};

use super::rat::{Mantissa, Rat};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;

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

    /// Does this expression SPELL an infinity anywhere (an `Ex::PosInf`/`Ex::NegInf` node)?
    /// Such a subtree can be infinite on a set of full measure, so the lossy blanket's
    /// premise ("finite almost everywhere") is not an approximation there but false (see
    /// `ac::search::distribute_product`).
    pub fn contains_infinity(&self) -> bool {
        match self {
            Ex::PosInf | Ex::NegInf => true,
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::NaN | Ex::Const | Ex::Leaf(_) => false,
            Ex::Add(v) | Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(Ex::contains_infinity),
            Ex::Pow(b, e) => b.contains_infinity() || e.contains_infinity(),
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
/// |   (magnitude 1: a bare sign)       |                  | coefficient-rides-free        |
/// |                                    |                  | asymmetry that minted the     |
/// |                                    |                  | materialization fork)         |
/// | rational `Pow` exponent            | as coefficient   | `x^-1` = division = structure |
/// | `<constant>`                       | c_free = 128     | a free real dof; the priciest |
/// |                                    |                  | atom (owner floor: >= special)|
///
/// (The sign rows of this preamble are superseded: a sign costs ONE BIT wherever it sits,
/// `mu_sign` below. A magnitude-1 exponent is division, not a sign, and stays free.)
/// The literal formula `max(2, bits(|p|) + bits(q) - 1)` is the unique reading
/// consistent with the ratified doc's worked examples: cost(2)=2 (T4: mu(2x)=18),
/// cost(5/2)=4 (T5: mu(2.5*pi)=20), a 52-bit dyadic ~105 (T1/T2), mu(0)=mu(1)=2 (T3).
///
/// (Historical preamble; later rulings amended it in place: milli-bit carrier and the
/// literal sign bit 2026-08-06, c_free DERIVED not floored 2026-08-06, and mu-PRIME at
/// D38/B2 2026-08-17 -- the numeric codebook gains the decimal-scientific codeword
/// behind a one-bit selector, and c_free' re-derives to 67 bits. Current authority:
/// `mu_rat`, `decimal_code`, `mu_numeric_str`, `mu_free` below.)
///
/// Worked examples: `x + y` = 24, `x*y` = 24, `2x` = 18, `x^2` = 18, `1/x` = 16,
/// `sin(x)` = 16, `sin(pi)` = 16, `E*x` = 24 < `2.718281828459045*x` = ~117,
/// `<constant>*x` = 144, `exp(pi*x)` = 32 < `pow(23.14..., x)` = ~121.
/// The symbol UNIT, in BITS: what one grammar symbol costs at the ratified 1/8 pin, and
/// the scale every entry of the symbol table below is expressed against. Since 2026-08-21
/// no node is priced at the bare unit except an unclassified function head -- the table
/// holds the per-kind entries, and this is what they are read AGAINST. Literal costs are
/// absolute bits and do not scale with it.
const MU_SYM_UNIT: u64 = 8;

/// A symbol-table entry, in MILLI-BITS: `bits` read at the ratified 8-bit unit.
///
/// EXACT in integer arithmetic at every unit -- `MU_MILLI / 8 = 125` divides evenly, so
/// no entry ever rounds. That is not cosmetic: the reduction ordering's well-foundedness
/// comes from mu taking values in a discrete set, and a table that rounded would owe a
/// fresh termination argument.
#[inline]
fn sym_bits(bits: u64) -> u64 {
    bits * (MU_MILLI / 8) * MU_SYM_UNIT
}

/// The mu SYMBOL TABLE (round-1, 2026-08-21), in BITS at the 8-bit unit.
///
/// mu' priced EVERY grammar symbol the same: one flat 8 bits for a bag, a `Pow`, a
/// function head, a variable leaf and a named constant alike. That flatness is what
/// minted the degenerate family `pi - 2 -> asin(sin 2)` (and its parametrised kin
/// `- k np.pi -> atan tan k`, `acos neg cos (-k)`). Those rules are CORRECT and
/// CERTIFIED -- `asin(sin x) = pi - x` on [pi/2, pi] -- and mu' genuinely preferred the
/// right-hand side, 19.000 against 19.585, because two transcendental heads plus a
/// literal cost exactly what a bag plus `pi` plus a literal cost.
///
/// A description length is not flat. `ac_node_census` over a 400-row corpus (10,996
/// charged nodes) reads the empirical class frequencies as
///
/// ```text
///   Leaf 3519 (1.65 bits)   Mul 1989 (2.47)    Num 1948 (2.50)   Pow 1250 (3.14)
///   Add  1088 (3.34)        Const 204 (5.75)   Fun:rootn 181 (5.92)
///   transcendental heads 40-68 each (7.3-8.1)
/// ```
///
/// -- structure CHEAP, a transcendental head EXPENSIVE, the exact opposite of what one
/// flat unit says. The table below is HEURISTIC, informed by that census and rounded to
/// whole bits: there is no standard corpus to derive it from, and the corpus that does
/// exist is a symbolic-regression skeleton set in which `pi` and `e` occur ZERO times, so
/// a straight frequency table would price them at the smoothing floor (~13.4 bits) and
/// make the defect family STRONGER rather than weaker.
///
/// Pi and E are fixed instead by a LAW the census cannot see: A NAMED CONSTANT MUST COST
/// LESS THAN THE CHEAPEST EXPRESSION DENOTING IT, or writing the name is never a descent.
/// The cheapest spellings here are `acos(-1)` = 11 bits and `exp(1)` = 9, so 4 has room --
/// and `np.e` now beats `exp 1` on COST rather than on the canonical tie-break, which is
/// what the fold filter's correctness had been resting on.
///
/// Entries are bits AT THE 8-BIT UNIT, while literal costs stay absolute bits.
const MU_BITS_LEAF: u64 = 6;
const MU_BITS_ADD: u64 = 3;
const MU_BITS_MUL: u64 = 3;
const MU_BITS_POW: u64 = 3;
const MU_BITS_PI: u64 = 4;
const MU_BITS_E: u64 = 4;
const MU_BITS_ELEMENTARY: u64 = 6;
const MU_BITS_TRANSCENDENTAL: u64 = 8;
const MU_BITS_INFINITY: u64 = 8;

/// A variable (or slot, or any other vocabulary) leaf.
#[inline]
pub fn mu_leaf() -> u64 {
    sym_bits(MU_BITS_LEAF)
}

/// The `Add` bag head. ONE node however many `+` the spelling shows.
#[inline]
pub fn mu_add() -> u64 {
    sym_bits(MU_BITS_ADD)
}

/// The `Mul` bag head. A negation `Mul[-1, x]` is no product and pays no head: it is a sign
/// (`mu_sign`).
#[inline]
pub fn mu_mul() -> u64 {
    sym_bits(MU_BITS_MUL)
}

/// A SIGN, in milli-bits: ONE BIT wherever it sits. The sign of a literal (`-2`, a coefficient
/// `-3`, an exponent `-2`), the coefficient -1 of a product beside its other factors
/// (`-x1*x2`), and a negation standing alone (`-x1`, the `-x2` of `x1 - x2`), which is a sign
/// and not a product, so it pays its operand and this bit and no `Mul` head. `+1` is no sign
/// and costs nothing, and a reciprocal (`Pow[y, -1]`) is division, not a sign.
///
/// A description length must tell a number from its negation, and one bit is what that
/// distinction costs; pricing it the same everywhere is what makes the placement of a sign
/// a question the sign key decides rather than the price (`-x1*tanh(3)` and
/// `x1*tanh(-3)` cost the same). An absolute bit, like a literal's, not a symbol-table
/// entry: it does not scale with `MU_SYM_UNIT`.
pub const MU_SIGN: u64 = MU_MILLI;

/// See [`MU_SIGN`].
#[inline]
pub fn mu_sign() -> u64 {
    MU_SIGN
}

/// The `Pow` head -- also the division wrapper `Pow[y, -1]` that spells `x / y`.
#[inline]
pub fn mu_pow() -> u64 {
    sym_bits(MU_BITS_POW)
}

/// `np.pi`. Below `acos(-1)` = 11 bits, its cheapest denoting expression.
#[inline]
pub fn mu_pi() -> u64 {
    sym_bits(MU_BITS_PI)
}

/// `np.e`. Below `exp(1)` = 9 bits, its cheapest denoting expression -- the inequality
/// the fold filter needs `exp 1 -> np.e` to be a strict improvement.
#[inline]
pub fn mu_e() -> u64 {
    sym_bits(MU_BITS_E)
}

/// `float("inf")`, `float("-inf")`, `float("nan")`. Left at the flat unit: the census
/// cannot see them either, and unlike `pi` and `e` no law pushes them down -- an infinity
/// is not a value a shorter expression denotes.
#[inline]
pub fn mu_infinity() -> u64 {
    sym_bits(MU_BITS_INFINITY)
}

/// A function HEAD, priced by name, in BITS at the 8-bit unit.
///
/// Two tiers, read off the census: the ELEMENTARY heads a skeleton reaches for constantly
/// (`rootn` alone is 181 of 10,996 charged nodes at 5.92 bits) and the TRANSCENDENTAL
/// heads it barely reaches at all (40-68 occurrences each, 7.3-8.1 bits). The gap between
/// the tiers is the whole point: it is what makes `pi - 2` cheaper than `asin(sin 2)`.
///
/// A head this table does not name is a USER-DEFINED operator, which no census can price;
/// it keeps the flat unit, so an unknown symbol is never made cheaper than mu' had it.
fn mu_fun_bits(name: &str) -> u64 {
    match name {
        "exp" | "log" | "abs" | "sin" | "cos" | "tan" | "rootn" => MU_BITS_ELEMENTARY,
        "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" => {
            MU_BITS_TRANSCENDENTAL
        }
        // The 0.15 heads, priced at the transcendental tier by ruling (no census yet).
        "floor" | "ceil" | "max" | "min" | "mod" | "harmonic" | "erf" | "atan2" | "besselj" => {
            MU_BITS_TRANSCENDENTAL
        }
        _ => 8,
    }
}

/// The free constant's price, DERIVED rather than chosen (contract §10.10(5), H-054) --
/// re-derived under the mu' codebook at D38/B2 (2026-08-17): the sup-construction
/// RECIPE is untouched, the codebook under the sup changed.
///
/// `<constant>` denotes an unknown value that a FIT will supply, i.e. an f64, and the ordering
/// requires it to dominate every literal it could be instantiated to: the price is the
/// SUPREMUM of mu' over f64 round-trip spellings (the exact rationals the shortest reprs
/// of finite f64s denote), plus the sign bit, ceiled.
///
/// ```text
///   max over f64 round-trip spellings m * 10^s of
///       min( rational codeword,  max(2, L(m)) + L(|s|) )
///     = 64.649 bits, attained at 8.9002954340287245e-308
///       (m = 89002954340287245, s = -324: the largest 17-digit mantissa any
///        shortest repr can carry -- the ulp-per-decimal-grid feasibility bound
///        m < 10 * 2^53 -- meeting the deepest scale a 17-digit spelling reaches)
///   + 1 sign bit + 1 selector bit  = 66.649 bits
///   rounded up                     = 67 bits
/// ```
///
/// The scientific codeword caps the sup analytically: `L(m) < L(10^17) = 56.475` (a
/// shortest repr has at most 17 significant digits) and `|s| <= 343` (the deepest
/// reachable spelling scale), so sup < 64.92 and the ceiling is robust to the exact
/// argmax. Swept: every binade top in the boundary decades walked by ulp (600k ulps
/// each), the full subnormal boundary, every binary exponent at six mantissas, and
/// 400,000 random bit patterns.
/// Under the fraction-only code the same construction gave 1133 bits, driven by the
/// 10^324-scale DENOMINATOR of the extreme spellings -- the D38 overcharge itself; the
/// re-derivation lands a free parameter at ~8 grammar symbols instead of ~142.
/// Executable derivations: `c_free_prime_derivation` below (this file) and
/// `tests/test_mu_prime.py::TestCfreePrimeDerivation` (independent model + the
/// engine-priced dominance sweep).
///
/// SCOPED TO THE f64 RANGE ON PURPOSE, as before: a beyond-f64 literal is not a value a
/// fit can produce, so it is not a value `<constant>` stands for. NOTE the direction of
/// the old "beyond-f64 literals outprice the free constant" consequence PARTLY inverts
/// under mu': a deep-but-thin beyond-f64 literal (`1e-400`, now 10.651 bits via its
/// scientific codeword) undercuts c_free'. Nothing load-bearing rests on the old
/// direction -- G3 refuses Const-introducing rules regardless of price, and the
/// dominance the ordering needs (over every FITTABLE value) is exactly what the sup
/// construction guarantees.
pub const MU_FREE_WORST_CASE_F64: u64 = 67 * MU_MILLI;

/// `<constant>`, the priciest atom: the worst case over every fittable f64.
pub const fn mu_free() -> u64 {
    MU_FREE_WORST_CASE_F64
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

    // x = m / 2^(b-1), in [1, 2), as an L_FRAC-bit fixed-point value.
    let x: u128 = if (b - 1) as u32 >= L_FRAC {
        m >> ((b - 1) as u32 - L_FRAC)
    } else {
        m << (L_FRAC - (b - 1) as u32)
    };
    l_millibits_fixed(b - 1, x)
}

/// `l_millibits` for an integer of any size: the same extraction on its top bits, so it is
/// bit-identical to `l_millibits` wherever both apply (the measure digest must not move for
/// numbers that fit 128 bits).
pub fn l_millibits_big(n: &BigUint) -> u64 {
    if let Some(small) = n.to_u128().filter(|&v| v < u128::MAX) {
        return l_millibits(small);
    }
    let m = n + 1u32;
    let b = m.bits(); // well beyond L_FRAC bits here
    let x = (&m >> (b - 1 - u64::from(L_FRAC)))
        .to_u128()
        .expect("the top L_FRAC + 1 bits fit");
    l_millibits_fixed(b - 1, x)
}

/// The fixed-point precision of `l_millibits`.
const L_FRAC: u32 = 48;

/// `1000 * (int_part + log2(x / 2^L_FRAC))`, rounded to the milli-bit, for `x` in
/// [2^L_FRAC, 2^(L_FRAC + 1)).
fn l_millibits_fixed(int_part: u64, mut x: u128) -> u64 {
    const FRAC: u32 = L_FRAC;
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

/// The DECIMAL-SCIENTIFIC codeword's cost components for a value with a finite decimal
/// expansion (`q = 2^a * 5^b`, which includes every integer): the cost of the integer
/// mantissa `m` of the SHORTEST exact decimal spelling `m * 10^k` and of the decimal
/// exponent `k`, each priced by `l_millibits` -- the exponent is just another integer
/// the spelling has to write down, `L(k) = log2(1 + |k|)`. The pair is the value's one
/// decimal decomposition ([`Rat::decimal_parts`]), the same the printer spells.
///
/// For a non-integer, `k = -max(a, b)` and `m = |p| * 10^|k| / q` (for a REDUCED
/// fraction `m` is never divisible by 10, so this IS the shortest spelling); for an
/// integer, `k >= 0` is its count of trailing base-ten zeros and `m` the remaining
/// digits (D38: the spec's "shortest exact decimal spelling" reading -- `1000` is the
/// codeword `(1, 3)`; an integer with no trailing zeros has `k = 0`, where the
/// codeword degenerates to the rational one and is refused as redundant).
///
/// `None` when the value does not terminate in base ten or when the codeword would be
/// degenerate (`k = 0`, zero included).
fn decimal_code(r: &Rat) -> Option<(u64, u64)> {
    let parts = r.decimal_parts()?;
    if parts.exp == 0 {
        return None;
    }
    let m = match &parts.mantissa {
        Mantissa::Small(m) if *m < u128::MAX => l_millibits(*m),
        Mantissa::Small(m) => l_millibits_big(&BigUint::from(*m)),
        Mantissa::Big(m) => l_millibits_big(m),
    };
    Some((m, l_millibits(u128::from(parts.exp.unsigned_abs()))))
}

/// Description length of the exact VALUE p/q (lowest terms), in MILLI-BITS and
/// spelling-free -- **mu' since D38 (B2, 2026-08-17): the numeric codebook carries TWO
/// codewords, chosen by a one-bit selector.**
///
/// ```text
/// mu'(value) = 1 selector bit + min( L(p) + L(q),  L(m) + L(k) ) + sign bit
/// ```
///
/// where `(p, q)` is the reduced rational spelling (an integer's denominator is
/// implicit and free) and `(m, k)` the decimal-scientific spelling of the SHORTEST
/// exact decimal print (`decimal_code`), with `L(k) = log2(1 + |k|)` -- the exponent is
/// an integer the spelling writes down, not a run of zeros. A value whose denominator
/// has a prime factor other than 2 and 5 has no decimal codeword and pays the rational
/// codeword alone -- still plus the selector, so every priced numeric leaf pays exactly
/// one selector bit and the codebook is a genuine prefix code.
///
/// Each codeword total carries the sign bit and the two-bit mantissa floor exactly as
/// the D38 rescorer priced them (`d38_muprime_rescore.dec_price`: the mantissa at the
/// integer price `max(2, L(m))`, the exponent at `L(k)`, the sign once); the NATIVE
/// measure adds the selector uniformly on top, which the study rescorer -- computing
/// `min(old, dec)` only -- deliberately omitted. Native mu' and study-rescored prices
/// therefore differ by a constant 1000 mB per priced numeric leaf, by design.
///
/// WHY A SECOND CODEWORD (D38). MDL codebooks should match the source distribution:
/// SR constants are decimal-printed f64s, and the fraction-only code forced them
/// through "two arbitrary integers", overcharging ~3.32 bits per decimal shift
/// (`2.177697277405848` priced 94.781 bits as p/q against 55.952 as mantissa+exponent
/// +selector). This REVOKES §10.10(1)/H-055 ("mu is value complexity, no min over
/// codes") -- the ratified resolution of the H-055 asymmetry is now symmetry at the
/// codeword level: `mu'(1000) == mu'(0.001)` (both are the codeword `(1, 3)`), where
/// H-055 had `9.967` against `10.967`.
///
/// WHY `L` AND NOT A BIT COUNT. `bits()` quantises exactly where it hurts: it cannot
/// separate 100 from 1000 once a scale is involved (both scales cost 2 bits), it priced
/// two spellings of one value up to a bit apart for no reason but rounding, and it made
/// mu non-monotone in magnitude at the `i128` boundary. `L` is the continuous lower
/// envelope of `bits` -- equal at 1, 3, 7, 127, never more than a bit below.
///
/// WHY THE SIGN COSTS A BIT. Without it `mu(2) == mu(-2)`: the measure could not tell a
/// number from its negation, which a description length may not do. `Rat` normalises to
/// `q > 0` with the sign on the numerator, so the bit is unambiguous and charged once,
/// inside EACH codeword total (it cancels in the min).
pub fn mu_rat_exact(r: &Rat) -> u64 {
    let (fraction, decimal) = mu_rat_codeword_totals(r);
    MU_MILLI
        + match decimal {
            Some(d) => fraction.min(d),
            None => fraction,
        }
}

/// THE PRICE OF A LITERAL: the lesser of the literal's own exact spelling ([`mu_rat_exact`])
/// and the shortest decimal of the float64 it reads as, priced by the same codewords. The
/// evaluator reads every literal as its nearest float64, so digits beyond float precision are
/// not information: the 64-digit fourth power of a 16-digit constant costs what its float's
/// shortest decimal (at most 17 significant digits) costs. A literal that is its own float's
/// shortest decimal keeps its price, and no price rises. The cap is that decimal (Rust's `{:?}`
/// spelling; on rare exact ties it can differ from Python's `repr` in the last digit), never a
/// fraction that reads as the same float: `0.3333333333333333` keeps its 16-digit price although
/// `1/3` reads as the same float. A literal beyond float64's range (or below its smallest
/// subnormal) keeps its exact price. The cap reaches every literal the exact domain holds as a
/// number, however a state holds it (`is_piece`, `piece_price`): a numeric-string leaf not
/// admissible in f64 mode, and a product, sum or power the f64 domain keeps in pieces, cost at
/// most what this function prices their value at; a leaf beyond the 1,100-bit cap keeps its
/// print's price (`mu_numeric_str`), so no ordering cliff opens between the two. The state keeps
/// the exact value and prints it: only the measure reads the float. The price is a function of the
/// value alone (the decimal is built in the exact number domain), so each thread remembers the
/// prices it computed, emptying the table at 65,536 entries.
pub fn mu_rat(r: &Rat) -> u64 {
    thread_local! {
        static FLOAT_PRICES: std::cell::RefCell<rustc_hash::FxHashMap<Rat, u64>> =
            std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    }
    if let Some(v) = FLOAT_PRICES.with(|m| m.borrow().get(r).copied()) {
        return v;
    }
    let v = mu_rat_float(r);
    FLOAT_PRICES.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() >= 1 << 16 {
            m.clear();
        }
        m.insert(r.clone(), v);
    });
    v
}

/// [`mu_rat`] without the table.
fn mu_rat_float(r: &Rat) -> u64 {
    let exact = mu_rat_exact(r);
    let y = r.to_f64_nearest();
    if !y.is_finite() || y == 0.0 {
        return exact;
    }
    match Rat::shortest_reading_as(y) {
        Some(d) => exact.min(mu_rat_exact(&d)),
        None => exact,
    }
}

/// The two codewords' TOTAL prices, in milli-bits, each carrying its own floor and the
/// sign bit, WITHOUT the selector (the caller adds it once): `(rational,
/// decimal_scientific_if_terminating)`.
///
/// Rational: `max(2 bits, L(p) [+ L(q) if q > 1] + sign)` -- an integer pays only its
/// own cost (implicit denominator), a genuine fraction pays both components. Decimal-
/// scientific: `max(2 bits, L(m)) + L(k) + sign` -- the mantissa at the integer price
/// (exactly `c([str(mant)])` of the rescorer), the exponent on top. These are ALSO the
/// exact quantities the serializer argmin compares (`decimal_spelling_wins`), so the
/// spelling emitted is by construction the codeword the measure priced.
/// The literal codeword FLOOR, in milli-bits: ONE bit, beyond the selector (owner ruling
/// 2026-08-22; it was two, and two was blind).
///
/// At two, `0`, `1`, `2` and `3` all priced 3000 -- `L` is under the clamp there -- so the
/// measure could not tell `x^2` from `x^3` from `x^4`. At one the clamp stops binding
/// except at `0` and `+-1`, where a codeword cannot cost less than a bit anyway, and the
/// exponent rows separate by `log2(3) - log2(2) = 415` milli-bits.
///
/// MEASURED before adoption, against the shipped floor of two: **0 of 5,451 shipped rules
/// change direction** and **0 of 105 printed literal spellings move** -- the serializer
/// argmin compares the same two codeword totals, and the floor shifts both. Four f64 rows
/// and six corpus rows of the 400-row benchmark re-spell, and the corpus rests cheaper by
/// 3,000 (f64) and 3,359 (permissive) milli-bits: a small genuine descent. The D38/B2
/// numbers move with it -- `mu(1000)` is 4 bits where the ratified note says 5 -- but
/// H-055's finding was the SYMMETRY, `mu(1000) == mu(0.001)`, and that holds at 4.
/// Evidence: `audit-2026-08-21/floor/`.
const MU_RAT_FLOOR: u64 = 1 * MU_MILLI;

fn mu_rat_codeword_totals(r: &Rat) -> (u64, Option<u64>) {
    // A big-form literal's codewords cost big-integer divisions and logarithms, and the same
    // values are priced again and again while candidates are compared (the sign placement in
    // `mul` alone prices whole factors per orientation). The price is a function of the value,
    // so a per-thread table answers repeats; it is emptied when it reaches 65,536 entries
    // (bounded per thread, also under the miner's thread pool).
    if r.small_parts().is_some() {
        return mu_rat_codeword_totals_uncached(r);
    }
    thread_local! {
        static BIG_PRICES: std::cell::RefCell<rustc_hash::FxHashMap<Rat, (u64, Option<u64>)>> =
            std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    }
    if let Some(v) = BIG_PRICES.with(|m| m.borrow().get(r).copied()) {
        return v;
    }
    let v = mu_rat_codeword_totals_uncached(r);
    BIG_PRICES.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() >= 1 << 16 {
            m.clear();
        }
        m.insert(r.clone(), v);
    });
    v
}

fn mu_rat_codeword_totals_uncached(r: &Rat) -> (u64, Option<u64>) {
    let sign = if r.is_negative() { MU_SIGN } else { 0 };
    let (pb, qb) = match r.small_parts() {
        Some((p, q)) => (l_millibits(p.unsigned_abs()), l_millibits(q as u128)),
        None => {
            let (p, q) = r.big_parts();
            (
                l_millibits_big(p.magnitude()),
                l_millibits_big(q.magnitude()),
            )
        }
    };
    let fraction_raw = if r.is_integer() { pb } else { pb + qb };
    let floor = MU_RAT_FLOOR;
    let fraction = (fraction_raw + sign).max(floor);
    let decimal = decimal_code(r).map(|(m, k)| m.max(floor) + k + sign);
    (fraction, decimal)
}

/// Whether the DECIMAL spelling is the canonical PRINT for this value: the serializer
/// argmin over the SAME two codeword totals `mu_rat_exact` mins over, so the print costs
/// the exact price; the measure (`mu_rat`) can be lower, by the float cap (the D38 state/serializer factoring:
/// states carry one exact rational per value and no spelling; the emitter realizes the
/// best codeword at print time, locally and closed-form).
///
/// TIES GO TO THE FRACTION -- the structurally distinguished caller-dialect member,
/// mirroring the sign-trade tier-2 rule ("what you typed survives"; states carry no
/// spelling, so the tie-break must be spelling-free, and either choice realizes the
/// same mu'). Measured: NO exact tie was found in the i128 lattice -- every
/// `(a, b)` with `q = 2^a * 5^b < 2^127` (3,563 pairs) scanned exhaustively over
/// numerators `p <= 2000` and at 29 spread samples up to ~3.5e9 beyond; the closest
/// calls observed are 3-4 milli-bits (`54321 / 2^45*5^14` and `19 / 2^28*5^9`, the
/// falsifier suite's boundary pair) -- so the clause is defensive. Ties would also
/// keep the powers of two (`1/2`, `5/8`) on the fraction side, exactly as before D38.
///
/// Callers gate integers upstream: an integer prints its decimal spelling
/// (`Rat::exact_decimal`, the rendering of BOTH an integer's codewords), positional in the
/// ordinary range (`1000`, never `1e3`) and scientific beyond it where shorter (`1e300`).
pub fn decimal_spelling_wins(r: &Rat) -> bool {
    let (fraction, decimal) = mu_rat_codeword_totals(r);
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
/// exactly the mu' rule `mu_rat_exact` applies in range (D38): one selector bit, then the
/// cheaper of the RATIONAL codeword (every integer the spelling writes down costs
/// `L(n)`, a fraction pays both components) and the DECIMAL-SCIENTIFIC codeword
/// (`max(floor, L(m)) + L(|scale|)` for the shortest spelling `m * 10^scale`, the
/// floor being `MU_RAT_FLOOR` -- one bit since the 2026-08-22 ruling), plus a sign
/// bit inside each codeword total.
///
/// Literals whose exact p/q exceeds 128 bits (`4.159653437657682e-35` has q = 10^50) live
/// as numeric-string LEAVES, invisible to `mu_rat` -- pricing them at one vocabulary symbol
/// would (a) invert the ordering at the `Rat` boundary (a ~220-bit object undercutting
/// `0.001`), (b) reopen the literal-respell hole the old `lit_size` tier closed in exactly
/// this range, and (c) decouple the fold's literal pricing from the in-range rule.
///
/// The rational codeword stays an approximation of the true reduced p/q by design (the
/// 2/5 gcd is ignored; the boundary this exists for cannot turn on a milli-bit) --
/// under mu' it is also almost never the argmin out here, every finite-decimal string
/// having a scientific codeword. Deterministic, and monotone in both the significand
/// and the scale within each codeword.
pub fn mu_numeric_str(s: &str) -> u64 {
    let negative = s.starts_with('-');
    let body = s.strip_prefix('-').unwrap_or(s);
    let sign = if negative { MU_SIGN } else { 0 };
    // Every priced numeric leaf pays the one selector bit on top of its cheapest
    // codeword total; each total carries the sign and the codeword floor itself,
    // mirroring `mu_rat_codeword_totals` exactly -- the SAME `MU_RAT_FLOOR`, or the
    // boundary becomes an ordering cliff: with this side clamped at two bits while
    // the in-range side moved to one, mu(1e-39) priced ABOVE mu(3e-38) (S15) -- the
    // simpler mantissa at the adjacent magnitude cost more, purely for having
    // crossed the i128 line.
    let finish = |fraction: u64, decimal: Option<u64>| {
        let f = fraction.saturating_add(sign).max(MU_RAT_FLOOR);
        MU_MILLI
            + match decimal {
                Some(d) => f.min(d.saturating_add(sign)),
                None => f,
            }
    };

    // Fraction-shaped beyond-`Rat` literals (`p/q` with a component past i128 -- in-range
    // fractions parse to `Num` and never reach the leaf pricing).
    if let Some((p, q)) = body.split_once('/') {
        let ps = p.trim_start_matches('0');
        let qs = q.trim_start_matches('0');
        if ps.is_empty() {
            return finish(0, None); // zero numerator: the value is 0
        }
        let lp = l_millibits_digits(ps);
        if qs == "1" {
            return finish(lp, None); // an implicit denominator is free, as for any integer
        }
        let fraction = lp.saturating_add(l_millibits_digits(qs));
        // The decimal-scientific codeword of a written p/q: strip the denominator's
        // trailing zeros (10^t), factor the u128-sized head into 2^a * 5^b, and price
        // the scaled mantissa `p * 2^(k-a') * 5^(k-b')`. Exactly `decimal_code`'s rule,
        // carried across the boundary so `1e-40` and `1/1<40 zeros>` price EQUAL (the
        // pre-D38 build pinned them one milli-bit apart under the fraction-only code).
        // Refusals (head beyond u128, scaled mantissa beyond u128, non-2^a5^b head) fall
        // back to the rational codeword -- the same bounded-effort boundary as in range.
        let dec = (|| -> Option<u64> {
            let head = qs.trim_end_matches('0');
            let t = (qs.len() - head.len()) as u32;
            let mut rest: u128 = head.parse().ok()?;
            let (mut a, mut b) = (0u32, 0u32);
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
            let k = a.max(b) + t;
            let (two, five) = (a.max(b) - a, a.max(b) - b);
            let m_cost = if two == 0 && five == 0 {
                // mantissa is p itself -- priced from its digits, any length
                l_millibits_digits(ps)
            } else {
                let mut m: u128 = ps.parse().ok()?;
                for _ in 0..two {
                    m = m.checked_mul(2)?;
                }
                for _ in 0..five {
                    m = m.checked_mul(5)?;
                }
                l_millibits(m)
            };
            Some(
                m_cost
                    .max(MU_RAT_FLOOR)
                    .saturating_add(l_millibits(u128::from(k))),
            )
        })();
        return finish(fraction, dec);
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
        return finish(0, None); // the string denotes zero
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
    // The RATIONAL codeword writes the whole power of ten out (numerator zeros or the
    // 10^|scale| denominator). Two regimes, one knee (B22, doc at `MU_SCALE_KNEE`): the
    // exact linear schedule up to the knee, then `KNEE_COST + L(|scale|)`, monotone and
    // far from u64::MAX where the old arms saturated and collided.
    let power_of_ten_cost = if exp_digits.is_none() && scale.unsigned_abs() <= MU_SCALE_KNEE {
        scale.unsigned_abs() * L10_MILLI // <= MU_SCALE_KNEE_COST: cannot overflow
    } else {
        MU_SCALE_KNEE_COST.saturating_add(scale_digits_cost)
    };
    let fraction = l_sig.saturating_add(power_of_ten_cost);
    // The DECIMAL-SCIENTIFIC codeword writes the SCALE as an integer: `sig_core` is
    // trailing-zero-stripped, so `(sig, scale)` IS the shortest exact spelling --
    // integers included, exactly as `decimal_code`'s integer arm reads them in range
    // (a `scale = 0` codeword degenerates to the rational one and is refused).
    let dec = if scale == 0 {
        None
    } else {
        Some(l_sig.max(MU_RAT_FLOOR).saturating_add(scale_digits_cost))
    };
    finish(fraction, dec)
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
/// CENSUS of the nodes `complexity` actually CHARGES, keyed by kind.
///
/// Mirrors the pricing arms exactly, including their silences: a +1 rational coefficient
/// inside a `Mul` bag is free, so it is not counted; a -1 coefficient is a `Sign`, and a
/// negation standing alone (`Mul[-1, x]`) is a `Sign` and no `Mul`; a degenerate
/// all-coefficient bag is the literal itself. Counting serialized TOKENS instead gets this
/// wrong in both directions -- `-` and `/` are not nodes (they are an inverted member of an
/// Add/Mul bag), and an n-ary bag is ONE node however many `+` the spelling shows.
pub fn node_census(e: &Ex, view: &TokenView, out: &mut std::collections::HashMap<String, u64>) {
    let mut bump = |k: &str| *out.entry(k.to_string()).or_insert(0) += 1;
    match e {
        Ex::Num(_) => bump("Num"),
        Ex::Pi => bump("Pi"),
        Ex::E => bump("E"),
        Ex::PosInf => bump("PosInf"),
        Ex::NegInf => bump("NegInf"),
        Ex::NaN => bump("NaN"),
        Ex::Const => bump("Const"),
        Ex::Leaf(_) => bump("Leaf"),
        Ex::Add(v) => {
            bump("Add");
            for x in v {
                node_census(x, view, out);
            }
        }
        Ex::Mul(v) => {
            if !is_negation(v) {
                bump("Mul");
            }
            for f in v {
                // The free +1 coefficient slot is NOT charged, so it is not counted; -1 is a sign.
                if let Ex::Num(r) = f {
                    if r.is_one() {
                        continue;
                    }
                    if *r == Rat::NEG_ONE {
                        *out.entry("Sign".to_string()).or_insert(0) += 1;
                        continue;
                    }
                }
                node_census(f, view, out);
            }
        }
        Ex::Pow(b, ex) => {
            bump("Pow");
            node_census(b, view, out);
            node_census(ex, view, out);
        }
        Ex::Fun(t, args) => {
            let name = view.with_str(*t, |s| s.to_string());
            *out.entry(format!("Fun:{name}")).or_insert(0) += 1;
            for a in args {
                node_census(a, view, out);
            }
        }
    }
}

/// Is this product a NEGATION standing alone, `Mul[-1, x]` (a sign, not a product)?
fn is_negation(v: &[Ex]) -> bool {
    v.len() == 2
        && v.iter()
            .any(|f| matches!(f, Ex::Num(r) if *r == Rat::NEG_ONE))
        && v.iter().any(|f| !matches!(f, Ex::Num(_)))
}

/// THE MEASURE `mu` of a canonical state, in milli-bits.
pub fn complexity(e: &Ex, view: &TokenView) -> u64 {
    priced(e, view).0
}

/// Is `s` a NUMERIC-STRING leaf (a literal the engine holds as written)? A first-byte filter
/// skips the full numeric parse for ordinary vocabulary leaves (`x0`, function names), which
/// never start with a digit, dot, or sign.
#[inline]
fn numeric_leaf(s: &str) -> bool {
    s.as_bytes()
        .first()
        .is_some_and(|b| b.is_ascii_digit() || *b == b'-' || *b == b'.' || *b == b'+')
        && crate::utils::is_numeric_string(s)
}

/// A PIECE OF A LITERAL: a number, a numeric-string leaf, an integer power of a piece, or a sum
/// or product of pieces -- variable-free arithmetic that denotes one rational value.
///
/// The exact number domain (`real`) holds every such value within the 1,100-bit cap as one
/// `Rat`, which [`mu_rat`] prices at most at its float's price. The f64 domain admits only
/// numbers whose components are at most 2^1022 (`rat::number_domain`) and holds the rest in
/// pieces: a literal whose exact fraction leaves it stays a leaf as written
/// (`1.2345678901234567e-305`, whose denominator is 10^321), and a product, sum or power that
/// leaves it stays a product, sum or power (`1.2345678901234567e-305 / 9.876543210987654e-10`
/// is the leaf times the number `1/9.876543210987654e-10`). Priced piece by piece, such a value
/// costs what its pieces cost, above its float. [`priced`] prices the pieces of a bag, and a
/// node that is pieces throughout, as the one literal they denote ([`piece_price`]).
fn is_piece(e: &Ex, view: &TokenView) -> bool {
    match e {
        Ex::Num(_) => true,
        Ex::Leaf(t) => view.with_str(*t, numeric_leaf),
        Ex::Pow(b, k) => matches!(&**k, Ex::Num(k) if k.is_integer()) && is_piece(b, view),
        Ex::Add(v) | Ex::Mul(v) => v.iter().all(|f| is_piece(f, view)),
        _ => false,
    }
}

/// The nodes of a piece ([`is_piece`]); an exponent is part of its power.
fn piece_nodes(e: &Ex) -> u64 {
    match e {
        Ex::Pow(b, _) => 1 + piece_nodes(b),
        Ex::Add(v) | Ex::Mul(v) => v.iter().fold(1, |n, f| n + piece_nodes(f)),
        _ => 1,
    }
}

/// The exact value of a piece ([`is_piece`]) as the exact number domain holds it: `None` where
/// that domain holds no number for it (a component beyond the 1,100-bit cap, a leaf no numeral
/// reader reads) or where `e` is no piece. The caller enters the exact domain.
fn piece_value(e: &Ex, view: &TokenView) -> Option<Rat> {
    match e {
        Ex::Num(r) => Some(r.clone()),
        Ex::Leaf(t) => view.with_str(*t, leaf_value),
        Ex::Pow(b, k) => match &**k {
            Ex::Num(k) => piece_value(b, view)?.checked_pow_integer(k),
            _ => None,
        },
        Ex::Add(v) => pieces_value(v.iter(), true, view),
        Ex::Mul(v) => pieces_value(v.iter(), false, view),
        _ => None,
    }
}

/// The exact value of a numeric-string leaf as the exact number domain holds it (see
/// [`piece_value`]). The caller enters the exact domain.
fn leaf_value(s: &str) -> Option<Rat> {
    if !numeric_leaf(s) {
        return None;
    }
    let (p, q) = super::rat::token_rational(s)?;
    Rat::from_big(p, q)
}

/// The exact sum (`sum`) or product of pieces ([`piece_value`]), formed exactly and capped once,
/// so it does not depend on the order of a bag's members. The caller enters the exact domain.
fn pieces_value<'a>(
    pieces: impl Iterator<Item = &'a Ex>,
    sum: bool,
    view: &TokenView,
) -> Option<Rat> {
    let mut p = BigInt::from(u8::from(!sum));
    let mut q = BigInt::from(1u8);
    for f in pieces {
        let (a, b) = piece_value(f, view)?.big_parts();
        if sum {
            p = p * &b + a * &q;
        } else {
            p *= a;
        }
        q *= b;
    }
    Rat::from_big(p, q)
}

/// The exact value of the pieces among `members` ([`is_piece`]), their sum or their product.
fn bag_literal_value(members: &[Ex], sum: bool, view: &TokenView) -> Option<Rat> {
    let _exact = super::rat::number_domain(false);
    pieces_value(members.iter().filter(|f| is_piece(f, view)), sum, view)
}

/// THE FLOAT CAP ON EVERY PATH. The price of a literal held in pieces ([`is_piece`]):
/// `structural`, its pieces priced one by one, at most what the exact domain prices the value
/// they denote ([`mu_rat`], so at most what its float costs), and never below one bit per node,
/// so that the price still bounds the node count (L5, docs/formal.md). `None`, a value the exact
/// domain holds no number for (beyond the 1,100-bit cap), keeps the structural price.
fn piece_price(structural: u64, value: Option<&Rat>, nodes: u64) -> u64 {
    match value {
        Some(v) => structural.min(mu_rat(v).max(nodes.saturating_mul(MU_MILLI))),
        None => structural,
    }
}

/// The price of a NUMERIC-STRING leaf: the price of its print ([`mu_numeric_str`]), at most what
/// the exact domain prices its value ([`piece_price`]): a literal not admissible in the f64
/// domain costs at most its float's price, as a `Num` does; one beyond the 1,100-bit cap keeps its
/// print's price. A function of the string, so each thread remembers the prices it computed,
/// emptying the table at 65,536 entries.
fn mu_numeric_leaf(s: &str) -> u64 {
    thread_local! {
        static LEAF_PRICES: std::cell::RefCell<rustc_hash::FxHashMap<Box<str>, u64>> =
            std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    }
    if let Some(v) = LEAF_PRICES.with(|m| m.borrow().get(s).copied()) {
        return v;
    }
    let value = {
        let _exact = super::rat::number_domain(false);
        leaf_value(s)
    };
    let v = piece_price(mu_numeric_str(s), value.as_ref(), 1);
    LEAF_PRICES.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() >= 1 << 16 {
            m.clear();
        }
        m.insert(s.into(), v);
    });
    v
}

/// [`complexity`], and whether `e` is a piece of a literal ([`is_piece`]).
fn priced(e: &Ex, view: &TokenView) -> (u64, bool) {
    match e {
        Ex::Num(r) => (mu_rat(r), true),
        Ex::Pi => (mu_pi(), false),
        Ex::E => (mu_e(), false),
        Ex::PosInf | Ex::NegInf | Ex::NaN => (mu_infinity(), false),
        // A vocabulary leaf is one symbol; a NUMERIC-STRING leaf is a literal the engine holds
        // as written and pays the description length of the value its print denotes (see
        // `mu_numeric_str` -- the boundary must not be an ordering cliff), capped
        // (`mu_numeric_leaf`). HOT PATH: no allocation (`with_str`).
        Ex::Leaf(t) => view.with_str(*t, |s| {
            if numeric_leaf(s) {
                (mu_numeric_leaf(s), true)
            } else {
                (mu_leaf(), false)
            }
        }),
        Ex::Const => (mu_free(), false),
        Ex::Add(v) => {
            // The sum's literal: its pieces, priced as the one literal they denote.
            let mut total = mu_add();
            let (mut lit, mut lit_nodes, mut pieces) = (0u64, 0u64, 0usize);
            for x in v {
                let (p, piece) = priced(x, view);
                if piece {
                    lit = lit.saturating_add(p);
                    lit_nodes += piece_nodes(x);
                    pieces += 1;
                } else {
                    total = total.saturating_add(p);
                }
            }
            let value = (pieces >= 2)
                .then(|| bag_literal_value(v, true, view))
                .flatten();
            let total = total.saturating_add(piece_price(lit, value.as_ref(), lit_nodes));
            if pieces == v.len() {
                (piece_price(total, value.as_ref(), lit_nodes + 1), true)
            } else {
                (total, false)
            }
        }
        Ex::Mul(v) => {
            let head = mu_mul();
            let mut total = head;
            let mut members = 0u64;
            // The product's literal: its coefficient slots and its other pieces, priced as the
            // one literal they denote.
            let (mut lit, mut lit_nodes, mut pieces) = (0u64, 0u64, 0usize);
            for f in v {
                match f {
                    // The rational coefficient slot: +1 is no sign (free), -1 is a bare sign
                    // (`mu_sign`, one bit, as a literal's sign); any other magnitude pays its
                    // description length, its sign bit included -- THE stage-2 decision (an
                    // `E` factor at 8 now undercuts a `2.718...` coefficient at ~109 instead
                    // of riding free below it).
                    Ex::Num(r) => {
                        if *r == Rat::NEG_ONE {
                            lit = lit.saturating_add(mu_sign());
                        } else if !r.is_one() {
                            lit = lit.saturating_add(mu_rat(r));
                        }
                        lit_nodes += 1;
                        pieces += 1;
                    }
                    _ => {
                        members += 1;
                        let (p, piece) = priced(f, view);
                        if piece {
                            lit = lit.saturating_add(p);
                            lit_nodes += piece_nodes(f);
                            pieces += 1;
                        } else {
                            total = total.saturating_add(p);
                        }
                    }
                }
            }
            let value = (pieces >= 2)
                .then(|| bag_literal_value(v, false, view))
                .flatten();
            let total = total.saturating_add(piece_price(lit, value.as_ref(), lit_nodes));
            let structural = if members == 0 {
                // Degenerate all-coefficient bag: the literal (or bare Const) itself.
                (total - head).max(2)
            } else if is_negation(v) {
                // A negation standing alone is a sign, not a product: its operand and one bit.
                total - head
            } else {
                total
            };
            if pieces == v.len() {
                (piece_price(structural, value.as_ref(), lit_nodes + 1), true)
            } else {
                (structural, false)
            }
        }
        Ex::Pow(b, ex) => {
            let head = mu_pow();
            if let Ex::Num(r) = &**ex {
                // Rational exponent as a coefficient slot: `x^-1` is division (pure
                // structure, the sign is free), `x^2` pays cost(2), `x^-2` the same.
                let mag = if r.is_one() || *r == Rat::NEG_ONE {
                    0
                } else {
                    mu_rat(r)
                };
                let (base, piece) = priced(b, view);
                let structural = head.saturating_add(base).saturating_add(mag);
                if piece && r.is_integer() {
                    // An integer power of a piece is a piece: the literal it denotes.
                    let value = {
                        let _exact = super::rat::number_domain(false);
                        piece_value(e, view)
                    };
                    return (
                        piece_price(structural, value.as_ref(), piece_nodes(e)),
                        true,
                    );
                }
                return (structural, false);
            }
            (
                head.saturating_add(complexity(b, view))
                    .saturating_add(complexity(ex, view)),
                false,
            )
        }
        // The head is priced by NAME (`mu_fun_bits`): the elementary/transcendental gap
        // is the table's load-bearing entry. Read without allocating, like the leaf above.
        Ex::Fun(t, args) => {
            let head = view.with_str(*t, |s| sym_bits(mu_fun_bits(s)));
            (
                args.iter()
                    .fold(head, |acc, x| acc.saturating_add(complexity(x, view))),
                false,
            )
        }
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
    /// The REGION certificate: the zero-set certificate, consulted only where a sum's zero sign is
    /// decided (`keeps_zero_sign`). A parse holds no other certificate, yet where it inverts a sum
    /// whose zero set may hold a region it distributes the inverse (`inverts_a_signed_zero`), a
    /// decision its canon cannot undo, so in `f64` it reads that region as the canon does while
    /// every licence of the parse stays certificate-free. `None`: reading A's test
    /// (`zero_set_null`).
    pub cert_region: Option<&'a dyn Fn(&Ex) -> bool>,
    /// WHICH SOUNDNESS this construction preserves. Was a `lossy: bool`, which could only
    /// ask "may I skip certification?" -- and that is the wrong question for a guard whose
    /// reason is f64 saturation rather than recall. `real` needs those guards OFF (the
    /// identities they protect are mathematically total) while keeping every recall
    /// certificate ON, and no boolean can say that. Both old readings are derived below,
    /// so the mode stays the single representation.
    pub mode: RuleMode,
    /// Whether the walk folds a ground node to the float the deployed evaluator computes for
    /// it (`Engine::ground_fold`), in `f64`. DERIVED from the mode by `Cx::folds_for`; the
    /// miner sets it explicitly (it canonicalises unfolded). No constructor folds a
    /// transcendental: folding is bottom-up there, so `exp(19.1...)` would become a 9-digit
    /// literal before the root over it is read, and `tanh 30` would become `1` before `atanh`
    /// sees it. The fold is a step of the pass, tried after the rules at each node and kept
    /// only where it descends; `permissive` always folds so (its lossy reading).
    pub fold_f64: bool,
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
    /// THE DERIVATION of `fold_f64` from a mode: `f64` folds a ground node to its float in the
    /// pass. `real` cannot (the result is not representable -- `tan(1)` is irrational and no
    /// literal spells it), and `permissive` folds in its pass by its lossy reading.
    #[inline]
    pub fn folds_for(mode: RuleMode) -> bool {
        matches!(mode, RuleMode::Default)
    }

    /// The RECALL reading of the mode: may a constructor gate pass uncertified? Only
    /// `permissive`. Derived from `RuleMode::wildcard_all` rather than restated, so the
    /// engine has ONE answer to "is this the permissive mode".
    #[inline]
    pub fn lossy(&self) -> bool {
        self.mode.wildcard_all()
    }

    /// The F64 reading: is this construction answerable only in what the deployed f64
    /// evaluator computes? True for `f64` alone. `real` reasons over the reals, where
    /// saturation does not exist, and `permissive` skips the gate anyway -- so a guard that
    /// exists BECAUSE of f64 saturation must ask this and not `lossy()`.
    #[inline]
    pub fn f64_semantics(&self) -> bool {
        matches!(self.mode, RuleMode::Default)
    }

    /// The number domain this construction runs in (`rat::number_domain`): exact in `real`
    /// mode, f64 in every other mode -- `permissive` is evaluated in float64 too -- and exact
    /// whenever the enclosing run is: a certificate-free `Cx::bare` inside a real-mode run
    /// builds and prints that run's numbers (it is mode Default only for its licences).
    #[inline]
    pub fn f64_numbers(&self) -> bool {
        !matches!(self.mode, RuleMode::Real) && super::rat::f64_numbers()
    }

    /// The SIGNED-ZERO reading: does this construction answer in float64's signed zeros (IEEE
    /// 754)? `f64` and `permissive` are evaluated in float64, where an exact zero carries the sign
    /// of what formed it (`x*0` is -0 wherever x < 0) and its inverse is the infinity of that sign;
    /// `real` reasons over the reals, whose one zero inverts to `+inf` whatever formed it.
    #[inline]
    pub fn ieee_zeros(&self) -> bool {
        !matches!(self.mode, RuleMode::Real)
    }

    /// In f64 mode, a numeral leaf the evaluator reads as +-inf or (underflowing) as 0:
    /// literals beyond float64's range, which stay leaves there (design 2c, review M4).
    fn f64_reads_degenerate(&self, t: Tok, zero_too: bool) -> bool {
        self.f64_numbers()
            && self.view.with_str(t, |s| {
                crate::utils::looks_numeric(s)
                    && crate::numeric::leaf_value(s)
                        .is_some_and(|v| v.is_infinite() || (zero_too && v == 0.0))
            })
    }

    /// A certificate-free context (conversions, pattern handling, tests).
    pub fn bare(view: &'a TokenView<'a>) -> Self {
        Cx {
            view,
            cert_fin: None,
            cert_finnz: None,
            cert_nzae: None,
            cert_nce: None,
            cert_region: None,
            mode: RuleMode::Default,
            fold_f64: false,
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
        if self.lossy() || self.certainly_nonneg(base) {
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
    pub(crate) fn certainly_finite(&self, e: &Ex) -> bool {
        match e {
            Ex::Num(_) | Ex::Pi | Ex::E | Ex::Const => true,
            // A numeral leaf beyond float64's range denotes a finite real (H-052), but in f64
            // mode the deployed evaluator reads it as inf (design 2c, review M4): there it
            // is not certainly finite (`0 * 1e400` is NaN to the evaluator).
            Ex::Leaf(t) => self.view.sigil(*t) == 0 && !self.f64_reads_degenerate(*t, false),
            Ex::Add(v) | Ex::Mul(v) => v.iter().all(|x| self.certainly_finite(x)),
            Ex::Pow(b, ex) => {
                self.certainly_finite(b)
                    && matches!(&**ex, Ex::Num(r) if r.is_integer() && !r.is_negative())
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
    /// list of an addend). Also the value-preservation licence of the D39 exploration
    /// moves (`ac::search::distribute_product`), which is why it is crate-visible.
    pub(crate) fn fin_licensed(&self, e: &Ex) -> bool {
        self.lossy() || self.fin_certified(e)
    }

    /// The certificate body of [`fin_licensed`] WITHOUT the lossy blanket: the answer the
    /// SOUND licence gives (the `nz_ae_certified` pattern). Lossy-mode callers use it where
    /// the blanket's premise cannot hold (`ac::search::distribute_product`, a piece that
    /// spells an infinity).
    pub(crate) fn fin_certified(&self, e: &Ex) -> bool {
        if self.certainly_finite(e) {
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
        if self.lossy() {
            return true;
        }
        match e {
            Ex::Pi | Ex::E => true,
            Ex::Num(r) => !r.is_zero(),
            // f64 mode: `1e400/1e400` and `1e-400/1e-400` are NaN to the evaluator (M4)
            Ex::Leaf(t) => self.view.sigil(*t) == 0 && !self.f64_reads_degenerate(*t, true),
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
        if self.lossy() {
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
                matches!(&**ex, Ex::Num(r) if r.is_even_integer()) || self.certainly_nonneg(b)
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
                                // EVEN index: defined only on [0, inf) and its
                                // principal value is non-negative there -- a range
                                // fact, exactly like `acos`. Off-domain it is NaN,
                                // which no licence reads as a sign. (A negative even
                                // index is the reciprocal of one, still non-negative;
                                // index 0 is excluded, it is not a root.)
                                Ex::Num(r) if r.is_even_integer() && !r.is_zero() => true,
                                // ODD index: an increasing odd bijection on the whole
                                // line, so it maps [0, inf) into [0, inf) -- the same
                                // clause as `sinh`/`asinh` below.
                                Ex::Num(r) if r.is_integer() => self.certainly_nonneg(&v[0]),
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
                    if r.is_even_integer() || (r.is_integer() && r.signum() > 0));
                if !ok {
                    return None;
                }
                self.sign_blind_rep(b).map(|s| pow(s, (**ex).clone(), self))
            }
            // A product's sign is erased here too, so each of its sign-trade sites takes its
            // cheaper spelling whatever sign that leaves, and on a tie the one with fewer signs
            // (`tie_cmp`): `|333 atan(u - 0.66)^5|` is `|333 atan(0.66 - u)^5|`, a sign bit
            // cheaper, where the product alone keeps its coefficient positive on the tie (a sign
            // bit for nothing), and `cos(x1*rootn(-x2*x3, 3))` is `cos(x1*rootn(x2*x3, 3))`.
            Ex::Mul(v) => {
                let mut out = v.clone();
                let mut moved = false;
                for f in out.iter_mut() {
                    if let Some(nf) = sign_trade_flip(f, self) {
                        let (mn, mf) = (complexity(&nf, self.view), complexity(f, self.view));
                        if mn < mf || (mn == mf && tie_cmp(&nf, f, self.view) == Ordering::Less) {
                            *f = nf;
                            moved = true;
                        }
                    }
                }
                if !moved {
                    return None;
                }
                let m = mul(out, self);
                Some(self.carried_negative(&m).unwrap_or(m))
            }
            _ => None,
        }
    }

    /// Does `e` certainly take values in [-1, 1] wherever defined? Range facts only:
    /// sin/cos/tanh land there for EVERY argument. (Literal arguments never reach this
    /// predicate -- ground trig compounds fold numerically before licences run.)
    /// A proven CEILING on `|e|`, in exact integers, or `None` when none is known.
    ///
    /// CONSERVATIVE BY CONSTRUCTION: a bound is returned only where it is provable and
    /// anything unrecognised yields `None`. Read on the INTERNAL form, never on the token
    /// spelling, so the canon unifying two spellings cannot change the answer (the F49
    /// hazard) -- the discipline `certainly_nonneg` already follows. Integer arithmetic
    /// throughout: this guard must not acquire an f64 reading of a `Rat`.
    ///
    /// EXISTS because the f64 evaluator rounds where the extended real line does not: an
    /// inverse pair that is a total bijection on the extended line drifts past the realised
    /// bar under the deployed arithmetic outside a band (`tanh` compresses its argument
    /// approaching +-1, `asinh`'s rounding returns as relative error through `sinh`), which
    /// is precisely the gap the contract judge cannot see (it models the extended reals,
    /// not f64). An overflow cuts no band (`inverse_pair_collapse`).
    fn magnitude_ceiling(&self, e: &Ex) -> Option<i128> {
        match e {
            Ex::Num(r) => r.ceil_abs(),
            Ex::Pi => Some(4),
            Ex::E => Some(3),
            Ex::Fun(f, a) => {
                let s = self.view.resolve_owned(*f);
                match s.as_str() {
                    // bounded RANGES, whatever the argument
                    "sin" | "cos" | "tanh" => Some(1),
                    "atan" | "asin" => Some(2),
                    "acos" => Some(4),
                    // magnitude-preserving
                    "abs" | "neg" => a.first().and_then(|t| self.magnitude_ceiling(t)),
                    _ => None,
                }
            }
            // |sum| <= sum of ceilings, |product| <= product of ceilings -- each needs
            // EVERY part bounded, and overflow abstains rather than wrapping.
            Ex::Add(v) => v.iter().try_fold(0i128, |acc, t| {
                self.magnitude_ceiling(t).and_then(|m| acc.checked_add(m))
            }),
            Ex::Mul(v) => v.iter().try_fold(1i128, |acc, t| {
                self.magnitude_ceiling(t).and_then(|m| acc.checked_mul(m))
            }),
            _ => None,
        }
    }

    /// Is `|e| <= bound` PROVEN? `false` whenever unknown -- refusal is the safe direction.
    fn certainly_in_band(&self, e: &Ex, bound: i128) -> bool {
        self.magnitude_ceiling(e).is_some_and(|m| m <= bound)
    }

    /// A PROVEN lower bound on |e| -- the mirror of `magnitude_ceiling`, for the
    /// licences that need the argument bounded AWAY from a flat point (task 46: the
    /// low-end collapse holes). Sound arms only: a literal's own floor, the named
    /// constants, `cosh >= 1`, magnitude-preserving heads, and a product of floors.
    /// A SUM abstains -- floors do not add across cancellation -- and everything
    /// unknown abstains, so refusal stays the safe direction.
    fn magnitude_floor(&self, e: &Ex) -> Option<i128> {
        match e {
            Ex::Num(r) => Some(r.floor_abs_saturating()),
            Ex::Pi => Some(3),
            Ex::E => Some(2),
            Ex::Fun(f, a) => {
                let s = self.view.resolve_owned(*f);
                match s.as_str() {
                    "cosh" => Some(1),
                    "abs" | "neg" => a.first().and_then(|t| self.magnitude_floor(t)),
                    _ => None,
                }
            }
            Ex::Mul(v) => v.iter().try_fold(1i128, |acc, t| {
                self.magnitude_floor(t).and_then(|m| acc.checked_mul(m))
            }),
            _ => None,
        }
    }

    /// Is `|e| >= bound` PROVEN? `false` whenever unknown.
    fn certainly_at_least(&self, e: &Ex, bound: i128) -> bool {
        self.magnitude_floor(e).is_some_and(|m| m >= bound)
    }

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
            // TOTAL on the extended real line -- but NOT under the deployed f64 algebra
            // everywhere, so each carries the band where its rounding stays within the
            // REALISED criterion (`_ulp_gap <= REALISED_ULP = 8`, verify._contract).
            // MODE-AWARE, and the guard asks the F64 question, not the recall one (owner
            // ruling 2026-08-19). These identities are TOTAL over the reals -- atanh(tanh x)
            // = x for every real x. OVERFLOW and UNDERFLOW cut no band: where the inner
            // function overflows or underflows (exp past 709.78 or below -708.4, sinh and
            // cosh past 710.48), a rewrite may give the evaluator's value or the true one,
            // and the collapse gives the true one. ROUNDING does cut bands. COMPRESSION near
            // a flat point of the inner function eats the LOW end -- exp rounds a small
            // argument into the neighbourhood of 1 and cosh is quadratically flat at 0, so
            // `log o exp` breaches the 8-ULP bar for all |t| < ~0.0625 (worst 6.7e7 ULP at
            // 1e-8) and `acosh o cosh` for all |t| < ~0.25 (worst ~4.5e18),
            // while at |t| >= 1 the measured worst wherever nothing overflows or underflows
            // is 0 and 1 ULP -- hence their `certainly_at_least(t, 1)` floor (task 46).
            // `asinh o sinh` is clean wherever sinh does not overflow, worst 2 ULP measured,
            // and carries no band. `sinh o asinh` never overflows short of the top of the
            // range, but asinh's ~|ln 2t| of absolute rounding returns as RELATIVE error
            // through sinh: worst 4 ULP for |t| <= 710, first breach past 8 ULP measured at
            // |t| ~ 4.4e6, so it keeps the 710 ceiling. `atanh o tanh` fails by COMPRESSION
            // at the HIGH end -- tanh loses ~2|t|/ln2 bits of the argument approaching +-1,
            // so the roundtrip drifts past 8 ULP at |t| ~ 2.7 (first breach measured 2.677
            // audit host / 2.808 this host, libm-dependent; by |t|=18 the drift is
            // ~0.03-0.06 ABSOLUTE), eleven-plus band-widths before saturation at 19.06 --
            // which is what the old attainment-derived band of 18 measured, and why it was
            // wrong (S3); saturation is not overflow, so the band stays. Each integer bound
            // below is the last magnitude ceiling on the SAFE side of every measured breach,
            // with >= 2x margin on both hosts. So:
            //   f64    -- band REQUIRED: the deployed evaluator is the authority.
            //   real   -- band SKIPPED: mathematics has no saturation to protect against,
            //             and requiring it here would refuse a rewrite that is simply true.
            //   permissive -- band SKIPPED: the training corpus is generated FROM the simplified form,
            //             so no external function exists to disagree with the collapse.
            // A `lossy: bool` could not express the middle row, which is why `Cx` now
            // carries the mode itself.
            ("atanh", "tanh") if !self.f64_semantics() || self.certainly_in_band(t, 2) => {
                Some(t.clone())
            }
            ("log", "exp") if !self.f64_semantics() || self.certainly_at_least(t, 1) => {
                Some(t.clone())
            }
            ("asinh", "sinh") => Some(t.clone()),
            ("sinh", "asinh") if !self.f64_semantics() || self.certainly_in_band(t, 710) => {
                Some(t.clone())
            }
            ("acosh", "cosh") if !self.f64_semantics() || self.certainly_at_least(t, 1) => {
                Some(fun(self.view.intern("abs"), vec![t.clone()], self))
            }
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
        // A positive odd power of such a function carries its sign out (`f(-u)^3 = -f(u)^3`).
        if let Ex::Pow(b, x) = e {
            if matches!(&**x, Ex::Num(n) if n.is_odd_integer() && n.signum() > 0)
                && odd_of_literal_multiple(b, view).is_some_and(|r| r.is_negative())
            {
                return unsign(b, view);
            }
            return false;
        }
        // An odd head of a negative literal multiple: the multiple's sign (`positive_head`).
        match odd_of_literal_multiple(e, view) {
            Some(r) if r.is_negative() => match positive_head(e, view) {
                Some(p) => {
                    *e = p;
                    true
                }
                None => false,
            },
            _ => false,
        }
    }
    let (c, key) = match e {
        Ex::Mul(v) => match v.first() {
            Some(Ex::Num(r)) if !r.is_zero() => {
                let r = r.clone();
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
/// A term with every class it holds as a factor -- a sum, an odd function of one or an odd root
/// of one, possibly under an integer power -- in the class's one orientation (the lower of `S`
/// and `-S` in the canonical order), the signs the flips move out gathered into the term's
/// coefficient. `None` where nothing flips. Raw, for matching: a repeated wildcard then binds a
/// class whichever orientation each of its terms carries (`?0/?2 + ?1/?2` reaches
/// `a/(x - y) + b/(y - x)`), and the replacement is re-canonicalized.
/// Are `x` and `y` (which differ) negations of each other through the orientation of the classes
/// they hold? Then a repeated wildcard bound to one meets the other: a near miss, which the
/// up-to-sign retry (`orient_term_classes`) may resolve. A sign-blind multiset of their leaves is
/// compared first, so a pair that differs in content is refused without building a key.
pub(crate) fn orientation_near_miss(x: &Ex, y: &Ex, view: &TokenView) -> bool {
    if leaf_sig(x) != leaf_sig(y) {
        return false;
    }
    let cx = Cx::bare(view);
    let (kx, mx) = up_to_sign_key(x, &cx);
    let (ky, my) = up_to_sign_key(y, &cx);
    (mx || my) && kx == ky
}

/// An order-free hash of the leaves of `e`, literals by magnitude and unit literals left out, so
/// that a sum and its negation, or a product and its negation, hash alike.
fn leaf_sig(e: &Ex) -> u64 {
    use std::hash::{Hash, Hasher};
    fn one<T: Hash>(t: &T) -> u64 {
        let mut h = rustc_hash::FxHasher::default();
        t.hash(&mut h);
        h.finish()
    }
    match e {
        Ex::Leaf(t) => one(t),
        Ex::Num(r) => {
            if r.is_one() || r.checked_neg().is_some_and(|n| n.is_one()) {
                0
            } else if r.is_negative() {
                r.checked_neg().map_or(1, |n| one(&n))
            } else {
                one(r)
            }
        }
        Ex::Add(v) | Ex::Mul(v) => v.iter().map(leaf_sig).fold(0, u64::wrapping_add),
        Ex::Fun(op, args) => args.iter().map(leaf_sig).fold(one(op), u64::wrapping_add),
        Ex::Pow(b, q) => leaf_sig(b).wrapping_add(leaf_sig(q)).wrapping_mul(31),
        other => one(&std::mem::discriminant(other)),
    }
}

/// `x` read up to sign: every class factor (a sum, an odd function or odd root of one, under an
/// integer power or not) in its one orientation, the coefficient's sign dropped; and whether a
/// class moved.
fn up_to_sign_key(x: &Ex, cx: &Cx) -> (Ex, bool) {
    let fs: Vec<&Ex> = match x {
        Ex::Mul(v) => v.iter().collect(),
        _ => vec![x],
    };
    let mut moved = false;
    let mut out: Vec<Ex> = Vec::with_capacity(fs.len());
    for f in fs {
        match f {
            Ex::Num(r) => {
                let a = if r.is_negative() {
                    match r.checked_neg() {
                        Some(a) => a,
                        None => return (x.clone(), false),
                    }
                } else {
                    r.clone()
                };
                if !a.is_one() {
                    out.push(Ex::Num(a));
                }
            }
            Ex::Pow(b, q) if matches!(&**q, Ex::Num(r) if r.is_integer()) => {
                match orient_base(b, cx, &[]) {
                    Some((nb, _)) => {
                        moved = true;
                        out.push(Ex::Pow(Box::new(nb), q.clone()));
                    }
                    None => out.push(f.clone()),
                }
            }
            _ => match orient_base(f, cx, &[]) {
                Some((nb, _)) => {
                    moved = true;
                    out.push(nb);
                }
                None => out.push(f.clone()),
            },
        }
    }
    out.sort_by(|a, b| cmp_ex(a, b, cx.view));
    let key = match out.len() {
        0 => Ex::Num(Rat::int(1)),
        1 => out.pop().expect("one member"),
        _ => Ex::Mul(out),
    };
    (key, moved)
}

pub(crate) fn orient_term_classes(t: &Ex, cx: &Cx) -> Option<Ex> {
    let (factors, coef): (Vec<Ex>, Rat) = match t {
        Ex::Mul(fs) => {
            let mut c = Rat::int(1);
            let mut v = Vec::new();
            for f in fs {
                match f {
                    Ex::Num(r) => c = r.clone(),
                    _ => v.push(f.clone()),
                }
            }
            (v, c)
        }
        Ex::Num(_) => return None,
        _ => (vec![t.clone()], Rat::int(1)),
    };
    // A class is reoriented only as a whole: each class the factors carry is written in its own
    // orientation (`orient_base`), unless the term keeps a value at the class's zero
    // (`zero_limit`) that this orientation leaves off the limit. Then it is written in its other
    // orientation where that restores the limit (an odd count of odd members, as beside a log of
    // the class, whose side fixes the orientation), else as it stands.
    let mut classes: Vec<Ex> = Vec::new();
    for f in &factors {
        carried_classes(f, cx, &mut classes);
    }
    let orient = |turns: &[(Ex, Turn)]| -> Vec<(Ex, bool, bool)> {
        factors
            .iter()
            .map(|f| match orient_factor(f, cx, turns) {
                Some((g, s)) => (g, true, s),
                None => (f.clone(), false, false),
            })
            .collect()
    };
    let mut turns: Vec<(Ex, Turn)> = Vec::new();
    let mut oriented = orient(&turns);
    for _ in 0..2 * classes.len() {
        let now: Vec<Ex> = oriented.iter().map(|(g, _, _)| g.clone()).collect();
        let off = classes.iter().find(|k| {
            !turns.contains(&((*k).clone(), Turn::Keep)) && zero_off(&now, k, cx) == Some(true)
        });
        let Some(k) = off else {
            break;
        };
        match turns.iter_mut().find(|(c, _)| c == k) {
            Some(t) => t.1 = Turn::Keep,
            None => turns.push((k.clone(), Turn::Other)),
        }
        oriented = orient(&turns);
    }
    let mut odd = false;
    let mut changed = false;
    let mut out: Vec<Ex> = Vec::with_capacity(oriented.len());
    for (g, moved, s) in oriented {
        changed |= moved;
        odd ^= s;
        out.push(g);
    }
    if !changed {
        return None;
    }
    let c = if odd { coef.checked_neg()? } else { coef };
    let mut v = Vec::with_capacity(out.len() + 1);
    if !c.is_one() {
        v.push(Ex::Num(c));
    }
    v.extend(out);
    Some(if v.len() == 1 {
        v.pop().expect("one member")
    } else {
        Ex::Mul(v)
    })
}

/// A class written against its own orientation (`orient_term_classes`): in the other one, or as
/// it stands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Turn {
    Other,
    Keep,
}

/// `f` in its class's orientation (`orient_base`), and whether the flip moved a sign out of it;
/// `None` where it already is, or holds no class.
fn orient_factor(f: &Ex, cx: &Cx, turns: &[(Ex, Turn)]) -> Option<(Ex, bool)> {
    let (b, k) = match f {
        Ex::Pow(b, e) => match &**e {
            Ex::Num(r) if r.is_integer() => (&**b, Some(r)),
            _ => return None,
        },
        _ => (f, None),
    };
    // A class under a negative power whose zero set may hold a region keeps its spelling: there
    // its inverse is infinite on the region, a value the answer keeps, and the sign of that
    // infinity follows the spelling (the products' fat-zero-set refusal, `power_bag`).
    if k.is_some_and(|r| r.is_negative()) && !zero_set_null(b, cx) {
        return None;
    }
    let (nb, flipped) = orient_base(b, cx, turns)?;
    let odd = k.is_none_or(|r| r.is_odd_integer());
    let g = match k {
        None => nb,
        Some(r) => Ex::Pow(Box::new(nb), Box::new(Ex::Num(r.clone()))),
    };
    Some((g, flipped && odd))
}

/// A class base (`class_base`: a sum, or an odd head of a negatable argument) in its class's
/// orientation, and whether it flipped (each is odd in its argument, so a flip moves a sign out);
/// `None` where it already has it. A sum's terms are read with every odd head of a negative literal
/// multiple turned positive (`sin(-x1)` as `-sin(x1)`), so that a sum and its negation spell their
/// terms alike; the sum's orientation is the structurally smaller one, or as `turns` gives it. An
/// odd head's argument takes its own orientation where it holds a class (`sin(-(x1 (x1 - 3)))` is
/// `-sin(x1 (x1 - 3))`), else its positive literal (`sin(-x1)` is `-sin(x1)`).
fn orient_base(b: &Ex, cx: &Cx, turns: &[(Ex, Turn)]) -> Option<(Ex, bool)> {
    match b {
        Ex::Add(_) => {
            let Some(ClassBase {
                arg: ClassArg::Sum(ts),
                ..
            }) = class_base(b, cx.view)
            else {
                return None;
            };
            let turn = if turns.is_empty() {
                None
            } else {
                class_of(ts, cx)
                    .and_then(|(k, _, _)| turns.iter().find(|(c, _)| *c == k).map(|(_, t)| *t))
            };
            if turn == Some(Turn::Keep) {
                return None;
            }
            let mut pos: Vec<Ex> = ts.iter().map(|t| odd_signs_out(t, cx)).collect();
            pos.sort_by(|x, y| add_term_cmp(x, y, cx.view));
            let mut neg: Vec<Ex> = pos
                .iter()
                .map(raw_negate_term)
                .collect::<Option<Vec<Ex>>>()?;
            neg.sort_by(|x, y| add_term_cmp(x, y, cx.view));
            let (pb, nb) = (Ex::Add(pos), Ex::Add(neg));
            if (cmp_ex(&nb, &pb, cx.view) == Ordering::Less) != (turn == Some(Turn::Other)) {
                Some((nb, true))
            } else {
                (pb != *b).then_some((pb, false))
            }
        }
        Ex::Fun(fk, args) => {
            let ClassBase { head: Some(_), arg } = class_base(b, cx.view)? else {
                return None;
            };
            let (a, s) = match (orient_base(&args[0], cx, turns), arg) {
                (Some(o), _) => o,
                (None, ClassArg::Multiple(m)) if multiple_literal(m)?.is_negative() => {
                    (positive_multiple(m)?, true)
                }
                _ => return None,
            };
            let mut v = args.clone();
            v[0] = a;
            Some((Ex::Fun(*fk, v), s))
        }
        // An integer power of a class carrier: odd powers move the sign out, even ones do not. A
        // negative power whose base's zero set may hold a region keeps its spelling
        // (`orient_factor`).
        Ex::Pow(pb, q) => {
            let Ex::Num(r) = &**q else { return None };
            if !r.is_integer() || (r.is_negative() && !zero_set_null(pb, cx)) {
                return None;
            }
            let (nb, s) = orient_base(pb, cx, turns)?;
            Some((Ex::Pow(Box::new(nb), q.clone()), s && r.is_odd_integer()))
        }
        // A product holding a class carrier: each carrier in its orientation and a negative
        // coefficient moved out, so that a product's sign is read as one (`sin(-(x1 (x1 - 3)))`
        // is `-sin(x1 (x1 - 3))`).
        Ex::Mul(fs) if fs.iter().any(|f| carries_class(f, cx)) => {
            let mut flipped = false;
            let mut changed = false;
            let mut v = Vec::with_capacity(fs.len());
            for f in fs {
                match f {
                    Ex::Num(r) if r.is_negative() => {
                        let a = r.checked_neg()?;
                        flipped = !flipped;
                        changed = true;
                        if !a.is_one() {
                            v.push(Ex::Num(a));
                        }
                    }
                    _ => match orient_base(f, cx, turns) {
                        Some((g, s)) => {
                            flipped ^= s;
                            changed = true;
                            v.push(g);
                        }
                        None => v.push(f.clone()),
                    },
                }
            }
            if !changed {
                return None;
            }
            v.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
            Some((
                if v.len() == 1 {
                    v.pop().expect("one member")
                } else {
                    Ex::Mul(v)
                },
                flipped,
            ))
        }
        _ => None,
    }
}

/// Does `f` carry a class that `orient_base` reorients: a class base (`class_base`), or an integer
/// power or product of one?
fn carries_class(f: &Ex, cx: &Cx) -> bool {
    match f {
        Ex::Pow(b, q) => matches!(&**q, Ex::Num(r) if r.is_integer()) && carries_class(b, cx),
        Ex::Mul(fs) => fs.iter().any(|g| carries_class(g, cx)),
        f => class_base(f, cx.view).is_some(),
    }
}

/// The sums `f` carries as classes where `orient_base` reaches them (`carries_class`), each in its
/// one orientation (`class_of`, the key a class's zero is read by).
fn carried_classes(f: &Ex, cx: &Cx, out: &mut Vec<Ex>) {
    match f {
        Ex::Pow(b, q) if matches!(&**q, Ex::Num(r) if r.is_integer()) => {
            carried_classes(b, cx, out)
        }
        Ex::Mul(fs) => fs.iter().for_each(|g| carried_classes(g, cx, out)),
        f => match class_base(f, cx.view).map(|c| c.arg) {
            Some(ClassArg::Sum(ts)) => {
                if let Some((k, _, _)) = class_of(ts, cx) {
                    if !out.contains(&k) {
                        out.push(k);
                    }
                }
            }
            Some(ClassArg::Multiple(m)) => carried_classes(m, cx, out),
            None => {}
        },
    }
}

/// `a` with its negative literal coefficient made positive, raw; `None` where it has none.
fn positive_multiple(a: &Ex) -> Option<Ex> {
    match a {
        Ex::Num(r) if r.is_negative() => Some(Ex::Num(r.checked_neg()?)),
        Ex::Mul(fs) => match fs.first() {
            Some(Ex::Num(r)) if r.is_negative() => {
                let c = r.checked_neg()?;
                let mut v: Vec<Ex> = Vec::with_capacity(fs.len());
                if !c.is_one() {
                    v.push(Ex::Num(c));
                }
                v.extend(fs[1..].iter().cloned());
                Some(if v.len() == 1 {
                    v.pop().expect("one member")
                } else {
                    Ex::Mul(v)
                })
            }
            _ => None,
        },
        _ => None,
    }
}

/// A term with each odd head of a negative literal multiple (possibly under an integer power)
/// turned positive (`positive_head`), the signs moved into the term's coefficient. Raw.
fn odd_signs_out(t: &Ex, cx: &Cx) -> Ex {
    let flip_fun = |f: &Ex| positive_head(f, cx.view);
    let (mut c, factors): (Rat, Vec<Ex>) = match t {
        Ex::Mul(fs) => match fs.first() {
            Some(Ex::Num(r)) => (r.clone(), fs[1..].to_vec()),
            _ => (Rat::int(1), fs.clone()),
        },
        Ex::Num(_) => return t.clone(),
        _ => (Rat::int(1), vec![t.clone()]),
    };
    let mut changed = false;
    let mut out: Vec<Ex> = Vec::with_capacity(factors.len());
    for f in factors {
        let (g, sign) = match &f {
            Ex::Pow(b, e) => match (&**e, flip_fun(b)) {
                (Ex::Num(k), Some(nb)) if k.is_integer() => {
                    (Some(Ex::Pow(Box::new(nb), e.clone())), k.is_odd_integer())
                }
                _ => (None, false),
            },
            _ => match flip_fun(&f) {
                Some(nf) => (Some(nf), true),
                None => (None, false),
            },
        };
        match g {
            Some(g) => {
                changed = true;
                if sign {
                    match c.checked_neg() {
                        Some(n) => c = n,
                        None => return t.clone(),
                    }
                }
                out.push(g);
            }
            None => out.push(f),
        }
    }
    if !changed {
        return t.clone();
    }
    let mut v: Vec<Ex> = Vec::with_capacity(out.len() + 1);
    if !c.is_one() {
        v.push(Ex::Num(c));
    }
    v.extend(out);
    if v.len() == 1 {
        v.pop().expect("one member")
    } else {
        Ex::Mul(v)
    }
}

/// A term negated in its coefficient, raw (no spelling choice: `-1 * f` stays a product).
fn raw_negate_term(t: &Ex) -> Option<Ex> {
    Some(match t {
        Ex::Num(r) => Ex::Num(r.checked_neg()?),
        Ex::Mul(fs) => match fs.first() {
            Some(Ex::Num(r)) => {
                let c = r.checked_neg()?;
                if c.is_one() {
                    if fs.len() == 2 {
                        fs[1].clone()
                    } else {
                        Ex::Mul(fs[1..].to_vec())
                    }
                } else {
                    let mut v = fs.clone();
                    v[0] = Ex::Num(c);
                    Ex::Mul(v)
                }
            }
            _ => {
                let mut v = Vec::with_capacity(fs.len() + 1);
                v.push(Ex::int(-1));
                v.extend(fs.iter().cloned());
                Ex::Mul(v)
            }
        },
        _ => Ex::Mul(vec![Ex::int(-1), t.clone()]),
    })
}

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
    // `erf` is odd bit for bit in float64 too, signed zero included: its kernel evaluates
    // |x| and restores the sign.
    const ODD: [&str; 9] = [
        "sin", "sinh", "tan", "tanh", "asin", "asinh", "atan", "atanh", "erf",
    ];
    ODD.iter().any(|s| view.tok_is(f, s))
}

/// The literal of an odd head of a literal multiple (`class_base`'s `{B, -B}` over a multiple):
/// `f(c u)` with `f` odd (`odd_fun`) or `rootn` of an odd index of at least 3, `c` the multiple's
/// literal (`multiple_literal`: 1 for a bare `u`, -1 for a negation). Such a head is a sign-trade
/// site: `f(-c u) = -f(c u)` exactly, so the sign moves between the product's coefficient and the
/// argument, and one owner (`sign_trade_flip`, `sign_place`) prices both spellings, the tie going
/// by the one sign key (`sign_key`). An odd head of a sum is a site of its own kind (its sum
/// flips).
fn odd_of_literal_multiple(e: &Ex, view: &TokenView) -> Option<Rat> {
    match class_base(e, view)? {
        ClassBase {
            head: Some(_),
            arg: ClassArg::Multiple(m),
        } => multiple_literal(m),
        _ => None,
    }
}

/// Does a unit-coefficient join of `key` have to go through `mul()`? Only a plain product
/// (no literal, infinity or Const member: those carry or eat the sign themselves) or a lone
/// factor that holds a sign-trade site. The shape screen keeps `sign_trade_flip` off the keys
/// that cannot hold one.
fn unit_join_trades(key: &Ex, cx: &Cx) -> bool {
    let may_be_site = |f: &Ex| match f {
        Ex::Add(_) => true,
        Ex::Pow(b, _) => matches!(**b, Ex::Add(_) | Ex::Fun(..)),
        Ex::Fun(_, args) => !args.is_empty(),
        _ => false,
    };
    match key {
        Ex::Mul(v) => {
            !v.iter().any(|f| {
                matches!(f, Ex::Num(_) | Ex::PosInf | Ex::NegInf | Ex::Const) || f.contains_const()
            }) && v
                .iter()
                .any(|f| may_be_site(f) && is_sign_trade_site(f, cx))
        }
        // A lone sum is `add()`'s to orient (`primitive_sum`), never a product's.
        Ex::Add(_) => false,
        k => may_be_site(k) && !k.contains_const() && is_sign_trade_site(k, cx),
    }
}

fn term_join(c: Rat, key: Ex, cx: &Cx) -> Ex {
    // The coefficient LANDS on its key here (permissive's ρ, [`land`]); a literal key is a
    // product of literals, which `mul` keeps exact.
    let c = if matches!(key, Ex::Num(_)) {
        c
    } else {
        land(c, cx)
    };
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
        // A UNIT coefficient still decides the orientation of the key's sign-trade sites:
        // a key oriented for its old coefficient (`k * (-a - b) * x` keeps the negated sum: at
        // |k| != 1 it ties with `-k * (a + b) * x`, and the tie goes to the positive
        // coefficient) is not the cheapest spelling once the coefficient is 1 -- `mul()`
        // reads `(-a - b) * x` as `-(a + b) * x`. Returned raw, the term printed
        // `-(a + b)/..` inside the sum and re-read with the sign on its coefficient (srbf
        // prediction 117892, permissive). Such a key takes the full assembly; a key without
        // a site cannot trade and is returned as it is.
        if unit_join_trades(&key, cx) {
            let items = match key {
                Ex::Mul(v) => v,
                k => vec![k],
            };
            let placed = mul(items, cx);
            debug_assert!(
                !matches!(placed, Ex::Add(_)),
                "term_join produced a bare Add term: {placed:?}"
            );
            return placed;
        }
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
            // A negative coefficient beside a sum that takes the sign itself (one holding an
            // infinity, a constant or a negation-absorbing member) also takes the full
            // assembly: `mul()` folds the sign into that sum, and `add()` re-settles such a
            // term the same way when a sum is read, so a raw join here would spell a term the
            // second read respells (`-x8 (log(x16) - inf)` is `x8 (inf - log(x16))`).
            let folds_sign = |f: &Ex| {
                matches!(f, Ex::Add(ts) if ts.iter().any(|m| {
                    m.contains_const()
                        || matches!(m, Ex::PosInf | Ex::NegInf)
                        || term_absorbs_negation(m)
                }))
            };
            if v.iter().any(|f| matches!(f, Ex::Num(_)))
                || (c.is_negative() && v.iter().any(|f| is_sign_trade_site(f, cx) || folds_sign(f)))
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
                if idx.is_even_integer() && idx.cmp_int(2) != Ordering::Less {
                    {
                        if let Some(r) = idx.checked_inv() {
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
            Ex::Num(r) => Some((&**b, r.clone())),
            _ => None,
        },
        Ex::Fun(op, args) if args.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            let Ex::Num(idx) = &args[1] else {
                return None;
            };
            if idx.is_even_integer() && idx.cmp_int(2) != Ordering::Less {
                idx.checked_inv().map(|r| (&args[0], r))
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
///     paying: `pow(exp x0, x1)` and `exp(x0*x1)` both price 24,000. UNDER THE SYMBOL
///     TABLE (2026-08-21) the LITERAL-exponent readings moved and no longer all agree:
///     `pow(exp x0, 3)` ties its composition at 18,000, but `rootn(exp x0, 3)` is 21,000
///     against 19,000 for `exp(x0/3)` -- a 2,000 DESCENT the arm declines, because a
///     `rootn` head (6, elementary in the table) is dearer than the `Mul` plus unit
///     fraction the composed spelling pays, where the flat unit priced them the other way
///     round (26,000 against 27,000). The arm's gate was never mu -- it is the FOLD
///     CONDITION, and that has not moved -- so this is a MISSED descent, not an unsound
///     one, and it stays where it already lives: with the RULES. `pow exp (-1) _0 ->
///     exp neg _0`
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
    let _domain = super::rat::number_domain(cx.f64_numbers());
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
                    (Ex::Num(r), Ex::Add(inner)) => (r.clone(), inner.clone()),
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

    // CANONICAL literal partition (design 2c, `Rat::partition_sum`): the literals summed as
    // far as the sums stay representable, as a fixed point a re-read reproduces.
    lits.sort_unstable_by(|a, b| a.cmp_exact(b));
    let mut acc = Rat::ZERO;
    let mut lits = Rat::partition_sum(lits);
    if lits.len() == 1 {
        acc = lits.pop().unwrap();
    } else {
        acc_overflow.extend(lits.into_iter().map(Ex::Num));
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

    // Like terms are like up to sign: terms that differ only in the orientation of their classes
    // (`1/(x3 - x2)` and `-1/(x2 - x3)`) are one term, read with every class in its one orientation
    // and the sign in the coefficient (`orient_term_classes`), so `x + f - f` cancels whatever
    // spelling each `f` arrived in. A sign-blind hash of the keys' leaves gates it: only keys that
    // collide pay for the orientation. A collected term is rebuilt through the constructors, which
    // decide its spelling.
    let mut rebuilt = vec![false; buckets.len()];
    if buckets.len() >= 2 {
        fn holds_sum(e: &Ex) -> bool {
            match e {
                Ex::Add(_) => true,
                Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(holds_sum),
                Ex::Pow(b, q) => holds_sum(b) || holds_sum(q),
                _ => false,
            }
        }
        let sig: Vec<Option<u64>> = buckets
            .iter()
            .map(|b| {
                (!b.key.contains_const()
                    && matches!(b.key, Ex::Mul(_) | Ex::Pow(..) | Ex::Fun(..))
                    && holds_sum(&b.key))
                .then(|| leaf_sig(&b.key))
            })
            .collect();
        let hit: Vec<usize> = (0..buckets.len())
            .filter(|&i| {
                sig[i].is_some() && sig.iter().enumerate().any(|(j, x)| j != i && *x == sig[i])
            })
            .collect();
        if !hit.is_empty() {
            let read: Vec<(Ex, bool)> = hit
                .iter()
                .map(|&i| match orient_term_classes(&buckets[i].key, cx) {
                    Some(o) => match term_split(o, cx.view) {
                        (c, k) if c.is_one() => (k, false),
                        (c, k) if c == Rat::NEG_ONE => (k, true),
                        _ => (buckets[i].key.clone(), false),
                    },
                    None => (buckets[i].key.clone(), false),
                })
                .collect();
            let mut gone = vec![false; buckets.len()];
            for a in 0..hit.len() {
                if gone[hit[a]] {
                    continue;
                }
                let group: Vec<usize> = (a..hit.len())
                    .filter(|&b| !gone[hit[b]] && read[b].0 == read[a].0)
                    .collect();
                if group.len() < 2 {
                    continue;
                }
                let mut coeffs: Vec<Rat> = Vec::new();
                let mut ok = true;
                for &b in &group {
                    for c in &buckets[hit[b]].coeffs {
                        match if read[b].1 {
                            c.checked_neg()
                        } else {
                            Some(c.clone())
                        } {
                            Some(c) => coeffs.push(c),
                            None => ok = false,
                        }
                    }
                }
                if !ok {
                    continue;
                }
                let target = hit[group[0]];
                buckets[target].key = read[a].0.clone();
                buckets[target].coeffs = coeffs;
                rebuilt[target] = true;
                for &b in &group[1..] {
                    gone[hit[b]] = true;
                }
            }
            if gone.iter().any(|&g| g) {
                let mut keep = gone.iter().map(|g| !g);
                buckets.retain(|_| keep.next().unwrap_or(true));
                let mut keep = gone.iter().map(|g| !g);
                rebuilt.retain(|_| keep.next().unwrap_or(true));
            }
        }
    }

    // Rebuild terms, applying the opposite-sign gate per bucket. A rebuilt term can itself be
    // an `Add`: when merged coefficients sum to exactly 1 on an Add-valued key
    // (`(1/4)A + (3/4)A -> A`), `term_join` returns the bare key -- SPLICE it, or the outer bag
    // would nest a same-kind bag and break the Flat invariant.
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
    for (b, rebuilt) in buckets.into_iter().zip(rebuilt) {
        // DETERMINISTIC per-key fold (design 2c, `Rat::partition_sum`). A bucket that mixes
        // signs on a key that is finite a.e. may cancel across them (`x - x = 0` needs the
        // licence), so its coefficients fold as ONE signed partition; otherwise each sign pool
        // folds on its own (a same-sign sum is total). Either way no two members left in a
        // pool fold, so the bucket re-reads to itself in any printed order. Folding the pools
        // first and cancelling only two folded totals did not: a printed bag
        // `x1 - C*x1 - D*x1` whose negative pool refused re-read `x1 - C*x1` first and
        // cancelled it.
        let Bucket { key, coeffs } = b;
        let mixed =
            coeffs.iter().any(|c| c.is_negative()) && coeffs.iter().any(|c| !c.is_negative());
        let pools: Vec<Vec<Rat>> = if mixed && cx.fin_licensed(&key) {
            vec![coeffs]
        } else {
            let (negs, poss): (Vec<Rat>, Vec<Rat>) =
                coeffs.into_iter().partition(|c| c.is_negative());
            vec![poss, negs]
        };
        for pool in pools {
            for c in Rat::partition_sum(pool) {
                // a fully cancelled bucket leaves nothing: 0 * (finite-a.e. t) -> 0 licensed
                let t = if rebuilt && !c.is_zero() {
                    mul(vec![Ex::Num(c), canon(key.clone(), cx)], cx)
                } else {
                    term_join(c, key.clone(), cx)
                };
                push_term(&mut out, &mut spliced, t);
            }
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
        // The constant term LANDS beside other terms here (permissive's ρ, [`land`]); a sum
        // of literals alone stays exact.
        let acc = if out.is_empty() { acc } else { land(acc, cx) };
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
            Ex::Num(r) => r.clone(),
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
    let g = if unanimous {
        magnitudes[0].clone()
    } else {
        Rat::ONE
    };
    // (b), owner ruling 2026-08-08: the filed orientation is the mu-CHEAPER one ("the
    // mirrors score equal; the bigger expression scores bigger"); the historical
    // first-in-sort-positive lex rule survives only as the exact-tie breaker. For a
    // content-1 sum the flip's wrapper cost is one sign (the negation `-1 x Add` exists
    // only in the flipped spelling); for a non-unit content the wrapper exists in BOTH
    // spellings and the flip pays only the coefficient's sign bit -- and the per-term
    // deltas must then be priced on the DIVIDED terms (the candidates actually stored),
    // not the raw ones, or g's sign bit is counted once per term instead of once.
    let sign_neg = if terms.iter().any(term_absorbs_negation) {
        false
    } else if g.is_one() {
        flipped_orientation_wins(&terms, Wrapper::Flipped, Tie::Keep, cx)
    } else {
        let Some(ng) = g.checked_neg() else {
            return Ex::Add(terms);
        };
        match divide_terms(&terms, &g, cx) {
            Some(divided_pos) => {
                let wsd = mu_rat(&ng) as i128 - mu_rat(&g) as i128;
                flipped_orientation_wins(&divided_pos, Wrapper::Both(wsd), Tie::Keep, cx)
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
/// What decides an EXACT mu tie between the two orientations that the sign key (`sign_key`)
/// leaves tied: at a SIGN-TRADE site one spelling is distinguished -- bare / positive
/// coefficient -- and it wins, so `Keep` (the current spelling is the
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

/// Where the wrapper of a sum's flipped orientation stands (`flipped_orientation_wins`).
#[derive(Clone, Copy)]
enum Wrapper {
    /// In neither spelling: a free-orientation site (an even power's base, an even function's
    /// argument), where both orientations are bare.
    Neither,
    /// In the flipped spelling only: a content-1 sum, whose flip is the negation `Mul[-1, .]`,
    /// one sign (`mu_sign`).
    Flipped,
    /// In both: a non-unit content `g`, written `g x .` and flipped `-g x .`; the flip pays the
    /// price of `-g` over `g`.
    Both(i128),
}

fn flipped_orientation_wins(terms: &[Ex], wrapper: Wrapper, tie: Tie, cx: &Cx) -> bool {
    // (b), owner ruling 2026-08-08: the mu-CHEAPER orientation wins. The wrapper's own price
    // delta, from the caller's side: one sign for a content-1 sum (the negation exists only
    // in the flipped spelling), the coefficient's sign-bit delta for a non-unit content
    // (wrapper in both spellings), nothing at a free-orientation site.
    let mut delta = match wrapper {
        Wrapper::Neither => 0,
        Wrapper::Flipped => mu_sign() as i128,
        Wrapper::Both(d) => d,
    };
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
    // EXACT mu tie -> the one sign key (`sign_key`): fewer signs, then outer ones. The flipped
    // spelling carries the wrapper's sign (`-1 x Add` for a content-1 sum, `-g x Add` beside a
    // current `g x Add`), except at the free-orientation sites (`Lex`: an even power's base),
    // where neither spelling has one.
    let terms_key = |ts: &[Ex], depth: u32| {
        ts.iter()
            .map(|t| sign_key(t, cx.view))
            .fold((0u32, 0u32), |a, (c, s)| (a.0 + c, a.1 + s + c * depth))
    };
    let (current, flip) = match tie {
        Tie::Lex => (terms_key(terms, 1), terms_key(&flipped, 1)),
        _ => {
            let depth = if matches!(wrapper, Wrapper::Both(_)) {
                2
            } else {
                1
            };
            let (c, s) = terms_key(&flipped, 2);
            (terms_key(terms, depth), (c + 1, s + 1))
        }
    };
    if current != flip {
        return flip < current;
    }
    // Equal signs too -> the caller's structural policy (owner ruling A). Only `Lex` --
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
        Ex::Num(r) => r.clone(),
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
    mul_with(items, cx, true)
}

/// Is `e` an odd root (`rootn` of an odd index), or a power of one? Its collection can leave its
/// class (`cbrt(u)^3` is `u`), which the certificate-free parse already decides; a divisor
/// holding one is distributed by its members' form there too (`power_bag`), so its members meet
/// that decision as the same members spelled one by one do.
fn holds_odd_root(e: &Ex, cx: &Cx) -> bool {
    match e {
        Ex::Pow(b, _) => holds_odd_root(b, cx),
        Ex::Fun(op, a) => {
            a.len() == 2
                && cx.view.tok_is(*op, "rootn")
                && matches!(&a[1], Ex::Num(k) if k.is_odd_integer())
        }
        _ => false,
    }
}

/// Is `e` an infinity, or a sum holding one (an infinite or NaN value, never zero)?
fn holds_infinity(e: &Ex) -> bool {
    match e {
        Ex::PosInf | Ex::NegInf => true,
        Ex::Add(ts) => ts
            .iter()
            .any(|t| matches!(t, Ex::PosInf | Ex::NegInf) || is_inf_carrying(t)),
        _ => false,
    }
}

/// The licence of an odd negative power's distribution over a product (`pow`): is `e` nonzero
/// almost everywhere by the certificate (`nz_ae_licensed`)? The parse, which has no certificate,
/// holds a divisor product, and `canon` distributes it where the certificate licenses it; a product
/// the certificate refuses stays one member, which the search may expand (`1/((a x + b)(c x + d))`
/// as one polynomial). Only a divisor holding an odd root is distributed by its form
/// (`power_bag`).
///
/// A negative power `b^-q` is zero only where `b` is infinite, so its zero set is null wherever `b`
/// is finite almost everywhere (`fin_licensed`); where `b` is undefined the power is too, and an
/// inverse of an undefined product is undefined as each member's is. So the inverse of a quotient
/// distributes as the quotient's members would: `1/(x12^(-1/2)/x14)` is `x14 x12^(1/2)`.
///
/// The verdict is a function of `e` and the context alone, so within a simplify call
/// (`RegionScope`) each distinct `e` is decided once per context however often the products
/// holding it are built.
fn zero_set_null(e: &Ex, cx: &Cx) -> bool {
    if cx.lossy() {
        return true;
    }
    let key = flip_cx_key(cx);
    let hit = REGION_MEMO.with(|m| {
        m.borrow()
            .as_ref()
            .and_then(|t| t.get(&key).and_then(|v| v.get(e).copied()))
    });
    if let Some(hit) = hit {
        return hit;
    }
    let r = cx.nz_ae_licensed(e)
        || matches!(e, Ex::Pow(b, q) if matches!(&**q, Ex::Num(r) if r.is_negative())
            && cx.fin_licensed(b));
    REGION_MEMO.with(|m| {
        if let Some(t) = m.borrow_mut().as_mut() {
            t.entry(key).or_default().insert(e.clone(), r);
        }
    });
    r
}

/// `zero_set_null`'s verdicts, per context (`flip_cx_key`) and expression.
type RegionMemo = rustc_hash::FxHashMap<FlipCxKey, rustc_hash::FxHashMap<Ex, bool>>;

thread_local! {
    static REGION_MEMO: std::cell::RefCell<Option<RegionMemo>> = const { std::cell::RefCell::new(None) };
}

/// Holds a fresh region memo (`zero_set_null`) and class-read memo (`zero_read`) open for one
/// simplify call and restores the
/// caller's when it drops. The memo's key holds the context's identities (its token view and
/// certificate closures), which live as long as the call that opens the scope, so no identity is
/// reused within one memo's extent; a nested call keys its own contexts in its own memo.
pub(crate) struct RegionScope(Option<RegionMemo>, Option<ReadMemo>);

impl RegionScope {
    pub(crate) fn open() -> Self {
        RegionScope(
            REGION_MEMO.with(|m| m.borrow_mut().replace(RegionMemo::default())),
            READ_MEMO.with(|m| m.borrow_mut().replace(ReadMemo::default())),
        )
    }
}

impl Drop for RegionScope {
    fn drop(&mut self) {
        let outer = self.0.take();
        REGION_MEMO.with(|m| *m.borrow_mut() = outer);
        let reads = self.1.take();
        READ_MEMO.with(|m| *m.borrow_mut() = reads);
    }
}

/// A SUM KEEPS THE SIGN OF ITS ZERO WHERE ITS ZERO SET MAY HOLD A REGION, in `f64`. Float64 adds
/// to `+0` wherever a sum cancels, however it is written: `x1 - |x1|` and `|x1| - x1` are both
/// `+0` for every `x1 >= 0`, so the negation `-(|x1| - x1)` is `-0` there. A sum and its terms
/// negated one by one (its other orientation) are one value everywhere else, the rounding being
/// symmetric in sign. Where the zero set is null its sign is no value the answer keeps (reading
/// A: poles are read by their limits); where it may hold a region, the sign is a value where the
/// zero is inverted (`region_signed_zero`, `inverts_a_signed_zero`). The region test is reading
/// A's (`zero_set_null`: the certificate where the context has one; a context without it reads
/// the form, `zero_set_null_by_form`). A sum without a region of the variables' space
/// (`Ex::has_measure_space`: a ground sum, or a pattern's) has no zero set to read, and a sum
/// whose sign an absorption owner takes into its terms whatever its zero set -- one holding a
/// `<constant>` (the refit, H-020), an infinity term or a term absorbing a negation (H-014,
/// H-030) -- keeps its owner. `real` reads one unsigned zero, and `permissive` a value up to the
/// sign of a zero.
pub(crate) fn keeps_zero_sign(s: &Ex, cx: &Cx) -> bool {
    let owned = s.contains_const()
        || matches!(s, Ex::Add(ts) if ts.iter().any(|t| {
            matches!(t, Ex::PosInf | Ex::NegInf) || term_absorbs_negation(t)
        }));
    cx.f64_semantics()
        && s.has_measure_space(cx.view)
        && !owned
        && !zero_set_null_by_form(s, cx)
        && !match cx.cert_region {
            Some(f) => f(s),
            None => zero_set_null(s, cx),
        }
}

/// Does `f` vanish on a region with the sign its orientation gives it: a sum that keeps the sign
/// of its zero (`keeps_zero_sign`), or an odd power, an odd function or an odd root of one, each
/// odd at a signed zero as everywhere else?
fn region_signed_zero(f: &Ex, cx: &Cx) -> bool {
    match f {
        Ex::Add(_) => keeps_zero_sign(f, cx),
        Ex::Pow(b, e) => {
            matches!(&**e, Ex::Num(r) if r.is_odd_integer() && !r.is_negative())
                && region_signed_zero(b, cx)
        }
        Ex::Fun(op, a) if a.len() == 1 && odd_fun(cx.view, *op) => region_signed_zero(&a[0], cx),
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            matches!(&a[1], Ex::Num(k) if k.is_odd_integer()) && region_signed_zero(&a[0], cx)
        }
        // The 0.15 heads. A step or lattice head vanishes on whole intervals with a sign of its
        // own (`ceil` is -0 on (-1, 0), `mod` takes the divisor's, `max(x, -0)` and `min(x, 0)`
        // hold a zero on a half-line), wherever it has a region to vanish on. `harmonic` is odd at
        // its zero, `atan2` in its first argument and `besselj` (under an odd order) in its
        // second, so each keeps its argument's.
        Ex::Fun(op, _)
            if ["floor", "ceil", "max", "min", "mod"]
                .iter()
                .any(|h| cx.view.tok_is(*op, h)) =>
        {
            cx.f64_semantics() && f.has_measure_space(cx.view)
        }
        Ex::Fun(op, a) if a.len() == 1 && cx.view.tok_is(*op, "harmonic") => {
            region_signed_zero(&a[0], cx)
        }
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "atan2") => {
            region_signed_zero(&a[0], cx)
        }
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "besselj") => {
            region_signed_zero(&a[1], cx)
        }
        _ => false,
    }
}

/// Is the zero set of `e` null by its form alone, without the certificate (which the parse has
/// not)? A variable, a nonzero literal, a `Const`, an infinity, a product of such factors, a
/// positive power or a root of one, a negative power of a function whose level sets are null
/// (zero only where that function is infinite), or such a function itself (`level_sets_null`).
/// A sufficient test: where it fails the certificate may still hold.
fn zero_set_null_by_form(e: &Ex, cx: &Cx) -> bool {
    match e {
        Ex::Num(r) => !r.is_zero(),
        Ex::Pi | Ex::E | Ex::Const | Ex::PosInf | Ex::NegInf => true,
        Ex::Leaf(_) => form_variable(e, cx),
        Ex::Mul(v) => v.iter().all(|f| zero_set_null_by_form(f, cx)),
        Ex::Pow(b, q) => match &**q {
            Ex::Num(q) if q.signum() > 0 => zero_set_null_by_form(b, cx),
            Ex::Num(q) if q.is_negative() => level_sets_null(b, cx),
            _ => false,
        },
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            zero_set_null_by_form(&a[0], cx)
        }
        _ => level_sets_null(e, cx),
    }
}

/// A variable of the form tests: a leaf that is no wildcard and no numeral (a literal the f64
/// domain keeps as a leaf may read as 0 or as an infinity).
fn form_variable(e: &Ex, cx: &Cx) -> bool {
    matches!(e, Ex::Leaf(t) if cx.view.sigil(*t) == 0
        && !cx.view.with_str(*t, crate::utils::looks_numeric))
}

/// Is every level set of `e` null: a variable, a nonconstant polynomial in variables (integer
/// exponents, rational coefficients; a `Const` term is a generic parameter, so each of its
/// values shifts one level), or a nonconstant analytic function, a nonzero power or a root of
/// such a function, each of whose levels is reached on countably many levels of its argument?
fn level_sets_null(e: &Ex, cx: &Cx) -> bool {
    const ANALYTIC: [&str; 14] = [
        "sin", "cos", "tan", "exp", "log", "sinh", "cosh", "tanh", "asin", "acos", "atan", "asinh",
        "acosh", "atanh",
    ];
    let variable = |f: &Ex| form_variable(f, cx);
    let power_of_variable = |f: &Ex| {
        variable(f)
            || matches!(f, Ex::Pow(b, q) if variable(b)
                && matches!(&**q, Ex::Num(q) if q.is_integer() && !q.is_zero()))
    };
    // A term of a polynomial: a literal or `Const` coefficient times powers of variables; does it
    // hold a variable?
    let monomial = |t: &Ex| -> Option<bool> {
        match t {
            Ex::Num(_) | Ex::Const => Some(false),
            Ex::Mul(v) => v
                .iter()
                .all(|f| matches!(f, Ex::Num(_) | Ex::Const) || power_of_variable(f))
                .then(|| v.iter().any(power_of_variable)),
            f => power_of_variable(f).then_some(true),
        }
    };
    match e {
        Ex::Leaf(_) => variable(e),
        Ex::Add(_) | Ex::Mul(_) => {
            let terms: &[Ex] = match e {
                Ex::Add(ts) => ts,
                _ => std::slice::from_ref(e),
            };
            let read: Option<Vec<bool>> = terms.iter().map(monomial).collect();
            read.is_some_and(|r| r.into_iter().any(|x| x))
        }
        Ex::Pow(b, q) => matches!(&**q, Ex::Num(q) if !q.is_zero()) && level_sets_null(b, cx),
        Ex::Fun(op, a) if a.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            level_sets_null(&a[0], cx)
        }
        Ex::Fun(op, a) => {
            a.len() == 1
                && ANALYTIC.iter().any(|s| cx.view.tok_is(*op, s))
                && level_sets_null(&a[0], cx)
        }
        _ => false,
    }
}

/// The members of a product under the negative integer power `k`, each raised, for the product
/// that decides them, exactly where `pow` distributes the power over the product they build
/// (`power_bag`): each member as it stands, or, where `pow` first folds a negative coefficient of
/// another magnitude than 1 into a sign-trade site (F63; the one zero's pole of `1/(-c S)` is
/// `+inf`, and `-1/(c S)` would be `-inf` there), the members of that bag; members that collect to
/// one factor give that factor raised. A pure sign stays a member, as a negated sum spells it.
/// `Err` holds their product, built without its decision (`mul_undecided`), where `pow` refuses.
pub(crate) fn raised_members(members: Vec<Ex>, k: &Rat, cx: &Cx) -> Result<Vec<Ex>, Ex> {
    let ex = Ex::Num(k.clone());
    if inverts_a_signed_zero(&members, cx) {
        return Ok(members
            .into_iter()
            .map(|f| pow_member(f, ex.clone(), cx))
            .collect());
    }
    let magnitude = |v: &[Ex]| {
        v.iter()
            .any(|f| matches!(f, Ex::Num(r) if r.is_negative() && *r != Rat::NEG_ONE))
    };
    match mul_undecided(members.clone(), cx) {
        Ex::Mul(v) => match power_bag(&v, k, cx) {
            Some(bag) if bag != v && magnitude(&v) => Ok(bag
                .into_iter()
                .map(|f| pow_member(f, ex.clone(), cx))
                .collect()),
            Some(_) => Ok(members
                .into_iter()
                .map(|f| pow_member(f, ex.clone(), cx))
                .collect()),
            None => Err(Ex::Mul(v)),
        },
        f => Ok(vec![pow_member(f, ex, cx)]),
    }
}

/// A ZERO KEEPS ITS SIGN WHERE IT IS INVERTED, in float64's signed zeros (`Cx::ieee_zeros`): does
/// the product of `v` hold an exact zero, whose sign is then the sign of its other members (`x*0`
/// is -0 wherever x < 0, `-(x - x)` is -0)? A negative integer power distributes over such a
/// product exactly in IEEE 754 arithmetic: the exact zero is +0, so `(0*f)^k` is `0^k * f^k`, that
/// is `inf * f^k`, for every `f`, a zero, an infinity and NaN included. Each member keeps its own
/// power and the zero the pole, so the answer keeps the infinity's sign: `1/(x1*(x2 - x2))` is
/// `inf/x1`, `1/-(x1 - x1)` is `-inf` and `x1/(0/x2)` is `inf*x1*x2`. `real` keeps its one zero:
/// there the product is the zero and its inverse `+inf`. Asked where the members of an inverted
/// product are still apart (`raised_members`), and by `pow`'s distribution licence
/// (`power_bag`), where a product that holds a zero is inverted whole.
///
/// So with a member that vanishes on a region with the sign its orientation gives it, in `f64`
/// (`region_signed_zero`): `1/(x2*-(|x1| - x1))` is `-1/x2 * 1/(|x1| - x1)`, `-inf/x2` for every
/// `x1 >= 0`, and each member's inverse keeps the orientation it was written in, a negative power
/// whose zero set may hold a region being no sign-trade site (`sign_trade_flip`).
pub(crate) fn inverts_a_signed_zero(v: &[Ex], cx: &Cx) -> bool {
    cx.ieee_zeros()
        && v.iter()
            .any(|f| matches!(f, Ex::Num(r) if r.is_zero()) || region_signed_zero(f, cx))
}

/// `mul()` without the opposite-sums decision, for a product about to be raised to a negative
/// integer power (`pow`, `canon`, the parse of a divisor). Its classes are decided where its
/// members meet that power: moved positive mass makes the product's zero `-0`, which the
/// inverse turns into the other infinity (`1/((x1 - 3)(3 - x1))` is no `-1/(x1 - 3)^2` at
/// `x1 = 3`), while each member's own inverse keeps its pole (`collect_opposite_sums`).
///
/// Its members over one base merge before a member that is itself a product (a negated sum is
/// `-1` times the sum) is spread into the bag, as their inverses merge in the product that
/// spells them one by one: `1/((1 - x)(1 - x)^(1/3))` is `(1 - x)^(-4/3)` as
/// `(1 - x)^-1 (1 - x)^(-1/3)` is, where spreading first would hand the sign to the coefficient
/// and leave `x - 1` beside `(1 - x)^(1/3)`.
pub(crate) fn mul_undecided(items: Vec<Ex>, cx: &Cx) -> Ex {
    if !items.iter().any(|f| matches!(f, Ex::Mul(_))) {
        return mul_with(items, cx, false);
    }
    let (mut plain, rest): (Vec<Ex>, Vec<Ex>) = items.into_iter().partition(|f| {
        matches!(
            f,
            Ex::Num(_) | Ex::PosInf | Ex::NegInf | Ex::Const | Ex::NaN
        )
    });
    let (merged, _) = collect_like_bases(rest, cx);
    plain.extend(merged);
    mul_with(plain, cx, false)
}

fn mul_with(items: Vec<Ex>, cx: &Cx, decide: bool) -> Ex {
    let _domain = super::rat::number_domain(cx.f64_numbers());
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
        // A product about to be inverted keeps its roots apart: the inverse reaches each of
        // them, and inverted roots are never joined (`mul_undecided`).
        if !decide {
            break;
        }
        let mut by_index: Vec<(Rat, Vec<usize>)> = Vec::new();
        for (i, f) in factors.iter().enumerate() {
            if let Ex::Fun(op, fargs) = f {
                if fargs.len() == 2 && cx.view.tok_is(*op, "rootn") {
                    if let Ex::Num(r) = &fargs[1] {
                        if r.is_odd_integer() && r.cmp_int(3) != Ordering::Less {
                            match by_index.iter_mut().find(|(gn, _)| gn == r) {
                                Some((_, idxs)) => idxs.push(i),
                                None => by_index.push((r.clone(), vec![i])),
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
                stack.push(fun(op_tok.unwrap(), vec![mul(args, cx), Ex::Num(n)], cx));
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
    // CANONICAL coefficient partition (design 2c, `Rat::partition_product`): the magnitudes
    // multiplied as far as the products stay representable, as a fixed point a re-read
    // reproduces; the sign stays on the coefficient slot below (F72).
    let mut coeff = Rat::ONE;
    let mut nums = Rat::partition_product(nums);
    if nums.len() == 1 {
        coeff = nums.pop().unwrap();
    } else {
        coeff_overflow.extend(nums.into_iter().map(Ex::Num));
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
        // The members a refused coefficient partition kept are positive finite magnitudes
        // (the sign is on `coeff`): each is absorbed like the coefficient (design 2c).
        coeff_overflow.clear();
        inf_sign = Some(sign);
        if has_const {
            // inf * c: c = 0 is reachable (0 * inf = nan), so the constant does NOT absorb the
            // infinity; keep both factors.
            factors.push(Ex::Const);
            has_const = false;
        }
    }

    // OPPOSITE SUMS: factors whose sums are each other's negation -- `(9.6 - x) * (x - 9.6)`,
    // and odd functions of such sums, `tanh(x1 - x2) * tanh(x2 - x1)` -- are one base up to a
    // sign, so their classes {B, -B} are decided on the exponent totals of the whole product and
    // collected again: `-(x - 9.6)^2`, and under the usual exponent licence
    // `(x - 9.6) / (9.6 - x)` to `-1` (`collect_opposite_sums`). Without this
    // the pair survived wherever it met in one bag (`sign_place` skips a flip that collides
    // with another base) but collected wherever the parse built a smaller product first, in
    // which the site traded alone: a state and the parse of its own print differed (srbf
    // ground truth 4174, in all three modes). A flip's sign goes to the bag's sign host: the
    // coefficient, a bare infinity, or a bare Const that absorbs it. A zero coefficient (which
    // eats every sign) and a coefficient partition keep their own owners. Only the scan runs on
    // every bag.
    let opposite = (decide
        && !coeff.is_zero()
        && (coeff_overflow.is_empty() || has_const)
        && holds_opposite_sums(&factors, cx))
    .then(|| factors.clone());
    // A zero coefficient keeps only the factors it cannot drop (below), read after the
    // collection and before it.
    let zero_members = (inf_sign.is_none() && coeff.is_zero() && coeff_overflow.is_empty())
        .then(|| factors.clone());

    let (mut out, spliced) = collect_like_bases(factors, cx);
    if has_const {
        // Const absorbs every nonzero rational factor (c * r = c'), the members a refused
        // coefficient partition kept included (design 2c, review M3).
        coeff_overflow.clear();
    }
    out.extend(coeff_overflow);
    // A spliced factor collects against the rest on a new pass, and so do two odd roots of one
    // index the collection left (`cbrt(x)^2 / cbrt(x)` beside `cbrt(y)`), which the join above
    // only takes on one.
    if spliced || (decide && holds_joinable_roots(&out, cx)) {
        if !coeff.is_one() {
            out.push(Ex::Num(coeff));
        }
        if has_const {
            out.push(Ex::Const);
        }
        if let Some(sign) = inf_sign {
            out.push(if sign { Ex::PosInf } else { Ex::NegInf });
        }
        return mul_with(out, cx, decide);
    }
    if let Some(raw) = opposite {
        let host = match inf_sign {
            Some(sign) => SignHost::Inf(sign),
            None if has_const => SignHost::Const,
            None => SignHost::Coeff(coeff),
        };
        return collect_opposite_sums(host, raw, out, cx);
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
        let kept = sign_place(Rat::ZERO, v, cx);
        // The licensed factors drop before the collection too: a collection can merge one into
        // a factor the zero cannot drop (`0 * x^4 * sqrt(x)` collects to `0 * x^(9/2)`, while
        // dropping first keeps `0 * sqrt(x)`), as it can cancel what dropping first keeps
        // (`0 * x / x`). Both read the product's members alone; the cheaper is kept, a tie
        // going to the collected one.
        if let Some(members) = zero_members {
            let w: Vec<Ex> = members
                .into_iter()
                .filter(|f| !cx.fin_licensed(f))
                .collect();
            let (w, spliced) = collect_like_bases(w, cx);
            if !spliced {
                let w: Vec<Ex> = w.into_iter().filter(|f| !cx.fin_licensed(f)).collect();
                if w.is_empty() {
                    return Ex::Num(Rat::ZERO);
                }
                let dropped = sign_place(Rat::ZERO, w, cx);
                if complexity(&dropped, cx.view) < complexity(&kept, cx.view) {
                    return dropped;
                }
            }
        }
        return kept;
    }

    if has_const {
        // Bare Const absorbs the (nonzero) rational coefficient: c * r = c'.
        coeff = Rat::ONE;
        out.push(Ex::Const);
    }
    if let Some(sign) = inf_sign {
        out.push(if sign { Ex::PosInf } else { Ex::NegInf });
    }
    if !decide {
        // No sign moves between the members either: a sign a site took would reach the
        // inverse as a flipped pole (`mul_undecided`).
        if !coeff.is_one() {
            out.push(Ex::Num(coeff));
        }
        return match out.len() {
            0 => Ex::Num(Rat::ONE),
            1 => out.pop().expect("one member"),
            _ => {
                out.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
                Ex::Mul(out)
            }
        };
    }
    finish_product(coeff, out, cx)
}

/// Like-base exponent collection, `mul()`'s step after the coefficient: the factors with one
/// base (and one symbolic exponent) merge their exponents, and each result is rebuilt through
/// [`pow`]; then the powers of e left over distinct bases merge into one (`merge_e_powers`).
/// Returns the rebuilt factors and whether one of them must re-enter the constructor
/// (`spliced`: a product whose members are in the list, or a factor of another kind, and the
/// caller re-runs the constructor on it).
fn collect_like_bases(factors: Vec<Ex>, cx: &Cx) -> (Vec<Ex>, bool) {
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
        // Integers merge totally -- in f64 mode only while the evaluator reads them exactly
        // (parity known, |n| <= 2^53): it reads `x^(2^53+1)` as `x^(2^53)`.
        let exact_integers = a.parity().is_some()
            && b.parity().is_some()
            && a.checked_add(b).is_some_and(|s| s.parity().is_some());
        if cx.lossy() || exact_integers {
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
    let mut spliced = false;
    let push_factor = |out: &mut Vec<Ex>, spliced: &mut bool, f: Ex| match f {
        Ex::Mul(v) => {
            out.extend(v);
            *spliced = true;
        }
        f => out.push(f),
    };
    // A rebuilt factor over another base (`cbrt(x)^3` is `x`) collects against the bucket of
    // that base on one more pass.
    let mut rebased = false;
    for b in buckets {
        let FBucket { base, sym, exps } = b;
        // DETERMINISTIC licence-gated fold (design 2c, `Rat::partition_with`): two exponents
        // merge when the branch-cut licence admits the pair and the sum is a number, until no
        // two merge -- a function of the exponent MULTISET that re-reads to itself in any
        // order (a greedy fold left pieces the next pass merged: `x^1e-306*x^1000*x^1001`).
        // Opposite signs share one pool only on a finite-nonzero base (`x^a * x^-a = 1`
        // needs the licence); a zero sum drops the factor.
        let sym_ok = match &sym {
            None => true,
            Some(y) => cx.sym_merge_licensed(&base, y),
        };
        let merge = |a: &Rat, b: &Rat| -> Option<Rat> {
            if sym_ok && branch_ok(cx, &base, a, b) {
                a.checked_add(b)
            } else {
                None
            }
        };
        let mixed = exps.iter().any(|r| r.is_negative()) && exps.iter().any(|r| !r.is_negative());
        let pools: Vec<Vec<Rat>> = if mixed && sym_ok && (cx.lossy() || cx.finnz_licensed(&base)) {
            vec![exps]
        } else {
            let (negs, poss): (Vec<Rat>, Vec<Rat>) =
                exps.into_iter().partition(|r| r.is_negative());
            vec![poss, negs]
        };
        for pool in pools {
            for r in Rat::partition_with(pool, merge, Rat::is_zero) {
                let f = rebuild_factor(base.clone(), sym.clone(), r, cx);
                if !matches!(f, Ex::Mul(_)) && factor_split(f.clone(), cx).0 != base {
                    rebased = true;
                }
                push_factor(&mut out, &mut spliced, f);
            }
        }
    }
    if rebased {
        let (out, again) = collect_like_bases(out, cx);
        return (out, spliced || again);
    }
    let merged = merge_e_powers(&mut out, cx);
    (out, spliced || merged)
}

/// The exponent of `f` read as a power of Euler's number: `exp(a)` is `e^a`, the leaf `E` is
/// `e^1`, and a rational power of either is `e^(a r)` -- a `Pow` with a literal exponent, or an
/// even root, which `factor_split` reads as a rational power too. `None` for anything else,
/// a symbolic power included: `(e^a)^b` is `e^(a b)` only where `a b` is no `0 * inf`.
fn e_exponent(f: &Ex, cx: &Cx) -> Option<Ex> {
    let (base, r) = as_rational_power(f, cx).unwrap_or((f, Rat::ONE));
    let a = match base {
        Ex::E => return Some(Ex::Num(r)),
        Ex::Fun(op, args) if args.len() == 1 && cx.view.tok_is(*op, "exp") => &args[0],
        _ => return None,
    };
    Some(if r.is_one() {
        a.clone()
    } else {
        mul(vec![Ex::Num(r), a.clone()], cx)
    })
}

/// POWERS OF e COLLECT BY THEIR EXPONENTS: `exp(a) exp(b) = exp(a + b)`. `exp` is a power of e,
/// so this is the like-base collection that merges `x^a x^b` into `x^(a + b)`, over the one base
/// e every power of e shares, whatever `exp` spells its exponent with: `collect_like_bases` merges
/// the powers over one `exp(a)` as over any base, and this merges what remains over distinct
/// ones. The base is positive, finite and nonzero, so every licence of that collection holds for
/// it (the branch cut and the finite-nonzero licence of a cancelling merge); the exponents sum
/// through [`add`], whose own licences decide what cancels: `exp(u) exp(-u)` is `exp(0) = 1` where
/// `u` is finite almost everywhere, as `exp(u)/exp(u)` is.
///
/// TOTAL on the extended real line, so no mode needs a licence of its own: at `a = +inf`,
/// `b = -inf` both sides are NaN (`inf * 0`, and `exp(inf - inf)`), and at one infinite exponent
/// both are that exponent's limit. Under the f64 evaluator the two sides differ where an `exp`
/// overflows or underflows (`exp(800) exp(-800)` is `inf * 0 = NaN`, while `exp(0)` is 1), which
/// is f64's range event (`numeric::RangeKind`): there a rewrite may give the true value, and the
/// merge gives it.
///
/// Rebuilt through [`pow`] over the base e, so the merged exponent takes the spelling `pow` gives
/// any power of e (`compose_e_power`): `exp(x + 1)`, or `E` and `1` at the exponents 1 and 0. A
/// single power of e stays as it stands. Returns whether the merged factor must re-enter the
/// constructor: when it is no power of e (a number, a product, a factor over another base).
fn merge_e_powers(out: &mut Vec<Ex>, cx: &Cx) -> bool {
    let at: Vec<(usize, Ex)> = out
        .iter()
        .enumerate()
        .filter_map(|(i, f)| e_exponent(f, cx).map(|x| (i, x)))
        .collect();
    if at.len() < 2 {
        return false;
    }
    let mut exponents = Vec::with_capacity(at.len());
    for (i, x) in at.into_iter().rev() {
        out.remove(i);
        exponents.push(x);
    }
    let merged = pow(Ex::E, add(exponents, cx), cx);
    let again = e_exponent(&merged, cx).is_none();
    out.push(merged);
    again
}

/// `mul()`'s last step: the sign owners, then the assembly, for a nonzero coefficient and a
/// collected factor list (bare `Const` and infinities included). `collect_opposite_sums` prices
/// its candidates through it, so a priced product is exactly the product built.
fn finish_product(mut coeff: Rat, mut out: Vec<Ex>, cx: &Cx) -> Ex {
    // The coefficient LANDS beside the other factors here (permissive's ρ, [`land`]); a
    // product of literals alone stays exact.
    if !out.is_empty() {
        coeff = land(coeff, cx);
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
    // coefficient's sign = entry sign times flip parity). The owner finds the argmin
    // of the orbit in closed form over whole classes (`sign_place_groups`), builds that
    // spelling and prices it with the real mu -- so the priced spelling IS the built
    // spelling (the row-120 law: any gap between the two re-opens construction-history
    // dependence), and the decision is a function of the ORBIT, not of the entry
    // spelling. Ties go to the sign key (`sign_key`: fewest signs, then the outer ones);
    // residual ties fall to a fixed order that is a function of the orbit. A negate_term overflow
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

/// Where a product's sign lives once its factors are collected (`mul()`'s tail): the rational
/// coefficient, a bare `Const` (whose refit absorbs any sign), or a bare infinity (which carries
/// it).
enum SignHost {
    Coeff(Rat),
    Const,
    Inf(bool),
}

impl SignHost {
    /// The product of `out` with the host's sign toggled when `neg`, assembled as `mul()` does;
    /// `None` where the coefficient's negation overflows.
    fn build(&self, neg: bool, mut out: Vec<Ex>, cx: &Cx) -> Option<Ex> {
        let coeff = match self {
            SignHost::Coeff(c) if neg => c.checked_neg()?,
            SignHost::Coeff(c) => c.clone(),
            SignHost::Const => {
                out.push(Ex::Const);
                Rat::ONE
            }
            SignHost::Inf(pos) => {
                out.push(if *pos != neg { Ex::PosInf } else { Ex::NegInf });
                Rat::ONE
            }
        };
        Some(finish_product(coeff, out, cx))
    }

    fn key(&self) -> HostKey {
        match self {
            SignHost::Coeff(c) => HostKey::Coeff(c.clone()),
            SignHost::Const => HostKey::Const,
            SignHost::Inf(pos) => HostKey::Inf(*pos),
        }
    }
}

/// A sign host as part of a memo key (`OPPOSITE_MEMO`).
#[derive(Clone, PartialEq, Eq, Hash)]
enum HostKey {
    Coeff(Rat),
    Const,
    Inf(bool),
}

/// OPPOSITE SUMS (see `mul()`): a class {B, -B} of a product's factors, where `B` is a sum `S`
/// (its powers, and its even roots read as powers) or an odd function of one (`sin(S)`,
/// `tanh(S)`, an odd `rootn(S, k)`, and their powers: `f(-S) = -f(S)`), and `-B` is the same
/// over the negation of `S`. An even function needs no class: `fun` files `cos(S)` and
/// `cos(-S)` in one orientation already.
///
/// A class is read as a TABLE of exponent totals per orientation, whatever factors delivered
/// them: on each side the positive integer exponents add up to `p` and the negative ones to `n`;
/// a fractional exponent gives its integer part (taken toward zero) to `p` or `n` and stays with
/// the rest (`S^(5/2) = S^2 * S^(1/2)`), and a symbolic one stays whole. What may move between
/// the two sides follows from the totals alone:
///
/// * all of `p`, with a sign to the sign host for every odd amount moved
///   (`(-B)^k = (-1)^k B^k`);
/// * the even part of `n`, freely (`(-B)^-2 = B^-2`, `+inf` on both sides at a zero);
/// * nothing else in a sound context. A side whose `n` is odd keeps one `B^-1`: moved, it would
///   flip the sign of the infinity at the zero (the pole trilemma). A lossy context reads a
///   function up to its values at a zero and moves that inverse too, with its sign. A fractional
///   or symbolic power never flips: `(-S)^(1/2)` is undefined where `S^(1/2)` is defined.
///
/// So `(3 - x1)^-3 * sqrt(x1 - 3)` may become `(3 - x1)^-1 * (x1 - 3)^(-3/2)` whether `^-3`
/// arrived as one factor or as `^-2 * ^-1`. Every placement of the movable totals (how each
/// class splits its positive and its movable negative mass between the sides, and on each side
/// between the integer power and the fractional powers of its sign) is priced as the whole
/// product it builds -- each class collected under the context's own licences, the rest of the
/// bag beside them, the sign placed by `finish_product` -- and the cheapest wins; a price tie
/// goes by `sign_place`'s own order (the sign key, then the structurally smaller product). The classes are decided together while their placements combine to at most
/// `JOINT_LIMIT` products, priced in the order of a lower bound on their prices, else one at a
/// time, from a start the tables decide, until none moves; a class with a large total offers its
/// two extreme placements and the one it holds. A product about to be inverted is built without
/// the decision (`mul_undecided`): its members are decided where they meet the inverse.
/// A class whose members all arrived over one sum is decided too where they hold exponents of
/// both signs (`-(x1 - 3) / (x1 - 3)^2` is `1/(3 - x1)`) or fractional powers that may split its
/// mass. Where a collection changes a table (the certified cancellation `S^2 * S^-3 = S^-1`), the
/// new tables are decided again, until none changes; where a collection leaves its class (an odd
/// root's cube is its radicand), the product is decided again from its own members. The decision
/// is a function of the tables, so the product does not depend on the order of its factors or on
/// how they were bracketed, and the product returned is the one `mul()` builds again from its own
/// factors.
///
/// A context without the `$`-certificate (the entry parse, the bare measure, a rule's sides) runs
/// the same decision on the same tables; its collection refuses the merges of opposite-sign
/// exponents the certificate licenses, and that is the only difference. The parse builds a
/// product once from all of its members, so no smaller product a spelling nests is decided on
/// its own first.
fn collect_opposite_sums(host: SignHost, raw: Vec<Ex>, out: Vec<Ex>, cx: &Cx) -> Ex {
    // Within an enclosing scope (a parse, a re-walk) each distinct product is decided once.
    let key = FLIP_MEMO
        .with(|m| m.borrow().is_some())
        .then(|| (flip_cx_key(cx), host.key(), raw.clone()));
    if let Some(hit) = key
        .as_ref()
        .and_then(|k| OPPOSITE_MEMO.with(|m| m.borrow().get(k).cloned()))
    {
        return hit;
    }
    // The flip memo spans the whole decision: every priced product places its signs over the
    // same sites.
    let _memo = FlipScope::open();
    let decided = decide_opposite_sums(host, raw, out, cx);
    if let Some(k) = key {
        OPPOSITE_MEMO.with(|m| m.borrow_mut().insert(k, decided.clone()));
    }
    decided
}

fn decide_opposite_sums(host: SignHost, raw: Vec<Ex>, out: Vec<Ex>, cx: &Cx) -> Ex {
    let mut classes: Vec<Class> = Vec::new();
    for f in &raw {
        let Some(read) = class_read(f, cx) else {
            continue;
        };
        if classes.iter_mut().any(|c| c.place(&read, cx).is_some()) {
            continue;
        }
        classes.push(Class::new(&read));
    }
    for c in classes.iter_mut().filter(|c| !c.multiple()) {
        c.kept = class_of(&c.sums[0], cx)
            .map(|(k, _, _)| k)
            .filter(|k| zero_limit(&raw, k, cx).is_some());
    }
    // A class whose factors all arrived over one sum is decided too where they hold exponents of
    // both signs, as the cancellation they meet may be cheaper over the other sum
    // (`-(x1 - 3) / (x1 - 3)^2` is `1/(3 - x1)`), or fractional powers that split its mass
    // (`(x1 - 3)^(1/2) (x1 - 3)^(9/2)` is `(x1 - 3)^(5/2) (x1 - 3)^(5/2)`).
    classes.retain_mut(|c| {
        if !c.sums[1].is_empty() {
            return true;
        }
        let t = c.table(
            raw.iter().filter_map(|f| class_read(f, cx).map(|r| (f, r))),
            cx,
        );
        (t.sides[0].p > 0 && t.sides[0].n < 0 || t.splits_mass(0, cx)) && c.complete(cx)
    });
    // Each class written over its first sum: the structurally smaller one, or for a literal
    // multiple the positive one (as an even power of it files, `pow`).
    for c in classes.iter_mut() {
        let first = if c.multiple() {
            c.sums[1]
                .first()
                .and_then(multiple_literal)
                .is_some_and(|r| !r.is_negative())
        } else {
            cmp_vec(&c.sums[1], &c.sums[0], cx.view) == Ordering::Less
        };
        if first {
            c.sums.swap(0, 1);
        }
    }
    classes.sort_by(|a, b| cmp_ex(&a.base(0), &b.base(0), cx.view));
    // The rest of the bag: the factors outside every class, collected as `mul()` collects them.
    // It is read from the factors as they arrived, not from `mul()`'s collection, where a class's
    // members can merge into a factor outside it (an odd root's cube is its radicand) that the
    // class's table still counts.
    let outside: Vec<Ex> = raw
        .iter()
        .filter(|f| {
            !class_read(f, cx).is_some_and(|r| classes.iter().any(|c| c.side_of(&r).is_some()))
        })
        .cloned()
        .collect();
    let (rest, spliced) = collect_like_bases(outside, cx);
    if spliced {
        return host
            .build(false, out, cx)
            .expect("the unflipped bag moves no sign");
    }
    // Each class's table, read from the factors as they arrived.
    let mut tables: Vec<Table> = classes
        .iter()
        .map(|c| {
            c.table(
                raw.iter().filter_map(|f| class_read(f, cx).map(|r| (f, r))),
                cx,
            )
        })
        .collect();
    // A class's factors collected until the collection moves nothing (a re-oriented even power
    // meets the factor it merges with on the next round, and a collection that leaves no two
    // factors over one base has nothing left to move); `None` if a rebuilt factor is a product,
    // which no power of a sum or of a function is.
    let pairs = |v: &[Ex]| {
        (0..v.len()).any(|i| (i + 1..v.len()).any(|j| same_collection_base(&v[i], &v[j], cx)))
    };
    let collect = |mut v: Vec<Ex>| -> Option<Vec<Ex>> {
        v.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
        for _ in 0..=v.len() {
            let (mut next, spliced) = collect_like_bases(v.clone(), cx);
            if spliced {
                return None;
            }
            next.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
            if next == v || !pairs(&next) {
                return Some(next);
            }
            v = next;
        }
        None
    };
    // Cheaper than `(q, m)`: the price, then `sign_place`'s own tie order (a non-negative
    // coefficient, then the structurally smaller product), so a product the sign owner alone
    // would build from either spelling is the one chosen.
    let better = |mu: u64, p: &Ex, q: &Ex, m: u64| {
        mu < m || (mu == m && tie_cmp(p, q, cx.view) == Ordering::Less)
    };
    // A lower bound on a candidate's price, from its factors alone: `complexity` prices a product
    // as its node, its members and its coefficient, and the sign placement can only choose each
    // member's orientation (at best its cheaper one) and the coefficient's sign (at best the
    // cheaper one). Candidates are priced in the order of their bounds, and none whose bound
    // exceeds the best price found can win. Where an absorption owner of `finish_product` may
    // re-orient a sum of the rest (a `Const`-bearing, negation-absorbing or infinite one) and take
    // the sign or the coefficient into it, the bound counts only the classes' factors and the
    // rest's other factors, at their cheaper orientations.
    // A piece of a literal (`is_piece`) is priced with the coefficient and the product's other
    // pieces as the one literal they denote, which can cost less than its pieces: where the
    // product holds one, the bound counts neither the pieces, nor the coefficient, nor the node.
    let pieces = raw.iter().any(|f| is_piece(f, cx.view));
    let factor_lb = |f: &Ex| {
        if is_piece(f, cx.view) {
            return 0;
        }
        let mu = complexity(f, cx.view);
        match sign_trade_flip(f, cx) {
            Some(nf) => mu.min(complexity(&nf, cx.view)),
            None => mu,
        }
    };
    let absorbing = |f: &Ex| {
        matches!(f, Ex::Add(ts) if ts.iter().any(|t| {
            t.contains_const() || term_absorbs_negation(t) || matches!(t, Ex::PosInf | Ex::NegInf)
        }))
    };
    let bounded = !rest.iter().any(absorbing);
    let rest_lb: u64 = rest
        .iter()
        .filter(|f| !absorbing(f))
        .map(factor_lb)
        .fold(0, u64::saturating_add);
    let (host_lb, host_members, unit) = match &host {
        SignHost::Coeff(c) => {
            let unit = c.is_one() || *c == Rat::NEG_ONE;
            (if unit { 0 } else { mu_rat(&c.abs()) }, 0, unit)
        }
        SignHost::Const => (complexity(&Ex::Const, cx.view), 1, true),
        SignHost::Inf(_) => (complexity(&Ex::PosInf, cx.view), 1, true),
    };
    let bound = |members: usize, lb: u64| -> u64 {
        if members == 0 {
            return 0;
        }
        if !bounded || pieces {
            return rest_lb.saturating_add(lb);
        }
        let node = if members >= 2 || !unit { mu_mul() } else { 0 };
        node.saturating_add(rest_lb)
            .saturating_add(host_lb)
            .saturating_add(lb)
    };
    // Every combination of one placement per class while there are at most this many; beyond,
    // the classes are decided one at a time, each against the others as they stand.
    const JOINT_LIMIT: usize = 4096;
    // Does a collected placement hold a factor over no side of its class (an odd root's cube is
    // its radicand)?
    let leaves = |k: usize, list: &Option<Vec<Ex>>| {
        list.as_ref().is_some_and(|l| {
            l.iter()
                .any(|f| !class_read(f, cx).is_some_and(|r| classes[k].side_of(&r).is_some()))
        })
    };
    let mut neg = false;
    let mut chosen: Option<Ex> = None;
    let mut escaped = false;
    for _ in 0..8 {
        // Each class's placements, collected (`None` where a collection splices), with the sign
        // each moves relative to the table and the bound of its factors; index 0 is the table's
        // own placement, with its mass beside its fractional powers. Placements that collect to
        // the same factors with the same sign are one option.
        let mut options: Vec<Vec<PlacedClass>> = Vec::with_capacity(classes.len());
        for (c, t) in classes.iter().zip(&tables) {
            let mut o: Vec<PlacedClass> = Vec::new();
            let placed = t
                .placements()
                .into_iter()
                .flat_map(|(pl, odd)| c.factors(t, &pl, cx).into_iter().map(move |v| (v, odd)));
            for (members, odd) in placed {
                let list = collect(members);
                if o.iter().any(|x| x.odd == odd && x.list == list) {
                    continue;
                }
                let lb = list
                    .as_ref()
                    .map_or(0, |l| l.iter().map(factor_lb).fold(0, u64::saturating_add));
                let members = list.as_ref().map_or(0, Vec::len);
                o.push(PlacedClass {
                    list,
                    odd,
                    lb,
                    members,
                });
            }
            options.push(o);
        }
        let held = vec![0usize; classes.len()];
        if chosen.is_none() && options.iter().all(|o| o.len() <= 1) {
            // Nothing can move: the bag as `mul()` collected it.
            escaped = options
                .iter()
                .enumerate()
                .any(|(k, o)| o.first().is_some_and(|x| leaves(k, &x.list)));
            break;
        }
        let bag = |choice: &[usize]| -> Option<(Vec<Ex>, bool)> {
            let mut v = rest.clone();
            let mut odd = false;
            for (k, &i) in choice.iter().enumerate() {
                let x = &options[k][i];
                v.extend(x.list.as_ref()?.iter().cloned());
                odd ^= x.odd;
            }
            Some((v, odd))
        };
        let lower = |choice: &[usize]| -> u64 {
            let (members, lb) = choice.iter().enumerate().fold(
                (rest.len() + host_members, 0u64),
                |(m, b), (k, &i)| {
                    (
                        m + options[k][i].members,
                        b.saturating_add(options[k][i].lb),
                    )
                },
            );
            bound(members, lb)
        };
        // A placement that leaves the product's value at a class's zero off the limit it keeps
        // there (`zero_limit`, as `sign_place` reads it) is priced above every other.
        let off = |p: &Ex| {
            let members = match p {
                Ex::Mul(v) => v.as_slice(),
                p => std::slice::from_ref(p),
            };
            classes
                .iter()
                .filter_map(|c| c.kept.as_ref())
                .any(|k| zero_off(members, k, cx) == Some(true))
        };
        let price = |choice: &[usize]| -> Option<(Ex, u64)> {
            let (v, odd) = bag(choice)?;
            let p = host.build(neg != odd, v, cx)?;
            let mu = if off(&p) {
                u64::MAX
            } else {
                complexity(&p, cx.view)
            };
            debug_assert!(mu >= lower(choice), "a candidate priced below its bound");
            Some((p, mu))
        };
        let Some(mut best) = price(&held) else {
            break;
        };
        let mut pick = held.clone();
        let limit = JOINT_LIMIT;
        let combos = options
            .iter()
            .try_fold(1usize, |n, o| n.checked_mul(o.len()))
            .filter(|n| *n <= limit);
        if let Some(n) = combos {
            let decode = |code: usize| -> Vec<usize> {
                let mut c = code;
                options
                    .iter()
                    .map(|o| {
                        let i = c % o.len();
                        c /= o.len();
                        i
                    })
                    .collect()
            };
            let mut order: Vec<(u64, usize)> =
                (1..n).map(|code| (lower(&decode(code)), code)).collect();
            order.sort_unstable();
            for (lb, code) in order {
                if lb > best.1 {
                    break;
                }
                let choice = decode(code);
                if let Some((p, mu)) = price(&choice) {
                    if better(mu, &p, &best.0, best.1) {
                        best = (p, mu);
                        pick = choice;
                    }
                }
            }
        } else {
            // One class at a time from a start the tables alone decide (each class's option
            // with the smallest bound, a tie by its factors), so the placement the sweeps settle
            // on does not depend on how the members arrived.
            let start: Vec<usize> = options
                .iter()
                .map(|o| {
                    (0..o.len())
                        .filter(|&i| o[i].list.is_some())
                        .min_by(|&a, &b| {
                            o[a].lb
                                .cmp(&o[b].lb)
                                .then_with(|| match (&o[a].list, &o[b].list) {
                                    (Some(x), Some(y)) => cmp_vec(x, y, cx.view),
                                    _ => Ordering::Equal,
                                })
                        })
                        .unwrap_or(0)
                })
                .collect();
            if let Some(s) = price(&start) {
                best = s;
                pick = start;
            }
            loop {
                let mut moved = false;
                for k in 0..options.len() {
                    for i in 0..options[k].len() {
                        if i == pick[k] {
                            continue;
                        }
                        let mut choice = pick.clone();
                        choice[k] = i;
                        if lower(&choice) > best.1 {
                            continue;
                        }
                        if let Some((p, mu)) = price(&choice) {
                            if better(mu, &p, &best.0, best.1) {
                                best = (p, mu);
                                pick = choice;
                                moved = true;
                            }
                        }
                    }
                }
                if !moved {
                    break;
                }
            }
        }
        chosen = Some(best.0);
        escaped = (0..classes.len()).any(|k| leaves(k, &options[k][pick[k]].list));
        if escaped || pick == held {
            break;
        }
        // The tables of the chosen placement, read from its collected factors: unchanged unless
        // the collection merged across a total (a cancellation), which is decided again.
        let mut changed = false;
        for k in 0..classes.len() {
            let Some(list) = &options[k][pick[k]].list else {
                break;
            };
            let t = classes[k].table(
                list.iter()
                    .filter_map(|f| class_read(f, cx).map(|r| (f, r))),
                cx,
            );
            neg ^= options[k][pick[k]].odd;
            changed |= !t.same_orbit(&tables[k]);
            tables[k] = t;
        }
        if !changed {
            break;
        }
    }
    let decided = chosen.unwrap_or_else(|| {
        host.build(false, out, cx)
            .expect("the unflipped bag moves no sign")
    });
    // A collection that left its class leaves a product whose classes are not the ones decided,
    // and a placement can leave two odd roots of one index, which `mul()` joins before any
    // decision: either product is built again from its own members, as the parse of its print
    // builds it. Each such round moves a multiple of an odd root's index out of that root's class
    // or joins two odd roots, so it ends.
    match decided {
        Ex::Mul(v) if escaped || holds_joinable_roots(&v, cx) => mul(v, cx),
        d => d,
    }
}

/// Does a product hold two odd roots of one index, which `mul()` joins into one?
fn holds_joinable_roots(v: &[Ex], cx: &Cx) -> bool {
    let mut seen: Vec<&Rat> = Vec::new();
    for f in v {
        if let Ex::Fun(op, a) = f {
            if a.len() == 2 && cx.view.tok_is(*op, "rootn") {
                if let Ex::Num(k) = &a[1] {
                    if k.is_odd_integer() && k.cmp_int(3) != Ordering::Less {
                        if seen.contains(&k) {
                            return true;
                        }
                        seen.push(k);
                    }
                }
            }
        }
    }
    false
}

/// How a factor reads as a member of a class {B, -B} (`collect_opposite_sums`): the head of `B`
/// (`None` for the sum itself, else the odd function and an odd root's index), the terms of its
/// argument (`class_base`; a literal multiple is a sum of one term: `sin(2 x1)` and `sin(-2 x1)`,
/// `sin(x1)` and `sin(-x1)`, `sin(2)` and `sin(-2)`), and the factor's exponent over `B` (`None`
/// when it is symbolic).
struct ClassRead<'a> {
    head: Option<(Tok, Option<&'a Ex>)>,
    terms: &'a [Ex],
    exp: Option<Rat>,
}

/// THE CLASS READING, the one definition every owner asks (the product table `class_read`, the
/// orientation of like terms and of shared factors `orient_base`, the sign trade's classes
/// `site_multiple`, a term read up to sign `odd_signs_out`, and `pow`'s even powers). A class is a
/// pair {B, -B}: a sum and its negation, or an odd head -- an odd function, or an odd root of an
/// odd index of at least 3 -- of a negatable argument and of its negation. A negatable argument is
/// a sum, or a literal multiple `c u` (`multiple_literal`: a bare `u` is `1 u`, a negation `-u` is
/// `-1 u`, a literal is itself). A sum is one only where the sign may pass through it: a sum
/// holding a `Const` or an infinity, or a term that absorbs a negation, has its own owners.
struct ClassBase<'a> {
    head: Option<(Tok, Option<&'a Ex>)>,
    arg: ClassArg<'a>,
}

/// The argument of a class base: a sum's terms, or a literal multiple.
enum ClassArg<'a> {
    Sum(&'a [Ex]),
    Multiple(&'a Ex),
}

impl<'a> ClassArg<'a> {
    /// The argument as the terms of a sum (a literal multiple is a sum of one term).
    fn terms(&self) -> &'a [Ex] {
        match self {
            ClassArg::Sum(ts) => ts,
            ClassArg::Multiple(m) => std::slice::from_ref(*m),
        }
    }
}

fn class_base<'a>(b: &'a Ex, view: &TokenView) -> Option<ClassBase<'a>> {
    let sum = |ts: &'a [Ex]| {
        (!ts.iter()
            .any(|t| t.contains_const() || t.contains_infinity() || term_absorbs_negation(t)))
        .then_some(ClassArg::Sum(ts))
    };
    match b {
        Ex::Add(ts) => Some(ClassBase {
            head: None,
            arg: sum(ts)?,
        }),
        Ex::Fun(op, a) => {
            let index = match a.as_slice() {
                [_] if odd_fun(view, *op) => None,
                [_, k]
                    if view.tok_is(*op, "rootn")
                        && matches!(k, Ex::Num(k) if k.is_odd_integer() && k.cmp_int(3) != Ordering::Less) =>
                {
                    Some(k)
                }
                _ => return None,
            };
            let arg = match &a[0] {
                Ex::Add(ts) => sum(ts)?,
                m => {
                    multiple_literal(m)?;
                    ClassArg::Multiple(m)
                }
            };
            Some(ClassBase {
                head: Some((*op, index)),
                arg,
            })
        }
        _ => None,
    }
}

/// THE SIGN TIE KEY (one key after the price, for every placement tie: the sign trade, the product
/// table, the reduction ordering): fewest signs, then outermost, then the structural order
/// (`tie_cmp`). A sign is a negative number -- a product's or a sum term's negative coefficient (a
/// negation is the coefficient -1) or a negative literal argument. A literal exponent is not a
/// sign (a negative one is a reciprocal, written as a division), nor is an infinity's sign, which
/// is its value (`-inf` and `log(0)` are one constant, not a placement). A sign inside a ground
/// argument of a function (no variable, wildcard or `<constant>`) belongs to that constant and is
/// not counted, so a constant keeps its sign on its literal (`sin(-2)` stays, `-sin(2)` is filed
/// as `sin(-2)`). The key is the number of the counted signs and the sum of their depths (the
/// root at depth 0): fewer, then outer.
pub(crate) fn sign_key(e: &Ex, view: &TokenView) -> (u32, u32) {
    fn walk(e: &Ex, depth: u32, view: &TokenView, acc: &mut (u32, u32)) {
        match e {
            Ex::Num(r) if r.is_negative() => {
                acc.0 += 1;
                acc.1 += depth;
            }
            Ex::Add(v) | Ex::Mul(v) => v.iter().for_each(|x| walk(x, depth + 1, view, acc)),
            Ex::Pow(b, q) => {
                walk(b, depth + 1, view, acc);
                if !matches!(**q, Ex::Num(_)) {
                    walk(q, depth + 1, view, acc);
                }
            }
            Ex::Fun(_, args) => {
                for a in args {
                    if a.has_measure_space(view) || a.contains_wildcard(view) {
                        walk(a, depth + 1, view, acc);
                    }
                }
            }
            _ => {}
        }
    }
    let mut acc = (0, 0);
    walk(e, 0, view, &mut acc);
    acc
}

/// The order of two spellings of an equal price (`sign_key`, then `cmp_ex`).
pub(crate) fn tie_cmp(a: &Ex, b: &Ex, view: &TokenView) -> Ordering {
    sign_key(a, view)
        .cmp(&sign_key(b, view))
        .then_with(|| cmp_ex(a, b, view))
}

/// The literal of a literal multiple `c u`: `c`, 1 for a bare `u`; `None` for a sum, a zero, and an
/// argument holding a `Const` or an infinity, whose signs have their own owners.
fn multiple_literal(m: &Ex) -> Option<Rat> {
    if matches!(m, Ex::Add(_)) || m.contains_const() || m.contains_infinity() {
        return None;
    }
    let c = match m {
        Ex::Num(r) => r.clone(),
        Ex::Mul(v) => match v.first() {
            Some(Ex::Num(r)) => r.clone(),
            _ => Rat::ONE,
        },
        _ => Rat::ONE,
    };
    (!c.is_zero()).then_some(c)
}

/// `b` over the positive literal of its class's argument, where `b` is an odd head over a literal
/// multiple with a negative literal (`sin(-2 x1)` is `-sin(2 x1)`); raw.
fn positive_head(b: &Ex, view: &TokenView) -> Option<Ex> {
    let ClassBase {
        head: Some(_),
        arg: ClassArg::Multiple(m),
    } = class_base(b, view)?
    else {
        return None;
    };
    let Ex::Fun(op, a) = b else { return None };
    let mut args = a.clone();
    args[0] = positive_multiple(m)?;
    Some(Ex::Fun(*op, args))
}

fn class_read<'a>(f: &'a Ex, cx: &Cx) -> Option<ClassRead<'a>> {
    let (base, exp) = match f {
        Ex::Pow(b, e) => (
            &**b,
            match &**e {
                Ex::Num(r) => Some(r.clone()),
                _ => None,
            },
        ),
        Ex::Fun(op, a)
            if a.len() == 2
                && cx.view.tok_is(*op, "rootn")
                && matches!(&a[1], Ex::Num(k) if k.is_even_integer() && k.cmp_int(2) != Ordering::Less) =>
        {
            let Ex::Num(k) = &a[1] else { return None };
            (&a[0], Some(k.checked_inv()?))
        }
        other => (other, Some(Rat::ONE)),
    };
    let ClassBase { head, arg } = class_base(base, cx.view)?;
    Some(ClassRead {
        head,
        terms: arg.terms(),
        exp,
    })
}

/// The class of a sign-trade site over a literal multiple (`class_read`'s {B, -B}): `B`, the
/// head over the multiple with its literal made positive, and whether the site spells `-B`.
fn site_multiple(f: &Ex, cx: &Cx) -> Option<(Ex, bool)> {
    let m = match f {
        Ex::Pow(b, q) if matches!(&**q, Ex::Num(_)) => &**b,
        _ => f,
    };
    let ClassBase {
        head: Some(_),
        arg: ClassArg::Multiple(arg),
    } = class_base(m, cx.view)?
    else {
        return None;
    };
    if multiple_literal(arg)?.is_negative() {
        Some((positive_head(m, cx.view)?, true))
    } else {
        Some((m.clone(), false))
    }
}

/// The class a sign-trade site is built over: the sum under the site's head (the sum itself, an
/// odd function or odd root of it, or a power of any of these) in its one orientation, and whether
/// the site spells it in the other one. `None` where the site is over no sum.
fn site_class(f: &Ex, cx: &Cx) -> Option<(Ex, bool)> {
    let m = match f {
        Ex::Pow(b, q) if matches!(&**q, Ex::Num(_)) => &**b,
        _ => f,
    };
    let terms = match m {
        Ex::Add(ts) => ts,
        Ex::Fun(_, a) => match a.first() {
            Some(Ex::Add(ts)) => ts,
            _ => return None,
        },
        _ => return None,
    };
    class_of(terms, cx).map(|(k, negative, _)| (k, negative))
}

/// THE CLASS OF A SUM IS ITS ZERO SET: sums equal up to a nonzero constant factor are one class
/// (`6 - 2 x1` is `x1 - 3`'s, `3 - x1` too), and so are sums equal up to a factor every term
/// shares (`3 x2 - x1 x2` is `x2 (3 - x1)`, a member of that class). Its key is the sum's primitive
/// part -- the shared factors divided out, then every term's coefficient divided by the leading
/// term's (the first by the order of the terms' non-numeric parts, a constant term never leading)
/// -- and the sum is that factor times the key: `true` where the constant is negative (the sum is
/// negative where the key is positive), and whether a shared factor was divided out (its sign is
/// not read). `None` where the sum holds a constant or an infinity, or a division overflows.
fn class_of(ts: &[Ex], cx: &Cx) -> Option<(Ex, bool, bool)> {
    if ts
        .iter()
        .any(|t| t.contains_const() || t.contains_infinity())
    {
        return None;
    }
    let mut parts: Vec<(Rat, Ex)> = ts
        .iter()
        .map(|t| match t {
            Ex::Num(r) => (r.clone(), Ex::Num(Rat::ONE)),
            // Each term read as the orientation reads it (`odd_signs_out`, as `orient_base` and
            // `up_to_sign_key` do): an odd function of a negated argument is the negated term.
            t => term_split(odd_signs_out(t, cx), cx.view),
        })
        .collect();
    // The factors every term shares, divided out.
    let factors = |k: &Ex| -> Vec<Ex> {
        match k {
            Ex::Mul(v) => v.clone(),
            Ex::Num(_) => Vec::new(),
            k => vec![k.clone()],
        }
    };
    let mut shared: Vec<Ex> = factors(&parts[0].1);
    for (_, k) in &parts[1..] {
        let fs = factors(k);
        shared.retain(|f| fs.contains(f));
    }
    let scaled = !shared.is_empty();
    if scaled {
        for (_, k) in parts.iter_mut() {
            let mut fs = factors(k);
            for f in &shared {
                if let Some(i) = fs.iter().position(|g| g == f) {
                    fs.remove(i);
                }
            }
            *k = match fs.len() {
                0 => Ex::Num(Rat::ONE),
                1 => fs.pop().expect("one factor"),
                _ => Ex::Mul(fs),
            };
        }
    }
    parts.sort_by(|a, b| cmp_ex(&a.1, &b.1, cx.view));
    let lead = parts
        .iter()
        .find(|(c, k)| !matches!(k, Ex::Num(_)) && !c.is_zero())?
        .0
        .clone();
    let inv = lead.checked_inv()?;
    let mut key = Vec::with_capacity(parts.len());
    for (c, k) in parts {
        let q = c.checked_mul(&inv)?;
        key.push(match k {
            Ex::Num(_) => Ex::Num(q),
            k if q.is_one() => k,
            k => Ex::Mul(vec![Ex::Num(q), k]),
        });
    }
    Some((Ex::Add(key), lead.is_negative(), scaled))
}

/// A hash of a sum's terms as `class_of` reads them: the factors every term shares left out, and
/// literals and signs ignored (`shape_sig`), so that two sums of one class always hash alike.
fn class_shape(ts: &[Ex]) -> u64 {
    let factors = |t: &Ex| -> Vec<u64> {
        match t {
            Ex::Num(_) => Vec::new(),
            Ex::Mul(v) => v
                .iter()
                .filter(|f| !matches!(f, Ex::Num(_)))
                .map(shape_sig)
                .collect(),
            t => vec![shape_sig(t)],
        }
    };
    let per: Vec<Vec<u64>> = ts.iter().map(factors).collect();
    let mut shared: Vec<u64> = per.first().cloned().unwrap_or_default();
    for fs in &per[1..] {
        shared.retain(|f| fs.contains(f));
    }
    let mut sig = 0u64;
    for mut fs in per {
        for f in &shared {
            if let Some(i) = fs.iter().position(|g| g == f) {
                fs.remove(i);
            }
        }
        let term = fs.iter().fold(0x9e37_79b9u64, |a, f| a.wrapping_add(*f));
        sig = sig.wrapping_add(term.wrapping_mul(0x0100_0000_01b3));
    }
    sig
}

/// A hash of a sum's shape blind to its literals and signs (`class_of` reads alike over a sum and
/// its constant multiples): two sums of one class always hash alike.
fn shape_sig(e: &Ex) -> u64 {
    use std::hash::{Hash, Hasher};
    fn one<T: Hash>(t: &T) -> u64 {
        let mut h = rustc_hash::FxHasher::default();
        t.hash(&mut h);
        h.finish()
    }
    match e {
        Ex::Num(_) => 0,
        Ex::Add(v) | Ex::Mul(v) => v.iter().map(shape_sig).fold(0, u64::wrapping_add),
        Ex::Fun(op, args) => args.iter().map(shape_sig).fold(one(op), u64::wrapping_add),
        Ex::Pow(b, q) => shape_sig(b).wrapping_add(shape_sig(q)).wrapping_mul(31),
        Ex::Leaf(t) => one(t),
        other => one(&std::mem::discriminant(other)),
    }
}

/// How a factor reads in a class `S` (a sum in its one orientation): free of it, even (alike over
/// `S` and `-S`), odd (negated with it), defined beside the zero of `S` on one side only, or
/// neither. Compositional: `S` and `-S` are odd; a product's parity is the product of its factors'
/// (a factor free of `S` is even); an odd function or an odd root keeps its argument's; an even
/// function of anything in `S`, an even integer power of it, or any function of something even in
/// it, is even; a sum has the parity its terms share. A log, an even root or a fractional or
/// symbolic power of something odd in `S` whose sign beside the zero is read is defined only on
/// the side where that is positive (`Side(true)`: where `S` is negative), and so is a product or
/// an integer power holding one; members defined on opposite sides leave no domain beside the
/// zero, and are not read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Parity {
    Free,
    Even,
    Odd,
    Side(bool),
    Unknown,
}

/// A factor read at the zero of a class `S` (READING A): its parity in `S`; whether its sign at
/// the zero differs from its sign beside the zero on the side where the oriented sum is positive
/// (for a factor defined on one side only, on that side); and that sign beside the zero
/// (negative?). The sum spelled in its other orientation is `+0` at its zero but negative beside
/// it; a product counts its members (a literal or a factor free of `S` signs both alike); an odd
/// power, an odd function and an odd root keep the sign of zero; an even power or an even function
/// differs nowhere, and neither does a one-sided head (`log(±0)` is `-inf`, a fractional power of
/// `±0` is `+0` or `+inf`, each with the sign it has beside the zero). `None` where that is not
/// read (a sum that holds `S` among other terms or times a factor its terms share,
/// `3 x2 - x1 x2`, any other head).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ZeroRead {
    parity: Parity,
    differs: Option<bool>,
    below: Option<bool>,
}

/// `zero_read`'s verdicts within a simplify call (`RegionScope`), per token view, class and factor.
type ReadMemo =
    rustc_hash::FxHashMap<usize, rustc_hash::FxHashMap<Ex, rustc_hash::FxHashMap<Ex, ZeroRead>>>;

thread_local! {
    static READ_MEMO: std::cell::RefCell<Option<ReadMemo>> = const { std::cell::RefCell::new(None) };
}

/// The one reading of a factor in a class: membership by parity and the sign at the zero, read
/// together and decided once per call for each factor and class however often products meet them.
fn zero_read(e: &Ex, class: &Ex, cx: &Cx) -> ZeroRead {
    let Ex::Add(cts) = class else {
        return ZeroRead {
            parity: Parity::Unknown,
            differs: None,
            below: None,
        };
    };
    let view = cx.view as *const TokenView as usize;
    let hit = READ_MEMO.with(|m| {
        m.borrow()
            .as_ref()
            .and_then(|t| t.get(&view)?.get(class)?.get(e).copied())
    });
    if let Some(hit) = hit {
        return hit;
    }
    let r = read_walk(e, class, cts.len(), class_shape(cts), cx);
    READ_MEMO.with(|m| {
        if let Some(t) = m.borrow_mut().as_mut() {
            t.entry(view)
                .or_default()
                .entry(class.clone())
                .or_default()
                .insert(e.clone(), r);
        }
    });
    r
}

fn parity_in(e: &Ex, class: &Ex, cx: &Cx) -> Parity {
    zero_read(e, class, cx).parity
}

/// The read of a product from its members' reads: its parity is theirs multiplied, and on the side
/// a one-sided member is defined on (the side where the class's key is positive otherwise) the
/// members whose sign at the zero differs from their sign beside it are counted, and so are those
/// negative beside it. An odd member read on the side where the key is negative changes both.
/// Members defined on opposite sides leave no domain beside the zero: not read.
fn product_read(reads: impl Iterator<Item = ZeroRead> + Clone) -> ZeroRead {
    use Parity::{Even, Free, Odd, Side, Unknown};
    let unread = ZeroRead {
        parity: Unknown,
        differs: None,
        below: None,
    };
    let mut side: Option<bool> = None;
    for r in reads.clone() {
        match r.parity {
            Unknown => return unread,
            Side(s) if side.is_some_and(|t| t != s) => return unread,
            Side(s) => side = Some(s),
            _ => {}
        }
    }
    let other = side == Some(true);
    let (mut odd, mut held) = (false, false);
    let (mut differs, mut below) = (Some(false), Some(false));
    for r in reads {
        let turn = match r.parity {
            Odd => {
                odd = !odd;
                held = true;
                other
            }
            Even | Side(_) => {
                held = true;
                false
            }
            _ => false,
        };
        differs = differs.zip(r.differs).map(|(a, b)| a ^ b ^ turn);
        below = below.zip(r.below).map(|(a, b)| a ^ b ^ turn);
    }
    let parity = match (side, odd, held) {
        (Some(s), _, _) => Side(s),
        (None, true, _) => Odd,
        (None, false, true) => Even,
        _ => Free,
    };
    ZeroRead {
        parity,
        differs,
        below,
    }
}

fn read_walk(e: &Ex, class: &Ex, n: usize, nk: u64, cx: &Cx) -> ZeroRead {
    use Parity::{Even, Free, Odd, Side, Unknown};
    let read = |parity, differs, below| ZeroRead {
        parity,
        differs,
        below,
    };
    let unread = read(Unknown, None, None);
    let free = |below| read(Free, Some(false), below);
    let even = read(Even, Some(false), Some(false));
    // A head defined where its argument is positive, over an argument odd in `S` whose sign beside
    // the zero is read: defined on the side where that sign is positive, where the head is
    // positive (a root or a power) or of the sign `below` gives (a log: negative where its argument
    // tends to zero, read for no argument here).
    let one_sided = |r: ZeroRead, below: Option<bool>| match (r.parity, r.below) {
        (Odd, Some(b)) => read(Side(b), Some(false), below),
        (Side(s), Some(false)) => read(Side(s), Some(false), below),
        _ => unread,
    };
    let is_even_fun = |op: Tok| {
        cx.view.tok_is(op, "abs") || cx.view.tok_is(op, "cos") || cx.view.tok_is(op, "cosh")
    };
    match e {
        Ex::Num(r) => free(Some(r.is_negative())),
        Ex::Add(ts) => {
            if ts.len() == n && class_shape(ts) == nk {
                if let Some((k, negative, scaled)) = class_of(ts, cx) {
                    if k == *class {
                        let sign = (!scaled).then_some(negative);
                        return read(Odd, sign, sign);
                    }
                }
            }
            let reads: Vec<ZeroRead> = ts.iter().map(|t| read_walk(t, class, n, nk, cx)).collect();
            // A sum holding a term defined on one side only reads as its one-sided terms where those
            // are infinite at the zero (of one side and one sign) and the rest finite there: they
            // dominate it at the zero and beside it (`(3 - x1)^(-1/2) + x2*(x1 - 3)`).
            if reads.iter().any(|r| matches!(r.parity, Side(_))) {
                let (mut side, mut below) = (None, None);
                for (t, r) in ts.iter().zip(&reads) {
                    match (r.parity, zero_mag(t, class, cx)) {
                        (Side(s), Mag::Inf) if side.is_none_or(|x| x == s) => {
                            let Some(b) = r.below.filter(|b| below.is_none_or(|x| x == *b)) else {
                                return unread;
                            };
                            (side, below) = (Some(s), Some(b));
                        }
                        (Free | Even | Odd, Mag::Zero | Mag::Finite) => {}
                        _ => return unread,
                    }
                }
                return side.map_or(unread, |s| read(Side(s), Some(false), below));
            }
            let (mut odd, mut even_t, mut free_t) = (false, false, false);
            for r in &reads {
                match r.parity {
                    Unknown | Side(_) => return unread,
                    Odd => odd = true,
                    Even => even_t = true,
                    Free => free_t = true,
                }
            }
            match (odd, even_t || free_t) {
                (true, true) => unread,
                (true, false) => read(Odd, None, None),
                _ if even_t => read(Even, None, None),
                _ => free(None),
            }
        }
        Ex::Mul(fs) => {
            let r = product_read(fs.iter().map(|f| read_walk(f, class, n, nk, cx)));
            if r.parity == Free {
                free(r.below)
            } else {
                r
            }
        }
        Ex::Pow(b, q) => {
            if read_walk(q, class, n, nk, cx).parity != Free {
                return unread;
            }
            let rb = read_walk(b, class, n, nk, cx);
            match (rb.parity, &**q) {
                (Free, Ex::Num(r)) if r.is_odd_integer() => free(rb.below),
                (Free, Ex::Num(r)) if r.is_even_integer() => free(Some(false)),
                (Free, _) => free(None),
                (Unknown, _) => unread,
                (Side(s), Ex::Num(r)) if r.is_even_integer() => {
                    read(Side(s), Some(false), Some(false))
                }
                (_, Ex::Num(r)) if r.is_even_integer() => even,
                (p, Ex::Num(r)) if r.is_odd_integer() => read(p, rb.differs, rb.below),
                (Even, _) => read(Even, None, None),
                // A fractional or symbolic power: defined where its base is positive.
                _ => one_sided(rb, Some(false)),
            }
        }
        Ex::Fun(op, args) if args.len() == 1 && odd_fun(cx.view, *op) => {
            let ra = read_walk(&args[0], class, n, nk, cx);
            match ra.parity {
                Free => free(None),
                Side(_) => unread,
                _ => ra,
            }
        }
        Ex::Fun(op, args) if args.len() == 2 && cx.view.tok_is(*op, "rootn") => {
            if read_walk(&args[1], class, n, nk, cx).parity != Free {
                return unread;
            }
            let ra = read_walk(&args[0], class, n, nk, cx);
            let odd_index = matches!(&args[1], Ex::Num(k) if k.is_odd_integer());
            let even_index = matches!(&args[1], Ex::Num(k) if k.is_even_integer());
            match ra.parity {
                Free if odd_index => free(ra.below),
                Free => free(Some(false)),
                Unknown => unread,
                _ if odd_index => ra,
                Even => even,
                _ if even_index => one_sided(ra, Some(false)),
                _ => unread,
            }
        }
        Ex::Fun(op, args) if args.len() == 1 && cx.view.tok_is(*op, "log") => {
            let ra = read_walk(&args[0], class, n, nk, cx);
            match ra.parity {
                Free => free(None),
                Even => read(Even, None, None),
                // negative beside the zero where its argument vanishes there, positive where it is
                // infinite
                _ => one_sided(
                    ra,
                    match zero_mag(&args[0], class, cx) {
                        Mag::Zero => Some(true),
                        Mag::Inf => Some(false),
                        _ => None,
                    },
                ),
            }
        }
        Ex::Fun(op, args) if args.len() == 1 && is_even_fun(*op) => {
            match read_walk(&args[0], class, n, nk, cx).parity {
                Free => free(None),
                Unknown | Side(_) => unread,
                _ => even,
            }
        }
        Ex::Fun(_, args) => {
            let mut acc = Free;
            for a in args {
                match read_walk(a, class, n, nk, cx).parity {
                    Free => {}
                    Even => acc = Even,
                    _ => return unread,
                }
            }
            if acc == Free {
                free(None)
            } else {
                read(Even, None, None)
            }
        }
        _ => free(None),
    }
}

/// READING A AT THE ZERO OF A CLASS `S`: the value a product keeps at the zero of `S` -- the side
/// whose limit it is (`true`: where the class's key is negative) and whether the factors as written
/// leave the value off that limit -- or `None` where it keeps none. It keeps one where that value
/// is a limit: where the product is even in `S` (two or more odd members, an even count, so its two
/// sides agree), or defined beside the zero on one side only (`Parity::Side`: a log, an even root,
/// a fractional or symbolic power of `S` among its members, which fixes the side), and the product
/// is not NaN there in every orientation (an infinity times a zero, `log(S) S`). An odd product
/// with no one-sided member has a two-signed pole or zero, whose sign is no value the answer keeps.
/// The float value at the zero is the limit exactly where the members whose sign there differs
/// from their sign beside the zero on that side are even in number (`ZeroRead::differs`): written
/// in the orientation positive on that side, every member's base is `+0` there. Where a member is
/// not read (an odd root of a product of unknown sign, `cbrt(x2 (x1 - 3))`, or a class scaled by a
/// factor of unknown sign), the product keeps nothing and its orientation is its cheapest. A lossy
/// context keeps no value at a pole. The product owners all ask this one predicate: the sign trade
/// (`sign_place`), the product table's lone inverse (`Class::table`) and the sum's like terms
/// (`orient_term_classes`).
fn zero_limit(factors: &[Ex], class: &Ex, cx: &Cx) -> Option<(bool, bool)> {
    if cx.lossy() {
        return None;
    }
    let reads: Vec<ZeroRead> = factors.iter().map(|f| zero_read(f, class, cx)).collect();
    let r = product_read(reads.iter().copied());
    let side = match r.parity {
        Parity::Side(s) => s,
        Parity::Even if reads.iter().filter(|r| r.parity == Parity::Odd).count() >= 2 => false,
        _ => return None,
    };
    let off = r.differs?;
    let mags = factors.iter().map(|f| zero_mag(f, class, cx));
    (product_mag(mags) != Mag::Nan).then_some((side, off))
}

/// Whether a product's value at the zero of a class is off the limit it keeps there
/// (`zero_limit`); `None` where it keeps none.
fn zero_off(factors: &[Ex], class: &Ex, cx: &Cx) -> Option<bool> {
    zero_limit(factors, class, cx).map(|(_, off)| off)
}

/// A factor's magnitude at the zero of a class `S` (`zero_limit`'s NaN test): zero, finite and
/// nonzero, infinite, NaN, or not read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mag {
    Zero,
    Finite,
    Inf,
    Nan,
    Unknown,
}

/// The magnitude of a product from its members': NaN where an infinity meets a zero.
fn product_mag(mags: impl Iterator<Item = Mag>) -> Mag {
    let (mut zero, mut inf) = (false, false);
    for m in mags {
        match m {
            Mag::Nan => return Mag::Nan,
            Mag::Unknown => return Mag::Unknown,
            Mag::Zero => zero = true,
            Mag::Inf => inf = true,
            Mag::Finite => {}
        }
    }
    match (zero, inf) {
        (true, true) => Mag::Nan,
        (true, false) => Mag::Zero,
        (false, true) => Mag::Inf,
        _ => Mag::Finite,
    }
}

/// `e`'s magnitude at the zero of a class `S`: a factor free of `S` is finite there (at a generic
/// point), a sum of the class and a sum of terms that vanish there are zero, and the heads carry it:
/// a negative power or a negative root index inverts it, a log of a zero is infinite, an odd
/// function keeps a zero, `exp`, `cos` and `cosh` of a finite argument are finite.
fn zero_mag(e: &Ex, class: &Ex, cx: &Cx) -> Mag {
    let inverse = |m: Mag| match m {
        Mag::Zero => Mag::Inf,
        Mag::Inf => Mag::Zero,
        m => m,
    };
    if parity_in(e, class, cx) == Parity::Free {
        return match e {
            Ex::PosInf | Ex::NegInf => Mag::Inf,
            _ => Mag::Finite,
        };
    }
    match e {
        Ex::Add(ts) if matches!(class_of(ts, cx), Some((k, _, _)) if k == *class) => Mag::Zero,
        Ex::Add(ts) => {
            let mut all_zero = true;
            for t in ts {
                match zero_mag(t, class, cx) {
                    Mag::Zero => {}
                    Mag::Finite => all_zero = false,
                    _ => return Mag::Unknown,
                }
            }
            if all_zero {
                Mag::Zero
            } else {
                Mag::Finite
            }
        }
        Ex::Mul(fs) => product_mag(fs.iter().map(|f| zero_mag(f, class, cx))),
        Ex::Pow(b, q) => match &**q {
            Ex::Num(r) if r.is_negative() => inverse(zero_mag(b, class, cx)),
            Ex::Num(_) => zero_mag(b, class, cx),
            _ => match zero_mag(b, class, cx) {
                Mag::Finite => Mag::Finite,
                _ => Mag::Unknown,
            },
        },
        Ex::Fun(op, args) if args.len() == 2 && cx.view.tok_is(*op, "rootn") => match &args[1] {
            Ex::Num(k) if k.is_negative() => inverse(zero_mag(&args[0], class, cx)),
            Ex::Num(_) => zero_mag(&args[0], class, cx),
            _ => Mag::Unknown,
        },
        Ex::Fun(op, args) if args.len() == 1 => {
            let m = zero_mag(&args[0], class, cx);
            let view = cx.view;
            if view.tok_is(*op, "log") {
                match m {
                    Mag::Zero | Mag::Inf => Mag::Inf,
                    m => m,
                }
            } else if odd_fun(view, *op) || view.tok_is(*op, "abs") {
                match m {
                    Mag::Zero | Mag::Finite => m,
                    _ => Mag::Unknown,
                }
            } else if ["exp", "cos", "cosh"].iter().any(|n| view.tok_is(*op, n)) {
                match m {
                    Mag::Zero | Mag::Finite => Mag::Finite,
                    _ => Mag::Unknown,
                }
            } else {
                Mag::Unknown
            }
        }
        _ => Mag::Unknown,
    }
}

/// The sites of one class in a product (`sign_place`): the class, and per site its index among the
/// sites and whether it spells the class against the class's own orientation.
type ClassGroup = (Option<Ex>, Vec<(usize, bool)>);

/// A class {B, -B}: the head of `B`, the terms of the sum on each side (the second empty until a
/// factor over it is seen), and the key and flip that recognise the negation of the first.
struct Class<'a> {
    head: Option<(Tok, Option<&'a Ex>)>,
    sums: [Vec<Ex>; 2],
    key: u64,
    flip: SumFlip,
    /// The class's key where the product keeps a value at its zero (`zero_limit`).
    kept: Option<Ex>,
}

impl<'a> Class<'a> {
    fn new(read: &ClassRead<'a>) -> Self {
        Class {
            head: read.head,
            sums: [read.terms.to_vec(), Vec::new()],
            key: negation_key(read.terms),
            flip: SumFlip::new(),
            kept: None,
        }
    }

    /// Is the class's argument a literal multiple (a sum of one term under a head)?
    fn multiple(&self) -> bool {
        self.head.is_some() && self.sums[0].len() == 1 && !matches!(self.sums[0][0], Ex::Add(_))
    }

    /// The second sum of a class whose factors all arrived over the first: its flip, where the
    /// sign may pass through it (`sign_trade_flip`; for a literal multiple, its head's).
    fn complete(&mut self, cx: &Cx) -> bool {
        let first = &self.sums[0];
        let multiple = self.multiple();
        let base = self.base(0);
        let flip = self.flip.get_or_init(|| {
            if multiple {
                // The head's own flip where it is a sign-trade site (a literal folds through
                // `fun()`), else the argument negated.
                let negated = match sign_trade_flip(&base, cx) {
                    Some(Ex::Fun(_, a)) => a[0].clone(),
                    _ => negate_term(&first[0], cx)?,
                };
                return multiple_literal(&negated).map(|_| vec![negated]);
            }
            match sign_trade_flip(&Ex::Add(first.clone()), cx) {
                Some(Ex::Add(f)) => Some(f),
                _ => None,
            }
        });
        match flip.clone() {
            Some(f) => {
                self.sums[1] = f;
                true
            }
            None => false,
        }
    }

    fn same_head(&self, read: &ClassRead) -> bool {
        match (&self.head, &read.head) {
            (None, None) => true,
            (Some((a, i)), Some((b, j))) => a == b && i == j,
            _ => false,
        }
    }

    /// The side a factor reads over, if it belongs to the class.
    fn side_of(&self, read: &ClassRead) -> Option<usize> {
        if !self.same_head(read) {
            return None;
        }
        (0..2).find(|&s| !self.sums[s].is_empty() && self.sums[s].as_slice() == read.terms)
    }

    /// The side a factor belongs to, recognising the second sum on first sight.
    fn place(&mut self, read: &ClassRead<'a>, cx: &Cx) -> Option<usize> {
        if let Some(s) = self.side_of(read) {
            return Some(s);
        }
        let opposite = self.same_head(read)
            && self.sums[1].is_empty()
            && negation_key(read.terms) == self.key
            && negates(&self.sums[0], &self.flip, read.terms, cx);
        if opposite {
            self.sums[1] = read.terms.to_vec();
            return Some(1);
        }
        None
    }

    /// The base over one side's sum (a literal multiple as itself).
    fn base(&self, side: usize) -> Ex {
        let sum = match self.sums[side].as_slice() {
            [m] if self.head.is_some() => m.clone(),
            ts => Ex::Add(ts.to_vec()),
        };
        match &self.head {
            None => sum,
            Some((op, None)) => Ex::Fun(*op, vec![sum]),
            Some((op, Some(k))) => Ex::Fun(*op, vec![sum, (*k).clone()]),
        }
    }

    /// The table of the class's members among `factors`.
    fn table<'b>(&self, factors: impl Iterator<Item = (&'b Ex, ClassRead<'b>)>, cx: &Cx) -> Table {
        let mut t = Table::default();
        for (f, read) in factors {
            let Some(s) = self.side_of(&read) else {
                continue;
            };
            let side = &mut t.sides[s];
            // An integer exponent is mass; a fractional one gives its integer part, taken toward
            // zero, and keeps the rest (`S^(5/2) = S^2 * S^(1/2)`: a merge of like signs, which a
            // context without the certificate makes, must not hide mass from the certified one);
            // a symbolic one, or one too large to count (or whose parity the f64 reading does not
            // know), stays as it is.
            let Some((whole, frac)) = read.exp.as_ref().and_then(whole_part) else {
                side.fixed.push(f.clone());
                continue;
            };
            let mass = if whole > 0 { &mut side.p } else { &mut side.n };
            match mass.checked_add(whole) {
                Some(m) if m.checked_abs().is_some_and(|a| a <= MASS_LIMIT) => *mass = m,
                _ => {
                    side.fixed.push(f.clone());
                    continue;
                }
            }
            if !frac.is_zero() {
                side.fixed.push(if whole == 0 {
                    f.clone()
                } else {
                    pow(self.base(s), Ex::Num(frac), cx)
                });
            }
        }
        // A lone inverse moves, its sign going to the product: the table writes the class in one
        // orientation, so at the zero of its sum (where every member's base is +0) the product's
        // value is the coefficient's infinity, the limit wherever the pole is one-signed
        // (`1/((x1 - 3)(3 - x1))` is `-1/(x1 - 3)^2`); where the product keeps a value at the
        // zero (`zero_limit`), only to a placement that keeps it. A class whose zero set may hold
        // a region keeps its inverse, infinite on the region.
        t.free_inverse = zero_set_null(&self.base(0), cx);
        for side in t.sides.iter_mut() {
            side.fixed.sort_by(|a, b| cmp_ex(a, b, cx.view));
        }
        t
    }

    /// The class's factors for a placement of its table: on each side the positive mass, the
    /// movable negative mass, the kept `B^-1` and the factors that never move. A fractional power
    /// may take the mass of its own sign (`B^2 * B^(1/2) * B^(1/2)` is `B^(5/2) * B^(1/2)` or
    /// `B^(3/2) * B^(3/2)`, merges of like signs), so every split of a side's mass between its
    /// integer power and its fractional powers is one spelling while a placement has at most
    /// `SPLIT_LIMIT` of them, else each mass goes whole to its integer power or spread evenly.
    fn factors(&self, t: &Table, pl: &Placement, cx: &Cx) -> Vec<Vec<Ex>> {
        // Per side, its fractional powers by sign (each with its position in `fixed` and its
        // exponent), and per side and sign the splits of the mass: the integer power's share,
        // then each fractional power's.
        let slots: Vec<[Vec<(usize, Rat)>; 2]> = t
            .sides
            .iter()
            .map(|side| {
                let mut by_sign: [Vec<(usize, Rat)>; 2] = [Vec::new(), Vec::new()];
                for (i, f) in side.fixed.iter().enumerate() {
                    if let Some(e) = class_read(f, cx)
                        .and_then(|r| r.exp)
                        .filter(|e| !e.is_integer())
                    {
                        by_sign[usize::from(e.is_negative())].push((i, e));
                    }
                }
                by_sign
            })
            .collect();
        let splits_with = |every: bool| -> Vec<[Vec<Vec<i128>>; 2]> {
            (0..2)
                .map(|s| {
                    [
                        mass_splits(pl.p[s], slots[s][0].len(), every),
                        mass_splits(pl.q[s], slots[s][1].len(), every),
                    ]
                })
                .collect()
        };
        let mut splits = splits_with(true);
        let count = splits
            .iter()
            .flatten()
            .try_fold(1usize, |n, v| n.checked_mul(v.len()));
        if count.is_none_or(|n| n > SPLIT_LIMIT) {
            splits = splits_with(false);
        }
        let mut lists: Vec<Vec<Ex>> = vec![Vec::new()];
        for s in 0..2 {
            let side = &t.sides[s];
            let b = self.base(s);
            let int_pow = |k: i128| -> Ex {
                if k == 1 {
                    b.clone()
                } else {
                    Ex::Pow(Box::new(b.clone()), Box::new(Ex::int(k)))
                }
            };
            let mut side_lists: Vec<Vec<Ex>> = Vec::new();
            for pos in &splits[s][0] {
                for neg in &splits[s][1] {
                    let mut v = Vec::new();
                    let mut taken = vec![0i128; side.fixed.len()];
                    for (sign, split) in [(0usize, pos), (1usize, neg)] {
                        for (&(i, _), &k) in slots[s][sign].iter().zip(&split[1..]) {
                            taken[i] = k;
                        }
                        if split[0] != 0 {
                            v.push(int_pow(split[0]));
                        }
                    }
                    let kept = t.kept(side.n);
                    if kept != 0 {
                        v.push(int_pow(kept));
                    }
                    for (i, f) in side.fixed.iter().enumerate() {
                        if taken[i] == 0 {
                            v.push(f.clone());
                            continue;
                        }
                        let e = slots[s]
                            .iter()
                            .flatten()
                            .find(|(j, _)| *j == i)
                            .and_then(|(_, e)| e.checked_add(&Rat::int(taken[i])));
                        match e {
                            Some(e) => v.push(pow(b.clone(), Ex::Num(e), cx)),
                            None => {
                                v.push(int_pow(taken[i]));
                                v.push(f.clone());
                            }
                        }
                    }
                    side_lists.push(v);
                }
            }
            lists = lists
                .into_iter()
                .flat_map(|l| {
                    side_lists.iter().map(move |sl| {
                        let mut l = l.clone();
                        l.extend(sl.iter().cloned());
                        l
                    })
                })
                .collect();
        }
        lists
    }
}

/// The most splits of its mass between integer and fractional powers a placement offers
/// (`Class::factors`).
const SPLIT_LIMIT: usize = 36;

/// The splits of a side's mass of one sign between its integer power and `slots` fractional
/// powers of that sign, the integer power's share first and the whole mass on the integer power
/// the first split: with `every`, every composition (while there are at most `SPLIT_LIMIT`);
/// else that one and the mass spread evenly over the fractional powers.
fn mass_splits(mass: i128, slots: usize, every: bool) -> Vec<Vec<i128>> {
    let mut whole = vec![0i128; slots + 1];
    whole[0] = mass;
    if mass == 0 || slots == 0 {
        return vec![whole];
    }
    let m = mass.unsigned_abs();
    let sign = mass.signum();
    if !every {
        let n = slots as u128;
        let even = (0..n)
            .map(|i| sign * ((m / n + u128::from(i < m % n)) as i128))
            .collect::<Vec<i128>>();
        return vec![whole, [vec![0], even].concat()];
    }
    // Every composition, the integer power's share falling: the whole mass on it comes first.
    fn fill(at: usize, left: u128, sign: i128, cur: &mut Vec<i128>, out: &mut Vec<Vec<i128>>) {
        if at + 1 == cur.len() {
            cur[at] = sign * left as i128;
            out.push(cur.clone());
            return;
        }
        for k in (0..=left).rev() {
            cur[at] = sign * k as i128;
            fill(at + 1, left - k, sign, cur, out);
        }
    }
    // C(m + slots, slots) compositions
    let mut count: u128 = 1;
    for i in 1..=slots as u128 {
        count = count.saturating_mul(m + i) / i;
        if count > SPLIT_LIMIT as u128 {
            return mass_splits(mass, slots, false);
        }
    }
    let mut out = Vec::new();
    fill(0, m, sign, &mut vec![0i128; slots + 1], &mut out);
    out
}

/// An exponent's integer part, taken toward zero, and the rest: `None` for one too large to count
/// or an integer whose parity the f64 reading does not know.
fn whole_part(r: &Rat) -> Option<(i128, Rat)> {
    let (p, q) = r.small_parts()?;
    if q == 1 && r.parity().is_none() {
        return None;
    }
    let whole = p / q;
    let frac = r.checked_add(&Rat::int(-whole))?;
    Some((whole, frac))
}

/// The largest total a table counts (a larger exponent stays as it is): sums and differences of
/// two totals stay far inside `i128`.
const MASS_LIMIT: i128 = 1 << 100;

/// A class's exponent totals on each side (`collect_opposite_sums`): the positive integer mass
/// `p`, the negative `n`, and the factors that never move; and whether a lone inverse moves too
/// (off a region).
#[derive(Default)]
struct Table {
    sides: [TableSide; 2],
    free_inverse: bool,
}

#[derive(Default)]
struct TableSide {
    p: i128,
    n: i128,
    fixed: Vec<Ex>,
}

/// A class's factors under one placement (`collect_opposite_sums`): collected (`None` where a
/// collection splices), whether the placement moves a sign, and the lower bound of their price
/// and their count.
struct PlacedClass {
    list: Option<Vec<Ex>>,
    odd: bool,
    lb: u64,
    members: usize,
}

/// Where a placement puts the movable mass, positive and negative, per side.
struct Placement {
    p: [i128; 2],
    q: [i128; 2],
}

impl Table {
    /// Does a side hold two fractional powers of one sign beside integer mass of that sign, which
    /// they may split between them?
    fn splits_mass(&self, s: usize, cx: &Cx) -> bool {
        let side = &self.sides[s];
        let mut count = [0usize; 2];
        for f in &side.fixed {
            if let Some(e) = class_read(f, cx)
                .and_then(|r| r.exp)
                .filter(|e| !e.is_integer())
            {
                count[usize::from(e.is_negative())] += 1;
            }
        }
        (count[0] >= 2 && side.p > 0) || (count[1] >= 2 && side.n - self.kept(side.n) < 0)
    }

    /// The inverse a side keeps: one `B^-1` where its negative total is odd, none where it may
    /// move (`Table::free_inverse`).
    fn kept(&self, n: i128) -> i128 {
        if self.free_inverse {
            0
        } else {
            n % 2
        }
    }

    /// The placements of the movable mass, the table's own first, each with whether it moves an
    /// odd amount of mass (a sign). Every split while there are few; else the two extremes beside
    /// the table's own.
    fn placements(&self) -> Vec<(Placement, bool)> {
        let [a, b] = &self.sides;
        let total_p = a.p + b.p;
        let q = |n: i128| n - self.kept(n);
        let total_q = q(a.n) + q(b.n);
        let step = if self.free_inverse { 1 } else { 2 };
        let fixed = self.kept(a.n) != 0
            || self.kept(b.n) != 0
            || !a.fixed.is_empty()
            || !b.fixed.is_empty();
        let mut ps: Vec<i128> = if total_p <= 7 {
            (0..=total_p).collect()
        } else {
            vec![0, a.p, total_p]
        };
        let mut qs: Vec<i128> = if -total_q <= 6 {
            (0..=-total_q / step).map(|k| -step * k).collect()
        } else {
            vec![0, q(a.n), total_q]
        };
        ps.sort_unstable();
        ps.dedup();
        qs.sort_unstable();
        qs.dedup();
        let own = (a.p, q(a.n));
        // Without a factor that stays, all the mass on one side mirrors all of it on the other,
        // and with even totals `pow` files the two alike: one of them is priced.
        let ends = [(total_p, total_q), (0, 0)];
        let mirrored = |c: (i128, i128)| {
            !fixed
                && total_p % 2 == 0
                && total_q % 2 == 0
                && ends.contains(&c)
                && (ends.contains(&own) || c == ends[1])
        };
        let mut out = vec![(
            Placement {
                p: [a.p, b.p],
                q: [q(a.n), q(b.n)],
            },
            false,
        )];
        for &x in &ps {
            for &y in &qs {
                if (x, y) == own || mirrored((x, y)) {
                    continue;
                }
                out.push((
                    Placement {
                        p: [x, total_p - x],
                        q: [y, total_q - y],
                    },
                    (x - a.p + y - q(a.n)) % 2 != 0,
                ));
            }
        }
        out
    }

    /// Do the two tables offer the same placements: the same totals, the same `B^-1` kept on
    /// each side, the same factors that never move?
    fn same_orbit(&self, o: &Table) -> bool {
        let [a, b] = &self.sides;
        let [c, d] = &o.sides;
        a.p + b.p == c.p + d.p
            && a.n + b.n == c.n + d.n
            && self.kept(a.n) == o.kept(c.n)
            && self.kept(b.n) == o.kept(d.n)
            && a.fixed == c.fixed
            && b.fixed == d.fixed
    }
}

/// Does the bag hold a class to decide (`collect_opposite_sums`): a factor over a sum, or over an
/// odd function of one, and one over its negation, or two over one of them with exponents of
/// opposite signs? Only sums that share a `negation_key` are compared, and each sum's flip is
/// computed at most once.
fn holds_opposite_sums(factors: &[Ex], cx: &Cx) -> bool {
    let reads: Vec<ClassRead> = factors.iter().filter_map(|f| class_read(f, cx)).collect();
    if reads.len() < 2 {
        return false;
    }
    let sign = |r: &ClassRead| r.exp.as_ref().and_then(whole_part).map(|(w, _)| w.signum());
    let fraction = |r: &ClassRead| r.exp.as_ref().filter(|e| !e.is_integer()).map(Rat::signum);
    let mut keyed: Vec<(u64, usize)> = reads
        .iter()
        .enumerate()
        .map(|(i, r)| (negation_key(r.terms), i))
        .collect();
    keyed.sort_unstable();
    for i in 0..keyed.len() {
        let first = &reads[keyed[i].1];
        let mut class = Class::new(first);
        for j in i + 1..keyed.len() {
            if keyed[j].0 != keyed[i].0 {
                break;
            }
            let other = &reads[keyed[j].1];
            match class.place(other, cx) {
                Some(1) => return true,
                Some(_) if sign(first).zip(sign(other)).is_some_and(|(a, b)| a * b < 0) => {
                    return true
                }
                // two fractional powers of one sign over one sum, which may split its mass
                Some(_) if fraction(first).is_some_and(|a| fraction(other) == Some(a)) => {
                    return true
                }
                _ => {}
            }
        }
    }
    false
}

/// A key a sum shares with its negation: its length and the multiset of its terms' shapes, where a
/// shape ignores every sign a negation can move (a coefficient's, and below it the orientation of
/// any sum) and reads four levels deep. Equal keys are only a necessary condition (`negates`).
fn negation_key(ts: &[Ex]) -> u64 {
    fn mix(x: u64) -> u64 {
        let z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn shape(e: &Ex, depth: u32) -> u64 {
        use std::hash::{Hash, Hasher};
        if depth == 0 {
            return 1;
        }
        let mut h = rustc_hash::FxHasher::default();
        match e {
            Ex::Num(r) => {
                0u8.hash(&mut h);
                if r.is_negative() {
                    r.checked_neg().hash(&mut h);
                } else {
                    Some(r).hash(&mut h);
                }
            }
            Ex::Add(ts) => {
                1u8.hash(&mut h);
                ts.iter()
                    .fold(0u64, |a, t| a.wrapping_add(mix(shape(t, depth - 1))))
                    .hash(&mut h);
            }
            Ex::Mul(fs) => {
                let unit = |f: &&Ex| matches!(f, Ex::Num(r) if r.is_one() || *r == Rat::NEG_ONE);
                if fs.iter().filter(|f| !unit(f)).count() == 1 {
                    if let Some(only) = fs.iter().find(|f| !unit(f)) {
                        return shape(only, depth);
                    }
                }
                2u8.hash(&mut h);
                fs.iter()
                    .filter(|f| !unit(f))
                    .fold(0u64, |a, f| a.wrapping_add(mix(shape(f, depth - 1))))
                    .hash(&mut h);
            }
            Ex::Pow(b, x) => {
                3u8.hash(&mut h);
                shape(b, depth - 1).hash(&mut h);
                x.hash(&mut h);
            }
            Ex::Fun(op, args) => {
                4u8.hash(&mut h);
                op.hash(&mut h);
                for a in args {
                    shape(a, depth - 1).hash(&mut h);
                }
            }
            Ex::PosInf | Ex::NegInf => 5u8.hash(&mut h),
            other => {
                6u8.hash(&mut h);
                other.hash(&mut h);
            }
        }
        h.finish()
    }
    ts.iter().fold(mix(ts.len() as u64), |a, t| {
        a.wrapping_add(mix(shape(t, 4)))
    })
}

/// A sum's flip (`sign_trade_flip`), computed once when a comparison needs it (`negates`).
type SumFlip = std::cell::OnceCell<Option<Vec<Ex>>>;

/// Is the sum `b` the negation of the sum `a`? Term by term (`opposite_sums`) where negation
/// keeps every key. A term holding a sum may take the sign into that sum instead
/// (`-sin(x1 - x2)` is `sin(x2 - x1)`) and so change its key; every other term keeps its key, so
/// those must pair up term by term before the sum's own flip decides (`sign_trade_flip`, computed
/// once into `flip`).
fn negates(a: &[Ex], flip: &SumFlip, b: &[Ex], cx: &Cx) -> bool {
    if a.len() != b.len() {
        return false;
    }
    if opposite_sums(a, b, cx.view) {
        return true;
    }
    if !a.iter().any(holds_sum) {
        return false;
    }
    fn plain(s: &[Ex]) -> Vec<&Ex> {
        s.iter().filter(|t| !holds_sum(t)).collect()
    }
    let (pa, pb) = (plain(a), plain(b));
    pa.len() == pb.len()
        && pa
            .iter()
            .zip(&pb)
            .all(|(x, y)| opposite_terms(x, y, cx.view))
        && flip
            .get_or_init(|| match sign_trade_flip(&Ex::Add(a.to_vec()), cx) {
                Some(Ex::Add(f)) => Some(f),
                _ => None,
            })
            .as_deref()
            == Some(b)
}

/// The base a factor is collected under (`factor_split`'s reading, borrowed): a power's base, an
/// even root's radicand, else the factor itself.
fn collection_base<'a>(f: &'a Ex, cx: &Cx) -> &'a Ex {
    match f {
        Ex::Pow(b, _) => b,
        Ex::Fun(op, args)
            if args.len() == 2
                && matches!(&args[1], Ex::Num(n)
                    if n.is_even_integer() && n.cmp_int(2) != Ordering::Less)
                && cx.view.tok_is(*op, "rootn") =>
        {
            &args[0]
        }
        _ => f,
    }
}

/// Is the sum `b` the negation of the sum `a`, term by term? Both are canonical bags, sorted
/// by their coefficient-stripped keys, so a negation keeps every term in its place.
fn opposite_sums(a: &[Ex], b: &[Ex], view: &TokenView) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| opposite_terms(x, y, view))
}

/// Is the term `y` the term `x` with its coefficient negated?
fn opposite_terms(x: &Ex, y: &Ex, view: &TokenView) -> bool {
    // A term's coefficient and the factors that remain (borrowed: no key is built).
    fn split(t: &Ex) -> (Rat, &[Ex]) {
        match t {
            Ex::Mul(v) => match v.first() {
                Some(Ex::Num(c)) => (c.clone(), &v[1..]),
                _ => (Rat::ONE, v.as_slice()),
            },
            other => (Rat::ONE, std::slice::from_ref(other)),
        }
    }
    match (x, y) {
        (Ex::Num(p), Ex::Num(q)) => p.checked_neg().is_some_and(|n| n == *q),
        (Ex::Num(_), _) | (_, Ex::Num(_)) => false,
        _ => {
            let ((cx_, kx), (cy, ky)) = (split(x), split(y));
            if kx == ky {
                return cx_.checked_neg().is_some_and(|n| n == cy);
            }
            // An odd function of a literal carries its sign inside (`sin(-2)`): only
            // `term_split`'s sign-normalized keys can match then.
            let literal_odd =
                |k: &[Ex]| k.iter().any(|f| odd_of_literal_multiple(f, view).is_some());
            if !literal_odd(kx) && !literal_odd(ky) {
                return false;
            }
            let (cx_, kx) = term_split(x.clone(), view);
            let (cy, ky) = term_split(y.clone(), view);
            kx == ky && cx_.checked_neg().is_some_and(|n| n == cy)
        }
    }
}

/// Does the expression hold a sum anywhere?
fn holds_sum(e: &Ex) -> bool {
    match e {
        Ex::Add(_) => true,
        Ex::Mul(v) | Ex::Fun(_, v) => v.iter().any(holds_sum),
        Ex::Pow(b, x) => holds_sum(b) || holds_sum(x),
        _ => false,
    }
}

/// Are the two factors collected under one base (`sign_place`'s collision test)?
fn same_collection_base(f: &Ex, g: &Ex, cx: &Cx) -> bool {
    cmp_ex(collection_base(f, cx), collection_base(g, cx), cx.view) == Ordering::Equal
}

/// Does the sum a site is built over occur again in the product, up to its orientation: as
/// another site's sum or inside any factor? Only then are its classes read (`sign_place`). One
/// pass over the product's sums with a hash of their shapes blind to literals and signs
/// (`shape_sig`'s), so a sum meets its constant multiples too; a hash that merely collides only
/// costs the full reading.
fn classes_meet(out: &[Ex], site_flips: &[(usize, Ex)]) -> bool {
    fn one<T: std::hash::Hash>(t: &T) -> u64 {
        use std::hash::Hasher;
        let mut h = rustc_hash::FxHasher::default();
        t.hash(&mut h);
        h.finish()
    }
    fn walk(e: &Ex, sums: &mut Vec<(usize, u64)>) -> u64 {
        match e {
            Ex::Num(_) => 0,
            Ex::Add(v) => {
                let s = v.iter().map(|x| walk(x, sums)).fold(0, u64::wrapping_add);
                sums.push((v.len(), class_shape(v)));
                s
            }
            Ex::Mul(v) => v.iter().map(|x| walk(x, sums)).fold(0, u64::wrapping_add),
            Ex::Fun(op, args) => args
                .iter()
                .map(|x| walk(x, sums))
                .fold(one(op), u64::wrapping_add),
            Ex::Pow(b, q) => walk(b, sums).wrapping_add(walk(q, sums)).wrapping_mul(31),
            Ex::Leaf(t) => one(t),
            other => one(&std::mem::discriminant(other)),
        }
    }
    let class_sites = site_flips
        .iter()
        .filter(|(pos, _)| site_sum_terms(&out[*pos]).is_some())
        .count();
    if class_sites == 0 {
        return false;
    }
    // Most products hold no sum besides the ones their sites are built over, each once: then
    // only two sites over sums of one length can meet.
    fn count(e: &Ex) -> usize {
        match e {
            Ex::Add(v) => 1 + v.iter().map(count).sum::<usize>(),
            Ex::Mul(v) | Ex::Fun(_, v) => v.iter().map(count).sum(),
            Ex::Pow(b, q) => count(b) + count(q),
            _ => 0,
        }
    }
    if out.iter().map(count).sum::<usize>() == class_sites {
        let lens: Vec<usize> = site_flips
            .iter()
            .filter_map(|(pos, _)| site_sum_terms(&out[*pos]).map(<[Ex]>::len))
            .collect();
        if (0..lens.len()).all(|i| lens[i + 1..].iter().all(|l| *l != lens[i])) {
            return false;
        }
    }
    let mut sums: Vec<(usize, u64)> = Vec::new();
    for f in out {
        walk(f, &mut sums);
    }
    if sums.len() <= 1 {
        return false;
    }
    site_flips.iter().any(|(pos, _)| {
        site_sum_terms(&out[*pos]).is_some_and(|ts| {
            let sig = class_shape(ts);
            sums.iter().filter(|&&x| x == (ts.len(), sig)).count() >= 2
        })
    })
}

/// The terms of the sum a site is built over, where it is one (the sum itself, an odd function or
/// odd root of it, a power of either).
fn site_sum_terms(f: &Ex) -> Option<&[Ex]> {
    let m = match f {
        Ex::Pow(b, _) => &**b,
        f => f,
    };
    match m {
        Ex::Add(ts) => Some(ts),
        Ex::Fun(_, a) => match a.first() {
            Some(Ex::Add(ts)) => Some(ts),
            _ => None,
        },
        _ => None,
    }
}

/// F63: assemble a product from `(coefficient, factor bag)` in its CANONICAL sign
/// placement -- the shared owner behind `mul()`'s final assembly and `term_join`'s
/// negative-coefficient joins, so every site that mints a product builds the SAME
/// spelling the orientation machinery prices (the row-120 law). With no trade site
/// (or a negation overflow, or no configuration that keeps the product's values at its zeros)
/// this is exactly the plain assembly: push the non-unit coefficient, sort, wrap.
fn sign_place(mut coeff: Rat, mut out: Vec<Ex>, cx: &Cx) -> Ex {
    // A sum holding a `Const` takes a sign into its terms (`negate_term`: a Const term absorbs it
    // in the refit, every other term negates), so it carries the product's sign exactly, as a
    // negation-absorbing sum does; only a bare `Const` factor or a zero coefficient eats a sign.
    // Beside such a sum the coefficient and the infinities are positive and the sum holds their
    // signs (H-020's fold, extended to the infinities).
    let const_sum = if !coeff.is_zero() && !out.iter().any(|f| matches!(f, Ex::Const)) {
        out.iter()
            .position(|f| matches!(f, Ex::Add(v) if v.iter().any(Ex::contains_const)))
    } else {
        None
    };
    if let Some(i) = const_sum {
        let negs = out.iter().filter(|f| matches!(f, Ex::NegInf)).count();
        if (negs % 2 == 1) != coeff.is_negative() {
            let Ex::Add(terms) = &out[i] else {
                unreachable!()
            };
            let flipped: Option<Vec<Ex>> = terms.iter().map(|t| negate_term(t, cx)).collect();
            if let Some(flipped) = flipped {
                let re = add(flipped, cx);
                if matches!(re, Ex::Add(_)) {
                    out[i] = re;
                    coeff = coeff.abs();
                    for f in out.iter_mut() {
                        if matches!(f, Ex::NegInf) {
                            *f = Ex::PosInf;
                        }
                    }
                }
            }
        } else if negs > 0 || coeff.is_negative() {
            coeff = coeff.abs();
            for f in out.iter_mut() {
                if matches!(f, Ex::NegInf) {
                    *f = Ex::PosInf;
                }
            }
        }
    }
    let assemble = |factors: Vec<Ex>, c: &Rat| -> Ex {
        let mut v = factors;
        if !c.is_one() {
            v.push(Ex::Num(c.clone()));
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
    //   * a bare Const factor: the sign VANISHES into the forall-exists refit -- every
    //     site orientation is value-equal as a family, so orientations are chosen
    //     freely (H-020);
    //   * a Const-bearing Add factor: the sign folds into that sum term by term
    //     (`negate_term`; H-020's fold), the coefficient and the infinities positive;
    //   * a bare +-Inf factor: the infinity carries the sign itself (H-014);
    //   * an Add factor with a negation-absorbing member: the sign folds into that
    //     sum member-wise (H-030);
    //   * a ZERO coefficient (the kept-zero bag `Mul[0, ..]`): the zero eats every
    //     sign -- `0 * X` and `0 * (-X)` are value-equal in EVERY case (finite -> 0,
    //     +-inf -> nan, nan -> nan), so the whole sign dimension collapses exactly
    //     like a Const-carrier's (F68; fuzz rows 392777/647852, introduced by F63:
    //     the kept-zero arm assembled RAW and froze the arrival mirror, so the
    //     direct build and the parse of its own rendering shipped two states);
    //   * otherwise: the rational coefficient.
    {
        // Each site's flip is materialized ONCE (the site test computes it anyway) and
        // reused by every orientation mask below: a flip negates a sum term by term, and
        // each negated term is built through `mul`, which places its own signs the same way,
        // so re-flipping per mask multiplied the work at every nesting level. Each mask's
        // price is the sum of per-factor prices (see `complexity`'s Mul arm), each factor and
        // each flip priced once.
        // A class trades its sign only as a whole: the sites built over one sum (`site_class`)
        // are written in one orientation and flip together, so a flip moves the product's sign
        // by the parity of the class's members. At the zero of the sum every member's base is +0,
        // so the product's value there is the coefficient's infinity -- the limit wherever the
        // pole is one-signed; a single member flipping alone would turn it (`1/(cbrt(u) u)`).
        // Beside a bare `Const`, which absorbs every constant factor of the product and so the
        // orientation sign of every class, every class is a site (`refit_trade_flip`).
        let refit = out.iter().any(|f| matches!(f, Ex::Const));
        let site_flips: Vec<(usize, Ex)> = out
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                let flip = if refit {
                    refit_trade_flip(f, cx)
                } else {
                    sign_trade_flip(f, cx)
                };
                flip.map(|nf| (i, nf))
            })
            .collect();
        // A class is read only where its sum meets another occurrence of itself: two sites over
        // sums of one shape, or a sum inside another factor (`classes_meet`). Elsewhere each site
        // is a class of its own, which nothing pins and no other member checks.
        let meets = classes_meet(&out, &site_flips);
        // Each group: its sites (indices into `site_flips`) and, per site, whether it spells the
        // class against the class's own orientation (`site_class`). A configuration bit picks one
        // orientation for the whole group; the sites spelled against it flip.
        let mut groups: Vec<ClassGroup> = Vec::new();
        for (bit, (pos, _)) in site_flips.iter().enumerate() {
            let class = match site_multiple(&out[*pos], cx) {
                Some(c) => Some(c),
                None if meets => site_class(&out[*pos], cx),
                None => None,
            };
            let joined = class.as_ref().and_then(|(key, flipped)| {
                groups
                    .iter_mut()
                    .find(|(k, _)| k.as_ref() == Some(key))
                    .map(|g| (g, *flipped))
            });
            match joined {
                Some(((_, members), flipped)) => members.push((bit, flipped)),
                None => {
                    let (key, flipped) = class.map_or((None, false), |(k, f)| (Some(k), f));
                    groups.push((key, vec![(bit, flipped)]));
                }
            }
        }
        // Reading A at the zero of each class (`zero_limit`): where the product keeps a value
        // there, a configuration that leaves it off the limit is no candidate. A configuration
        // changes the count of members whose sign at the zero differs from their sign beside it by
        // the sites it flips whose own count changes; a member built over the sum's other
        // orientation inside a product (`sin(-(x1 (x1 - 3)))` for `sin(x1 (3 - x1))`) is counted
        // where it stands, so the product restores the limit an inner orientation moved.
        let checks: Vec<(bool, Vec<Option<bool>>)> = groups
            .iter()
            .filter_map(|(k, _)| {
                let k = k.as_ref()?;
                let (_, now) = zero_limit(&out, k, cx)?;
                let toggles: Vec<Option<bool>> = site_flips
                    .iter()
                    .map(|(pos, nf)| {
                        let r = zero_read(&out[*pos], k, cx);
                        if r.parity == Parity::Free {
                            return Some(false);
                        }
                        Some(r.differs? != zero_read(nf, k, cx).differs?)
                    })
                    .collect();
                Some((now, toggles))
            })
            .collect();
        let factor_price = |f: &Ex| -> u64 {
            match f {
                Ex::Num(r) if r.is_one() => 0,
                Ex::Num(r) if *r == Rat::NEG_ONE => mu_sign(),
                Ex::Num(r) => mu_rat(r),
                _ => complexity(f, cx.view),
            }
        };
        if let (false, Some(nc)) = (site_flips.is_empty(), coeff.checked_neg()) {
            // Priority: Free FIRST -- a bare Const eats EVERY sign (coefficient and
            // bare-infinity signs alike, by the forall-exists refit), so with one present
            // the whole sign dimension collapses and orientations are chosen freely.
            // Checking Inf first split the orbit into two components joined only through
            // the refit (fuzz row 39122: {PosInf, x4-x1} and {PosInf, x1-x4} both stable,
            // one value class). A zero coefficient is Free by the same collapse (see the
            // carrier list above). A Const-bearing sum comes next: it holds the signs of
            // the coefficient and the infinities (normalized above), so every toggle goes
            // into it and the orbit has one component.
            let carrier = if coeff.is_zero() || out.iter().any(|f| matches!(f, Ex::Const)) {
                Carrier::Free
            } else if let Some(i) = const_sum {
                Carrier::Absorb(i)
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
            let base_price: Vec<u64> = out.iter().map(&factor_price).collect();
            let flip_price: Vec<u64> = site_flips.iter().map(|(_, nf)| factor_price(nf)).collect();
            let orbit = SignOrbit {
                out: &out,
                coeff: &coeff,
                nc: &nc,
                site_flips: &site_flips,
                groups: &groups,
                checks: &checks,
                base_price: &base_price,
                flip_price: &flip_price,
                carrier,
                assemble: &assemble,
            };
            let chosen = sign_place_groups(&orbit, cx);
            // The closed form's price is the enumeration's optimum: checked on every small orbit.
            #[cfg(debug_assertions)]
            if groups.len() <= 6 {
                debug_assert_eq!(
                    chosen.as_ref().map(|c| complexity(c, cx.view)),
                    sign_place_enumerated(&orbit, cx),
                    "the sign orbit's closed form prices off its enumeration"
                );
            }
            if let Some(chosen) = chosen {
                return chosen;
            }
        }
    }
    assemble(out, &coeff)
}

/// The negation of a sum that carries a product's sign (`sign_place`): a sum holding a `Const`
/// is negated through `add`, which settles a negated member that holds a Const-bearing sum itself
/// (its sign folds into that sum, as `mul()` folds it); any other carrier term by term. `None`
/// where a negation overflows or the sum does not stay one.
fn flip_carrier_sum(f: &Ex, cx: &Cx) -> Option<Ex> {
    let Ex::Add(ts) = f else {
        return None;
    };
    let mut fl = ts
        .iter()
        .map(|x| negate_term(x, cx))
        .collect::<Option<Vec<Ex>>>()?;
    if ts.iter().any(Ex::contains_const) {
        return Some(add(fl, cx)).filter(|s| matches!(s, Ex::Add(_)));
    }
    fl.sort_by(|a, b| add_term_cmp(a, b, cx.view));
    Some(Ex::Add(fl))
}

/// A product's sign carrier (`sign_place`): where a sign toggle lives.
#[derive(Clone, Copy)]
enum Carrier {
    Inf(usize),
    Free,
    Absorb(usize),
    Coeff,
}

/// A product's sign orbit (`sign_place`): its factors and coefficient (and the coefficient's
/// negation), its sign-trade sites with their flips, the classes they form, the zero checks, the
/// prices of every factor and flip, and the sign carrier.
struct SignOrbit<'a> {
    out: &'a [Ex],
    coeff: &'a Rat,
    nc: &'a Rat,
    site_flips: &'a [(usize, Ex)],
    groups: &'a [ClassGroup],
    checks: &'a [(bool, Vec<Option<bool>>)],
    base_price: &'a [u64],
    flip_price: &'a [u64],
    carrier: Carrier,
    assemble: &'a dyn Fn(Vec<Ex>, &Rat) -> Ex,
}

/// One orientation of a group of sites (`SignOrbit::group_options`): the sites it flips, their
/// price, whether it moves the product's sign, and its sites as they then stand (the tie order).
#[derive(Clone)]
struct SiteOption {
    flips: Vec<usize>,
    cost: u64,
    odd: bool,
    key: Vec<Ex>,
    /// The signs of its sites as they then stand (`sign_key`, summed).
    signs: (u32, u32),
}

impl SignOrbit<'_> {
    /// The product the flipped sites give, the carrier taking the sign their count moves, and
    /// whether its coefficient is negative; `None` where the carrier cannot take it.
    fn candidate(&self, flipped: &[bool], cx: &Cx) -> Option<(Ex, bool)> {
        let mut factors = self.out.to_vec();
        let mut toggled = false;
        for (bit, (pos, nf)) in self.site_flips.iter().enumerate() {
            if flipped[bit] {
                factors[*pos] = nf.clone();
                toggled = !toggled;
            }
        }
        let c = match self.carrier {
            Carrier::Coeff if toggled => self.nc.clone(),
            Carrier::Coeff => self.coeff.clone(),
            Carrier::Inf(i) => {
                if toggled {
                    factors[i] = match &factors[i] {
                        Ex::PosInf => Ex::NegInf,
                        _ => Ex::PosInf,
                    };
                }
                self.coeff.clone()
            }
            // The refit eats the toggle: every configuration is value-equal as a fitted family.
            // The whole sign dimension goes into the refit: coefficient magnitude only, bare
            // infinities positive.
            Carrier::Free => {
                for f in factors.iter_mut() {
                    if matches!(f, Ex::NegInf) {
                        *f = Ex::PosInf;
                    }
                }
                if self.coeff.is_negative() {
                    self.nc.clone()
                } else {
                    self.coeff.clone()
                }
            }
            Carrier::Absorb(i) => {
                if toggled {
                    factors[i] = flip_carrier_sum(&factors[i], cx)?;
                }
                self.coeff.clone()
            }
        };
        let negative = c.is_negative();
        Some(((self.assemble)(factors, &c), negative))
    }

    /// Each group's orientations: the class written along its key or against it, every site
    /// spelled the other way flipping. An orientation that flips a site onto the base of another
    /// factor is none: the bag would be an unmerged spelling that collection merges, so the
    /// product would not read back as itself (two factors the entry already holds over one base
    /// are how `mul()` emits them, and block nothing).
    fn group_options(&self, cx: &Cx) -> Vec<Vec<SiteOption>> {
        let view = cx.view;
        let mut site_of: Vec<Option<usize>> = vec![None; self.out.len()];
        let mut group_of = vec![0usize; self.site_flips.len()];
        for (bit, (pos, _)) in self.site_flips.iter().enumerate() {
            site_of[*pos] = Some(bit);
        }
        for (g, (_, members)) in self.groups.iter().enumerate() {
            for (bit, _) in members {
                group_of[*bit] = g;
            }
        }
        let state = |bit: usize, flipped: bool| -> &Ex {
            if flipped {
                &self.site_flips[bit].1
            } else {
                &self.out[self.site_flips[bit].0]
            }
        };
        self.groups
            .iter()
            .enumerate()
            .map(|(g, (_, members))| {
                [false, true]
                    .into_iter()
                    .filter_map(|against| {
                        let flips: Vec<usize> = members
                            .iter()
                            .filter(|(_, other)| *other != against)
                            .map(|(bit, _)| *bit)
                            .collect();
                        for &bit in &flips {
                            let (pos, nf) = &self.site_flips[bit];
                            for j in (0..self.out.len()).filter(|j| j != pos) {
                                let hit = match site_of[j] {
                                    Some(b) if group_of[b] == g => {
                                        same_collection_base(nf, state(b, flips.contains(&b)), cx)
                                    }
                                    Some(b) => {
                                        same_collection_base(nf, state(b, false), cx)
                                            || same_collection_base(nf, state(b, true), cx)
                                    }
                                    None => same_collection_base(nf, &self.out[j], cx),
                                };
                                if hit {
                                    return None;
                                }
                            }
                        }
                        let cost = members.iter().fold(0u64, |acc, (bit, _)| {
                            acc.saturating_add(if flips.contains(bit) {
                                self.flip_price[*bit]
                            } else {
                                self.base_price[self.site_flips[*bit].0]
                            })
                        });
                        let mut key: Vec<Ex> = members
                            .iter()
                            .map(|(bit, _)| state(*bit, flips.contains(bit)).clone())
                            .collect();
                        key.sort_by(|a, b| cmp_ex(a, b, view));
                        let signs = key
                            .iter()
                            .map(|f| sign_key(f, view))
                            .fold((0, 0), |a, k| (a.0 + k.0, a.1 + k.1));
                        Some(SiteOption {
                            odd: flips.len() % 2 == 1,
                            flips,
                            cost,
                            key,
                            signs,
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Does a configuration (the sites it flips) keep every value the product keeps at a class's
    /// zero (`zero_limit`)? A site whose own count is not read may not flip.
    fn keeps(&self, checks: &[usize], flipped: &[bool]) -> bool {
        checks.iter().all(|&c| {
            let (now, toggles) = &self.checks[c];
            let mut off = *now;
            for (bit, t) in toggles.iter().enumerate() {
                if flipped[bit] {
                    match t {
                        Some(t) => off ^= t,
                        None => return false,
                    }
                }
            }
            !off
        })
    }
}

/// THE SIGN ORBIT'S ARGMIN, for any number of classes. A product's price is the sum of its
/// factors' prices and the carrier's (`complexity`'s Mul arm; pieces of a literal, `is_piece`,
/// price together, which a site that is one reads as its own price), so the cheapest spelling is
/// separable up to the one coupling the classes share, the PARITY of the sites they flip (the
/// sign the carrier takes). Each class flips as a whole and prices as the sum of its members; a
/// zero check (`zero_limit`) constrains the classes whose sites it counts, and the classes one
/// check couples are decided together (a component, its orientations enumerated: in practice a
/// check counts the sites of its own class only). Per component its cheapest orientation, and per
/// parity of the sign the one component whose switch to the other parity costs least: the
/// cheapest product of either sign, priced exactly, the cheaper winning (then a non-negative
/// coefficient, then the structurally smaller product). Ties within a parity resolve by a fixed
/// order that is a function of the orbit, never of the entry spelling: each component's
/// orientation by its sites as they then stand, the switch by the components' smallest site.
/// `None` where no configuration keeps the product's values at its zeros, or a component
/// couples more than `COMPONENT_LIMIT` classes: the caller keeps the entry spelling.
fn sign_place_groups(o: &SignOrbit, cx: &Cx) -> Option<Ex> {
    const COMPONENT_LIMIT: usize = 12;
    let view = cx.view;
    let n = o.site_flips.len();
    let options = o.group_options(cx);
    if options.iter().any(Vec::is_empty) {
        return None;
    }
    let mut group_of = vec![0usize; n];
    for (g, (_, members)) in o.groups.iter().enumerate() {
        for (bit, _) in members {
            group_of[*bit] = g;
        }
    }
    // Components: the groups each check couples (the sites whose flip changes its count, or whose
    // count is not read), joined.
    let mut root: Vec<usize> = (0..o.groups.len()).collect();
    fn find(root: &mut [usize], g: usize) -> usize {
        let mut r = g;
        while root[r] != r {
            r = root[r];
        }
        root[g] = r;
        r
    }
    let mut check_groups: Vec<Vec<usize>> = Vec::with_capacity(o.checks.len());
    for (now, toggles) in o.checks {
        let mut gs: Vec<usize> = toggles
            .iter()
            .enumerate()
            .filter(|(_, t)| **t != Some(false))
            .map(|(bit, _)| group_of[bit])
            .collect();
        gs.sort_unstable();
        gs.dedup();
        if gs.is_empty() && *now {
            return None;
        }
        for w in gs.windows(2) {
            let (a, b) = (find(&mut root, w[0]), find(&mut root, w[1]));
            root[a.max(b)] = a.min(b);
        }
        check_groups.push(gs);
    }
    let mut components: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    for g in 0..o.groups.len() {
        let r = find(&mut root, g);
        match components
            .iter_mut()
            .find(|(gs, _)| find(&mut root, gs[0]) == r)
        {
            Some((gs, _)) => gs.push(g),
            None => components.push((vec![g], Vec::new())),
        }
    }
    for (c, gs) in check_groups.iter().enumerate() {
        if let Some(&g) = gs.first() {
            let r = find(&mut root, g);
            if let Some((_, cs)) = components
                .iter_mut()
                .find(|(groups, _)| find(&mut root, groups[0]) == r)
            {
                cs.push(c);
            }
        }
    }
    // Each component's orientations, cheapest first (then by its sites as they stand), and its
    // rank: its smallest site in either orientation.
    let mut decided: Vec<(Ex, Vec<SiteOption>)> = Vec::with_capacity(components.len());
    for (gs, cs) in &components {
        if gs.len() > COMPONENT_LIMIT {
            return None;
        }
        let mut opts: Vec<SiteOption> = vec![SiteOption {
            flips: Vec::new(),
            cost: 0,
            odd: false,
            key: Vec::new(),
            signs: (0, 0),
        }];
        for &g in gs {
            opts = opts
                .into_iter()
                .flat_map(|a| {
                    options[g].iter().map(move |b| {
                        let mut flips = a.flips.clone();
                        flips.extend(b.flips.iter().copied());
                        let mut key = a.key.clone();
                        key.extend(b.key.iter().cloned());
                        SiteOption {
                            flips,
                            cost: a.cost.saturating_add(b.cost),
                            odd: a.odd ^ b.odd,
                            key,
                            signs: (a.signs.0 + b.signs.0, a.signs.1 + b.signs.1),
                        }
                    })
                })
                .collect();
        }
        if !cs.is_empty() {
            opts.retain(|opt| {
                let mut flipped = vec![false; n];
                for &bit in &opt.flips {
                    flipped[bit] = true;
                }
                o.keeps(cs, &flipped)
            });
        }
        if opts.is_empty() {
            return None;
        }
        opts.sort_by(|a, b| {
            (a.cost, a.signs)
                .cmp(&(b.cost, b.signs))
                .then_with(|| cmp_vec(&a.key, &b.key, view))
        });
        let rank = gs
            .iter()
            .flat_map(|g| o.groups[*g].1.iter())
            .flat_map(|(bit, _)| [&o.out[o.site_flips[*bit].0], &o.site_flips[*bit].1])
            .min_by(|a, b| cmp_ex(a, b, view))
            .cloned()
            .unwrap_or(Ex::Num(Rat::ONE));
        decided.push((rank, opts));
    }
    decided.sort_by(|a, b| cmp_ex(&a.0, &b.0, view));
    // Per parity, the cheapest configuration: every component's cheapest orientation, and where
    // their parity is the other one, the cheapest switch of one component to its cheapest
    // orientation of the other parity (among switches of one cost, the one whose product is
    // structurally smallest).
    let best: Vec<&SiteOption> = decided.iter().map(|(_, opts)| &opts[0]).collect();
    let build = |pick: &[&SiteOption]| {
        let mut flipped = vec![false; n];
        for opt in pick {
            for &bit in &opt.flips {
                flipped[bit] = true;
            }
        }
        o.candidate(&flipped, cx)
    };
    // Where the sign is absorbed (a bare `Const`, which absorbs every constant factor of its
    // product, or a zero coefficient) the parity is no constraint: every class takes its own
    // cheapest orientation, ties by its sites as they then stand, whatever else the product holds.
    if matches!(o.carrier, Carrier::Free) {
        return build(&best).map(|(e, _)| e);
    }
    let parity = best.iter().fold(false, |p, opt| p ^ opt.odd);
    type Delta = (u64, i64, i64);
    let switches: Vec<(Delta, usize, &SiteOption)> = decided
        .iter()
        .enumerate()
        .filter_map(|(i, (_, opts))| {
            let other = opts.iter().find(|opt| opt.odd != opts[0].odd)?;
            let delta = (
                other.cost - opts[0].cost,
                i64::from(other.signs.0) - i64::from(opts[0].signs.0),
                i64::from(other.signs.1) - i64::from(opts[0].signs.1),
            );
            Some((delta, i, other))
        })
        .collect();
    let least = switches.iter().map(|(delta, _, _)| *delta).min();
    let mut chosen: Option<(u64, Ex)> = None;
    for target in [false, true] {
        let built = if target == parity {
            build(&best)
        } else {
            switches
                .iter()
                .filter(|(delta, _, _)| Some(*delta) == least)
                .filter_map(|(_, i, other)| {
                    let mut pick = best.clone();
                    pick[*i] = other;
                    build(&pick)
                })
                .min_by(|a, b| tie_cmp(&a.0, &b.0, view))
        };
        let Some((cand, _)) = built else {
            continue;
        };
        let mu = complexity(&cand, view);
        let better = match &chosen {
            None => true,
            Some((bmu, bex)) => {
                mu < *bmu || (mu == *bmu && tie_cmp(&cand, bex, view) == Ordering::Less)
            }
        };
        if better {
            chosen = Some((mu, cand));
        }
    }
    chosen.map(|(_, e)| e)
}

/// The debug build's cross-check of `sign_place_groups`: the cheapest price over every
/// configuration of whole classes, under the same orientations and checks.
#[cfg(debug_assertions)]
fn sign_place_enumerated(o: &SignOrbit, cx: &Cx) -> Option<u64> {
    let n = o.site_flips.len();
    let options = o.group_options(cx);
    let all: Vec<usize> = (0..o.checks.len()).collect();
    let mut best: Option<u64> = None;
    for code in 0usize..(1usize << o.groups.len()) {
        let mut flipped = vec![false; n];
        let mut allowed = true;
        for (g, (_, members)) in o.groups.iter().enumerate() {
            let against = code & (1 << g) != 0;
            let flips: Vec<usize> = members
                .iter()
                .filter(|(_, other)| *other != against)
                .map(|(bit, _)| *bit)
                .collect();
            allowed &= options[g].iter().any(|opt| opt.flips == flips);
            for bit in flips {
                flipped[bit] = true;
            }
        }
        if !allowed || !o.keeps(&all, &flipped) {
            continue;
        }
        if let Some((cand, _)) = o.candidate(&flipped, cx) {
            let mu = complexity(&cand, cx.view);
            best = Some(best.map_or(mu, |b| b.min(mu)));
        }
    }
    best
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
/// extended reals: the sum itself; Pow(S, odd n >= 3); rootn(S, odd m >= 3) (the
/// SIGNED total root is an odd bijection); the eight odd functions (`odd_fun`); and
/// Pow(c, odd n >= 3) of either of the last two over a sum. A negative odd power of any of them
/// is one too, its identity failing only at the pole: there the two sides disagree, so the sign
/// of the pole's infinity is no value the answer keeps (reading A). A product flips the sites of
/// one class only together (`sign_place`), so a one-signed pole keeps its infinity.
///
/// MEMOIZED for the dynamic extent of the outermost call: the flip is a pure function of
/// `(f, cx)` and the thread's number domain, and computing it re-enters itself FOUR times per
/// nesting level -- it negates each term through `term_join`, whose site test
/// (`is_sign_trade_site`) and `mul()` assembly (`sign_place`) each ask the flip of the nested
/// sum, and the files-bare gate (`flipped_orientation_wins`) negates the flipped terms again
/// through the same two. Without the memo a sum nested `d` deep cost `4^d` flips (measured:
/// `x1*(x1 + (N + 1)/x1)` nested 9 deep took 7 s with the search off, all of it here).
fn sign_trade_flip(f: &Ex, cx: &Cx) -> Option<Ex> {
    trade_flip(f, cx, false)
}

/// The flip of `f` as a factor of a product that holds a bare `Const` (`sign_place`). A
/// `<constant>` absorbs every constant factor of its product, and a class's orientation sign is
/// one, the constant factor -1: beside one, the orientation of EVERY class is free, so the classes
/// whose sign an absorption owner holds elsewhere (a sum holding a `Const`, one with a
/// negation-absorbing member, one with an infinity) are sites too, their sign the constant's.
fn refit_trade_flip(f: &Ex, cx: &Cx) -> Option<Ex> {
    trade_flip(f, cx, true)
}

/// [`sign_trade_flip`] and [`refit_trade_flip`], memoized; `refit`: the product's sign is a bare
/// `Const`'s.
fn trade_flip(f: &Ex, cx: &Cx, refit: bool) -> Option<Ex> {
    if !matches!(f, Ex::Add(_) | Ex::Pow(..) | Ex::Fun(..)) {
        return None;
    }
    let key = (flip_cx_key(cx), f.clone(), refit);
    let outer = FLIP_MEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.is_none() {
            *m = Some(rustc_hash::FxHashMap::default());
            true
        } else {
            false
        }
    });
    let _scope = FlipScope(outer);
    if let Some(hit) = FLIP_MEMO.with(|m| m.borrow().as_ref().and_then(|t| t.get(&key).cloned())) {
        return hit;
    }
    let r = sign_trade_flip_raw(f, cx, refit);
    FLIP_MEMO.with(|m| {
        if let Some(t) = m.borrow_mut().as_mut() {
            t.insert(key, r.clone());
        }
    });
    r
}

/// The memo key's context part: everything `sign_trade_flip` reads besides the factor -- the
/// token view, the four certificate closures (by identity), the mode, the two fold switches and
/// the thread's number domain. All of them outlive the outermost call, so no identity is reused
/// within the memo's extent.
type FlipCxKey = (usize, [usize; 5], u8);
type FlipMemo = rustc_hash::FxHashMap<(FlipCxKey, Ex, bool), Option<Ex>>;

/// The opposite-sums decision of a product (`collect_opposite_sums`), keyed like a flip: by the
/// context, the sign host and the factors as they arrived. It lives exactly as long as the flip
/// memo, so a parse or a re-walk decides each distinct product once however often it occurs.
type OppositeKey = (FlipCxKey, HostKey, Vec<Ex>);

thread_local! {
    static FLIP_MEMO: std::cell::RefCell<Option<FlipMemo>> = const { std::cell::RefCell::new(None) };
    static OPPOSITE_MEMO: std::cell::RefCell<rustc_hash::FxHashMap<OppositeKey, Ex>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::default());
}

/// Drops the memos when the outermost `sign_trade_flip`, opposite-sums decision, parse or
/// re-walk returns (or unwinds).
pub(crate) struct FlipScope(bool);

impl FlipScope {
    /// Holds the memos open until the guard drops (unless an enclosing scope already does), so
    /// the flips of one decision are computed once however many products it prices, and a parse
    /// or re-walk decides each distinct product once.
    pub(crate) fn open() -> Self {
        let outer = FLIP_MEMO.with(|m| {
            let mut m = m.borrow_mut();
            if m.is_none() {
                *m = Some(rustc_hash::FxHashMap::default());
                true
            } else {
                false
            }
        });
        FlipScope(outer)
    }
}

impl Drop for FlipScope {
    fn drop(&mut self) {
        if self.0 {
            FLIP_MEMO.with(|m| *m.borrow_mut() = None);
            OPPOSITE_MEMO.with(|m| m.borrow_mut().clear());
        }
    }
}

fn flip_cx_key(cx: &Cx) -> FlipCxKey {
    fn id(c: Option<&dyn Fn(&Ex) -> bool>) -> usize {
        c.map_or(0, |f| f as *const dyn Fn(&Ex) -> bool as *const () as usize)
    }
    let mode = match cx.mode {
        RuleMode::Default => 0u8,
        RuleMode::Real => 1,
        RuleMode::Permissive => 2,
    };
    let bits = mode
        | (u8::from(cx.fold_f64) << 2)
        | (u8::from(cx.sentinels_expired) << 3)
        | (u8::from(super::rat::f64_numbers()) << 4);
    (
        cx.view as *const TokenView as usize,
        [
            id(cx.cert_fin),
            id(cx.cert_finnz),
            id(cx.cert_nzae),
            id(cx.cert_nce),
            id(cx.cert_region),
        ],
        bits,
    )
}

fn sign_trade_flip_raw(f: &Ex, cx: &Cx, refit: bool) -> Option<Ex> {
    let flip_sum = |ts: &[Ex]| -> Option<Vec<Ex>> {
        // Exclusions = the absorption owners' classes: Const-bearing (H-020),
        // absorbing-member (H-030), and BARE-infinity terms (the inf carries the
        // sign itself, H-014 -- `term_absorbs_negation` only matches inf-carrying
        // PRODUCTS, so the bare member needs its own test; missing it let an
        // odd-fun trade flip an inf-bearing sum and inflate mu: fuzz row 131227,
        // P5, the one violation in 200k). Beside a bare `Const` (`refit`) the
        // constant owns every class's sign, theirs too: such a sum negates through
        // `add`, which files it bare (`primitive_sum` never wraps one).
        if ts.iter().any(Ex::contains_const)
            || ts.iter().any(term_absorbs_negation)
            || ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
        {
            if !refit {
                return None;
            }
            let negated = ts
                .iter()
                .map(|t| negate_term(t, cx))
                .collect::<Option<Vec<Ex>>>()?;
            return match add(negated, cx) {
                Ex::Add(v) => Some(v),
                _ => None,
            };
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
        if flipped_orientation_wins(&out, Wrapper::Flipped, Tie::Keep, cx) {
            return None;
        }
        out.sort_by(|a, b| add_term_cmp(a, b, cx.view));
        Some(out)
    };
    let odd_int_ge3 = |e: &Ex| -> bool {
        matches!(e, Ex::Num(r) if r.is_odd_integer() && r.cmp_int(3) != Ordering::Less)
    };
    // A negative odd power's pole is two-signed, so its orientation is not pinned there
    // (`-1/(x1 - 3)` is `1/(3 - x1)`).
    let odd_power = |e: &Ex| -> bool {
        odd_int_ge3(e) || matches!(e, Ex::Num(r) if r.is_odd_integer() && r.is_negative())
    };
    match f {
        Ex::Add(ts) => flip_sum(ts).map(Ex::Add),
        Ex::Pow(b, e) if odd_power(e) => {
            let flipped = match &**b {
                Ex::Add(ts) => flip_sum(ts).map(|s| Ex::Pow(Box::new(Ex::Add(s)), e.clone())),
                // An odd power of an odd head of a negatable argument (`class_base`) is one too:
                // `f(-u)^n = -f(u)^n`, for a negative `n` as for a positive one. A constant
                // argument keeps its sign on its literal under a power (`1/tanh(-2)`), as the
                // sign key files it.
                Ex::Fun(_, args)
                    if args.first().is_some_and(|a| {
                        a.has_measure_space(cx.view) || a.contains_wildcard(cx.view)
                    }) =>
                {
                    trade_flip(b, cx, refit).map(|nb| Ex::Pow(Box::new(nb), e.clone()))
                }
                _ => None,
            }?;
            // A negative power's pole pins the orientation where the base's zero set may hold a
            // region: there the inverse is infinite on the region, a value the answer keeps. Asked
            // last, of a base whose flip exists.
            (cx.lossy() || !matches!(&**e, Ex::Num(r) if r.is_negative()) || zero_set_null(b, cx))
                .then_some(flipped)
        }
        Ex::Fun(op, args)
            if args.len() == 2 && cx.view.tok_is(*op, "rootn") && odd_int_ge3(&args[1]) =>
        {
            let Ex::Add(ts) = &args[0] else {
                return literal_multiple_flip(f, cx);
            };
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
            _ => literal_multiple_flip(f, cx),
        },
        Ex::Fun(..) => literal_multiple_flip(f, cx),
        _ => None,
    }
}

/// The other spelling of an odd function or odd root of a literal multiple
/// (`odd_of_literal_multiple`): the argument negated through `mul()`, rebuilt through `fun()`;
/// `None` where the rebuild leaves the shape (a fold, or a negation the argument's own sites took
/// instead of the literal).
fn literal_multiple_flip(f: &Ex, cx: &Cx) -> Option<Ex> {
    let c = odd_of_literal_multiple(f, cx.view)?;
    let Ex::Fun(op, args) = f else { return None };
    let mut items = vec![Ex::Num(c.checked_neg()?)];
    match &args[0] {
        Ex::Num(_) => {}
        Ex::Mul(v) if matches!(v.first(), Some(Ex::Num(_))) => items.extend(v[1..].iter().cloned()),
        u => items.push(u.clone()),
    }
    let mut flipped_args = args.clone();
    flipped_args[0] = mul(items, cx);
    let flipped = fun(*op, flipped_args, cx);
    matches!(
        class_base(&flipped, cx.view),
        Some(ClassBase { head: Some(_), .. })
    )
    .then_some(flipped)
}

// F72 (2026-08-09): the residual equal-mu same-coefficient-sign tie in the
// sign-placement owner breaks on `cmp_ex` -- the ONE canonical total order, which
// compares Leaf/Fun tokens by their STRINGS. The dedicated `ex_struct_cmp` this
// replaced compared them by raw token ID, i.e. by INTERNING ORDER, so on a fully
// mu-tied orbit (the two mirror pairings price alike) the winner was a function of which
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
fn rejoin_reciprocals(settled: Ex, signed: bool, cx: &Cx) -> Ex {
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
            (r.is_integer() && r.is_negative()).then_some(i)
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
            if *r == Rat::NEG_ONE {
                (**b).clone()
            } else {
                pow((**b).clone(), Ex::Num(r.checked_neg().unwrap()), cx)
            }
        })
        .collect();
    let inner = mul(inner_members.clone(), cx);
    // The joined base's own decision can trade a member's sign into its coefficient (a sum
    // whose other orientation is cheaper once it stands in a product); that sign leaves the base
    // (`joined_base_sign`).
    let (inner, base_sign) = joined_base_sign(inner, cx);
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
    let mut rest: Vec<Ex> = members
        .iter()
        .enumerate()
        .filter(|(i, _)| !eligible.contains(i))
        .map(|(_, f)| f.clone())
        .collect();
    if base_sign {
        // The rest of the bag takes the sign, placed by its own decision (`mul()`): the parse of the
        // joined spelling decides the same members with the base's inverted members beside them,
        // which carry no sign of their own.
        rest.push(Ex::Num(Rat::NEG_ONE));
        rest = match mul(rest, cx) {
            Ex::Mul(v) => v,
            Ex::Num(r) if r.is_one() => Vec::new(),
            f => vec![f],
        };
        // A bag that may not take a new sign (`signed`) keeps its working spelling.
        if !signed && gains_a_sign(&rest, &settled) {
            return settled;
        }
    }
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
            let inner2 = mul(vec![Ex::Num(cinv.clone()), inner.clone()], cx);
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
    let mu_settled = complexity(&settled, cx.view);
    let image = resettle(&best);
    let mu_image = complexity(&image, cx.view);
    if mu_best < mu_settled && mu_best < mu_image {
        best
    } else if base_sign
        && mu_image < mu_settled
        && (signed || !gains_a_sign(std::slice::from_ref(&image), &settled))
    {
        // The base's decision found a cheaper orientation for a member the working state holds
        // under its inverse: the re-settle image keeps it, distributed, and re-derives itself.
        image
    } else {
        settled
    }
}

/// Does `members` (a bag's members, or the bag itself) carry a negative coefficient where the bag
/// `settled` carries none?
fn gains_a_sign(members: &[Ex], settled: &Ex) -> bool {
    let negative = |f: &Ex| match f {
        Ex::Num(r) => r.is_negative(),
        Ex::Mul(v) => v.iter().any(|g| matches!(g, Ex::Num(r) if r.is_negative())),
        _ => false,
    };
    members.iter().any(negative) && !negative(settled)
}

/// The joined base without a negative coefficient of its own, and whether one left it: a sign the
/// joined base's own product placed in its coefficient leaves the base for the rest of the bag
/// (`rejoin_reciprocals`). The base is decided as a product of its own, so a sum the distributed
/// bag held under its inverse can take its cheaper orientation there and hand the sign to the
/// base's coefficient, while the bag keeps the sign it had. Kept in the base, the two signs would
/// print as `(-a)/(-(b (x - u)))`, which the parse reads as `a/(b (x - u))`, so a second call would
/// print another spelling. In the lossy context the projection runs in, the inverse of `-P` is
/// minus the inverse of `P` (they differ only at `P`'s zero), so the sign joins the rest of the bag,
/// whose own decision places it. A bag the parse reads with its sign kept (the answer, or a term of
/// a sum) takes it as it comes; inside a function's argument or a power's base, where the parse
/// moves or drops a product's sign, a bag that would gain a sign keeps its working spelling.
fn joined_base_sign(inner: Ex, cx: &Cx) -> (Ex, bool) {
    if !cx.lossy() {
        return (inner, false);
    }
    let Ex::Mul(mut v) = inner else {
        return (inner, false);
    };
    if !matches!(v.first(), Some(Ex::Num(r)) if r.is_negative()) {
        return (Ex::Mul(v), false);
    }
    let Ex::Num(r) = &v[0] else { unreachable!() };
    match r.checked_neg() {
        Some(p) if p.is_one() => {
            v.remove(0);
        }
        Some(p) => v[0] = Ex::Num(p),
        None => return (Ex::Mul(v), false),
    }
    let base = if v.len() == 1 {
        v.pop().expect("one member")
    } else {
        Ex::Mul(v)
    };
    (base, true)
}

/// The output-boundary rejoin walk (see [`rejoin_reciprocals`]): apply the reciprocal
/// join decision bottom-up to every `Mul` node of the final lossy state. STRUCTURAL
/// rebuild on purpose -- routing through the constructors would re-distribute the joins
/// (`pow`'s lossy blanket). Bags whose members changed are re-sorted under their own
/// comparator so the projected tree spells deterministically.
pub fn rejoin_projection(e: Ex, cx: &Cx) -> Ex {
    rejoin_walk(e, true, cx)
}

/// [`rejoin_projection`] at a node; `signed`: does the parse keep a sign the node's bag takes (the
/// answer itself, a term of a sum, a member of a product), or move or drop it (a function's
/// argument, a power's base or exponent)?
fn rejoin_walk(e: Ex, signed: bool, cx: &Cx) -> Ex {
    match e {
        Ex::Add(v) => {
            let mut terms: Vec<Ex> = v.into_iter().map(|x| rejoin_walk(x, true, cx)).collect();
            terms.sort_by(|a, b| add_term_cmp(a, b, cx.view));
            Ex::Add(terms)
        }
        Ex::Mul(v) => {
            let mut m: Vec<Ex> = v.into_iter().map(|x| rejoin_walk(x, true, cx)).collect();
            m.sort_by(|a, b| mul_factor_cmp(a, b, cx.view));
            rejoin_reciprocals(Ex::Mul(m), signed, cx)
        }
        Ex::Pow(b, ex) => Ex::Pow(
            Box::new(rejoin_walk(*b, false, cx)),
            Box::new(rejoin_walk(*ex, false, cx)),
        ),
        Ex::Fun(f, v) => Ex::Fun(
            f,
            v.into_iter().map(|x| rejoin_walk(x, false, cx)).collect(),
        ),
        leaf => leaf,
    }
}

/// Rebuild `base^exp` (rational or symbolic-with-count) through the power constructor so
/// integer-power distribution and exact folds apply; as a member of the product being built
/// ([`pow_member`]), whose sign placement orients it.
fn rebuild_factor(base: Ex, sym: Option<Ex>, r: Rat, cx: &Cx) -> Ex {
    match sym {
        None => pow_member(base, Ex::Num(r), cx),
        Some(e) => {
            if r.is_one() {
                pow_member(base, e, cx)
            } else {
                let scaled = mul(vec![Ex::Num(r), e], cx);
                pow_member(base, scaled, cx)
            }
        }
    }
}

/// `pow`'s licence to distribute the integer power `e` over a product of `v`: the bag it distributes
/// over (a negative coefficient folded into a sign-trade site first, F63), or `None` where it
/// refuses. The divisor parse and `canon` ask it too (`distributes_power`, `inverted_members`),
/// so a quotient is distributed exactly where `pow` would distribute it.
/// Does `pow` distribute the integer power `e` over the product of `v` (`power_bag`'s licence)?
pub(crate) fn distributes_power(v: &[Ex], e: &Rat, cx: &Cx) -> bool {
    power_bag(v, e, cx).is_some()
}

fn power_bag(v: &[Ex], e: &Rat, cx: &Cx) -> Option<Vec<Ex>> {
    let licensed = !e.is_negative()
        || e.is_even_integer()
        || inverts_a_signed_zero(v, cx)
        || v.iter().any(holds_infinity)
        || v.iter().all(|f| zero_set_null(f, cx))
        || (v.iter().any(|f| holds_odd_root(f, cx))
            && v.iter()
                .all(|f| cx.nz_ae_licensed(f) || zero_set_null_by_form(f, cx)))
        || v.iter().all(|f| cx.certainly_nonneg(f));
    if !licensed {
        return None;
    }
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
    let bag: &[Ex] = if e.is_negative() && e.is_odd_integer() && !inverts_a_signed_zero(v, cx) {
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
                        let mut w = v.to_vec();
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
    Some(bag.to_vec())
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
/// * a power that is a sign-trade site takes the orientation it takes as a product's one factor
///   (`lone_site`).
pub fn pow(base: Ex, exp: Ex, cx: &Cx) -> Ex {
    match pow_member(base, exp, cx) {
        node @ Ex::Pow(..) => lone_site(node, cx),
        other => other,
    }
}

/// [`pow`] for a member of a product, whose sign placement decides the member's orientation with
/// the product's other members (`sign_place`): the same power, a sign-trade site left in the
/// orientation it was built in.
pub(crate) fn pow_member(base: Ex, exp: Ex, cx: &Cx) -> Ex {
    let _domain = super::rat::number_domain(cx.f64_numbers());
    // A literal base or exponent beside a non-literal one LANDS here (permissive's ρ,
    // [`land`]); a power of two literals folds exactly or stays as written.
    let (base, exp) = if matches!(base, Ex::Num(_)) != matches!(exp, Ex::Num(_)) {
        (land_operand(base, cx), land_operand(exp, cx))
    } else {
        (base, exp)
    };
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
        if e.is_integer() {
            if let Some(r) = b.checked_pow_integer(e) {
                return Ex::Num(r);
            }
        } else if !b.is_negative() {
            // b^(p/q): exact iff the q-th root of b is rational.
            if let Some(root) = b.checked_root_integer(&e.denom()) {
                if let Some(r) = root.checked_pow_integer(&e.numer()) {
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
    if !cx.lossy() || cx.sentinels_expired {
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
    // A NEGATED SITE UNDER A POWER IS THE POWER OF ITS FLIP: `-F` and its flip (`sign_trade_flip`)
    // are one class in its two orientations, one value wherever the flip is a site's other spelling,
    // so `(-F)^s` is `flip(F)^s` for every exponent. A site standing alone may take the negated
    // orientation (`lone_site`); a power over it reads the class as it reads the site spelled
    // without the sign: `(S^3)^-1` is `S^-3` and `(S^3)^(1/2)` is `S^(3/2)` whichever orientation
    // `S^3` stands in. (A negative coefficient beside other members folds into a site where the
    // power distributes, `power_bag`.)
    // An odd negative power inverts the zero of a site that vanishes on a region with the sign
    // its orientation gives it (`region_signed_zero`): `1/-(S^3)` is `-inf` there and
    // `1/flip(S)^3` is `+inf`, so the power distributes instead (`inverts_a_signed_zero`).
    let inverts = matches!(&exp, Ex::Num(s) if s.is_negative() && s.is_odd_integer());
    if let Ex::Mul(v) = &base {
        if let [Ex::Num(c), f] | [f, Ex::Num(c)] = v.as_slice() {
            if *c == Rat::NEG_ONE && !(inverts && region_signed_zero(f, cx)) {
                if let Some(flipped) = sign_trade_flip(f, cx) {
                    return pow_member(flipped, exp, cx);
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
    // certificate; a polynomial or an analytic function of one passes by its form,
    // `zero_set_null_by_form`, which the certificate-free parse reads too), a bag with an
    // infinite member (never zero), or an all-certainly-nonneg bag (at a zero every
    // co-factor is >= 0, so both spellings agree on +inf). Refusal keeps Pow(Mul[..], n):
    // on the refused set the two spellings are genuinely DIFFERENT functions, so
    // canonicalization must not identify them. The licence is `power_bag`.
    if let (Ex::Mul(v), Ex::Num(e)) = (&base, &exp) {
        if e.is_integer() {
            if let Some(bag) = power_bag(v, e, cx) {
                let parts: Vec<Ex> = bag
                    .into_iter()
                    .map(|f| pow_member(f, Ex::Num(e.clone()), cx))
                    .collect();
                return mul(parts, cx);
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
            if e.is_integer() && e.cmp_int(-1) == Ordering::Less {
                let m = e.checked_neg().expect("a negative integer negates");
                let parts: Vec<Ex> = v
                    .iter()
                    .cloned()
                    .map(|f| pow_member(f, Ex::Num(m.clone()), cx))
                    .collect();
                return pow_member(mul(parts, cx), Ex::int(-1), cx);
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
            let licence = if cx.lossy() {
                true
            } else if r.is_integer() {
                s.is_integer()
                    || (r.is_odd_integer()
                        && (r.signum() > 0 || cx.never_infinite(inner_base))
                        && r.checked_mul(s).is_some_and(|p| !p.is_integer()))
                    || cx.certainly_nonneg(inner_base)
            } else {
                let rs = r.checked_mul(s);
                rs.is_some_and(|p| !p.is_integer()) || cx.certainly_nonneg(inner_base)
            };
            if licence {
                if let Some(p) = r.checked_mul(s) {
                    return pow_member(inner_base.clone(), Ex::Num(p), cx);
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
            if let (Ex::Num(idx), true) = (&rargs[1], e.is_integer()) {
                let n = e.abs();
                if n.cmp_int(2) != Ordering::Less
                    && *idx == n
                    && (n.is_odd_integer() || cx.certainly_nonneg(&rargs[0]))
                {
                    if !e.is_negative() {
                        return rargs[0].clone();
                    }
                    return pow_member(rargs[0].clone(), Ex::int(-1), cx);
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
                if idx.is_odd_integer() && idx.cmp_int(3) != Ordering::Less {
                    if let Some(t) = idx.checked_inv().and_then(|inv| s.checked_mul(&inv)) {
                        return pow_member(rargs[0].clone(), Ex::Num(t), cx);
                    }
                }
            }
        }
    }
    // A NEGATIVE EVEN UNIT FRACTION over a `rootn` whose index has no known parity (f64 mode,
    // beyond 2^53; review M of #65): `rootn(b, m)^(-1/n) -> 1/rootn(rootn(b, m), n)` for even
    // n. The odd-root composition above needs the parity and refuses, while the printed
    // `inv pow rootn b m 1/n` re-reads through the parity-free even-unit-fraction and
    // root-composition arms; construction takes that route directly. Sound: on R = rootn(b, m)
    // >= 0 both sides are 1/R^(1/n), on R < 0 both are NaN, at R = 0 both are +inf.
    if let (Ex::Fun(rop, rargs), Ex::Num(s)) = (&base, &exp) {
        if rargs.len() == 2 && cx.view.tok_is(*rop, "rootn") && s.is_negative() && !s.is_integer() {
            if let Ex::Num(idx) = &rargs[1] {
                let n = s.denom();
                if idx.is_integer()
                    && idx.parity().is_none()
                    && s.numer() == Rat::NEG_ONE
                    && n.is_even_integer()
                {
                    if let Some(unit) = s.checked_neg() {
                        let root = pow_member(base.clone(), Ex::Num(unit), cx);
                        return pow_member(root, Ex::int(-1), cx);
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
                let mag = bv.abs().cmp_int(1);
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
                return pow_member((**inner).clone(), flipped, cx);
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
                return pow_member(r, positive, cx);
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
                return pow_member(r, (**ie).clone(), cx);
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
        let n = e.denom();
        if e.numer().is_one() && n.is_even_integer() && n.cmp_int(2) != Ordering::Less {
            return fun(cx.view.intern("rootn"), vec![base, Ex::Num(n)], cx);
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
        if r.is_even_integer()
            && !r.is_zero()
            && !ts.iter().any(Ex::contains_const)
            && !ts.iter().any(term_absorbs_negation)
            && !ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
            && flipped_orientation_wins(ts, Wrapper::Neither, Tie::Lex, cx)
        {
            if let Some(mut flipped) = ts
                .iter()
                .map(|t| negate_term(t, cx))
                .collect::<Option<Vec<Ex>>>()
            {
                // FILES-BARE gate (same rule as sign_trade_flip): only adopt an
                // orientation primitive_sum itself would file bare.
                if !flipped_orientation_wins(&flipped, Wrapper::Flipped, Tie::Keep, cx) {
                    flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
                    return pow_member(Ex::Add(flipped), exp, cx);
                }
            }
        }
    }
    // The same orientation for an odd function of a sum under an even exponent: f(-S)^2k =
    // (-f(S))^2k = f(S)^2k exactly, so `tanh(x2 - x1)^2` and `tanh(x1 - x2)^2` are one value,
    // filed with the argument the sum itself would take.
    if let (Ex::Fun(op, args), Ex::Num(r)) = (&base, &exp) {
        let odd = (args.len() == 1 && odd_fun(cx.view, *op))
            || (args.len() == 2
                && cx.view.tok_is(*op, "rootn")
                && matches!(&args[1], Ex::Num(k) if k.is_odd_integer() && k.cmp_int(3) != Ordering::Less));
        if let (true, Ex::Add(ts)) = (odd && r.is_even_integer() && !r.is_zero(), &args[0]) {
            if !ts.iter().any(Ex::contains_const)
                && !ts.iter().any(term_absorbs_negation)
                && !ts.iter().any(|x| matches!(x, Ex::PosInf | Ex::NegInf))
                && flipped_orientation_wins(ts, Wrapper::Neither, Tie::Lex, cx)
            {
                if let Some(mut flipped) = ts
                    .iter()
                    .map(|t| negate_term(t, cx))
                    .collect::<Option<Vec<Ex>>>()
                {
                    if !flipped_orientation_wins(&flipped, Wrapper::Flipped, Tie::Keep, cx) {
                        flipped.sort_by(|a, b| add_term_cmp(a, b, cx.view));
                        let mut args = args.clone();
                        args[0] = Ex::Add(flipped);
                        return pow_member(Ex::Fun(*op, args), exp, cx);
                    }
                }
            }
        }
        // And over a negative literal multiple, the positive one: `sin(-2 x1)^2` is
        // `sin(2 x1)^2`, as the class {sin(2 x1), sin(-2 x1)} is written (`collect_opposite_sums`).
        if r.is_even_integer() && !r.is_zero() {
            if let Some(Ex::Fun(fk, fa)) = positive_head(&base, cx.view) {
                return pow_member(fun(fk, fa, cx), exp, cx);
            }
        }
    }
    Ex::Pow(Box::new(base), Box::new(exp))
}

/// ONE ORIENTATION FOR A CLASS, WHEREVER IT STANDS. A sign-trade site (`sign_trade_flip`: an odd
/// power or an odd root of a sum, an odd function of one) spells its class in either
/// orientation, `f(S)` or `-f(-S)`; as a factor of a product the sign placement picks between
/// them over the whole product (`sign_place`: the cheaper, then the sign key, fewest signs and
/// then the outer ones). A site standing alone takes the spelling that placement gives it as the
/// one factor of a product, so `S^3` and `1*S^3` are one answer, and so are `1/S` and `1/(1*S)`:
/// `(x3*(x3 - x1) - x2)^3` is `-(x2 + x3*(x1 - x3))^3`, whose two signs stand further out.
fn lone_site(node: Ex, cx: &Cx) -> Ex {
    // A site standing alone may be the one an inverse reads (`1/S^3`): over a sum that vanishes on
    // a region it keeps the orientation it was written in, which the inverse keeps
    // (`region_signed_zero`, `inverts_a_signed_zero`).
    if region_signed_zero(&node, cx) {
        return node;
    }
    // One scope for the site test and the placement, so the flip is computed once.
    let _memo = FlipScope::open();
    if sign_trade_flip(&node, cx).is_some() {
        sign_place(Rat::ONE, vec![node], cx)
    } else {
        node
    }
}

/// THE PERMISSIVE LITERAL FOLD. An exact rational the constructors produced -- the quotient of two
/// 16-digit decimals, a coefficient cleared into `p/q`, a 30-digit integer -- prices under
/// mu' as the rational codeword alone once its expansion stops terminating, and the
/// emitter must spell it as that fraction, because a spelling denotes the state's EXACT
/// value (`decimal_spelling_wins` chooses among exact codewords only). On the permissive
/// tier the value is allowed to move -- that tier's licence -- so the literal folds to the
/// f64 nearest to it (as that float's shortest decimal) WHEN the exact codewords
/// ([`mu_rat_exact`]) price that spelling strictly cheaper -- the float price ([`mu_rat`]) of the
/// two is equal by construction, the fold changes what is PRINTED:
/// `2e29/426738538271436458205631863649` (196.8 bits) becomes `0.46867105279529636`
/// (57.1 bits); `1/2`, `15/37` and `4366/8875` stay, their fractions being cheaper than
/// any 17-digit float; a literal that already IS a shortest f64 spelling is its own fold
/// and never moves, so the drawn constants of a training stream are untouched, and the fold
/// is idempotent. It is applied where a literal lands ([`land`]).
///
/// THE FLOAT IT LANDS ON: the value lies between two floats, and permissive promises float
/// precision, not the nearest float, so the literal lands on whichever of the two prints
/// shorter (the cheaper shortest spelling; the nearest on a tie). A literal combined from
/// literals that already landed can sit an ulp from the single rounding of its exact sources,
/// on the float whose shortest spelling carries a digit more (`0.016213652755841255` where the
/// sources give `0.01621365275584125`); the shorter neighbour is that rounding wherever it is
/// one of the two, and no worse a float anywhere. Values whose shortest
/// spelling is no number of the f64 domain refuse (fail-closed, as the ground fold does);
/// non-finite values
/// never fold.
pub fn lossy_literal(r: &Rat) -> Option<Rat> {
    let (y, other) = r.float_bracket()?;
    // Rust's `{:?}` for f64 is the shortest string that round-trips, which is precisely the
    // spelling whose f64 reading is this value again.
    let snapped = Rat::parse_decimal(&format!("{y:?}"))?;
    if snapped == *r {
        return None;
    }
    let snapped = match other.and_then(|o| Rat::parse_decimal(&format!("{o:?}"))) {
        Some(s) if mu_rat_exact(&s) < mu_rat_exact(&snapped) => s,
        _ => snapped,
    };
    (mu_rat_exact(&snapped) < mu_rat_exact(r)).then_some(snapped)
}

/// ρ, PERMISSIVE'S ONE ROUNDING: a literal is rounded once, when it becomes part of the
/// expression, not at every operation. Arithmetic BETWEEN
/// literals stays exact -- an all-literal node folds exactly, as in every mode -- and a literal
/// is moved by [`lossy_literal`] at the one place it LANDS as an operand of a composite node:
/// a sum's constant term or a term's coefficient (`add`, `term_join`), a product's coefficient
/// (`finish_product`), a power's literal base or exponent beside a non-literal one (`pow`), a
/// function's argument (`fun`), and the root of a state (the walk, [`land_root`]). So
/// `3*(1/3 + 1e-20)` folds exactly to `1 + 3e-20` and rounds once, to `1`; rounding inside
/// each operation would give `0.9999999999999999`.
///
/// ρ is the identity outside permissive, idempotent (a moved literal is a shortest float
/// spelling, which [`lossy_literal`] never moves), and never raises a price: the float cap prices
/// a literal and its nearest float alike ([`mu_rat`]), and the shorter neighbour costs less, so
/// the reduction ordering, termination and every descent argument are unchanged, and every state
/// the walk holds prints the codeword it is priced at. It moves a value to one of the two floats
/// around it, less than an ulp, inside permissive's licence; it never creates an infinity or a
/// zero (non-finite values refuse, and both floats around a nonzero value are nonzero).
#[inline]
pub fn land(r: Rat, cx: &Cx) -> Rat {
    if !cx.lossy() {
        return r;
    }
    match lossy_literal(&r) {
        Some(s) => s,
        None => r,
    }
}

/// [`land`] on a literal operand; any other operand is returned as it is.
#[inline]
fn land_operand(e: Ex, cx: &Cx) -> Ex {
    match e {
        Ex::Num(r) => Ex::Num(land(r, cx)),
        e => e,
    }
}

/// [`land`] at the root of a state: a state that is one literal is that literal landed.
pub fn land_root(e: Ex, cx: &Cx) -> Ex {
    land_operand(e, cx)
}

/// `floor` (or `ceil`) of a rational, exactly; `None` when the result leaves `Rat`.
fn rat_floor(r: &Rat, ceil: bool) -> Option<Rat> {
    use num_integer::Integer;
    if r.is_integer() {
        return Some(r.clone());
    }
    let (p, q) = r.big_parts();
    let f = p.div_floor(&q);
    Rat::from_big(if ceil { f + 1 } else { f }, BigInt::from(1))
}

/// The exact value of `max` or `min` at rational arguments, and with `real` of `floor`, `ceil`
/// or `mod` too (the real `mod`: `a - b floor(a/b)`, NaN at `b = 0`); `None` for any other head
/// or a result that leaves `Rat`.
fn exact_step_fold(op: Tok, args: &[Ex], real: bool, cx: &Cx) -> Option<Ex> {
    let v = cx.view;
    match args {
        [Ex::Num(r)] if real && v.tok_is(op, "floor") => rat_floor(r, false).map(Ex::Num),
        [Ex::Num(r)] if real && v.tok_is(op, "ceil") => rat_floor(r, true).map(Ex::Num),
        [Ex::Num(a), Ex::Num(b)] if v.tok_is(op, "max") || v.tok_is(op, "min") => {
            let a_wins = (a.cmp_exact(b) == Ordering::Less) != v.tok_is(op, "max");
            Some(Ex::Num(if a_wins { a.clone() } else { b.clone() }))
        }
        [Ex::Num(a), Ex::Num(b)] if real && v.tok_is(op, "mod") => {
            if b.is_zero() {
                return Some(Ex::NaN);
            }
            let k = rat_floor(&a.checked_mul(&b.checked_inv()?)?, false)?;
            a.checked_add(&b.checked_mul(&k)?.checked_neg()?)
                .map(Ex::Num)
        }
        _ => None,
    }
}

pub fn fun(op: Tok, args: Vec<Ex>, cx: &Cx) -> Ex {
    let _domain = super::rat::number_domain(cx.f64_numbers());
    // A literal argument LANDS here (permissive's ρ, [`land`]).
    let args: Vec<Ex> = if cx.lossy() {
        args.into_iter().map(|a| land_operand(a, cx)).collect()
    } else {
        args
    };
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
                return Ex::Num(r.clone());
            }
        }
    }
    // THE STEP AND LATTICE HEADS OF RATIONALS ARE EXACT ARITHMETIC, so they fold here as
    // `abs` does: `max` and `min` in every mode (they pick an operand, and rounding to the
    // nearest float keeps the order, so the float of the pick is the pick of the floats), and
    // `floor`, `ceil` and `mod` (`mod(a, 0)` is NaN) in `real` only, where a literal is its
    // rational: `f64` and `permissive` read a literal at its nearest float
    // (`floor(0.99999999999999999)` is 1 there), so their ground fold computes these with the
    // deployed kernels instead.
    if !args.is_empty() && args.iter().all(|a| matches!(a, Ex::Num(_))) {
        if let Some(folded) = exact_step_fold(op, &args, matches!(cx.mode, RuleMode::Real), cx) {
            return folded;
        }
    }
    // `max` AND `min` ARE COMMUTATIVE AND IDEMPOTENT, bit for bit: their kernels are IEEE
    // 754-2019's maximum and minimum, NaN-propagating in both orders and ordering -0 below +0,
    // and so are the real functions. The arguments take the canonical order, and `max(a, a)` is
    // `a` (two `<constant>`s are independent, so a `Const`-bearing pair stays). Nesting stays
    // as written: `max(a, max(b, c))` is no bag.
    let mut args = args;
    if args.len() == 2 && (cx.view.tok_is(op, "max") || cx.view.tok_is(op, "min")) {
        if args[0] == args[1] && !args[0].contains_const() {
            return args.into_iter().next().unwrap();
        }
        if cmp_ex(&args[1], &args[0], cx.view) == Ordering::Less {
            args.swap(0, 1);
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
                && flipped_orientation_wins(ts, Wrapper::Neither, Tie::Lex, cx)
            {
                if let Some(mut flipped) = ts
                    .iter()
                    .map(|t| negate_term(t, cx))
                    .collect::<Option<Vec<Ex>>>()
                {
                    // FILES-BARE gate (same rule as sign_trade_flip).
                    if !flipped_orientation_wins(&flipped, Wrapper::Flipped, Tie::Keep, cx) {
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
                if r.is_integer() {
                    let n = r.clone();
                    if n.is_zero() {
                        return Ex::NaN;
                    }
                    if n.is_one() {
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
                    // census that came out of it (the research harness) found
                    // its sibling still blind on the very next base: `pow(exp(-1), 1/2)` was
                    // shipping as `rootn(exp(-1), 2)` at mu 20,000 where `exp(-1/2)` is
                    // 11,585. One base is a patch; the family is the principle.
                    if let Some(composed) = n
                        .checked_inv()
                        .and_then(|inv| compose_e_power(&args[0], &Ex::Num(inv), cx))
                    {
                        return composed;
                    }
                    // n >= 2 EVEN: kept as `rootn`. The conversion to `pow(x, 1/n)` used
                    // to live here; it now runs the OTHER way (see `pow`), because mu
                    // prefers the root spelling once a genuine fraction pays for its
                    // denominator -- `rootn x 2` prices 18 against `pow x 1/2` at 19.
                    if n.is_negative() {
                        // checked_neg, not `-n`: a MIN index cannot be negated (release
                        // wraps silently -> the still-negative index would recurse here
                        // forever). Unreachable while the no-MIN Rat invariant holds;
                        // refusing to fold (stay symbolic) is sound either way.
                        if let Some(pos) = n.checked_neg() {
                            let base = args.into_iter().next().unwrap();
                            let inner = fun(op, vec![base, Ex::Num(pos)], cx);
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
                                if m.is_integer() && m.cmp_int(2) != Ordering::Less {
                                    {
                                        if let Some(prod) = m.checked_mul(&n) {
                                            let inner_base = inner_args[0].clone();
                                            return fun(op, vec![inner_base, Ex::Num(prod)], cx);
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
                    if n.is_even_integer() {
                        let composed = if let Ex::Pow(inner, ex) = &args[0] {
                            match &**ex {
                                Ex::Num(k) => {
                                    n.checked_inv().and_then(|s| k.checked_mul(&s)).filter(|p| {
                                        cx.lossy()
                                            || cx.certainly_nonneg(inner)
                                            || if k.is_integer() {
                                                k.is_odd_integer()
                                                    && (k.signum() > 0 || cx.never_infinite(inner))
                                                    && !p.is_integer()
                                            } else {
                                                !p.is_integer()
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
                    if n.is_odd_integer() {
                        if let Ex::Pow(b, ex) = &args[0] {
                            let _ = b;
                            if let Ex::Num(rr) = &**ex {
                                if *rr == n {
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
                        if let Some(root) = r.checked_root_integer(&n) {
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
    // THE SIGN OF AN ODD HEAD (the one tie key, `sign_key`): `f(-u)` and `-f(u)` are one value
    // for an odd head of a negatable argument (`class_base`); the cheaper spelling stands, and on
    // an equal price the sign stands outside (fewer signs, then the outer one). A product the head
    // stands in decides again over all its members (`sign_place`). A constant argument keeps its
    // sign on its literal (`sin(-2)`).
    let head = Ex::Fun(op, args);
    if let (Some(r), Ex::Fun(_, a)) = (odd_of_literal_multiple(&head, cx.view), &head) {
        if r.is_negative() && (a[0].has_measure_space(cx.view) || a[0].contains_wildcard(cx.view)) {
            if let Some(p) = positive_head(&head, cx.view) {
                let mut v = vec![Ex::Num(Rat::NEG_ONE), p];
                v.sort_by(|x, y| mul_factor_cmp(x, y, cx.view));
                let outside = Ex::Mul(v);
                let (inside_mu, outside_mu) =
                    (complexity(&head, cx.view), complexity(&outside, cx.view));
                if outside_mu < inside_mu
                    || (outside_mu == inside_mu
                        && tie_cmp(&outside, &head, cx.view) == Ordering::Less)
                {
                    return outside;
                }
            }
        }
    }
    // A head of a sum is a class site: alone it takes the orientation it takes as a product's one
    // factor (`lone_site`). Its flip negates the sum in place, so asking it builds no head.
    if matches!(&head, Ex::Fun(_, a) if matches!(a.first(), Some(Ex::Add(_)))) {
        return lone_site(head, cx);
    }
    head
}

/// Full recursive re-canonicalization: rebuild `e` bottom-up through the canonical
/// constructors. Used on freshly substituted rule RHS instances and at the parse boundary.
pub fn canon(e: Ex, cx: &Cx) -> Ex {
    // One scope for the whole re-walk: a product met again is decided once (`FlipScope`).
    let _memo = FlipScope::open();
    canon_walk(e, cx)
}

fn canon_walk(e: Ex, cx: &Cx) -> Ex {
    match e {
        Ex::Add(v) => {
            let parts = v.into_iter().map(|x| canon_walk(x, cx)).collect();
            add(parts, cx)
        }
        Ex::Mul(v) => {
            // A member that is a product under a negative integer power gives the enclosing
            // product its own members, each raised (`inverted_members`), so that one product
            // decides them all.
            let mut parts = Vec::with_capacity(v.len());
            for x in v {
                match x {
                    Ex::Pow(b, k) if inverted_product(&b, &k) => {
                        let (Ex::Mul(w), Ex::Num(k)) = (*b, *k) else {
                            unreachable!()
                        };
                        match inverted_members(w, &k, cx) {
                            Ok(raised) => parts.extend(raised),
                            Err(kept) => parts.push(kept),
                        }
                    }
                    x => parts.push(canon_walk(x, cx)),
                }
            }
            mul(parts, cx)
        }
        Ex::Pow(b, k) if inverted_product(&b, &k) => {
            let (Ex::Mul(w), Ex::Num(k)) = (*b, *k) else {
                unreachable!()
            };
            match inverted_members(w, &k, cx) {
                Ok(raised) => mul(raised, cx),
                Err(kept) => kept,
            }
        }
        Ex::Pow(b, ex) => {
            let b = canon_walk(*b, cx);
            let ex = canon_walk(*ex, cx);
            pow(b, ex, cx)
        }
        Ex::Fun(f, v) => {
            let parts = v.into_iter().map(|x| canon_walk(x, cx)).collect();
            fun(f, parts, cx)
        }
        leaf => leaf,
    }
}

/// Is `b^k` a product under a negative integer power?
fn inverted_product(b: &Ex, k: &Ex) -> bool {
    matches!((b, k), (Ex::Mul(_), Ex::Num(k)) if k.is_integer() && k.is_negative())
}

/// The members of a product `w` raised to the negative integer `k`, for the product that decides
/// them (`raised_members`), so the members meet the power as they would spelled one by one. Where
/// `pow` does not distribute it, the product stays one member under the power (`Err`), built as
/// every product is (`mul`): the canonical state its print reads back to.
fn inverted_members(w: Vec<Ex>, k: &Rat, cx: &Cx) -> Result<Vec<Ex>, Ex> {
    let w: Vec<Ex> = w.into_iter().map(|x| canon_walk(x, cx)).collect();
    raised_members(w.clone(), k, cx).map_err(|_| pow(mul(w, cx), Ex::Num(k.clone()), cx))
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
                cert_region: None,
                mode: RuleMode::Default,
                fold_f64: Cx::folds_for(RuleMode::Default),
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
    fn a_constant_absorbs_the_orientation_sign_of_every_class() {
        with_view(|view| {
            let sin = view.intern("sin");
            for mode in [RuleMode::Default, RuleMode::Real, RuleMode::Permissive] {
                let mut cx = Cx::bare(view);
                cx.mode = mode;
                let neg = |a: Ex| mul(vec![Ex::int(-1), a], &cx);
                // A sum holding a `<constant>` takes a sign into its terms: -(x0 + x1^c) is
                // -x0 - x1^c.
                let s = add(vec![x(view), pow(y(view), Ex::Const, &cx)], &cx);
                let neg_s = neg(s.clone());
                assert!(matches!(&neg_s, Ex::Add(_)), "{mode:?}: {neg_s:?}");
                // Beside a bare `<constant>`, which absorbs every constant factor of its
                // product, that sign is the constant's, whatever route built the product, and the
                // sum takes the orientation with the fewer signs: c*-(x0 + x1^c) is c*(x0 + x1^c).
                let want = mul(vec![Ex::Const, s.clone()], &cx);
                assert_eq!(want, Ex::Mul(vec![Ex::Const, s.clone()]), "{mode:?}");
                assert_eq!(mul(vec![neg_s.clone(), Ex::Const], &cx), want, "{mode:?}");
                assert_eq!(
                    mul(vec![Ex::Const, Ex::int(-1), s.clone()], &cx),
                    want,
                    "{mode:?}"
                );
                // So does an odd function of one: c*sin(c - x0) is c*sin(x0 + c).
                let t = add(vec![x(view), Ex::Const], &cx);
                assert_eq!(
                    mul(vec![Ex::Const, fun(sin, vec![neg(t.clone())], &cx)], &cx),
                    mul(vec![Ex::Const, fun(sin, vec![t], &cx)], &cx),
                    "{mode:?}"
                );
                // Without a bare `<constant>` the sum keeps the sign the product gave it.
                let kept = mul(vec![y(view), neg_s.clone()], &cx);
                assert!(
                    matches!(&kept, Ex::Mul(v) if v.contains(&neg_s)),
                    "{mode:?}: {kept:?}"
                );
            }
        });
    }

    #[test]
    fn a_lone_site_takes_the_orientation_of_a_products_one_factor() {
        with_view(|view| {
            let sin = view.intern("sin");
            let z = || Ex::Leaf(view.intern("x2"));
            let half = Ex::Num(Rat::new(1, 2).unwrap());
            for mode in [RuleMode::Default, RuleMode::Real, RuleMode::Permissive] {
                let mut cx = Cx::bare(view);
                cx.mode = mode;
                let neg = |a: Ex| mul(vec![Ex::int(-1), a], &cx);
                // x0 - x1 - x2: two signs; -(x1 + x2 - x0) has two too, the outer one
                let s = add(vec![x(view), neg(y(view)), neg(z())], &cx);
                let raw_cube = Ex::Pow(Box::new(s.clone()), Box::new(Ex::int(3)));
                let raw_sin = Ex::Fun(sin, vec![s.clone()]);
                // alone, as the one factor of a product: (x0 - x1 - x2)^3 is
                // -(x1 + x2 - x0)^3, and sin(x0 - x1 - x2) is -sin(x1 + x2 - x0)
                let cube = pow(s.clone(), Ex::int(3), &cx);
                assert_eq!(
                    cube,
                    mul(vec![Ex::int(1), raw_cube.clone()], &cx),
                    "{mode:?}"
                );
                assert!(
                    matches!(&cube, Ex::Mul(v) if v.contains(&Ex::int(-1))),
                    "{mode:?}: {cube:?}"
                );
                let sine = fun(sin, vec![s.clone()], &cx);
                assert_eq!(
                    sine,
                    mul(vec![Ex::int(1), raw_sin.clone()], &cx),
                    "{mode:?}"
                );
                // a power over a lone site reads its class, whichever orientation it stands in
                assert_eq!(
                    pow(cube.clone(), Ex::int(-1), &cx),
                    pow(s.clone(), Ex::int(-3), &cx),
                    "{mode:?}"
                );
                assert_eq!(
                    pow(cube, half.clone(), &cx),
                    pow(s.clone(), Ex::Num(Rat::new(3, 2).unwrap()), &cx),
                    "{mode:?}"
                );
                assert_eq!(
                    pow(sine, Ex::int(-1), &cx),
                    pow(raw_sin, Ex::int(-1), &cx),
                    "{mode:?}"
                );
            }
        });
    }

    #[test]
    fn a_zero_keeps_its_ieee_sign_where_it_is_inverted() {
        with_view(|view| {
            let lg = |cx: &Cx| fun(view.intern("log"), vec![x(view)], cx);
            for mode in [RuleMode::Default, RuleMode::Permissive, RuleMode::Real] {
                let mut cx = Cx::bare(view);
                cx.mode = mode;
                let ieee = !matches!(mode, RuleMode::Real);
                let inverse = |members: Vec<Ex>| -> Ex {
                    let raised = raised_members(members, &Rat::NEG_ONE, &cx)
                        .unwrap_or_else(|held| vec![pow(held, Ex::int(-1), &cx)]);
                    mul(raised, &cx)
                };
                // 1/(x0*0) is inf/x0 in the float64 modes (-inf wherever x0 < 0), +inf in real
                let want = if ieee {
                    mul(vec![Ex::PosInf, pow(x(view), Ex::int(-1), &cx)], &cx)
                } else {
                    Ex::PosInf
                };
                assert_eq!(inverse(vec![x(view), Ex::int(0)]), want, "{mode:?}");
                // 1/(-1*0) = 1/-0 is -inf in the float64 modes
                let want = if ieee { Ex::NegInf } else { Ex::PosInf };
                assert_eq!(inverse(vec![Ex::int(-1), Ex::int(0)]), want, "{mode:?}");
                // a product held whole around its zero distributes the inverse too
                let held = pow(Ex::Mul(vec![Ex::int(0), lg(&cx)]), Ex::int(-1), &cx);
                if ieee {
                    let want = mul(vec![Ex::PosInf, pow(lg(&cx), Ex::int(-1), &cx)], &cx);
                    assert_eq!(held, want, "{mode:?}");
                } else {
                    assert!(matches!(held, Ex::Pow(..)), "{mode:?}: {held:?}");
                }
            }
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
    fn powers_of_e_collect_by_their_exponents() {
        with_view(|view| {
            let exp = view.intern("exp");
            let y = || Ex::Leaf(view.intern("x1"));
            for mode in [RuleMode::Default, RuleMode::Real, RuleMode::Permissive] {
                let mut cx = Cx::bare(view);
                cx.mode = mode;
                let e = |a: Ex| fun(exp, vec![a], &cx);
                let neg = |a: Ex| mul(vec![Ex::int(-1), a], &cx);
                // exp(a) exp(b) = exp(a + b)
                assert_eq!(
                    mul(vec![e(x(view)), e(y())], &cx),
                    e(add(vec![x(view), y()], &cx)),
                    "{mode:?}"
                );
                // exp(u) exp(-u) = exp(0) = 1, u finite almost everywhere
                assert_eq!(mul(vec![e(x(view)), e(neg(x(view)))], &cx), Ex::int(1));
                // e is e^1, and a rational power of an exp is a power of e
                assert_eq!(
                    mul(vec![Ex::E, e(x(view))], &cx),
                    e(add(vec![x(view), Ex::int(1)], &cx))
                );
                let sq = pow(e(x(view)), Ex::int(2), &cx);
                assert_eq!(
                    mul(vec![sq.clone(), e(y())], &cx),
                    e(add(vec![mul(vec![Ex::int(2), x(view)], &cx), y()], &cx))
                );
                // a quotient of powers of e, inside a wider product
                let q = mul(vec![y(), e(x(view)), pow(e(y()), Ex::int(-1), &cx)], &cx);
                assert_eq!(q, mul(vec![y(), e(add(vec![x(view), neg(y())], &cx))], &cx));
                // f64's exp(800) exp(-800) is inf * 0 = NaN, an overflow, where the true value
                // is given: exp(0) = 1
                assert_eq!(
                    mul(vec![e(Ex::int(800)), e(Ex::int(-800))], &cx),
                    Ex::int(1)
                );
                // a lone power of e stays, and one base's powers collect over that base as
                // `pow` spells them: exp(x) exp(x) is exp(x)^2
                assert_eq!(
                    mul(vec![sq.clone(), Ex::int(3)], &cx),
                    Ex::Mul(vec![Ex::int(3), sq.clone()])
                );
                assert_eq!(mul(vec![e(x(view)), e(x(view))], &cx), sq);
            }
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

    /// Permissive's one rounding lands on whichever of the two floats around the value prints
    /// shorter: a value between 0.1 and the next float, nearer the next, lands on 0.1 (the
    /// nearest, 0.10000000000000002, prints a digit more). A float's own shortest spelling never
    /// moves.
    #[test]
    fn a_landed_literal_takes_the_shorter_of_its_two_floats() {
        let r = Rat::from_big(
            num_bigint::BigInt::from(28823037615171179u64),
            num_bigint::BigInt::from(288230376151711744u64),
        )
        .unwrap();
        assert_eq!(r.to_f64_nearest(), 0.10000000000000002);
        assert_eq!(lossy_literal(&r), Rat::parse_decimal("0.1"));
        assert_eq!(lossy_literal(&Rat::parse_decimal("0.1").unwrap()), None);
        assert_eq!(
            lossy_literal(&Rat::parse_decimal("0.10000000000000002").unwrap()),
            None
        );
    }

    /// ONE PRICE FOR A SIGN: one bit wherever it sits -- on a literal, on a product's
    /// coefficient beside other factors, or a negation standing alone (a sign, not a product,
    /// so no `Mul` head). `+1` is no sign, and a reciprocal is division, not a sign.
    #[test]
    fn a_sign_costs_one_bit_wherever_it_sits() {
        with_view(|view| {
            let cx = Cx::bare(view);
            let c = |e: &Ex| super::complexity(e, view);
            let (x_, y_) = (x(view), y(view));
            let neg = |e: Ex| mul(vec![Ex::int(-1), e], &cx);
            let sin = |e: Ex| fun(view.intern("sin"), vec![e], &cx);
            // a negation standing alone: its operand and one bit
            assert_eq!(c(&neg(x_.clone())), c(&x_) + 1_000);
            assert_eq!(c(&neg(x_.clone())), 7_000);
            // the term of a difference
            let diff = add(vec![x_.clone(), neg(y_.clone())], &cx);
            let sum = add(vec![x_.clone(), y_.clone()], &cx);
            assert_eq!(c(&diff), c(&sum) + 1_000);
            // the -1 coefficient of a product beside other factors
            let prod = mul(vec![x_.clone(), y_.clone()], &cx);
            assert_eq!(c(&neg(prod.clone())), c(&prod) + 1_000);
            // the sign of a literal, a coefficient and an exponent: unchanged, one bit each
            assert_eq!(c(&Ex::int(-2)), c(&Ex::int(2)) + 1_000);
            let three_x = mul(vec![Ex::int(3), x_.clone()], &cx);
            let minus_three_x = mul(vec![Ex::int(-3), x_.clone()], &cx);
            assert_eq!(c(&minus_three_x), c(&three_x) + 1_000);
            assert_eq!(
                c(&pow(x_.clone(), Ex::int(-2), &cx)),
                c(&pow(x_.clone(), Ex::int(2), &cx)) + 1_000
            );
            // a reciprocal is division, not a sign
            assert_eq!(c(&pow(x_.clone(), Ex::int(-1), &cx)), 9_000);
            // so the two placements of a constant's sign cost the same, and the sign key
            // decides: the sign stays on the constant's literal
            let outside = Ex::Mul(vec![
                Ex::int(-1),
                x_.clone(),
                Ex::Fun(view.intern("sin"), vec![Ex::int(3)]),
            ]);
            let inside = Ex::Mul(vec![
                x_.clone(),
                Ex::Fun(view.intern("sin"), vec![Ex::int(-3)]),
            ]);
            assert_eq!(c(&outside), c(&inside));
            let built = mul(vec![Ex::int(-1), x_.clone(), sin(Ex::int(3))], &cx);
            assert_eq!(built, inside);
        });
    }

    #[test]
    fn complexity_weight_table() {
        // The mu weight table in MILLI-BITS, at the SYMBOL TABLE (2026-08-21): a leaf 6
        // bits, Add/Mul/Pow 3, Pi/E 4, an elementary head 6, a transcendental head 8,
        // an infinity 8 -- where mu' (D38/B2) charged a flat 8 for every one of them.
        // Unchanged by the table: a priced literal pays ONE selector bit + its cheaper codeword
        // (rational L(p)[+L(q)] vs decimal-scientific max(2,L(m))+L(|k|)) + a bit for a
        // negative sign, mantissa floored at two bits; magnitude-1 coefficient and
        // exponent slots stay free (a bare sign writes no number, so no selector
        // either); <constant> = MU_FREE_WORST_CASE_F64 = 67 bits (the sup-construction
        // re-derived under the mu' codebook -- DERIVED, not chosen).
        with_view(|view| {
            let cx = Cx::bare(view);
            let c = |e: &Ex| super::complexity(e, view);
            // x - y = 16: Add(3) + x(6) + the negation of y (a sign, one bit, and no Mul
            // head) + y(6).
            // x*y = 15; x/y = 18 (the Pow(y, -1) wrapper at 3, exponent -1 free).
            let x_ = x(view);
            let y_ = y(view);
            let x_minus_y = add(
                vec![x_.clone(), mul(vec![Ex::int(-1), y_.clone()], &cx)],
                &cx,
            );
            assert_eq!(c(&x_minus_y), 16_000);
            assert_eq!(c(&mul(vec![x_.clone(), y_.clone()], &cx)), 15_000);
            let x_over_y = mul(vec![x_.clone(), pow(y_.clone(), Ex::int(-1), &cx)], &cx);
            assert_eq!(c(&x_over_y), 18_000);
            // 2x = 11.585 (the coefficient PAYS selector + its bits: 3 + 2.585 + 6),
            // x^2 = 11.585, 1/x = 9, sin(x) = 12. The literal floor is ONE bit since
            // 2026-08-22, so cost(2) = 2.585 clears it and no longer ties cost(3).
            assert_eq!(c(&mul(vec![Ex::int(2), x_.clone()], &cx)), 11_585);
            assert_eq!(c(&pow(x_.clone(), Ex::int(2), &cx)), 11_585);
            // THE EXPONENTS ARE ORDERED AGAIN. Under the two-bit floor `x^2`, `x^3` and
            // `x^4` all priced alike, because the clamp swallowed cost(2), cost(3) and
            // cost(4) together -- the measure could not tell a square from a cube.
            assert!(
                c(&pow(x_.clone(), Ex::int(2), &cx)) < c(&pow(x_.clone(), Ex::int(3), &cx))
                    && c(&pow(x_.clone(), Ex::int(3), &cx)) < c(&pow(x_.clone(), Ex::int(4), &cx))
            );
            assert_eq!(c(&pow(x_.clone(), Ex::int(-1), &cx)), 9_000);
            assert_eq!(c(&fun(view.intern("sin"), vec![x_.clone()], &cx)), 12_000);
            // x^-2 costs as x^2 plus the FULL sign bit (the -2 exponent is a priced
            // literal: selector + L(2) + sign = 1 + 1.585 + 1, all of it clear of the
            // one-bit floor; under the two-bit floor the clamp absorbed all but 0.585).
            assert_eq!(c(&pow(x_.clone(), Ex::int(-2), &cx)), 12_585);
            // Mul(3) + <constant>(67_000) + x(6) = 76_000. The free constant is a LITERAL
            // price, so the table does not touch it: it now outweighs its own structure by
            // more than it did, which is the direction the ordering wants.
            assert_eq!(
                c(&mul(vec![Ex::Const, x_.clone()], &cx)),
                mu_mul() + MU_FREE_WORST_CASE_F64 + mu_leaf()
            );
            assert_eq!(mu_free(), 67 * MU_MILLI);
            assert_eq!(c(&Ex::int(-5)), 4_585); // selector + L(5) + 1 for the sign
                                                // x8 + 1.2*x3 = 3 + 6 + (3 + cost(6/5) + 6).
                                                // cost(6/5) = selector + min(L(6)+L(5) = 5.392, max(2,L(12))+L(1) = 4.700)
                                                // = 5.700: the decimal-scientific codeword takes it (D38 revoked the
                                                // Sec 10.10(1) fraction-only rule).
            let t = mul(vec![Ex::Num(Rat::new(6, 5).unwrap()), y_.clone()], &cx);
            assert_eq!(c(&add(vec![x_.clone(), t], &cx)), 23_700);
            // The stage-2 point, in one line: an E factor undercuts its f64 image --
            // and by MORE than before, since the table cheapened the structure the
            // symbolic spelling is made of and left the image's 56.272 bits alone.
            let sym = c(&mul(vec![Ex::E, x_.clone()], &cx));
            let mat = c(&mul(
                vec![
                    Ex::Num(Rat::new(2718281828459045, 1000000000000000).unwrap()),
                    x_.clone(),
                ],
                &cx,
            ));
            assert!(sym == 13_000 && sym < mat, "{sym} vs {mat}");
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
        // only when there is no genuine denominator), plus the mu' selector bit; the
        // denominator 7 offers no decimal codeword: 1 + L(111...1) + L(7) = 137.031 bits.
        assert_eq!(c, 137_031);
        // Far above any in-range literal; the boundary that matters is the ~105-BIT
        // 52-bit dyadics.
        assert!(c > 105 * MU_MILLI);
        // Monotone in each component.
        assert!(mu_numeric_str(&format!("{big41}1/7")) > c);
        assert!(mu_numeric_str(&format!("{big41}/71")) > c);
        // The sign costs exactly one bit here, as it does in range (owner 2026-08-06);
        // a zero numerator floors at the one-bit codeword floor (+ the selector) --
        // the 2026-08-22 ruling, mirrored across the i128 boundary (S15).
        assert_eq!(mu_numeric_str(&format!("-{big41}/7")), c + MU_MILLI);
        assert_eq!(mu_numeric_str("0/7"), 2 * MU_MILLI);
        // The decimal path is case-insensitive in the exponent marker.
        assert_eq!(
            mu_numeric_str("4.159653437657682e-35"),
            mu_numeric_str("4.159653437657682E-35")
        );
        // D38 RESTORES the minimum over codes (revoking Sec 10.10(1)): `1e-40` takes its
        // decimal-scientific codeword, selector + max(floor, L(1)) + L(40) = 7.358 bits
        // under the one-bit floor, not the 133.880 the fraction-only rule charged. The boundary property this test
        // exists for -- the token pricer agrees with `mu_rat`'s rule, and the SAME value
        // prices the SAME from both spellings -- now holds EXACTLY: the pre-D38 build
        // pinned a one-milli-bit seam between these two spellings (133.880 vs 133.879,
        // the rounded-L10 residue of the fraction code); both now take the decimal
        // codeword through the same `l_millibits(40)` and the seam is CLOSED.
        let e_notation = mu_numeric_str("1e-40");
        let as_fraction = mu_numeric_str(&format!("1/1{}", "0".repeat(40)));
        assert_eq!(e_notation, 7_358);
        assert_eq!(as_fraction, 7_358);
        assert_eq!(
            e_notation, as_fraction,
            "the two spellings of 1e-40 drifted apart: {e_notation} vs {as_fraction}"
        );
    }

    /// c_free' derivation (D38/B2): `MU_FREE_WORST_CASE_F64` is the CEILED sup of mu'
    /// over f64 round-trip spellings plus the sign bit -- recomputed here from the
    /// engine's own token pricer, so the published constant cannot drift from the code
    /// that defines it. The sweep: every f64 exponent field at six mantissa patterns
    /// (binade edges and mid-grid probes), plus the recorded argmax
    /// `8.9002954340287245e-308` (found offline by ulp-walking every binade top in the
    /// boundary decades; the analytic feasibility cap -- a shortest repr's mantissa is
    /// below `10 * 2^53` and its scale within +-343 -- bounds the sup under 64.92
    /// bits, so the 67-bit ceiling is robust to argmax misses).
    #[test]
    fn c_free_prime_derivation() {
        let price = |x: f64| -> u64 {
            let tok = format!("{x:e}");
            mu_numeric_str(&tok) - MU_MILLI // the codeword min, without the selector
        };
        let mantissas: [u64; 6] = [
            0,                     // the binade bottom (a power of two)
            1,                     // one ulp up
            0x000F_FFFF_FFFF_FFFF, // the binade top
            0x0008_0000_0000_0000,
            0x0005_5555_5555_5555,
            0x000A_AAAA_AAAA_AAAA,
        ];
        let mut sup = 0u64;
        for exp in 0..2047u64 {
            for m in mantissas {
                let x = f64::from_bits((exp << 52) | m);
                if x == 0.0 || !x.is_finite() {
                    continue;
                }
                sup = sup.max(price(x));
            }
        }
        // The recorded argmax reproduces the recorded supremum, and nothing in the
        // sweep prices above it (the load-bearing safety direction).
        let rec = price(8.9002954340287245e-308);
        assert_eq!(rec, 64_649);
        assert!(
            sup <= rec,
            "the sweep beat the recorded supremum: {sup} > {rec}"
        );
        assert!(
            sup > rec - 2_000,
            "the sweep never reached the extreme region: {sup}"
        );
        // + the sign bit + the selector bit, ceiled: the published constant, re-derived.
        let derived = (rec + MU_MILLI + MU_MILLI).div_ceil(MU_MILLI) * MU_MILLI;
        assert_eq!(derived, MU_FREE_WORST_CASE_F64);
        assert_eq!(mu_free(), 67 * MU_MILLI);
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
            // The saturation CROSSING is no longer materializable (mu', D38): under
            // the two-codeword pricing every leaf is bounded by ~3.32 bits per DIGIT
            // of its own print (the scientific codeword prices the scale as an
            // integer, so the old ~1.4e13-milli-bit astronomic leaves now price a few
            // dozen bits -- this leaf is {ca} ~ 55 bits), and reaching u64::MAX would
            // take a tree whose serialization is petabytes. The old crossing pin
            // (`vec![leaf; u64::MAX / ca]`, ~1.3M members pre-D38) would today demand
            // ~3e14 members and is deleted as physically unreachable; the
            // `saturating_add` accumulation stays as pure defense-in-depth. What
            // remains checkable at reachable scale: totals are exact (no wrap, no
            // saturation) and strictly monotone in member count on the largest
            // feasible astronomic bags.
            let q = 100_000usize;
            let mut prev = 0u64;
            for n in [q - 1, q, q + 1] {
                let cs = complexity(&Ex::Add(vec![a.clone(); n]), view);
                assert!(
                    cs > prev,
                    "the Add total failed to grow at {n} members: {prev} -> {cs}"
                );
                assert_eq!(
                    cs,
                    mu_add() + ca * n as u64,
                    "an astronomic Add bag mispriced at {n} members"
                );
                prev = cs;
            }
            for bag in [
                Ex::Mul(vec![a.clone(); q]),
                Ex::Fun(view.intern("sin"), vec![a.clone(); q]),
            ] {
                let cs = complexity(&bag, view);
                assert!(
                    cs >= ca && cs >= bare,
                    "a large astronomic bag priced below its parts: {cs}"
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
    /// interning order. The mu-tied orbit of `-1 * (a-b) * (c-d)` (its two mirror
    /// pairings price alike) must land on the same resolved spelling
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
    /// product is not representable), the sign has exactly ONE canonical host, a function
    /// of the value -- not of which member carried it on arrival. Before the
    /// sign-factored accumulation, `-N * P` kept the sign on the `-N` partial while
    /// the negation of `N * P` hosted it on the coefficient slot: two stable states
    /// for one value (the extreme lane's P1-idempotence family, 88 -> 30 rows).
    #[test]
    fn f72_partition_sign_host_is_route_invariant() {
        with_view(|view| {
            let cx = Cx::bare(view);
            // 0.9999999999999999 and 1e-300: the product's denominator 10^316 exceeds the
            // f64 domain's 2^1022 (phase 2c; it used to be an i128 overflow), so the bag keeps
            // BOTH as members.
            let n = Rat::new(9999999999999999, 10000000000000000).unwrap();
            let p = Rat::from_big(
                num_bigint::BigInt::from(1),
                num_traits::pow(num_bigint::BigInt::from(10), 300),
            )
            .unwrap();
            let route_signed_member = mul(
                vec![Ex::Num(n.checked_neg().unwrap()), Ex::Num(p.clone())],
                &cx,
            );
            let route_negated_bag =
                negate_term(&mul(vec![Ex::Num(n.clone()), Ex::Num(p.clone())], &cx), &cx).unwrap();
            let route_other_host = term_join(p.checked_neg().unwrap(), Ex::Num(n.clone()), &cx);
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

    /// Integers beyond 128 bits are integers (number plan phase 2). The map of the 128-bit
    /// boundary found the sites that read "does not fit i128" as "is not an integer", each
    /// unsound once such an integer reaches it. Real mode has the exact parity; the f64 domain
    /// knows no parity beyond 2^53 (phase 2c), and every site must then refuse as well.
    #[test]
    fn big_integers_keep_their_integer_reading() {
        for mode in [RuleMode::Real, RuleMode::Default] {
            with_view(|view| big_integer_sites(view, mode));
        }
    }

    fn big_integer_sites(view: &TokenView, mode: RuleMode) {
        {
            let mut cx = Cx::bare(view);
            cx.mode = mode;
            let _domain = super::super::rat::number_domain(cx.f64_numbers());
            let big =
                |p: num_bigint::BigInt| Rat::from_big(p, num_bigint::BigInt::from(1)).unwrap();
            let ten40 = big(num_traits::pow(num_bigint::BigInt::from(10), 40));
            let odd = ten40.checked_add(&Rat::ONE).unwrap(); // 10^40 + 1
            assert!(ten40.small_int().is_none() && ten40.is_integer());
            if mode == RuleMode::Real {
                assert!(ten40.is_even_integer() && odd.is_odd_integer());
            } else {
                assert!(ten40.parity().is_none() && odd.parity().is_none());
            }
            let rootn = view.intern("rootn");

            // 1. rootn(x, 10^40) is a root with an integer index, not the invalid operation.
            let r = fun(rootn, vec![x(view), Ex::Num(ten40.clone())], &cx);
            assert!(matches!(&r, Ex::Fun(..)), "{r:?}");

            // 2. Pow-of-pow: (x^(10^40))^(1/3) must not become x^(10^40/3). The inner power is
            //    an even power, defined and non-negative at x < 0, where x^(10^40/3) is NaN.
            let third = Rat::new(1, 3).unwrap();
            let inner = pow(x(view), Ex::Num(ten40.clone()), &cx);
            let composed = pow(inner, Ex::Num(third.clone()), &cx);
            let wrong = pow(x(view), Ex::Num(ten40.checked_mul(&third).unwrap()), &cx);
            assert_ne!(composed, wrong);

            // 3. Power through an even root: rootn(x^(2(10^40+1)), 4) must not become
            //    x^((10^40+1)/2), for the same reason.
            let k = odd.checked_mul(&Rat::int(2)).unwrap();
            let through = fun(
                rootn,
                vec![pow(x(view), Ex::Num(k.clone()), &cx), Ex::int(4)],
                &cx,
            );
            let wrong = pow(
                x(view),
                Ex::Num(k.checked_mul(&Rat::new(1, 4).unwrap()).unwrap()),
                &cx,
            );
            assert_ne!(through, wrong);
        }
    }

    /// The float cap: a literal costs at most what its float's shortest decimal costs.
    #[test]
    fn float_cap_a_literal_costs_at_most_its_float() {
        let long = {
            let _exact = crate::ac::rat::number_domain(false);
            Rat::parse_decimal("0.0766541268471677307861497420516056347394916180597539647375669841")
                .unwrap()
        };
        let short = Rat::shortest_reading_as(long.to_f64_nearest()).unwrap();
        assert_eq!(mu_rat(&long), mu_rat_exact(&short));
        assert!(mu_rat(&long) < mu_rat_exact(&long));
        // a fraction is never the cap: the 16-digit decimal keeps its price
        let third = Rat::parse_decimal("0.3333333333333333").unwrap();
        assert_eq!(mu_rat(&third), mu_rat_exact(&third));
        assert!(mu_rat(&Rat::new(1, 3).unwrap()) < mu_rat(&third));
        for t in [
            "1000",
            "0.2",
            "2",
            "-7.5",
            "1e-40",
            "0.5",
            "3",
            "0.001",
            "2.177697277405848",
        ] {
            let r = Rat::parse_decimal(t).unwrap();
            assert_eq!(mu_rat(&r), mu_rat_exact(&r), "{t}");
        }
    }

    /// The shortest decimal reads back as the float.
    #[test]
    fn float_cap_shortest_decimal_reads_back() {
        for y in [
            0.1,
            1.0 / 3.0,
            std::f64::consts::PI,
            1e-300,
            5e-324,
            f64::MAX,
            -0.75,
            1e22,
            0.45445993041202365,
        ] {
            let d = Rat::shortest_reading_as(y).unwrap();
            assert_eq!(d.to_f64_nearest(), y, "{y:?}");
        }
        assert_eq!(Rat::shortest_reading_as(f64::INFINITY), None);
    }

    /// The price is a function of the value alone: the same in the f64 and the exact domain,
    /// and never above the exact price.
    #[test]
    fn float_cap_price_is_mode_free_and_never_rises() {
        let values: Vec<Rat> = {
            let _exact = crate::ac::rat::number_domain(false);
            [
                "0.3333333333333333",
                "1e-310",
                "123456789012345678901234567890",
                "0.1",
                "2.718281828459045235360287471352662497757",
                "-4.2e17",
            ]
            .iter()
            .map(|t| Rat::parse_decimal(t).unwrap())
            .chain([Rat::new(355, 113).unwrap(), Rat::new(-22, 7).unwrap()])
            .collect()
        };
        for r in &values {
            let in_f64 = {
                let _d = crate::ac::rat::number_domain(true);
                mu_rat_float(r)
            };
            let in_exact = {
                let _d = crate::ac::rat::number_domain(false);
                mu_rat_float(r)
            };
            assert_eq!(in_f64, in_exact, "{r:?}");
            assert!(in_f64 <= mu_rat_exact(r), "{r:?}");
        }
    }

    /// The float cap on every path: a value the f64 domain holds in pieces costs what the exact
    /// domain prices the one number at, alone or beside other factors.
    #[test]
    fn float_cap_reaches_a_literal_held_in_pieces() {
        with_view(|view| {
            let cx = Cx::bare(view); // f64's number domain
            let (a, b) = {
                let _exact = crate::ac::rat::number_domain(false);
                (
                    Rat::parse_decimal("1.2345678901234567e-305").unwrap(),
                    Rat::parse_decimal("9.876543210987654e-10").unwrap(),
                )
            };
            let inv_b = b.checked_inv().unwrap();
            let exact = {
                let _exact = crate::ac::rat::number_domain(false);
                a.checked_mul(&inv_b).unwrap()
            };
            // a's denominator, 10^321, passes 2^1022: a stays a leaf as written, beside 1/b
            let leaf = Ex::Leaf(view.intern("1.2345678901234567e-305"));
            let quotient = mul(
                vec![leaf.clone(), pow(Ex::Num(b.clone()), Ex::int(-1), &cx)],
                &cx,
            );
            assert_eq!(
                quotient,
                Ex::Mul(vec![Ex::Num(inv_b.clone()), leaf.clone()])
            );
            let pieces = mu_numeric_str("1.2345678901234567e-305") + mu_rat(&inv_b);
            assert!(mu_rat(&exact) < pieces);
            assert_eq!(complexity(&quotient, view), mu_rat(&exact));
            let beside = mul(vec![quotient, x(view)], &cx);
            assert_eq!(
                complexity(&beside, view),
                mu_mul() + mu_rat(&exact) + mu_leaf()
            );
            // A product deciding opposite sums beside a piece prices its candidates against
            // bounds below their prices (`decide_opposite_sums`' debug assertion).
            let s = add(
                vec![x(view), Ex::Num(Rat::parse_decimal("-9.6").unwrap())],
                &cx,
            );
            let t = mul(vec![Ex::int(-1), s.clone()], &cx);
            let p = mul(vec![Ex::int(3), leaf, s, t], &cx);
            let Ex::Mul(v) = &p else {
                panic!("not a product: {p:?}")
            };
            assert_eq!(v.iter().filter(|f| is_piece(f, view)).count(), 2, "{p:?}");
            let literal = bag_literal_value(v, false, view).unwrap();
            let rest: u64 = v
                .iter()
                .filter(|f| !is_piece(f, view))
                .map(|f| complexity(f, view))
                .sum();
            assert_eq!(complexity(&p, view), mu_mul() + mu_rat(&literal) + rest);
        });
    }
}
