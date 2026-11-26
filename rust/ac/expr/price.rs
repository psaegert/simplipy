//! THE MEASURE: the description length `mu` of an expression in milli-bits, the codeword of every
//! node and every number ([`complexity`], [`priced`]).

use super::*;

/// The COMPLEXITY functional IS the unified simplicity measure mu (stage 2, ratified in
/// design/UNIFIED_SIMPLICITY_MEASURE.md): description length under an exactness-respecting
/// cost model, measured on the CANONICAL internal form and never on a serialization.
/// "Simplify is lossless compression" -- pi is one symbol with zero degrees of freedom,
/// `23.14069263277927` is a ~52-bit object, `<constant>` is a full real degree of freedom.
/// Costs are scaled integers in units of 1/8 (one grammar symbol = 8), so all arithmetic
/// stays exact and the order is discrete -- no float comparisons anywhere in the ordering.
///
/// The weight table (each row a ratified decision):
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
/// consistent with the ratified doc's worked examples: cost(2)=2 (mu(2x)=18),
/// cost(5/2)=4 (T5: mu(2.5*pi)=20), a 52-bit dyadic ~105, mu(0)=mu(1)=2.
///
/// (Historical preamble; later rulings amended it in place: milli-bit carrier and the
/// literal sign bit, c_free DERIVED not floored, and mu-PRIME, where
/// the numeric codebook gains the decimal-scientific codeword
/// behind a one-bit selector, and c_free' re-derives to 67 bits. Current authority:
/// `mu_rat`, `decimal_code`, `mu_numeric_str`, `mu_free` below.)
///
/// Worked examples: `x + y` = 24, `x*y` = 24, `2x` = 18, `x^2` = 18, `1/x` = 16,
/// `sin(x)` = 16, `sin(pi)` = 16, `E*x` = 24 < `2.718281828459045*x` = ~117,
/// `<constant>*x` = 144, `exp(pi*x)` = 32 < `pow(23.14..., x)` = ~121.
/// The symbol UNIT, in BITS: what one grammar symbol costs at the ratified 1/8 pin, and
/// the scale every entry of the symbol table below is expressed against;
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

/// The mu SYMBOL TABLE, in BITS at the 8-bit unit.
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
/// The cheapest spellings here are `acos(-1)` = 11 bits and `exp(1)` = 8, so 4 has room --
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

/// `np.e`. Below `exp(1)` = 8 bits, its cheapest denoting expression -- the inequality
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
/// Two tiers: the ELEMENTARY heads at `rootn`'s price, the one head a skeleton reaches for
/// constantly (181 of 10,996 charged nodes at 5.92 bits), and the TRANSCENDENTAL heads at the
/// price of those it barely reaches (40-68 occurrences each, 7.3-8.1 bits). Which head sits in
/// which tier is a choice the census does not make: `exp`, `log`, `abs`, `sin`, `cos` and `tan`
/// are as rare in it as the transcendental heads. The gap between the tiers is the whole point:
/// it is what makes `pi - 2` cheaper than `asin(sin 2)`.
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
        "floor" | "ceil" | "max" | "min" | "mod" | "harmonic" | "erf" | "erfc" | "atan2"
        | "besselj" | "bessely" | "besseli" | "besselk" => MU_BITS_TRANSCENDENTAL,
        _ => 8,
    }
}

/// A declared function head's price in milli-bits ([`mu_fun_bits`] at the symbol unit): what
/// [`complexity`] charges a `Fun` node's head, and what the written price charges a head written.
pub(super) fn mu_head(name: &str) -> u64 {
    sym_bits(mu_fun_bits(name))
}

/// The head of a DEFINED FUNCTION (ruling 40: a config's `defined_functions:`, `crate::defined`),
/// in milli-bits, priced by name like every head, at the tier of the function its definition
/// applies: `sqrt` at rootn's (`rootn(u, 2)`, without the 2), `cot` at tan's (`1/tan(u)`), `sec` at
/// cos's (`1/cos(u)`) and `csc` at sin's (`1/sin(u)`), the elementary tier, and `acot` at atan's
/// (`atan(1/u)`), `asec` at acos's (`acos(1/u)`) and `acsc` at asin's (`asin(1/u)`), the
/// transcendental tier. Another defined
/// name keeps the flat unit, as an unknown head does. Only the written price reads it
/// (`written_complexity`): the canonical state holds a defined function's definition, never its
/// head, and a config that declares an operator of one of these names keeps [`mu_fun_bits`]'s
/// price for it.
pub(super) fn mu_defined_head(name: &str) -> u64 {
    sym_bits(match name {
        "sqrt" | "cot" | "sec" | "csc" => MU_BITS_ELEMENTARY,
        "acot" | "asec" | "acsc" => MU_BITS_TRANSCENDENTAL,
        _ => 8,
    })
}

/// The free constant's price, DERIVED rather than chosen (contract §10.10(5)) --
/// derived under the mu' codebook: the sup-construction
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
/// 10^324-scale DENOMINATOR of the extreme spellings -- the overcharge itself; the
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
    // The small arguments (a literal's digits, exponents and denominators are mostly small) are
    // read from a table the same extraction fills once.
    if n < L_SMALL {
        static TABLE: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
        return TABLE.get_or_init(|| (0..L_SMALL).map(l_millibits_computed).collect())[n as usize];
    }
    l_millibits_computed(n)
}

/// How many small arguments [`l_millibits`] reads from its table.
const L_SMALL: u128 = 4096;

/// [`l_millibits`], computed.
fn l_millibits_computed(n: u128) -> u64 {
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
/// digits (the spec's "shortest exact decimal spelling" reading -- `1000` is the
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
/// spelling-free -- **mu': the numeric codebook carries TWO
/// codewords, chosen by a one-bit selector**, the fraction codeword with a second form whose
/// denominator is written as a decimal (the reciprocal codeword, below).
///
/// ```text
/// mu'(value) = 1 selector bit + min( L(p) + L(q),  L(m) + L(k),  L(1) + L(m') + L(k') ) + sign bit
/// ```
///
/// with `(m', k')` the decimal code of `1/value` where that decimal terminates and the value's
/// reduced denominator passes the recognition list's bound.
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
/// the reference rescorer priced them (the mantissa at the
/// integer price `max(2, L(m))`, the exponent at `L(k)`, the sign once); the NATIVE
/// measure adds the selector uniformly on top, which the study rescorer -- computing
/// `min(old, dec)` only -- deliberately omitted. Native mu' and study-rescored prices
/// therefore differ by a constant 1000 mB per priced numeric leaf, by design.
///
/// WHY A SECOND CODEWORD. MDL codebooks should match the source distribution:
/// SR constants are decimal-printed f64s, and the fraction-only code forced them
/// through "two arbitrary integers", overcharging ~3.32 bits per decimal shift
/// (`2.177697277405848` priced 94.781 bits as p/q against 55.952 as mantissa+exponent
/// +selector). This replaces the fraction-only rule ("mu is value complexity, no min over
/// codes"): the asymmetry between `1000` and `0.001` resolves to symmetry at the
/// codeword level: `mu'(1000) == mu'(0.001)` (both are the codeword `(1, 3)`), where
/// fraction-only code had `9.967` against `10.967`.
///
/// THE RECIPROCAL CODEWORD. A non-integer whose reciprocal `c` has a terminating decimal, and
/// whose reduced denominator passes the recognition list's bound (`ground::RECOGNISED_K`), is
/// also written `1/c`, the divisor-side spelling (`x1/0.225147640565301`): its codeword is
/// `L(1)`, the numerator of the fraction codeword, plus `c`'s decimal codeword, so `mu(1/c)` is
/// one bit above `c`'s decimal codeword, as `mu(1/n)` is one bit above `mu(n)`. Where it is the
/// cheapest of the three, the serializer prints it (`reciprocal_spelling_wins`): the reciprocal
/// of a long decimal costs what its divisor costs, not its fraction's two long integers. A short
/// fraction keeps its own codewords (`25/2`, `10/3`).
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
    let (fraction, decimal, reciprocal) = mu_rat_codeword_totals(r);
    MU_MILLI
        + fraction
            .min(decimal.unwrap_or(u64::MAX))
            .min(reciprocal.unwrap_or(u64::MAX))
}

/// THE PRICE OF A LITERAL: its own exact spelling ([`mu_rat_exact`]), the codeword the
/// serializer prints. Every number is priced as it is written, so the printed spelling is the
/// priced one everywhere: the 34-digit exact product `0.0900000000000000240000000000000016` costs
/// its 34 digits, and a mode that may write a cheaper value writes it (permissive the double
/// around it, [`land`]; strict float64's value of the operation that made it,
/// [`strict_number`]). A literal held in pieces (`is_piece`, `piece_price`) costs at most what
/// this function prices the value they denote at.
pub fn mu_rat(r: &Rat) -> u64 {
    mu_rat_exact(r)
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
/// The literal codeword FLOOR, in milli-bits: ONE bit, beyond the selector (a floor of
/// two was blind).
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
/// 3,000 (f64) and 3,359 (permissive) milli-bits: a small genuine descent. The
/// codeword prices move with it -- `mu(1000)` is 4 bits, not 5 -- but
/// what matters is the SYMMETRY, `mu(1000) == mu(0.001)`, and that holds at 4.
pub(super) const MU_RAT_FLOOR: u64 = MU_MILLI;

/// The codeword totals of a literal: (fraction, decimal, reciprocal), each without the selector.
type CodewordTotals = (u64, Option<u64>, Option<u64>);

pub(super) fn mu_rat_codeword_totals(r: &Rat) -> CodewordTotals {
    // A literal's codewords cost integer divisions and logarithms (128-bit or big), and the same
    // values are priced again and again while candidates are compared (the sign placement in
    // `mul` alone prices whole factors per orientation). The price is a function of the value,
    // so a per-thread table answers repeats; it is emptied when it reaches 65,536 entries
    // (bounded per thread, also under the miner's thread pool). A small integer is priced
    // directly.
    if r.small_parts()
        .is_some_and(|(p, q)| q == 1 && p.unsigned_abs() < 1 << 20)
    {
        return mu_rat_codeword_totals_uncached(r);
    }
    thread_local! {
        static BIG_PRICES: std::cell::RefCell<rustc_hash::FxHashMap<Rat, CodewordTotals>> =
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

fn mu_rat_codeword_totals_uncached(r: &Rat) -> CodewordTotals {
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
    // The reciprocal codeword: `1/c`, `c` a terminating decimal, priced by `c`'s decimal code
    // (an integer `c` with trailing zeros by its scientific code, `1/30` as `/3e1`; one without
    // has no decimal code, and its reciprocal is the fraction codeword `1/n` itself). A value
    // whose reduced denominator is within the recognition list's bound keeps its fraction and
    // decimal codewords alone: `25/2`, `10/3` and `1/20` are short fractions or decimals, not
    // reciprocals.
    let bound = crate::ground::RECOGNISED_K;
    let short_denominator = match r.small_parts() {
        Some((_, q)) => q <= i128::from(bound),
        None => r.big_parts().1 <= BigInt::from(bound),
    };
    // `1/c` terminates only where the numerator has no prime factor but 2 and 5 (zero has no
    // reciprocal)
    let smooth_numerator = match r.small_parts() {
        Some((0, _)) => false,
        Some((p, _)) => {
            let mut m = p.unsigned_abs();
            m >>= m.trailing_zeros();
            while m % 5 == 0 {
                m /= 5;
            }
            m == 1
        }
        None => true,
    };
    let reciprocal = if r.is_integer() || short_denominator || !smooth_numerator {
        None
    } else {
        r.abs()
            .checked_inv()
            .and_then(|c| decimal_code(&c))
            .map(|(m, k)| MU_MILLI + m.max(floor) + k + sign)
    };
    (fraction, decimal, reciprocal)
}

/// Whether the DECIMAL spelling is the canonical PRINT for this value: the serializer
/// argmin over the SAME two codeword totals `mu_rat_exact` mins over, so the print costs
/// the price the measure (`mu_rat`) charges (the state/serializer factoring: states carry one
/// exact rational per value and no spelling; the emitter realizes the best codeword at print
/// time, locally and closed-form).
///
/// TIES GO TO THE FRACTION -- the structurally distinguished caller-dialect member,
/// mirroring the sign-trade tier-2 rule ("what you typed survives"; states carry no
/// spelling, so the tie-break must be spelling-free, and either choice realizes the
/// same mu'). Measured: NO exact tie was found in the i128 lattice -- every
/// `(a, b)` with `q = 2^a * 5^b < 2^127` (3,563 pairs) scanned exhaustively over
/// numerators `p <= 2000` and at 29 spread samples up to ~3.5e9 beyond; the closest
/// calls observed are 3-4 milli-bits (`54321 / 2^45*5^14` and `19 / 2^28*5^9`, the
/// falsifier suite's boundary pair) -- so the clause is defensive. Ties would also
/// keep the powers of two (`1/2`, `5/8`) on the fraction side.
///
/// Callers gate integers upstream: an integer prints its decimal spelling
/// (`Rat::exact_decimal`, the rendering of BOTH an integer's codewords), positional in the
/// ordinary range (`1000`, never `1e3`) and scientific beyond it where shorter (`1e300`).
pub fn decimal_spelling_wins(r: &Rat) -> bool {
    let (fraction, decimal, reciprocal) = mu_rat_codeword_totals(r);
    decimal.is_some_and(|d| d < fraction && reciprocal.is_none_or(|c| d <= c))
}

/// Whether the RECIPROCAL spelling `1/c` is the canonical print for this value: its codeword is
/// strictly the cheapest of the three (`mu_rat_exact`), so the serializer writes the divisor
/// `c`, a terminating decimal, wherever it writes the value (`x1/0.225147640565301`, alone
/// `1/0.225147640565301`). Ties go to the fraction and the decimal.
pub fn reciprocal_spelling_wins(r: &Rat) -> bool {
    let (fraction, decimal, reciprocal) = mu_rat_codeword_totals(r);
    reciprocal.is_some_and(|c| c < fraction && decimal.is_none_or(|d| c < d))
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

/// The ASTRONOMIC knee. The exact linear schedule `|scale| * L10_MILLI`
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
/// exactly the mu' rule `mu_rat_exact` applies in range: one selector bit, then the
/// cheaper of the RATIONAL codeword (every integer the spelling writes down costs
/// `L(n)`, a fraction pays both components) and the DECIMAL-SCIENTIFIC codeword
/// (`max(floor, L(m)) + L(|scale|)` for the shortest spelling `m * 10^scale`, the
/// floor being `MU_RAT_FLOOR`, one bit), plus a sign
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
    // the in-range side moved to one, mu(1e-39) priced ABOVE mu(3e-38) -- the
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
        // fraction-only code priced them one milli-bit apart).
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
    // 10^|scale| denominator). Two regimes, one knee (doc at `MU_SCALE_KNEE`): the
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

// SATURATING accumulation throughout: every per-node price is bounded far
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
/// negation standing alone (`Mul[-1, x]`) is a `Sign` and no `Mul`. Counting serialized
/// TOKENS instead gets this wrong in both directions -- `-` and `/` are not nodes (they are an inverted member of an
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
        Ex::Const(_) => bump("Const"),
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

/// A PIECE OF A LITERAL: a number, a numeric-string leaf, an integer power of a piece, or a sum
/// or product of pieces -- variable-free arithmetic that denotes one rational value.
///
/// The exact number domain holds every such value within the 1,100-bit cap as one
/// `Rat`, which [`mu_rat`] prices as its spelling. The f64 domain admits only
/// numbers whose components are at most 2^1022 (`rat::number_domain`) and holds the rest in
/// pieces: a literal whose exact fraction leaves it stays a leaf as written
/// (`1.2345678901234567e-305`, whose denominator is 10^321), and a product, sum or power that
/// leaves it stays a product, sum or power (`1.2345678901234567e-305 / 9.876543210987654e-10`
/// is the leaf times the number `1/9.876543210987654e-10`). Priced piece by piece, such a value
/// costs what its pieces cost, above its float. [`priced`] prices the pieces of a bag, and a
/// node that is pieces throughout, as the one literal they denote ([`piece_price`]).
pub(super) fn is_piece(e: &Ex, view: &TokenView) -> bool {
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
    let (p, q) = crate::ac::rat::token_rational(s)?;
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
pub(super) fn bag_literal_value(members: &[Ex], sum: bool, view: &TokenView) -> Option<Rat> {
    let _exact = crate::ac::rat::number_domain(false);
    pieces_value(members.iter().filter(|f| is_piece(f, view)), sum, view)
}

/// THE PRICE OF A LITERAL HELD IN PIECES ([`is_piece`]): `structural`, its pieces priced one
/// by one, at most what the exact domain prices the value they denote ([`mu_rat`]), and never
/// below one bit per node,
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
/// domain costs what the same value costs as a `Num`; one beyond the 1,100-bit cap keeps its
/// print's price. A function of the string, so each thread remembers the prices it computed,
/// emptying the table at 65,536 entries.
pub(super) fn mu_numeric_leaf(s: &str) -> u64 {
    thread_local! {
        static LEAF_PRICES: std::cell::RefCell<rustc_hash::FxHashMap<Box<str>, u64>> =
            std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    }
    if let Some(v) = LEAF_PRICES.with(|m| m.borrow().get(s).copied()) {
        return v;
    }
    // The written negative zero is the zero and one sign bit (its value, 0, has no sign).
    if s == NEG_ZERO {
        return mu_rat(&Rat::ZERO).saturating_add(MU_SIGN);
    }
    let value = {
        let _exact = crate::ac::rat::number_domain(false);
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
        // A NAMED CONSTANT is priced as a fitted constant at every occurrence, whether the
        // expression uses it once (it is then `Ex::Const`) or more than once (a leaf, read as a
        // variable): its price does not depend on how often the expression uses it, so an answer
        // that leaves a name in one place, read then as a fitted constant, never prices above
        // the expression it came from.
        Ex::Leaf(t) if view.is_named(*t) => (mu_free(), false),
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
        Ex::Const(_) => (mu_free(), false),
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
            // A product of numbers alone is numbers the number domain holds apart (it admits no
            // number for their product, `1e200*1e200`): it pays its product charge, as the same
            // product beside a variable does, and a known value caps it below.
            let structural = if is_negation(v) {
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
                        let _exact = crate::ac::rat::number_domain(false);
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
