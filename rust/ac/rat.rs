//! Exact rational arithmetic for the AC core.
//!
//! Contract D1: "number literals denote their exact rational values" -- so the core's numeric
//! type is an exact rational, not an f64. Coefficient and exponent arithmetic thereby becomes
//! COMPUTATION that cannot be wrong (replacing the mined-and-sampled coefficient rule family).
//!
//! Representation: `p/q` with `q > 0` and `gcd(|p|, q) == 1`, in one of two forms. The SMALL
//! form holds both components in `i128` (never `i128::MIN`) and is the fast path every common
//! number takes. The BIG form holds them as big integers, up to a size cap of `CAP_BITS` bits
//! each (number plan phase 2: exact numbers without a fixed width). A value that fits the small
//! form is ALWAYS small, so the derived `Eq` and `Hash` are value equality (the ground-rule
//! index, the normal-form memo, bucket keys and the matcher rely on that).
//!
//! Every operation is CHECKED: when the result would leave the representable set it returns
//! `None` and the caller keeps the symbolic form instead of folding -- refusing to compute is
//! always sound here, computing wrongly never is.
//!
//! `WIDE_RESULTS` is on (phase 2c): an operation whose result leaves the small form computes it
//! in the big form, and the result stands if it is REPRESENTABLE in the current number domain
//! (`number_domain`): within the cap in the exact domain (`real` mode), and in the f64 domain
//! (every other mode) only when the deployed float64 evaluator reads it within one rounding and
//! every printed spelling of it reads back as itself. Every refusal keeps the symbolic form, as
//! a 128-bit overflow always did.

use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// The size cap of the big form: numerator and denominator have at most this many bits. Every
/// float64 is an exact fraction within it (the largest double is below 2^1024, the smallest
/// subnormal is 2^-1074), and srbf's largest number needs 375 bits (plan §2).
pub const CAP_BITS: u64 = 1100;

/// Whether a result of small operands that leaves the small form is computed in the big form
/// (phase 2c) or refused, as before phase 2 (phase 2a).
pub(crate) const WIDE_RESULTS: bool = true;

thread_local! {
    /// The current thread's number domain: f64 (`true`, the default and every mode but `real`)
    /// or exact (`false`, `real`). The constructors enter it from their context's mode.
    static F64_NUMBERS: Cell<bool> = const { Cell::new(true) };
}

/// Restores the previous number domain when dropped (see [`number_domain`]).
pub struct DomainGuard(bool);

impl Drop for DomainGuard {
    fn drop(&mut self) {
        F64_NUMBERS.with(|c| c.set(self.0));
    }
}

/// Enter a number domain on this thread until the returned guard drops.
///
/// In the f64 domain (`true`) a number is ADMISSIBLE when it is zero or its numerator and its
/// denominator are both at most 2^1022 in magnitude (design 2c; reviews H1, H2 and #65's M1).
/// Such a number is normal and so is its reciprocal, so every printed spelling -- one token,
/// `/ p q`, a divisor-side reciprocal -- reads back within one rounding as the same number,
/// where a subnormal reads far from its exact value (`5e-324` is 4.94e-324) and a number beyond
/// DBL_MAX reads as inf. Literals outside the set (`5e307`, `1e400`, `5e-324`) stay leaves as
/// written, as in main. Every 128-bit value is admissible, so only `from_big` checks. In the
/// f64 domain an integer beyond 2^53 also has no known parity ([`Rat::parity`]). The exact
/// domain (`false`) admits every number within the cap.
pub fn number_domain(f64_numbers: bool) -> DomainGuard {
    DomainGuard(F64_NUMBERS.with(|c| c.replace(f64_numbers)))
}

/// Whether this thread is in the f64 number domain.
pub(crate) fn f64_numbers() -> bool {
    F64_NUMBERS.with(|c| c.get())
}

/// `n <= 2^1022`, from the bit length (a second test only at exactly 1,023 bits).
fn le_two_1022(n: &BigUint) -> bool {
    let bits = n.bits();
    bits <= 1022 || (bits == 1023 && n.trailing_zeros() == Some(1022))
}

/// The f64 domain's admissibility of a reduced nonzero `p/q` (see [`number_domain`]): both
/// components at most 2^1022. The magnitude then lies in [2^-1022, 2^1022], inside the normal
/// range, and the set is closed under negation and reciprocals, so every spelling the printer
/// can choose (one token, `/ p q`, a divisor-side reciprocal) re-reads as the same number.
/// (With `|p| <= DBL_MAX` alone, `5e307` had no admissible reciprocal: `5e307/5e307` could not
/// cancel, and a coefficient beside it re-read differently.)
fn f64_admissible(p: &BigInt, q: &BigInt) -> bool {
    le_two_1022(p.magnitude()) && le_two_1022(q.magnitude())
}

/// The largest |log2| a number of the current domain can have: 1,022 in the f64 domain, the
/// cap in the exact one (a reduced component of more bits is refused either way).
fn log2_limit() -> i64 {
    if f64_numbers() {
        1022
    } else {
        CAP_BITS as i64
    }
}

/// An exact rational `p/q`, normalized (`q > 0`, `gcd(|p|, q) == 1`).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Rat(Repr);

#[derive(Clone, PartialEq, Eq, Hash)]
enum Repr {
    Small { p: i128, q: i128 },
    Big(Arc<BigParts>),
}

/// The big form's components: reduced, `q > 0`, at least one of them outside the small form,
/// both within `CAP_BITS`.
#[derive(Clone, PartialEq, Eq, Hash)]
struct BigParts {
    p: BigInt,
    q: BigInt,
}

impl fmt::Debug for Rat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (p, q) = self.big_parts();
        write!(f, "Rat {{ p: {p}, q: {q} }}")
    }
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

/// `x` as a small-form component (`i128`, never `i128::MIN`).
fn small_of(x: &BigInt) -> Option<i128> {
    x.to_i128().filter(|&v| v != i128::MIN)
}

/// Exact 128x128 -> 256-bit unsigned multiplication by the schoolbook double-word method:
/// split each operand into 64-bit halves, take the four partial products (each fits a
/// u128 exactly: 64x64 -> 128), and assemble with carries. Returns (hi, lo); two results
/// compare as the mathematical products via lexicographic (hi, lo) order.
#[inline]
fn widening_mul_u128(a: u128, b: u128) -> (u128, u128) {
    const MASK: u128 = (1u128 << 64) - 1;
    let (a_hi, a_lo) = (a >> 64, a & MASK);
    let (b_hi, b_lo) = (b >> 64, b & MASK);
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    let mid = lh.wrapping_add(hl);
    let carry_mid = (mid < lh) as u128; // overflow out of the middle sum: worth 2^128
    let lo = ll.wrapping_add(mid << 64);
    let carry_lo = (lo < ll) as u128;
    let hi = hh + (mid >> 64) + (carry_mid << 64) + carry_lo;
    (hi, lo)
}

impl Rat {
    pub const ZERO: Rat = Rat(Repr::Small { p: 0, q: 1 });
    pub const ONE: Rat = Rat(Repr::Small { p: 1, q: 1 });
    pub const NEG_ONE: Rat = Rat(Repr::Small { p: -1, q: 1 });

    /// The small form, trusted: `q > 0`, reduced, neither component `i128::MIN`.
    #[inline]
    const fn small(p: i128, q: i128) -> Rat {
        Rat(Repr::Small { p, q })
    }

    /// Build `p/q` normalized. `None` if `q == 0` (not a rational) or normalization overflows
    /// (`p == i128::MIN` cannot be negated).
    pub fn new(p: i128, q: i128) -> Option<Rat> {
        if q == 0 || p == i128::MIN || q == i128::MIN {
            return None;
        }
        let (p, q) = if q < 0 {
            (p.checked_neg()?, -q)
        } else {
            (p, q)
        };
        let g = gcd(p, q);
        // g == 0 only when p == 0 and q == 0; q != 0 here, so g >= 1.
        Some(Rat::small(p / g, q / g))
    }

    /// Build `p/q` from big integers, normalized and in the small form whenever it fits.
    /// `None` if `q == 0`, a reduced component exceeds `CAP_BITS`, or -- in the f64 domain -- the
    /// number is not admissible (see [`number_domain`]).
    pub fn from_big(p: BigInt, q: BigInt) -> Option<Rat> {
        if q.is_zero() {
            return None;
        }
        let (p, q) = if q.is_negative() { (-p, -q) } else { (p, q) };
        let g = p.gcd(&q);
        let (p, q) = if g.is_one() || g.is_zero() {
            (p, q)
        } else {
            (p / &g, q / &g)
        };
        if let (Some(ps), Some(qs)) = (small_of(&p), small_of(&q)) {
            return Some(Rat::small(ps, qs));
        }
        if p.bits() > CAP_BITS || q.bits() > CAP_BITS {
            return None;
        }
        if f64_numbers() && !f64_admissible(&p, &q) {
            return None;
        }
        Some(Rat(Repr::Big(Arc::new(BigParts { p, q }))))
    }

    pub fn int(n: i128) -> Rat {
        // "No small Rat ever holds i128::MIN" is a LOAD-BEARING invariant (MIN cannot be
        // negated or abs'd; release builds have no overflow checks, so a violation would WRAP
        // silently downstream). `Rat::new` refuses MIN; this bypass constructor must too,
        // loudly. Every real caller passes bounded values.
        assert!(
            n != i128::MIN,
            "Rat cannot represent i128::MIN (unnegatable)"
        );
        Rat::small(n, 1)
    }

    /// The components in the small form, or `None` for a big number. For code with a 128-bit
    /// fast path; a caller must not read `None` as "not an integer" or "not a number".
    #[inline]
    pub fn small_parts(&self) -> Option<(i128, i128)> {
        match &self.0 {
            Repr::Small { p, q } => Some((*p, *q)),
            Repr::Big(_) => None,
        }
    }

    /// The numerator as an integer. A component of an existing number: within the cap and
    /// independent of the current number domain.
    pub fn numer(&self) -> Rat {
        match &self.0 {
            Repr::Small { p, .. } => Rat::small(*p, 1),
            Repr::Big(b) => Rat::integer(b.p.clone()),
        }
    }

    /// The denominator as a (positive) integer (see [`Rat::numer`]).
    pub fn denom(&self) -> Rat {
        match &self.0 {
            Repr::Small { q, .. } => Rat::small(*q, 1),
            Repr::Big(b) => Rat::integer(b.q.clone()),
        }
    }

    /// An integer known to be within the cap, small whenever it fits, without the domain check.
    fn integer(n: BigInt) -> Rat {
        match small_of(&n) {
            Some(v) => Rat::small(v, 1),
            None => Rat(Repr::Big(Arc::new(BigParts {
                p: n,
                q: BigInt::one(),
            }))),
        }
    }

    /// The numerator's decimal digits (with a leading `-` when negative).
    pub fn numer_string(&self) -> String {
        match &self.0 {
            Repr::Small { p, .. } => p.to_string(),
            Repr::Big(b) => b.p.to_string(),
        }
    }

    /// The denominator's decimal digits.
    pub fn denom_string(&self) -> String {
        match &self.0 {
            Repr::Small { q, .. } => q.to_string(),
            Repr::Big(b) => b.q.to_string(),
        }
    }

    /// The components as big integers (allocates for the small form).
    pub fn big_parts(&self) -> (BigInt, BigInt) {
        match &self.0 {
            Repr::Small { p, q } => (BigInt::from(*p), BigInt::from(*q)),
            Repr::Big(b) => (b.p.clone(), b.q.clone()),
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        matches!(self.0, Repr::Small { p: 0, .. })
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        matches!(self.0, Repr::Small { p: 1, q: 1 })
    }

    #[inline]
    pub fn is_integer(&self) -> bool {
        match &self.0 {
            Repr::Small { q, .. } => *q == 1,
            Repr::Big(b) => b.q.is_one(),
        }
    }

    #[inline]
    pub fn is_negative(&self) -> bool {
        match &self.0 {
            Repr::Small { p, .. } => *p < 0,
            Repr::Big(b) => b.p.is_negative(),
        }
    }

    /// -1, 0 or 1.
    #[inline]
    pub fn signum(&self) -> i32 {
        match &self.0 {
            Repr::Small { p, .. } => p.signum() as i32,
            Repr::Big(b) => match b.p.sign() {
                Sign::Minus => -1,
                Sign::NoSign => 0,
                Sign::Plus => 1,
            },
        }
    }

    /// The parity of an integer: `Some(true)` odd, `Some(false)` even. `None` for a non-integer
    /// and, in the f64 domain, for an integer beyond 2^53: the float64 evaluator reads such an
    /// integer as an even float, so its parity is not known there (design 2c, review H3) and
    /// every parity-dependent rewrite refuses. The exact domain has the exact parity.
    pub fn parity(&self) -> Option<bool> {
        match &self.0 {
            Repr::Small { p, q } => {
                if *q != 1 || (f64_numbers() && p.unsigned_abs() > 1u128 << 53) {
                    return None;
                }
                Some(p % 2 != 0)
            }
            Repr::Big(b) => (b.q.is_one() && !f64_numbers()).then(|| b.p.is_odd()),
        }
    }

    /// CERTAINLY an odd integer (see [`Rat::parity`]).
    pub fn is_odd_integer(&self) -> bool {
        self.parity() == Some(true)
    }

    /// CERTAINLY an even integer (see [`Rat::parity`]); zero is even.
    pub fn is_even_integer(&self) -> bool {
        self.parity() == Some(false)
    }

    /// The integer value if this is an integer in the small form. For counts, indices and
    /// loop bounds only: a big integer gives `None`, so a caller asking "is this an integer?",
    /// "is it odd?" or "is it negative?" must use `is_integer`, `is_odd_integer`,
    /// `is_even_integer` or `signum` instead (map §2.4: reading `None` as "not an integer"
    /// is unsound at four sites once integers exceed 128 bits).
    #[inline]
    pub fn small_int(&self) -> Option<i128> {
        match &self.0 {
            Repr::Small { p, q: 1 } => Some(*p),
            _ => None,
        }
    }

    /// `self` compared with the integer `n`, exactly.
    pub fn cmp_int(&self, n: i128) -> Ordering {
        self.cmp_exact(&Rat::int(n))
    }

    /// `ceil(|p/q|)` exactly, in integers -- `None` on overflow. Used by the inverse-pair
    /// band guard, which needs a magnitude comparison and must not acquire an f64 reading
    /// (f64 has authority nowhere in the engine; see `to_f64`).
    pub(crate) fn ceil_abs(&self) -> Option<i128> {
        let (p, q) = self.small_parts()?;
        let p = p.checked_abs()?;
        let q = q.abs();
        if q == 0 {
            return None;
        }
        p.checked_add(q - 1).map(|n| n / q)
    }

    /// `floor(|p/q|)`, saturated at `i128::MAX` (still a valid lower bound on `|self|`).
    pub(crate) fn floor_abs_saturating(&self) -> i128 {
        match &self.0 {
            Repr::Small { p, q } => (p.unsigned_abs() / (*q as u128)) as i128,
            Repr::Big(b) => (b.p.abs() / &b.q).to_i128().unwrap_or(i128::MAX),
        }
    }

    /// The canonical members of a product bag (units dropped; `[]` is the product 1), by
    /// [`Rat::partition_with`] with exact products admitted by the number domain (in the f64
    /// domain: the result is admissible, see [`number_domain`]).
    pub fn partition_product(mut members: Vec<Rat>) -> Vec<Rat> {
        members.retain(|m| !m.is_one());
        if members.iter().any(Rat::is_zero) {
            return vec![Rat::ZERO];
        }
        Self::partition_with(members, Rat::checked_mul, Rat::is_one)
    }

    /// The canonical members of a sum bag (zeros dropped; `[]` is the sum 0), by the same
    /// rule as [`Rat::partition_product`].
    pub fn partition_sum(mut members: Vec<Rat>) -> Vec<Rat> {
        members.retain(|m| !m.is_zero());
        Self::partition_with(members, Rat::checked_add, Rat::is_zero)
    }

    /// A bag folded as far as `fold` allows, to members of which NO TWO fold: sorted
    /// ascending, each member is folded into the lowest kept member it folds with (the result
    /// re-enters the queue; a `neutral` result vanishes), otherwise it is kept. The result is
    /// sorted and a function of the multiset.
    ///
    /// Why no two members may fold: a printed bag re-reads as a left-nested product (sum) in
    /// its printed order, which need not be the sorted order (a sum prints its positive terms
    /// first), so every subset of the members is canonicalised on its own on the way. A bag
    /// in which no two members fold is the partition of each of its subsets, so it re-reads
    /// to itself in any order. Weaker rules failed: the greedy fold of main stopped after one
    /// pass (three rows of the phase-1 review), all-or-nothing folding broke on a prefix that
    /// folds where the whole bag does not (`x1*1e300*1e10*1e-100`), and folding sorted
    /// neighbours only left two members apart that the printed order brought together
    /// (`x1 + 912.../295... - 4e28 - 1e-325`, real mode). Each member is tried against at most
    /// the kept members and every fold removes one, so the cost is quadratic in the bag, and
    /// a refusal that the magnitudes decide costs no big-integer arithmetic (`log2_window`).
    pub fn partition_with(
        mut members: Vec<Rat>,
        fold: impl Fn(&Rat, &Rat) -> Option<Rat>,
        neutral: impl Fn(&Rat) -> bool,
    ) -> Vec<Rat> {
        members.sort_unstable_by(|a, b| a.cmp_exact(b));
        let mut queue: std::collections::VecDeque<Rat> = members.into();
        let mut kept: Vec<Rat> = Vec::new();
        while let Some(m) = queue.pop_front() {
            match kept
                .iter()
                .enumerate()
                .find_map(|(i, k)| fold(k, &m).map(|x| (i, x)))
            {
                Some((i, x)) => {
                    kept.remove(i);
                    if !neutral(&x) {
                        queue.push_front(x);
                    }
                }
                None => {
                    let at = kept.partition_point(|k| k.cmp_exact(&m) == Ordering::Less);
                    kept.insert(at, m);
                }
            }
        }
        kept
    }

    pub fn checked_add(&self, o: &Rat) -> Option<Rat> {
        if let (Some((p1, q1)), Some((p2, q2))) = (self.small_parts(), o.small_parts()) {
            // p1/q1 + p2/q2 = (p1*q2 + p2*q1) / (q1*q2), then normalize.
            let small = (|| {
                let a = p1.checked_mul(q2)?;
                let b = p2.checked_mul(q1)?;
                Rat::new(a.checked_add(b)?, q1.checked_mul(q2)?)
            })();
            if small.is_some() || !WIDE_RESULTS {
                return small;
            }
        }
        // A same-sign sum is at least its larger member: beyond the domain's magnitude it
        // refuses without computing (bounds from bit lengths only, see `log2_window`).
        if self.is_negative() == o.is_negative()
            && self.log2_window().0.max(o.log2_window().0) >= log2_limit()
        {
            return None;
        }
        let ((p1, q1), (p2, q2)) = (self.big_parts(), o.big_parts());
        Rat::from_big(&p1 * &q2 + &p2 * &q1, q1 * q2)
    }

    /// Exclusive bounds `(lo, hi)` with `2^lo < |self| < 2^hi`, from the bit lengths of the
    /// components alone (`self != 0`). Lets a fold that surely leaves the domain refuse before
    /// any big-integer arithmetic: a bag of many refused big members costs comparisons, not
    /// products and gcds.
    fn log2_window(&self) -> (i64, i64) {
        let (bp, bq) = match &self.0 {
            Repr::Small { p, q } => (
                128 - i64::from(p.unsigned_abs().leading_zeros()),
                128 - i64::from(q.unsigned_abs().leading_zeros()),
            ),
            Repr::Big(b) => (b.p.bits() as i64, b.q.bits() as i64),
        };
        (bp - bq - 1, bp - bq + 1)
    }

    pub fn checked_mul(&self, o: &Rat) -> Option<Rat> {
        if let (Some((p1, q1)), Some((p2, q2))) = (self.small_parts(), o.small_parts()) {
            // Cross-reduce first so intermediates stay small: (p1/q2')·(p2/q1').
            let small = (|| {
                let g1 = gcd(p1, q2).max(1);
                let g2 = gcd(p2, q1).max(1);
                let p = (p1 / g1).checked_mul(p2 / g2)?;
                let q = (q1 / g2).checked_mul(q2 / g1)?;
                Rat::new(p, q)
            })();
            if small.is_some() || !WIDE_RESULTS {
                return small;
            }
        }
        if self.is_zero() || o.is_zero() {
            return Some(Rat::ZERO);
        }
        let ((l1, h1), (l2, h2)) = (self.log2_window(), o.log2_window());
        let limit = log2_limit();
        if l1 + l2 >= limit || h1 + h2 <= -limit {
            return None; // surely beyond the domain's magnitude: no product computed
        }
        let ((p1, q1), (p2, q2)) = (self.big_parts(), o.big_parts());
        Rat::from_big(p1 * p2, q1 * q2)
    }

    pub fn checked_neg(&self) -> Option<Rat> {
        match &self.0 {
            Repr::Small { p, q } => Some(Rat::small(p.checked_neg()?, *q)),
            // A sign flip keeps both magnitudes: within the cap, and admissible in every
            // domain the number already is, so no domain check. It stays big: a big
            // magnitude is at least 2^127 and `i128::MIN` is not small.
            Repr::Big(b) => Some(Rat(Repr::Big(Arc::new(BigParts {
                p: -b.p.clone(),
                q: b.q.clone(),
            })))),
        }
    }

    /// Multiplicative inverse. `None` for zero (1/0 is not a rational -- the caller keeps the
    /// symbolic `Pow(0, -1)` for the rules/value-set machinery to judge).
    pub fn checked_inv(&self) -> Option<Rat> {
        if self.is_zero() {
            return None;
        }
        match &self.0 {
            Repr::Small { p, q } => Rat::new(*q, *p),
            Repr::Big(b) => Rat::from_big(b.q.clone(), b.p.clone()),
        }
    }

    /// `self^n` for an integer exponent. `None` on overflow or `0^negative`.
    /// `0^0 == 1` here on purpose: it matches both Python's `0.0**0 == 1.0` and the engine's
    /// deployed fold, and the exponent 0 only arises from exact arithmetic, never from data.
    pub fn checked_pow_int(&self, n: i128) -> Option<Rat> {
        if n == 0 {
            return Some(Rat::ONE);
        }
        if self.is_zero() {
            return if n > 0 { Some(Rat::ZERO) } else { None };
        }
        let (base, n) = if n < 0 {
            (self.checked_inv()?, n.checked_neg()?)
        } else {
            (self.clone(), n)
        };
        // (+-1)^n: exact for ANY exponent magnitude -- but (-1)^n needs n's parity, which the
        // f64 domain does not know beyond 2^53.
        if base == Rat::ONE {
            return Some(Rat::ONE);
        }
        if base == Rat::NEG_ONE {
            if f64_numbers() && n.unsigned_abs() > 1u128 << 53 {
                return None;
            }
            return Some(if n % 2 == 0 { Rat::ONE } else { Rat::NEG_ONE });
        }
        if base.small_parts().is_some() {
            // Exponentiation by squaring, checked throughout. Cap the exponent so a
            // pathological `pow(x, 10^30)` never spins here -- i128 overflow would refuse it
            // anyway for any |base| != 1, and |base| == 1 is handled exactly.
            let small = (|| {
                if n > 512 {
                    return None;
                }
                let mut acc = Rat::ONE;
                let mut b = base.clone();
                let mut e = n;
                while e > 0 {
                    if e & 1 == 1 {
                        acc = acc
                            .checked_mul(&b)?
                            .small_parts()
                            .map(|(p, q)| Rat::small(p, q))?;
                    }
                    e >>= 1;
                    if e > 0 {
                        b = b
                            .checked_mul(&b)?
                            .small_parts()
                            .map(|(p, q)| Rat::small(p, q))?;
                    }
                }
                Some(acc)
            })();
            if small.is_some() || !WIDE_RESULTS {
                return small;
            }
        }
        // The big form: the result is exactly p^n / q^n (a reduced fraction stays reduced), so
        // its size is known before computing. With b bits, p^n has between n(b-1)+1 and nb bits;
        // refuse once the lower bound leaves the cap.
        let (p, q) = base.big_parts();
        let n_u = u64::try_from(n).ok()?;
        for x in [&p, &q] {
            let b = x.bits();
            if b > 1 && n_u.checked_mul(b - 1).is_none_or(|lo| lo >= CAP_BITS) {
                return None;
            }
        }
        let n32 = u32::try_from(n_u).ok()?;
        Rat::from_big(
            num_traits::pow(p, n32 as usize),
            num_traits::pow(q, n32 as usize),
        )
    }

    /// `self^n` for an integer `n` of any size; `None` unless `n` is an integer or when the
    /// result is not representable. Beyond `i128` only 0, 1 and -1 have a power within the
    /// cap.
    pub fn checked_pow_integer(&self, n: &Rat) -> Option<Rat> {
        if !n.is_integer() {
            return None;
        }
        if let Some(k) = n.small_int() {
            return self.checked_pow_int(k);
        }
        if self.is_zero() {
            return (!n.is_negative()).then_some(Rat::ZERO);
        }
        if *self == Rat::ONE {
            return Some(Rat::ONE);
        }
        if *self == Rat::NEG_ONE {
            return n
                .parity()
                .map(|odd| if odd { Rat::NEG_ONE } else { Rat::ONE });
        }
        None
    }

    /// The exact `k`-th root for a positive integer `k` of any size (see `checked_root`).
    /// Beyond `i128` only 0, 1 and (at an odd index) -1 have a rational root: every other
    /// root would need a component beyond the cap.
    pub fn checked_root_integer(&self, k: &Rat) -> Option<Rat> {
        if !k.is_integer() {
            return None;
        }
        if let Some(k) = k.small_int() {
            return self.checked_root(k);
        }
        if k.is_negative() {
            return None;
        }
        if *self == Rat::NEG_ONE {
            return k.is_odd_integer().then_some(Rat::NEG_ONE);
        }
        (self.is_zero() || self.is_one()).then(|| self.clone())
    }

    /// `|self|`.
    pub fn abs(&self) -> Rat {
        if self.is_negative() {
            self.checked_neg().expect("a negation stays representable")
        } else {
            self.clone()
        }
    }

    /// Exact k-th root, if it exists as a rational: `self == r^k` with matching sign rules
    /// (`k` even requires `self >= 0`; the even root returned is the non-negative one).
    pub fn checked_root(&self, k: i128) -> Option<Rat> {
        if k <= 0 {
            return None;
        }
        if k == 1 {
            return Some(self.clone());
        }
        if self.is_negative() && k % 2 == 0 {
            return None;
        }
        if let Some((p, q)) = self.small_parts() {
            let sign: i128 = if p < 0 { -1 } else { 1 };
            let rp = int_root(p.checked_abs()?, k)?;
            let rq = int_root(q, k)?;
            return Rat::new(sign * rp, rq);
        }
        // A root of a big number: every root index beyond the cap has no integer root but 0/1.
        let k32 = u32::try_from(k)
            .ok()
            .filter(|&k| u64::from(k) <= CAP_BITS)?;
        let (p, q) = self.big_parts();
        let root = |x: &BigInt| -> Option<BigInt> {
            let r = x.abs().nth_root(k32);
            (num_traits::pow(r.clone(), k32 as usize) == x.abs()).then_some(r)
        };
        let (rp, rq) = (root(&p)?, root(&q)?);
        Rat::from_big(if p.is_negative() { -rp } else { rp }, rq)
    }

    /// Compare exactly (no float detour): `p1/q1 <=> p2/q2` == `p1*q2 <=> p2*q1` with q > 0.
    /// EXACT at every magnitude: the fast path cross-multiplies in i128; on overflow the
    /// wide path computes both cross-products exactly in 256 bits (the former f64 fallback
    /// broke totality -- Equal on 6.6% and inverted 0.35% of adjacent sub-1e-18 literal
    /// pairs -- and the canonical sort's totality is load-bearing even though ordering is
    /// a CANONICALIZATION concern, not a soundness one; see `expr.rs` on why order never
    /// changes denotation). A big operand compares by big cross-products.
    pub fn cmp_exact(&self, o: &Rat) -> Ordering {
        let (Some((p1, q1)), Some((p2, q2))) = (self.small_parts(), o.small_parts()) else {
            let ((p1, q1), (p2, q2)) = (self.big_parts(), o.big_parts());
            return (p1 * q2).cmp(&(p2 * q1));
        };
        // Fast path: both cross-products fit in i128 (every common magnitude). This is
        // byte-identical to the historical exact path, so the hot path costs nothing new.
        if let (Some(a), Some(b)) = (p1.checked_mul(q2), p2.checked_mul(q1)) {
            return a.cmp(&b);
        }
        // EXACT wide path: an f64 estimate is not monotone in the true rational order at
        // these magnitudes (three roundings), and this comparison backs a TOTAL order --
        // the sort's totality check panics on any cycle -- so the cross-products are
        // computed exactly in 256 bits. Signs first: q > 0 is a Rat invariant, so
        // sign(p1*q2) = sign(p1).
        let (s1, s2) = (p1.signum(), p2.signum());
        if s1 != s2 {
            return s1.cmp(&s2);
        }
        let a = widening_mul_u128(p1.unsigned_abs(), q2 as u128);
        let b = widening_mul_u128(p2.unsigned_abs(), q1 as u128);
        let mag = a.cmp(&b); // (hi, lo) tuples compare lexicographically -- exact
        if s1 < 0 {
            mag.reverse()
        } else {
            mag
        }
    }

    /// The f64 reading of this rational.
    ///
    /// This used to be `#[cfg(test)]`, carrying the note "f64 has authority nowhere in
    /// the engine". That was true of a single-mode engine and is exactly the sentence the
    /// mode split overturns: in `Mode.f64` the deployed f64 evaluator IS the authority,
    /// so the constructor must be able to ask it. `Mode.real` still never calls this --
    /// there, f64 has authority nowhere, as before. A big number reads as its nearest f64.
    pub fn to_f64(&self) -> f64 {
        match &self.0 {
            Repr::Small { p, q } => *p as f64 / *q as f64,
            Repr::Big(b) => big_ratio_to_f64(&b.p, &b.q),
        }
    }

    /// The f64 NEAREST to this value. `to_f64` rounds each component to f64 before the
    /// division, which is off by an ulp or two once either leaves 53 bits (the permissive
    /// literal fold's inputs routinely do: `2e29 / 426738538271436458205631863649`); here
    /// that candidate is walked to the correctly rounded neighbour by EXACT midpoint tests
    /// (`cmp_dyadic`), so the fold really lands on the nearest float -- the value
    /// `float(Fraction(p, q))` produces -- at every magnitude. A big number is rounded
    /// directly from its components.
    pub fn to_f64_nearest(&self) -> f64 {
        self.nearest_walk().0
    }

    /// `to_f64_nearest`, but `None` unless the walk settled, i.e. the result is certified to
    /// be the correctly rounded f64. Callers that build an enclosure around the value (the
    /// interval kernel's one-ulp leaf bracket) need the certificate. Every midpoint test is
    /// exact, so this fails only if the walk does not settle within its step budget, which
    /// `to_f64`'s candidate (within a few ulps) never needs.
    pub fn to_f64_nearest_certified(&self) -> Option<f64> {
        match self.nearest_walk() {
            (y, true) => Some(y),
            (_, false) => None,
        }
    }

    /// The midpoint walk behind both readers: `(candidate, certified)`.
    fn nearest_walk(&self) -> (f64, bool) {
        let Some((p, q)) = self.small_parts() else {
            let (p, q) = self.big_parts();
            let y = big_ratio_to_f64(&p, &q);
            return (y, y.is_finite());
        };
        let mut y = self.to_f64();
        if !y.is_finite() {
            return (y, false);
        }
        for _ in 0..8 {
            let up = next_up(y);
            if cmp_dyadic(p, q, dyadic_midpoint(y, up)) == Ordering::Greater {
                y = up;
                continue;
            }
            let down = next_down(y);
            if cmp_dyadic(p, q, dyadic_midpoint(down, y)) == Ordering::Less {
                y = down;
                continue;
            }
            return (y, true);
        }
        (y, false)
    }

    /// The shortest exact decimal string, if one exists (`q == 2^a * 5^b`): `1/2 -> "0.5"`,
    /// `-7/4 -> "-1.75"`, `3 -> "3"`. `None` for e.g. `1/3` (the serializer then spells the
    /// division structurally). Exactness is by construction: multiply p by 2s and 5s until the
    /// denominator is a power of ten, then place the decimal point.
    pub fn exact_decimal(&self) -> Option<String> {
        let Some((p, q)) = self.small_parts() else {
            let (p, q) = self.big_parts();
            return big_exact_decimal(&p, &q);
        };
        if q == 1 {
            return Some(p.to_string());
        }
        let (mut a, mut b) = (0u32, 0u32);
        let mut rest = q;
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
        // Scale numerator so denominator becomes 10^k with k = max(a, b).
        let k = a.max(b);
        let scaled = (|| {
            let mut scaled = p.checked_abs()?;
            for _ in 0..(k - a) {
                scaled = scaled.checked_mul(2)?;
            }
            for _ in 0..(k - b) {
                scaled = scaled.checked_mul(5)?;
            }
            Some(scaled)
        })();
        let Some(scaled) = scaled else {
            return if WIDE_RESULTS {
                big_exact_decimal(&BigInt::from(p), &BigInt::from(q))
            } else {
                None
            };
        };
        Some(place_point(p < 0, scaled.to_string(), k as usize))
    }

    /// Parse a decimal token EXACTLY: `"7" -> 7`, `"-1.75" -> -7/4`, `"0.2" -> 1/5`,
    /// `"1e-3" -> 1/1000`, `"1." -> 1`. This is a DECIMAL parse, not a float parse -- `"0.2"`
    /// means one fifth, exactly, even though the f64 nearest to it does not.
    pub fn parse_decimal(s: &str) -> Option<Rat> {
        // Only the numeral grammar (B2): the parse below used to accept `-+5` as -5.
        if !crate::utils::is_decimal_numeral(s) {
            return None;
        }
        // Split off an exponent part (`e`/`E`).
        let (mant, exp) = match s.find(['e', 'E']) {
            Some(i) => (&s[..i], s[i + 1..].parse::<i32>().ok()?),
            None => (s, 0i32),
        };
        let (sign, mant) = match mant.strip_prefix('-') {
            Some(rest) => (-1i128, rest),
            None => (1i128, mant),
        };
        let mant = mant.strip_prefix('+').unwrap_or(mant);
        let (int_part, frac_part) = match mant.find('.') {
            Some(i) => (&mant[..i], &mant[i + 1..]),
            None => (mant, ""),
        };
        if int_part.is_empty() && frac_part.is_empty() {
            return None;
        }
        if !int_part.chars().all(|c| c.is_ascii_digit())
            || !frac_part.chars().all(|c| c.is_ascii_digit())
        {
            return None;
        }
        let shift = exp as i64 - frac_part.len() as i64;
        let small = parse_decimal_small(sign, int_part, frac_part, shift);
        if small.is_some() || !WIDE_RESULTS {
            return small;
        }
        parse_decimal_big(sign < 0, int_part, frac_part, shift)
    }
}

/// `parse_decimal` in the small form: `None` once a component leaves `i128`.
fn parse_decimal_small(sign: i128, int_part: &str, frac_part: &str, shift: i64) -> Option<Rat> {
    let mut p: i128 = 0;
    for c in int_part.chars().chain(frac_part.chars()) {
        p = p.checked_mul(10)?.checked_add((c as u8 - b'0') as i128)?;
    }
    p = p.checked_mul(sign)?;
    // Zero mantissa: bail BEFORE the scaling loops. The positive-exponent loop relies
    // on checked_mul OVERFLOW to bound absurd exponents, and 0 * 10 never overflows,
    // so "0e2147483647" would otherwise spin the full exponent count.
    if p == 0 {
        return Some(Rat::ZERO);
    }
    let mut q: i128 = 1;
    if shift >= 0 {
        for _ in 0..shift {
            p = p.checked_mul(10)?;
        }
    } else {
        // p / 10^k with q = 10^k materialized NAIVELY overflows i128 at k = 39 even
        // when the REDUCED fraction fits: the engine's own exact-decimal emitter
        // writes e.g. (1/4)^25 = 1/2^50 as a 50-digit decimal (p = 5^50, 35 digits,
        // fits), and refusing to read it back demoted the exact rational to an opaque
        // overlay leaf, which sorts as a key factor among the variables instead of as
        // the stripped coefficient, so bag order changed across calls (the 1M-gate
        // idempotence pair 274133/514869). Cancel the common 2s and 5s from p against
        // the exponent FIRST; only a fraction whose reduced denominator genuinely
        // exceeds i128 still falls back.
        let k = -shift;
        let mut a = k; // remaining factor-2 exponent of the denominator
        let mut b = k; // remaining factor-5 exponent of the denominator
        while a > 0 && p != 0 && p % 2 == 0 {
            p /= 2;
            a -= 1;
        }
        while b > 0 && p != 0 && p % 5 == 0 {
            p /= 5;
            b -= 1;
        }
        // p != 0 here: the zero mantissa bailed before the branch, and exact division
        // of a nonzero p by 2 or 5 cannot reach zero.
        for _ in 0..a {
            q = q.checked_mul(2)?;
        }
        for _ in 0..b {
            q = q.checked_mul(5)?;
        }
    }
    Rat::new(p, q)
}

/// `parse_decimal` in the big form. The size is bounded from the digit count and the exponent
/// BEFORE anything is built, so `1e2147483647` and a million-digit mantissa refuse at once.
///
/// Bounds (after stripping leading and trailing zeros, so the mantissa `m` has no factor 10):
/// - a value within the cap has at most 1,101 significant digits (`m = N * 2^(K-x) * 5^(K-y)`
///   with `N` and the denominator `2^x * 5^y` within 1,100 bits), so more digits refuse;
/// - `m * 10^s` for `s >= 0` has more than `3 * (digits - 1 + s)` bits;
/// - `m / 10^s` keeps a denominator of at least `2^s` (only 2s or only 5s can cancel).
fn parse_decimal_big(negative: bool, int_part: &str, frac_part: &str, shift: i64) -> Option<Rat> {
    let digits: String = int_part.chars().chain(frac_part.chars()).collect();
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Some(Rat::ZERO);
    }
    let trimmed = digits.trim_end_matches('0');
    let shift = shift.checked_add((digits.len() - trimmed.len()) as i64)?;
    let n = trimmed.len() as i64;
    if n > 1_200
        || (shift >= 0 && 3 * (n - 1 + shift) > CAP_BITS as i64)
        || (shift < 0 && -shift > CAP_BITS as i64)
    {
        return None;
    }
    let m: BigInt = trimmed.parse().ok()?;
    let m = if negative { -m } else { m };
    let ten = BigInt::from(10);
    if shift >= 0 {
        Rat::from_big(m * num_traits::pow(ten, shift as usize), BigInt::one())
    } else {
        Rat::from_big(m, num_traits::pow(ten, (-shift) as usize))
    }
}

/// `digits` (an unsigned integer string) divided by 10^k, as a positional decimal.
fn place_point(negative: bool, digits: String, k: usize) -> String {
    let (int_part, frac_part) = if digits.len() > k {
        (
            digits[..digits.len() - k].to_string(),
            digits[digits.len() - k..].to_string(),
        )
    } else {
        ("0".to_string(), format!("{:0>width$}", digits, width = k))
    };
    // A normalized fraction with q != 1 has a nonzero fractional part, and 10^k is the
    // MINIMAL power (k = max(a,b)), so the last digit is nonzero: no trailing-zero trim.
    let sign = if negative { "-" } else { "" };
    if k == 0 {
        format!("{sign}{int_part}")
    } else {
        format!("{sign}{int_part}.{frac_part}")
    }
}

/// `exact_decimal` for big components.
fn big_exact_decimal(p: &BigInt, q: &BigInt) -> Option<String> {
    if q.is_one() {
        return Some(p.to_string());
    }
    let a = q.trailing_zeros().unwrap_or(0);
    let mut rest: BigInt = q >> a;
    let five = BigInt::from(5);
    let mut b = 0u64;
    while (&rest % &five).is_zero() {
        rest /= &five;
        b += 1;
    }
    if !rest.is_one() {
        return None;
    }
    let k = a.max(b);
    let scaled = p.abs()
        * num_traits::pow(BigInt::from(2), (k - a) as usize)
        * num_traits::pow(five, (k - b) as usize);
    Some(place_point(p.is_negative(), scaled.to_string(), k as usize))
}

/// The digit limit of the exact token readers, CPython's integer-string limit: a longer
/// component keeps the old floating reading rather than spend time building it.
pub const READER_DIGITS: usize = 4300;

/// The exact rational a numeral token spells -- a decimal or a `p/q` fraction, the grammar of
/// `utils::is_numeric_string` -- as unreduced `(p, q)` with `q > 0`, at any size up to
/// `READER_DIGITS` digits and a decimal exponent of at most 4,000 in magnitude; `None` for
/// anything else. The readers outside the AC core (offline evaluator, interval kernel) use it,
/// so a token reads as its exact value however it was printed.
pub fn token_rational(t: &str) -> Option<(BigInt, BigInt)> {
    if let Some((p, q)) = crate::utils::split_fraction(t) {
        if p.trim_start_matches(['+', '-']).len() > READER_DIGITS || q.len() > READER_DIGITS {
            return None;
        }
        let p: BigInt = p.strip_prefix('+').unwrap_or(p).parse().ok()?;
        let q: BigInt = q.parse().ok()?;
        return (!q.is_zero()).then_some((p, q));
    }
    if !crate::utils::is_decimal_numeral(t) {
        return None;
    }
    let (mant, exp) = match t.find(['e', 'E']) {
        Some(i) => (&t[..i], t[i + 1..].parse::<i64>().ok()?),
        None => (t, 0),
    };
    let (negative, mant) = match mant.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mant.strip_prefix('+').unwrap_or(mant)),
    };
    let (int_part, frac_part) = mant.split_once('.').unwrap_or((mant, ""));
    let digits: String = int_part.chars().chain(frac_part.chars()).collect();
    if digits.is_empty() || digits.len() > READER_DIGITS {
        return None;
    }
    let shift = exp.checked_sub(frac_part.len() as i64)?;
    if shift.abs() > 4000 {
        return None;
    }
    let m: BigInt = digits.parse().ok()?;
    let m = if negative { -m } else { m };
    let ten = BigInt::from(10);
    Some(if shift >= 0 {
        (m * num_traits::pow(ten, shift as usize), BigInt::one())
    } else {
        (m, num_traits::pow(ten, (-shift) as usize))
    })
}

/// The f64 nearest to the numeral token `t`, correctly rounded at any size (`+-inf` beyond
/// float64's range, signed zero below it), or `None` where `token_rational` refuses.
pub fn token_nearest_f64(t: &str) -> Option<f64> {
    let (p, q) = token_rational(t)?;
    Some(big_ratio_to_f64(&p, &q))
}

/// Whether the numeral token `t` denotes exactly the integer value of the finite,
/// integer-valued f64 `v`, at any size.
pub fn token_denotes_integer(t: &str, v: f64) -> bool {
    if !v.is_finite() || v.fract() != 0.0 {
        return false;
    }
    let Some((p, q)) = token_rational(t) else {
        return false;
    };
    let (m, e) = f64_parts(v);
    let mut vi = BigInt::from(m);
    // exact: an integer-valued f64 has no set bit below 2^0
    if e >= 0 {
        vi <<= e as u64;
    } else {
        vi >>= (-e) as u64;
    }
    p == vi * q
}

/// `p/q` (`q > 0`) correctly rounded to f64 (ties to even), with subnormals and overflow to
/// +-inf: the value `float(Fraction(p, q))` gives, except that Python raises OverflowError
/// where this returns +-inf.
fn big_ratio_to_f64(p: &BigInt, q: &BigInt) -> f64 {
    if p.is_zero() {
        return 0.0;
    }
    let negative = p.is_negative();
    let (a, b): (BigUint, BigUint) = (p.magnitude().clone(), q.magnitude().clone());
    // a/b lies in (2^(e-1), 2^(e+1)) for e = bits(a) - bits(b). Scale so that the integer
    // quotient has 55 or 56 bits: two guard bits beyond the 53-bit significand.
    let e = a.bits() as i64 - b.bits() as i64;
    let k = e - 55;
    let (num, den) = if k >= 0 {
        (a, b << k as u64)
    } else {
        (a << (-k) as u64, b)
    };
    let (quot, rem) = num.div_rem(&den);
    let sticky = !rem.is_zero();
    let nb = quot.bits() as i64;
    // Keep 53 bits for a normal result; fewer when the least significant bit would fall
    // below 2^-1074, the subnormal floor.
    let drop = (nb - 53).max(-1074 - k).max(0);
    let mut m = (&quot >> drop as u64).to_u64().unwrap_or(u64::MAX);
    if drop > 0 {
        let half = BigUint::one() << (drop - 1) as u64;
        let low = &quot & ((BigUint::one() << drop as u64) - 1u32);
        let up = match low.cmp(&half) {
            Ordering::Greater => true,
            Ordering::Equal => sticky || m & 1 == 1,
            Ordering::Less => false,
        };
        if up {
            m += 1;
        }
    }
    let mut exp = k + drop; // value = m * 2^exp
    if m == 1 << 53 {
        m >>= 1;
        exp += 1;
    }
    let bits = if m == 0 {
        0
    } else if m >= 1 << 52 {
        let biased = exp + 52 + 1023;
        if biased >= 2047 {
            return if negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        ((biased as u64) << 52) | (m - (1 << 52))
    } else {
        m // subnormal: exp == -1074 by construction
    };
    let y = f64::from_bits(bits);
    if negative {
        -y
    } else {
        y
    }
}

/// Compare `p/q` exactly against the dyadic `s * 2^k`. A midpoint between two doubles near
/// this value can need a denominator far beyond `i128` (2^1075 at the subnormal floor),
/// so it is never built as a `Rat` -- that overflowed below about 2^-11 and left the walk
/// uncertified. Instead `p` vs `s * q * 2^k` is decided in 256 bits: `s * q` is below
/// 2^182 and `p` below 2^127, and a shift that leaves 256 bits makes its side the larger
/// one outright.
fn cmp_dyadic(p: i128, q: i128, (s, k): (i128, i32)) -> Ordering {
    let (sp, ss) = (p.signum(), s.signum());
    if sp != ss {
        return sp.cmp(&ss);
    }
    if sp == 0 {
        return Ordering::Equal;
    }
    let left = shl_256((0, p.unsigned_abs()), (-k).max(0) as u32);
    let right = shl_256(
        widening_mul_u128(s.unsigned_abs(), q as u128),
        k.max(0) as u32,
    );
    let mag = match (left, right) {
        (Some(a), Some(b)) => a.cmp(&b),
        (None, _) => Ordering::Greater, // only one side is ever shifted
        (_, None) => Ordering::Less,
    };
    if sp < 0 {
        mag.reverse()
    } else {
        mag
    }
}

/// A finite f64 as `(signed mantissa, exponent)` with `x == m * 2^e` exactly (`m` below 2^53).
fn f64_parts(x: f64) -> (i128, i32) {
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = (bits & ((1u64 << 52) - 1)) as i128;
    let (m, e) = if exp == 0 {
        (frac, -1074)
    } else {
        (frac | (1i128 << 52), exp - 1075)
    };
    (if bits >> 63 == 1 { -m } else { m }, e)
}

/// The exact midpoint of two finite f64s as the dyadic `(s, k)`, `s * 2^k` (`s` below 2^55):
/// both on their smaller exponent, summed, halved by the exponent.
fn dyadic_midpoint(a: f64, b: f64) -> (i128, i32) {
    let ((ma, ea), (mb, eb)) = (f64_parts(a), f64_parts(b));
    let e = ea.min(eb);
    // Adjacent doubles differ in exponent by at most one, and a zero's exponent is the
    // subnormal floor, so the shifts stay small.
    ((ma << (ea - e)) + (mb << (eb - e)), e - 1)
}

/// `x * 2^k` for a 256-bit `(hi, lo)`, `None` when the result leaves 256 bits.
fn shl_256((hi, lo): (u128, u128), k: u32) -> Option<(u128, u128)> {
    if k == 0 {
        return Some((hi, lo));
    }
    if k >= 256 {
        return ((hi, lo) == (0, 0)).then_some((0, 0));
    }
    if k >= 128 {
        let s = k - 128;
        return (hi == 0 && (s == 0 || lo >> (128 - s) == 0)).then(|| (lo << s, 0));
    }
    if hi >> (128 - k) != 0 {
        return None;
    }
    Some(((hi << k) | (lo >> (128 - k)), lo << k))
}

/// Integer k-th root, exact or nothing: the r >= 0 with `r^k == n` (n >= 0), else `None`.
/// The next f64 toward +inf (`f64::next_up`, which is stable only from Rust 1.86; the MSRV is 1.83).
fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    f64::from_bits(if x > 0.0 { bits + 1 } else { bits - 1 })
}

/// The next f64 toward -inf (`f64::next_down`, likewise).
fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = x.to_bits();
    f64::from_bits(if x > 0.0 { bits - 1 } else { bits + 1 })
}

fn int_root(n: i128, k: i128) -> Option<i128> {
    if n == 0 || n == 1 {
        return Some(n);
    }
    if k == 1 {
        return Some(n);
    }
    // n >= 2 needs r >= 2, and r^k >= 2^127 > i128::MAX for k >= 127: no root can exist.
    // This guard also confines the cast below to k in [2, 126], where it is lossless --
    // `k as u32` TRUNCATES mod 2^32 (k = 2^32, the denominator of (1/2)^32, would divide
    // by zero).
    if !(2..127).contains(&k) {
        return None;
    }
    // Binary search on r in [1, min(n, hi)]. hi = 2^(floor(127/k) + 1) >= 2^(127/k) is a
    // TRUE upper bound on the root; a floor-only bound misses exact roots in the band
    // (2^floor, 2^(127/k)] and silently demotes them to the f64 fold. Max shift is 64
    // (k = 2); checked_pow refuses overflow, so a generous hi is safe.
    let mut lo: i128 = 1;
    let mut hi: i128 = 1i128 << (127 / k as u32 + 1);
    if hi > n {
        hi = n;
    }
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        match Rat::int(mid)
            .checked_pow_int(k)
            .and_then(|v| v.small_parts())
        {
            Some((v, _)) => match v.cmp(&n) {
                Ordering::Equal => return Some(mid),
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid - 1,
            },
            None => hi = mid - 1, // overflowed: mid too big
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The nearest f64 is certified at every magnitude a `Rat` can have (references:
    /// Python's correctly rounded `float(Fraction(p, q))`). Building the midpoints as `Rat`s
    /// overflowed below about 2^-11 and left these on `float(p) / float(q)`.
    #[test]
    fn nearest_f64_is_certified_at_every_magnitude() {
        let cases: [(i128, i128, f64); 5] = [
            (8159658314, 59197280150402860313, 1.3783839888029838e-10),
            (
                1,
                30000000000000000000000000000000000007,
                3.3333333333333336e-38,
            ),
            (-7, 100000000000000000000000000000000000003, -7e-38),
            (
                673107593011939307760027002528,
                810572757194796821120128085049,
                0.8304098392615706,
            ),
            (i128::MAX, 3, 5.671372782015641e+37),
        ];
        for (p, q, want) in cases {
            let r = Rat::new(p, q).unwrap();
            assert_eq!(r.to_f64_nearest_certified(), Some(want), "{p}/{q}");
        }
    }

    #[test]
    fn shl_256_shifts_and_detects_overflow() {
        assert_eq!(shl_256((0, 1), 0), Some((0, 1)));
        assert_eq!(shl_256((0, 1), 127), Some((0, 1 << 127)));
        assert_eq!(shl_256((0, 1), 128), Some((1, 0)));
        assert_eq!(shl_256((0, 3), 200), Some((3 << 72, 0)));
        assert_eq!(shl_256((0, 1), 255), Some((1 << 127, 0)));
        assert_eq!(shl_256((0, 2), 255), None);
        assert_eq!(shl_256((1, 0), 128), None);
        assert_eq!(shl_256((0, 1), 256), None);
        assert_eq!(shl_256((0, 0), 300), Some((0, 0)));
    }

    /// Continued-fraction (Euclidean-descent) comparison: an INDEPENDENT exact oracle for
    /// cmp_exact's wide path -- a different algorithm cannot share a bug with the
    /// schoolbook 256-bit multiply. Positive operands; terminates like the Euclidean gcd.
    fn euclid_pos(mut p1: u128, mut q1: u128, mut p2: u128, mut q2: u128) -> Ordering {
        let mut flip = false;
        loop {
            let (d1, r1) = (p1 / q1, p1 % q1);
            let (d2, r2) = (p2 / q2, p2 % q2);
            if d1 != d2 {
                let o = d1.cmp(&d2);
                return if flip { o.reverse() } else { o };
            }
            match (r1 == 0, r2 == 0) {
                (true, true) => return Ordering::Equal,
                (true, false) => {
                    return if flip {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                }
                (false, true) => {
                    return if flip {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                }
                _ => {}
            }
            // fractional parts r1/q1 vs r2/q2 = reciprocals q1/r1 vs q2/r2, REVERSED.
            (p1, q1, p2, q2) = (q1, r1, q2, r2);
            flip = !flip;
        }
    }

    fn euclid_cmp(a: &Rat, b: &Rat) -> Ordering {
        let ((ap, aq), (bp, bq)) = (a.small_parts().unwrap(), b.small_parts().unwrap());
        let (s1, s2) = (ap.signum(), bp.signum());
        if s1 != s2 {
            return s1.cmp(&s2);
        }
        if s1 == 0 {
            return Ordering::Equal;
        }
        let o = euclid_pos(ap.unsigned_abs(), aq as u128, bp.unsigned_abs(), bq as u128);
        if s1 < 0 {
            o.reverse()
        } else {
            o
        }
    }

    /// cmp_exact agrees with an INDEPENDENT Euclidean-descent oracle over three
    /// magnitude regimes (including the overflow-certain one), plus transitivity
    /// triples: the exactness and totality the canonical order relies on.
    #[test]
    fn cmp_exact_agrees_with_the_euclidean_oracle_at_every_magnitude() {
        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut rat = |bits: u32, signed: bool| -> Rat {
            let mask = if bits >= 64 {
                u64::MAX as u128 | ((next() as u128) << 64) >> (128 - bits)
            } else {
                (next() >> (64 - bits)) as u128
            };
            let p_mag = mask as i128 & (i128::MAX);
            let q = ((next() as u128 | ((next() as u128) << 64)) >> (128 - bits.max(2))).max(1)
                as i128
                & i128::MAX;
            let p = if signed && next() % 2 == 0 {
                -p_mag
            } else {
                p_mag
            };
            Rat::new(p, q.max(1)).unwrap()
        };
        for &bits in &[20u32, 63, 126] {
            let mut pairs_checked = 0usize;
            for _ in 0..20_000 {
                let (a, b) = (rat(bits, true), rat(bits, true));
                assert_eq!(
                    a.cmp_exact(&b),
                    euclid_cmp(&a, &b),
                    "disagreement at {bits} bits: {a:?} vs {b:?}"
                );
                assert_eq!(a.cmp_exact(&b), b.cmp_exact(&a).reverse(), "antisymmetry");
                assert_eq!(a.cmp_exact(&a), Ordering::Equal, "reflexivity");
                pairs_checked += 1;
            }
            assert!(pairs_checked == 20_000);
        }
        // Transitivity triples in the overflow regime (the old fallback's 3-cycles).
        for _ in 0..20_000 {
            let (mut a, mut b, mut c) = (rat(126, true), rat(126, true), rat(126, true));
            // order the triple by cmp_exact, then verify pairwise consistency
            if a.cmp_exact(&b) == Ordering::Greater {
                std::mem::swap(&mut a, &mut b);
            }
            if b.cmp_exact(&c) == Ordering::Greater {
                std::mem::swap(&mut b, &mut c);
                if a.cmp_exact(&b) == Ordering::Greater {
                    std::mem::swap(&mut a, &mut b);
                }
            }
            assert_ne!(
                a.cmp_exact(&c),
                Ordering::Greater,
                "3-cycle: {a:?} {b:?} {c:?}"
            );
        }
        // The audit's concrete failure class: ADJACENT sub-1e-18 decimal literals must be
        // DISTINCT and consistently ordered (the f64 fallback said Equal for 6.6% and
        // inverted 0.35%).
        let lo = Rat::parse_decimal("6.4495375319922606e-18").unwrap();
        let hi = Rat::parse_decimal("6.449537531992261e-18").unwrap();
        assert_eq!(lo.cmp_exact(&hi), Ordering::Less);
        assert_eq!(hi.cmp_exact(&lo), Ordering::Greater);
    }

    /// COST measurement (owner request): the fast path is byte-identical to the historical
    /// exact path; the wide path replaces an f64 fallback with the 256-bit multiply.
    /// Prints ns/op for all three under --nocapture.
    #[test]
    fn cmp_exact_cost_measurement() {
        use std::time::Instant;
        let small: Vec<Rat> = (1..2_000i128)
            .map(|k| Rat::new(k * 7 - 9_000, k + 1).unwrap())
            .collect();
        let huge: Vec<Rat> = (1..2_000i128)
            .map(|k| {
                Rat::new(
                    i128::MAX / (k + 3) - k * 12_345,
                    i128::MAX / (k * 5 + 11) - k,
                )
                .unwrap()
            })
            .collect();
        let bench = |name: &str, data: &[Rat], f: &dyn Fn(&Rat, &Rat) -> Ordering| {
            let t = Instant::now();
            let mut acc = 0usize;
            const REPS: usize = 500;
            for _ in 0..REPS {
                for w in data.windows(2) {
                    acc += (f(&w[0], &w[1]) == Ordering::Less) as usize;
                }
            }
            let n = REPS * (data.len() - 1);
            eprintln!(
                "cmp_exact cost [{name}]: {:.1} ns/op ({n} ops, checksum {acc})",
                t.elapsed().as_nanos() as f64 / n as f64
            );
        };
        let exact = |a: &Rat, b: &Rat| a.cmp_exact(b);
        let old_f64 = |a: &Rat, b: &Rat| a.to_f64().total_cmp(&b.to_f64());
        bench("fast path (common magnitudes)", &small, &exact);
        bench("wide path, NEW 256-bit exact", &huge, &exact);
        bench("wide path, OLD f64 fallback  ", &huge, &old_f64);
    }

    /// The exact-decimal emitter and `parse_decimal` must agree on the representable set:
    /// every decimal the emitter writes (mantissa <= i128) reads back to the SAME Rat.
    /// (1/4)^25 = 1/2^50 emits 50 fractional digits whose naive denominator 10^50
    /// overflows i128 -- the parser must cancel the 2s/5s first. A decimal that stayed
    /// unreadable demoted the exact coefficient to an opaque overlay Leaf, which sorts as
    /// a KEY factor instead of a stripped coefficient: bag order then differed between a
    /// fresh computation and a re-parse of its own output (1M-gate idempotence rows
    /// 274133/514869).
    #[test]
    fn parse_decimal_reads_back_every_emitted_decimal() {
        let tiny = Rat::new(1, 1i128 << 50).unwrap(); // 1/2^50
        let s = tiny.exact_decimal().expect("2^a*5^b denominator emits");
        assert_eq!(Rat::parse_decimal(&s), Some(tiny));
        let neg = Rat::new(-3, 1i128 << 50).unwrap(); // p carrying its own factors
        let s = neg.exact_decimal().expect("emits");
        assert_eq!(Rat::parse_decimal(&s), Some(neg));
        let five = Rat::new(5, 1i128 << 40).unwrap(); // p divisible by 5, q pure 2s
        let s = five.exact_decimal().expect("emits");
        assert_eq!(Rat::parse_decimal(&s), Some(five));
        // A reduced denominator genuinely beyond i128 still refuses (Leaf fallback), and
        // an absurd exponent bails fast instead of looping.
        assert_eq!(Rat::parse_decimal("1e-2000000000"), None);
    }

    #[test]
    fn normalization_and_arith() {
        assert_eq!(Rat::new(2, 4).and_then(|r| r.small_parts()), Some((1, 2)));
        assert_eq!(Rat::new(1, -2).and_then(|r| r.small_parts()), Some((-1, 2)));
        assert_eq!(Rat::new(1, 0), None);
        let half = Rat::new(1, 2).unwrap();
        let third = Rat::new(1, 3).unwrap();
        assert_eq!(half.checked_add(&third), Rat::new(5, 6));
        assert_eq!(half.checked_mul(&third), Rat::new(1, 6));
        assert_eq!(half.checked_inv(), Rat::new(2, 1));
        assert_eq!(Rat::ZERO.checked_inv(), None);
        assert_eq!(Rat::int(2).checked_pow_int(10), Some(Rat::int(1024)));
        assert_eq!(Rat::int(2).checked_pow_int(-2), Rat::new(1, 4));
        assert_eq!(Rat::ZERO.checked_pow_int(0), Some(Rat::ONE));
        assert_eq!(Rat::ZERO.checked_pow_int(-1), None);
        // (-1)^huge stays exact.
        assert_eq!(Rat::NEG_ONE.checked_pow_int(1_000_001), Some(Rat::NEG_ONE));
    }

    #[test]
    fn exact_roots() {
        assert_eq!(Rat::int(8).checked_root(3), Some(Rat::int(2)));
        assert_eq!(Rat::int(-8).checked_root(3), Some(Rat::int(-2)));
        assert_eq!(Rat::int(-4).checked_root(2), None);
        assert_eq!(Rat::new(4, 9).unwrap().checked_root(2), Rat::new(2, 3));
        assert_eq!(Rat::int(10).checked_root(2), None);
        assert_eq!(Rat::int(1 << 40).checked_root(4), Some(Rat::int(1 << 10)));
    }

    /// A floor-only search bound `2^floor(127/k)` would miss every exact root in the
    /// band up to the TRUE ceiling `2^(127/k)`, and `k as u32` truncation would panic
    /// on k = 2^32. The oracle sweep finds the largest representable perfect power for
    /// every k where roots can exist and requires the round-trip.
    #[test]
    fn exact_roots_above_the_old_floor_bound() {
        // The three audit reproductions, at the Rat level.
        let n5000_10 = Rat::int(5000).checked_pow_int(10).unwrap(); // 9.765625e36; old k=10 bound 2^12 = 4096
        assert_eq!(n5000_10.checked_root(10), Some(Rat::int(5000)));
        let n5e12_3 = Rat::int(5_000_000_000_000).checked_pow_int(3).unwrap(); // 1.25e38; old k=3 bound 2^42
        assert_eq!(n5e12_3.checked_root(3), Some(Rat::int(5_000_000_000_000)));
        // k = 2 near the top: floor(sqrt(i128::MAX)) is above the old 2^63 cap.
        let r_max: i128 = 13_043_817_825_332_782_212; // floor(sqrt(2^127 - 1))
        assert!(r_max.checked_mul(r_max).is_some());
        assert!((r_max + 1).checked_mul(r_max + 1).is_none());
        assert_eq!(
            Rat::int(r_max * r_max).checked_root(2),
            Some(Rat::int(r_max))
        );
        // The panic input: denominator 2^32 truncated to 0 in u32. Now a plain None
        // (2 is not a rational 2^32-th power), no panic.
        assert_eq!(Rat::int(2).checked_root(1i128 << 32), None);
        assert_eq!(Rat::int(2).checked_root(1i128 << 64), None);
        // k >= 127: r >= 2 overflows i128, so no root exists for any n >= 2...
        assert_eq!(Rat::int(2).checked_root(127), None);
        // ...but 0 and 1 keep their roots at EVERY index.
        assert_eq!(Rat::ZERO.checked_root(1 << 40), Some(Rat::ZERO));
        assert_eq!(Rat::ONE.checked_root(127), Some(Rat::ONE));
        // Oracle sweep: for every k where roots can exist, find r_k = the LARGEST r whose
        // k-th power is representable (independent binary search on representability --
        // a different question than int_root's equality search, so no shared bug), and
        // require the round-trip. Before the fix this failed for every k with
        // r_k > 2^floor(127/k) -- most k, since 127 is prime.
        // (Representability is i128's here, by the test's construction: the largest r whose
        // k-th power fits i128, found with plain integer arithmetic.)
        for k in 2i128..127 {
            let fits = |r: i128| r.checked_pow(k as u32).is_some();
            let (mut lo, mut hi) = (1i128, 1i128 << (127 / k as u32 + 1));
            while lo < hi {
                let mid = lo + (hi - lo + 1) / 2;
                if fits(mid) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            let n = Rat::int(lo.pow(k as u32));
            assert!(!fits(lo + 1));
            assert_eq!(n.checked_root(k), Some(Rat::int(lo)), "k = {k}, r_k = {lo}");
        }
    }

    /// `Rat::int` is the one constructor that could bypass `Rat::new`'s i128::MIN
    /// refusal -- the no-MIN invariant is load-bearing (unchecked negations downstream
    /// would WRAP in release builds), so the bypass must fail loudly.
    #[test]
    #[should_panic(expected = "i128::MIN")]
    fn rat_int_refuses_the_unnegatable_min() {
        let _ = Rat::int(i128::MIN);
    }

    /// A zero mantissa never overflows the positive-exponent scaling loop, so without
    /// the explicit bail "0e2147483647" spins the full exponent count. Value AND time
    /// are the regression here.
    #[test]
    fn zero_mantissa_parses_instantly_at_any_exponent() {
        let t0 = std::time::Instant::now();
        assert_eq!(Rat::parse_decimal("0e2147483647"), Some(Rat::ZERO));
        assert_eq!(Rat::parse_decimal("-0.00e2147483647"), Some(Rat::ZERO));
        assert_eq!(Rat::parse_decimal("0.0E999999999"), Some(Rat::ZERO));
        assert!(
            t0.elapsed().as_millis() < 1000,
            "zero-mantissa parse must not spin"
        );
        // Nonzero mantissas keep their bound from checked_mul overflow (stay symbolic).
        assert_eq!(Rat::parse_decimal("1e2147483647"), None);
    }

    #[test]
    fn decimal_round_trip() {
        for (p, q, s) in [
            (1i128, 2i128, "0.5"),
            (-7, 4, "-1.75"),
            (3, 1, "3"),
            (1, 5, "0.2"),
            (1, 1000, "0.001"),
            (-2, 1, "-2"),
        ] {
            let r = Rat::new(p, q).unwrap();
            assert_eq!(r.exact_decimal().as_deref(), Some(s));
            assert_eq!(Rat::parse_decimal(s), Some(r));
        }
        assert_eq!(Rat::new(1, 3).unwrap().exact_decimal(), None);
        assert_eq!(Rat::parse_decimal("1e-3"), Rat::new(1, 1000));
        assert_eq!(Rat::parse_decimal("1."), Some(Rat::ONE));
        assert_eq!(Rat::parse_decimal("2.5e2"), Some(Rat::int(250)));
        assert_eq!(Rat::parse_decimal("abc"), None);
        assert_eq!(Rat::parse_decimal(""), None);
    }

    #[test]
    fn exact_compare() {
        let a = Rat::new(1, 3).unwrap();
        let b = Rat::new(333333333333, 1000000000000).unwrap();
        assert_eq!(a.cmp_exact(&b), Ordering::Greater); // 1/3 > 0.333333333333 exactly
        assert_eq!(a.cmp_exact(&a), Ordering::Equal);
    }

    fn big(p: &str, q: &str) -> Rat {
        Rat::from_big(p.parse().unwrap(), q.parse().unwrap()).unwrap()
    }

    /// The small form is canonical: a value that fits is small however it was built, so
    /// derived `Eq`/`Hash` are value equality.
    #[test]
    fn from_big_demotes_every_value_that_fits() {
        let _domain = number_domain(false); // the exact domain: every number within the cap
        let r = big(
            "340282366920938463463374607431768211456",
            "680564733841876926926749214863536422912",
        );
        assert_eq!(r.small_parts(), Some((1, 2)));
        assert_eq!(r, Rat::new(1, 2).unwrap());
        assert_eq!(big("-6", "-4"), Rat::new(3, 2).unwrap());
        // i128::MIN stays out of the small form
        let min = big("-170141183460469231731687303715884105728", "1");
        assert_eq!(min.small_parts(), None);
        assert_eq!(min.checked_neg().unwrap().small_parts(), None);
        assert_eq!(Rat::from_big(BigInt::from(1), BigInt::zero()), None);
    }

    #[test]
    fn the_cap_bounds_both_components() {
        let _domain = number_domain(false); // the exact domain: every number within the cap
        let two = BigInt::from(2);
        let at_cap = num_traits::pow(two, CAP_BITS as usize - 1); // CAP_BITS bits
        assert!(Rat::from_big(at_cap.clone(), BigInt::from(3)).is_some());
        assert!(Rat::from_big(BigInt::from(3), at_cap.clone()).is_some());
        let over = at_cap * 2u32; // CAP_BITS + 1 bits
        assert_eq!(Rat::from_big(over.clone(), BigInt::from(3)), None);
        assert_eq!(Rat::from_big(BigInt::from(3), over), None);
    }

    /// Arithmetic with a big operand, against Python's `fractions.Fraction`.
    #[test]
    fn big_arithmetic_is_exact() {
        let _domain = number_domain(false); // the exact domain: every number within the cap
        let a = big("1000000000000000000000000000000000000000", "3"); // 10^39/3
        let b = big("1", "1000000000000000000000000000000000000000"); // 10^-39
        assert_eq!(a.checked_mul(&b), Rat::new(1, 3));
        assert_eq!(
            a.checked_add(&b).unwrap(),
            big(
                "1000000000000000000000000000000000000000000000000000000000000000000000000000003",
                "3000000000000000000000000000000000000000"
            )
        );
        assert_eq!(
            a.checked_inv().unwrap(),
            big("3", "1000000000000000000000000000000000000000")
        );
        assert_eq!(a.checked_neg().unwrap().signum(), -1);
        assert_eq!(a.cmp_exact(&b), Ordering::Greater);
        assert_eq!(b.cmp_exact(&Rat::ZERO), Ordering::Greater);
        assert_eq!(a.cmp_exact(&a.clone()), Ordering::Equal);
        // powers: (10^39/3)^2, and a power whose lower size bound leaves the cap
        assert_eq!(
            a.checked_pow_int(2).unwrap(),
            big(
                "1000000000000000000000000000000000000000000000000000000000000000000000000000000",
                "9"
            )
        );
        assert_eq!(a.checked_pow_int(9), None);
        assert_eq!(a.checked_pow_int(-1), a.checked_inv());
        // exact roots of big numbers, and inexact ones refuse
        let sq = a.checked_pow_int(2).unwrap();
        assert_eq!(sq.checked_root(2), Some(a.clone()));
        assert_eq!(a.checked_root(2), None);
        let neg = a.checked_neg().unwrap();
        assert_eq!(neg.checked_pow_int(3).unwrap().checked_root(3), Some(neg));
        // integer queries at any size
        let odd = big("1000000000000000000000000000000000000001", "1");
        assert!(odd.is_integer() && odd.is_odd_integer() && !odd.is_even_integer());
        assert_eq!(odd.small_int(), None);
        assert_eq!(odd.cmp_int(5), Ordering::Greater);
        assert!(!a.is_integer() && !a.is_odd_integer() && !a.is_even_integer());
    }

    /// The big form's nearest float agrees with the small form's certified walk wherever both
    /// apply, and handles what the walk cannot: components beyond f64, subnormals, overflow.
    #[test]
    fn big_nearest_float_is_correctly_rounded() {
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let bits = 1 + next() % 126;
            let p = ((next() as u128 | ((next() as u128) << 64)) >> (128 - bits)) as i128;
            let qbits = 1 + next() % 126;
            let q = (((next() as u128 | ((next() as u128) << 64)) >> (128 - qbits)) as i128).max(1);
            let p = if next() % 2 == 0 { -p } else { p };
            let r = Rat::new(p, q).unwrap();
            let (bp, bq) = r.big_parts();
            assert_eq!(
                big_ratio_to_f64(&bp, &bq),
                r.to_f64_nearest_certified().unwrap(),
                "{p}/{q}"
            );
        }
        let p10 = |k: usize| num_traits::pow(BigInt::from(10), k);
        let p2 = |k: usize| num_traits::pow(BigInt::from(2), k);
        assert_eq!(big_ratio_to_f64(&p10(400), &p10(399)), 10.0);
        assert_eq!(big_ratio_to_f64(&BigInt::one(), &p2(1074)), 5e-324);
        assert_eq!(big_ratio_to_f64(&BigInt::one(), &p2(1075)), 0.0); // a tie, to even
        assert_eq!(big_ratio_to_f64(&BigInt::from(3), &p2(1076)), 5e-324);
        assert_eq!(
            big_ratio_to_f64(&BigInt::one(), &p2(1022)),
            f64::MIN_POSITIVE
        );
        assert_eq!(big_ratio_to_f64(&p2(1023), &BigInt::one()), 2f64.powi(1023));
        assert_eq!(big_ratio_to_f64(&p2(1024), &BigInt::one()), f64::INFINITY);
        // halfway between f64::MAX and 2^1024 rounds to even, i.e. up to infinity
        let max_half = p2(1024) - p2(970);
        assert_eq!(big_ratio_to_f64(&max_half, &BigInt::one()), f64::INFINITY);
        assert_eq!(big_ratio_to_f64(&(max_half - 1), &BigInt::one()), f64::MAX);
        assert_eq!(
            big_ratio_to_f64(&-p10(400), &BigInt::one()),
            f64::NEG_INFINITY
        );
        // 10^-320 is subnormal
        assert_eq!(big_ratio_to_f64(&BigInt::one(), &p10(320)), 1e-320);
    }

    #[test]
    fn big_decimals_print_and_parse_exactly() {
        let _domain = number_domain(false); // the exact domain: every number within the cap
        let r = big("1", "100000000000000000000000000000000000000000"); // 1e-41
        let s = r.exact_decimal().unwrap();
        assert_eq!(s, format!("0.{}1", "0".repeat(40)));
        assert_eq!(parse_decimal_big(false, "0", &s[2..], -41), Some(r.clone()));
        assert_eq!(parse_decimal_big(false, "1", "", -41), Some(r.clone()));
        assert_eq!(big("-7", "1").exact_decimal().as_deref(), Some("-7"));
        assert_eq!(
            big("1", "3000000000000000000000000000000000000000").exact_decimal(),
            None
        );
        let e40 = parse_decimal_big(false, "1", "", 40).unwrap();
        assert_eq!(e40.exact_decimal().unwrap(), format!("1{}", "0".repeat(40)));
        assert_eq!(parse_decimal_big(true, "25", "", -41).unwrap().signum(), -1);
        // size bounds: refused before building, at once
        let t0 = std::time::Instant::now();
        assert_eq!(parse_decimal_big(false, "1", "", 2_147_483_647), None);
        assert_eq!(parse_decimal_big(false, "1", "", -2_000_000_000), None);
        assert_eq!(
            parse_decimal_big(false, &"7".repeat(1_000_000), "", 0),
            None
        );
        assert!(t0.elapsed().as_millis() < 1000);
        // trailing zeros of the mantissa do not count as digits
        assert_eq!(
            parse_decimal_big(false, &format!("1{}", "0".repeat(5000)), "", -5000),
            Some(Rat::ONE)
        );
        // the shortest spellings of float64s parse (the smallest subnormal, the smallest normal,
        // the largest double); a longer decimal spelling of a subnormal can leave the cap
        // (4.9406564584124654e-324 needs a 1,129-bit denominator)
        assert!(parse_decimal_big(false, "5", "", -324).is_some());
        assert!(parse_decimal_big(false, "2", "2250738585072014", -324).is_some());
        assert!(parse_decimal_big(false, "1", "7976931348623157", 292).is_some());
        assert!(parse_decimal_big(false, "4", "9406564584124654", -340).is_none());
    }

    /// The token readers outside the AC core read a numeral exactly at any size (phase 2b).
    #[test]
    fn token_readers_are_exact_at_any_size() {
        assert_eq!(
            token_nearest_f64("673107593011939307760027002528/810572757194796821120128085049"),
            Some(0.8304098392615706)
        );
        let p10 = |k: usize| format!("1{}", "0".repeat(k));
        assert_eq!(
            token_nearest_f64(&format!("{}/{}", p10(400), p10(399))),
            Some(10.0)
        );
        assert_eq!(token_nearest_f64(&p10(400)), Some(f64::INFINITY));
        assert_eq!(token_nearest_f64("1e-400"), Some(0.0));
        assert!(token_nearest_f64("-1e-400").unwrap().is_sign_negative());
        assert_eq!(token_nearest_f64("2.5e-1"), Some(0.25));
        assert_eq!(token_nearest_f64("-.5"), Some(-0.5));
        assert_eq!(token_nearest_f64("x0"), None);
        assert_eq!(token_nearest_f64("1/0"), None);
        assert_eq!(token_nearest_f64(&"7".repeat(5000)), None); // beyond the digit limit
        assert_eq!(token_nearest_f64("1e99999"), None); // beyond the exponent limit

        // integers: 2^200 written out is exactly the f64 2^200; 1e40's nearest f64 is not 10^40
        let two200 = num_traits::pow(BigInt::from(2), 200).to_string();
        assert!(token_denotes_integer(&two200, 2f64.powi(200)));
        assert!(!token_denotes_integer("1e40", 1e40));
        assert!(token_denotes_integer("1e22", 1e22)); // 10^22 is exact in f64
        assert!(token_denotes_integer("6/3", 2.0));
        assert!(token_denotes_integer("-4.0", -4.0));
        assert!(!token_denotes_integer("7/3", 2.0));
        assert!(!token_denotes_integer("2.5", 2.0));
    }

    /// The 2a switch: a result of small operands that leaves the small form refuses while
    /// `WIDE_RESULTS` is off (byte identity with the 128-bit engine) and is computed exactly once
    /// it is on (phase 2c).
    #[test]
    fn a_small_overflow_follows_the_switch() {
        let big = Rat::int(i128::MAX / 2 + 1); // doubled: 2^127, one beyond i128
        let sum = big.checked_add(&big);
        let product = big.checked_mul(&Rat::int(4));
        let tiny = Rat::new(1, i128::MAX / 3).unwrap();
        let tiny_sq = tiny.checked_mul(&tiny);
        if WIDE_RESULTS {
            assert_eq!(sum.unwrap().small_parts(), None);
            assert_eq!(product.unwrap().small_parts(), None);
            assert_eq!(tiny_sq.unwrap().small_parts(), None);
        } else {
            assert_eq!(sum, None);
            assert_eq!(product, None);
            assert_eq!(tiny_sq, None);
        }
    }

    /// The power size bound never overflows: with `b - 1 = 255` and `n = (2^64 - 1) / 255`,
    /// `n * (b - 1)` is exactly `u64::MAX`, where `lo + 1` used to overflow (review of 2a).
    #[test]
    fn the_power_size_bound_does_not_overflow() {
        let base = Rat::from_big(num_traits::pow(BigInt::from(2), 255), BigInt::one()).unwrap();
        let n = (u64::MAX / 255) as i128;
        assert_eq!(base.checked_pow_int(n), None);
    }
}
