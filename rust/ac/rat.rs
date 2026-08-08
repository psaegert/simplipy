//! Exact rational arithmetic for the AC core.
//!
//! Contract D1: "number literals denote their exact rational values" -- so the core's numeric
//! type is an exact rational, not an f64. Coefficient and exponent arithmetic thereby becomes
//! COMPUTATION that cannot be wrong (replacing the mined-and-sampled coefficient rule family).
//!
//! Representation: `p/q` over `i128` with `q > 0` and `gcd(|p|, q) == 1`. Every operation is
//! CHECKED: on overflow it returns `None` and the caller keeps the symbolic form instead of
//! folding -- refusing to compute is always sound here, computing wrongly never is. No big-int
//! dependency: the corpus' coefficients are tiny, and anything that overflows i128 is better
//! left symbolic anyway.

use std::cmp::Ordering;

/// An exact rational `p/q`, normalized (`q > 0`, `gcd(|p|, q) == 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rat {
    p: i128,
    q: i128,
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
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
    pub const ZERO: Rat = Rat { p: 0, q: 1 };
    pub const ONE: Rat = Rat { p: 1, q: 1 };
    pub const NEG_ONE: Rat = Rat { p: -1, q: 1 };

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
        Some(Rat { p: p / g, q: q / g })
    }

    pub fn int(n: i128) -> Rat {
        // "No Rat ever holds i128::MIN" is a LOAD-BEARING invariant (MIN cannot be negated
        // or abs'd; release builds have no overflow checks, so a violation would WRAP
        // silently downstream). `Rat::new` refuses MIN; this bypass constructor must too,
        // loudly. Every real caller passes bounded values.
        assert!(
            n != i128::MIN,
            "Rat cannot represent i128::MIN (unnegatable)"
        );
        Rat { p: n, q: 1 }
    }

    #[inline]
    pub fn num(&self) -> i128 {
        self.p
    }

    #[inline]
    pub fn den(&self) -> i128 {
        self.q
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.p == 0
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.p == 1 && self.q == 1
    }

    #[inline]
    pub fn is_integer(&self) -> bool {
        self.q == 1
    }

    /// The integer value, if this is an integer.
    #[inline]
    pub fn as_integer(&self) -> Option<i128> {
        if self.q == 1 {
            Some(self.p)
        } else {
            None
        }
    }

    #[inline]
    pub fn is_negative(&self) -> bool {
        self.p < 0
    }

    pub fn checked_add(&self, o: &Rat) -> Option<Rat> {
        // p1/q1 + p2/q2 = (p1*q2 + p2*q1) / (q1*q2), then normalize.
        let a = self.p.checked_mul(o.q)?;
        let b = o.p.checked_mul(self.q)?;
        Rat::new(a.checked_add(b)?, self.q.checked_mul(o.q)?)
    }

    pub fn checked_mul(&self, o: &Rat) -> Option<Rat> {
        // Cross-reduce first so intermediates stay small: (p1/q2')·(p2/q1').
        let g1 = gcd(self.p, o.q).max(1);
        let g2 = gcd(o.p, self.q).max(1);
        let p = (self.p / g1).checked_mul(o.p / g2)?;
        let q = (self.q / g2).checked_mul(o.q / g1)?;
        Rat::new(p, q)
    }

    pub fn checked_neg(&self) -> Option<Rat> {
        Some(Rat {
            p: self.p.checked_neg()?,
            q: self.q,
        })
    }

    /// Multiplicative inverse. `None` for zero (1/0 is not a rational -- the caller keeps the
    /// symbolic `Pow(0, -1)` for the rules/value-set machinery to judge).
    pub fn checked_inv(&self) -> Option<Rat> {
        if self.p == 0 {
            return None;
        }
        Rat::new(self.q, self.p)
    }

    /// `self^n` for an integer exponent. `None` on overflow or `0^negative`.
    /// `0^0 == 1` here on purpose: it matches both Python's `0.0**0 == 1.0` and the engine's
    /// deployed fold, and the exponent 0 only arises from exact arithmetic, never from data.
    pub fn checked_pow_int(&self, n: i128) -> Option<Rat> {
        if n == 0 {
            return Some(Rat::ONE);
        }
        if self.p == 0 {
            return if n > 0 { Some(Rat::ZERO) } else { None };
        }
        let (base, n) = if n < 0 {
            (self.checked_inv()?, n.checked_neg()?)
        } else {
            (*self, n)
        };
        // Exponentiation by squaring, checked throughout. Cap the exponent so a pathological
        // `pow(x, 10^30)` never spins here -- i128 overflow would refuse it anyway for any
        // |base| != 1, and |base| == 1 is handled exactly.
        if base.p.abs() == 1 && base.q == 1 {
            // (+-1)^n: exact for ANY exponent magnitude.
            return Some(if base.p == 1 || n % 2 == 0 {
                Rat::ONE
            } else {
                Rat::NEG_ONE
            });
        }
        if n > 512 {
            return None;
        }
        let mut acc = Rat::ONE;
        let mut b = base;
        let mut e = n;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.checked_mul(&b)?;
            }
            e >>= 1;
            if e > 0 {
                b = b.checked_mul(&b)?;
            }
        }
        Some(acc)
    }

    /// Exact k-th root, if it exists as a rational: `self == r^k` with matching sign rules
    /// (`k` even requires `self >= 0`; the even root returned is the non-negative one).
    pub fn checked_root(&self, k: i128) -> Option<Rat> {
        if k <= 0 {
            return None;
        }
        if k == 1 {
            return Some(*self);
        }
        if self.p < 0 && k % 2 == 0 {
            return None;
        }
        let sign: i128 = if self.p < 0 { -1 } else { 1 };
        let rp = int_root(self.p.checked_abs()?, k)?;
        let rq = int_root(self.q, k)?;
        Rat::new(sign * rp, rq)
    }

    /// Compare exactly (no float detour): `p1/q1 <=> p2/q2` == `p1*q2 <=> p2*q1` with q > 0.
    /// EXACT at every magnitude: the fast path cross-multiplies in i128; on overflow the
    /// wide path computes both cross-products exactly in 256 bits (the former f64 fallback
    /// broke totality -- Equal on 6.6% and inverted 0.35% of adjacent sub-1e-18 literal
    /// pairs -- and the canonical sort's totality is load-bearing even though ordering is
    /// a CANONICALIZATION concern, not a soundness one; see `expr.rs` on why order never
    /// changes denotation).
    pub fn cmp_exact(&self, o: &Rat) -> Ordering {
        // Fast path: both cross-products fit in i128 (every common magnitude). This is
        // byte-identical to the historical exact path, so the hot path costs nothing new.
        if let (Some(a), Some(b)) = (self.p.checked_mul(o.q), o.p.checked_mul(self.q)) {
            return a.cmp(&b);
        }
        // EXACT wide path: an f64 estimate is not monotone in the true rational order at
        // these magnitudes (three roundings), and this comparison backs a TOTAL order --
        // the sort's totality check panics on any cycle -- so the cross-products are
        // computed exactly in 256 bits. Signs first: q > 0 is a Rat invariant, so
        // sign(p1*q2) = sign(p1).
        let (s1, s2) = (self.p.signum(), o.p.signum());
        if s1 != s2 {
            return s1.cmp(&s2);
        }
        let a = widening_mul_u128(self.p.unsigned_abs(), self.q_u128_of(o));
        let b = widening_mul_u128(o.p.unsigned_abs(), o.q_u128_of(self));
        let mag = a.cmp(&b); // (hi, lo) tuples compare lexicographically -- exact
        if s1 < 0 {
            mag.reverse()
        } else {
            mag
        }
    }

    /// The OTHER operand's denominator as u128 (q > 0 by invariant). A helper so the two
    /// cross-products in `cmp_exact` read symmetrically.
    #[inline]
    fn q_u128_of(&self, o: &Rat) -> u128 {
        debug_assert!(o.q > 0);
        o.q as u128
    }

    /// The f64 reading of this rational -- f64 has authority nowhere in the engine; this
    /// exists only as the comparator benchmark's model of the replaced fallback.
    #[cfg(test)]
    pub fn to_f64(self) -> f64 {
        self.p as f64 / self.q as f64
    }

    /// The shortest exact decimal string, if one exists (`q == 2^a * 5^b`): `1/2 -> "0.5"`,
    /// `-7/4 -> "-1.75"`, `3 -> "3"`. `None` for e.g. `1/3` (the serializer then spells the
    /// division structurally). Exactness is by construction: multiply p by 2s and 5s until the
    /// denominator is a power of ten, then place the decimal point.
    pub fn exact_decimal(&self) -> Option<String> {
        if self.q == 1 {
            return Some(self.p.to_string());
        }
        let (mut a, mut b) = (0u32, 0u32);
        let mut rest = self.q;
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
        let mut scaled = self.p.checked_abs()?;
        for _ in 0..(k - a) {
            scaled = scaled.checked_mul(2)?;
        }
        for _ in 0..(k - b) {
            scaled = scaled.checked_mul(5)?;
        }
        let digits = scaled.to_string();
        let k = k as usize;
        let (int_part, frac_part) = if digits.len() > k {
            (
                digits[..digits.len() - k].to_string(),
                digits[digits.len() - k..].to_string(),
            )
        } else {
            ("0".to_string(), format!("{:0>width$}", digits, width = k))
        };
        // q != 1 guarantees a nonzero fractional part (p/q is normalized), no trailing-zero trim
        // needed: 10^k is the MINIMAL power (k = max(a,b)), so the last digit is nonzero.
        let sign = if self.p < 0 { "-" } else { "" };
        Some(format!("{sign}{int_part}.{frac_part}"))
    }

    /// Parse a decimal token EXACTLY: `"7" -> 7`, `"-1.75" -> -7/4`, `"0.2" -> 1/5`,
    /// `"1e-3" -> 1/1000`, `"1." -> 1`. This is a DECIMAL parse, not a float parse -- `"0.2"`
    /// means one fifth, exactly, even though the f64 nearest to it does not.
    pub fn parse_decimal(s: &str) -> Option<Rat> {
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
        let shift = exp as i64 - frac_part.len() as i64;
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
            // overlay leaf -- which sorts by mint order, so bag order changed across
            // calls (the 1M-gate idempotence pair 274133/514869). Cancel the common 2s
            // and 5s from p against the exponent FIRST; only a fraction whose reduced
            // denominator genuinely exceeds i128 still falls back.
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
}

/// Integer k-th root, exact or nothing: the r >= 0 with `r^k == n` (n >= 0), else `None`.
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
        match Rat::int(mid).checked_pow_int(k) {
            Some(v) => match v.num().cmp(&n) {
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
        let (s1, s2) = (a.p.signum(), b.p.signum());
        if s1 != s2 {
            return s1.cmp(&s2);
        }
        if s1 == 0 {
            return Ordering::Equal;
        }
        let o = euclid_pos(
            a.p.unsigned_abs(),
            a.q as u128,
            b.p.unsigned_abs(),
            b.q as u128,
        );
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
        assert_eq!(Rat::new(2, 4), Some(Rat { p: 1, q: 2 }));
        assert_eq!(Rat::new(1, -2), Some(Rat { p: -1, q: 2 }));
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
        for k in 2i128..127 {
            let (mut lo, mut hi) = (1i128, 1i128 << (127 / k as u32 + 1));
            while lo < hi {
                let mid = lo + (hi - lo + 1) / 2;
                if Rat::int(mid).checked_pow_int(k).is_some() {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            let n = Rat::int(lo).checked_pow_int(k).unwrap();
            assert!(Rat::int(lo + 1).checked_pow_int(k).is_none());
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
}
