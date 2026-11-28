//! Exact polynomials over opaque atoms, for the class reading's polynomial multiples
//! (`expr::class_multiple`): a sum read as a polynomial with exact rational coefficients in its
//! atoms (variables and any other subexpression, each an indeterminate), divided exactly by a
//! class's sum. An identity of formal polynomials holds for every value of the atoms, so a factor
//! found here is proven. Every operation refuses (`None`) past its size bounds or on a rational
//! overflow, which the caller reads as no factor.

use super::rat::Rat;
use std::cmp::Ordering;

/// A monomial: `(atom, exponent)` pairs, ascending by atom, every exponent positive.
pub(crate) type Mono = Vec<(u16, u32)>;

/// The most terms any polynomial built here may hold.
pub(crate) const POLY_TERM_CAP: usize = 64;

/// A polynomial: its terms in descending lexicographic order of their monomials (the lowest atom
/// most significant), no zero coefficient, no monomial twice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Poly(pub(crate) Vec<(Mono, Rat)>);

/// The lexicographic monomial order, the lowest atom most significant: a monomial order (a total
/// order compatible with multiplication, 1 the least), so division by one polynomial decides
/// divisibility (one polynomial is a Groebner basis of the ideal it generates).
fn cmp_mono(a: &Mono, b: &Mono) -> Ordering {
    let (mut i, mut j) = (0, 0);
    loop {
        match (a.get(i), b.get(j)) {
            (Some(&(x, e)), Some(&(y, f))) => {
                if x == y {
                    if e != f {
                        return e.cmp(&f);
                    }
                    i += 1;
                    j += 1;
                } else if x < y {
                    return Ordering::Greater;
                } else {
                    return Ordering::Less;
                }
            }
            (Some(_), None) => return Ordering::Greater,
            (None, Some(_)) => return Ordering::Less,
            (None, None) => return Ordering::Equal,
        }
    }
}

fn mul_mono(a: &Mono, b: &Mono) -> Option<Mono> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(&(x, e)), Some(&(y, f))) if x == y => {
                out.push((x, e.checked_add(f)?));
                i += 1;
                j += 1;
            }
            (Some(&(x, e)), Some(&(y, _))) if x < y => {
                out.push((x, e));
                i += 1;
            }
            (Some(&p), None) => {
                out.push(p);
                i += 1;
            }
            (_, Some(&q)) => {
                out.push(q);
                j += 1;
            }
            (None, None) => unreachable!(),
        }
    }
    Some(out)
}

/// `a / b` where `b` divides `a`, else `None`.
fn div_mono(a: &Mono, b: &Mono) -> Option<Mono> {
    let mut out = Vec::with_capacity(a.len());
    let mut j = 0;
    for &(x, e) in a {
        match b.get(j) {
            Some(&(y, f)) if y == x => {
                if f > e {
                    return None;
                }
                if e > f {
                    out.push((x, e - f));
                }
                j += 1;
            }
            Some(&(y, _)) if y < x => return None,
            _ => out.push((x, e)),
        }
    }
    (j == b.len()).then_some(out)
}

impl Poly {
    pub(crate) fn zero() -> Poly {
        Poly(Vec::new())
    }

    pub(crate) fn constant(c: Rat) -> Poly {
        if c.is_zero() {
            Poly::zero()
        } else {
            Poly(vec![(Vec::new(), c)])
        }
    }

    pub(crate) fn atom(a: u16, e: u32) -> Poly {
        Poly(vec![(vec![(a, e)], Rat::ONE)])
    }

    /// The constant this polynomial is, if it is one.
    pub(crate) fn as_constant(&self) -> Option<Rat> {
        match self.0.as_slice() {
            [] => Some(Rat::ZERO),
            [(m, c)] if m.is_empty() => Some(c.clone()),
            _ => None,
        }
    }

    /// Terms in any order, like monomials collected, zeros dropped, sorted; `None` past the cap.
    fn normalize(mut terms: Vec<(Mono, Rat)>) -> Option<Poly> {
        terms.sort_by(|a, b| cmp_mono(&b.0, &a.0));
        let mut out: Vec<(Mono, Rat)> = Vec::with_capacity(terms.len());
        for (m, c) in terms {
            match out.last_mut() {
                Some((n, d)) if *n == m => *d = d.checked_add(&c)?,
                _ => out.push((m, c)),
            }
        }
        out.retain(|(_, c)| !c.is_zero());
        (out.len() <= POLY_TERM_CAP).then_some(Poly(out))
    }

    pub(crate) fn add(&self, o: &Poly) -> Option<Poly> {
        Poly::normalize(self.0.iter().chain(&o.0).cloned().collect())
    }

    pub(crate) fn mul(&self, o: &Poly) -> Option<Poly> {
        if self.0.len().saturating_mul(o.0.len()) > POLY_TERM_CAP * POLY_TERM_CAP {
            return None;
        }
        let mut terms = Vec::with_capacity(self.0.len() * o.0.len());
        for (m, c) in &self.0 {
            for (n, d) in &o.0 {
                terms.push((mul_mono(m, n)?, c.checked_mul(d)?));
            }
        }
        Poly::normalize(terms)
    }

    pub(crate) fn pow(&self, k: u32) -> Option<Poly> {
        let mut acc = Poly::constant(Rat::ONE);
        for _ in 0..k {
            acc = acc.mul(self)?;
        }
        Some(acc)
    }

    /// The exact quotient `self / d`, or `None` where `d` does not divide `self` (or a bound is
    /// passed). Division by one polynomial: the remainder is zero exactly where `d` divides.
    pub(crate) fn div_exact(&self, d: &Poly) -> Option<Poly> {
        let (lm, lc) = d.0.first()?;
        let inv = lc.checked_inv()?;
        let mut rem = self.clone();
        let mut quot: Vec<(Mono, Rat)> = Vec::new();
        while let Some((m, c)) = rem.0.first() {
            if quot.len() >= POLY_TERM_CAP {
                return None;
            }
            // A leading term `d`'s does not divide stays in the remainder: no exact quotient.
            let qm = div_mono(m, lm)?;
            let qc = c.checked_mul(&inv)?;
            let mut step: Vec<(Mono, Rat)> = Vec::with_capacity(d.0.len());
            for (n, e) in &d.0 {
                step.push((mul_mono(&qm, n)?, qc.checked_mul(e)?.checked_neg()?));
            }
            rem = rem.add(&Poly(step))?;
            quot.push((qm, qc));
        }
        Poly::normalize(quot)
    }

    /// Where this polynomial is `c a + r` with `c` a nonzero constant and `r` free of `a`: `c`
    /// and `r`.
    pub(crate) fn linear_in(&self, a: u16) -> Option<(Rat, Poly)> {
        let mut c: Option<Rat> = None;
        let mut rest: Vec<(Mono, Rat)> = Vec::new();
        for (m, k) in &self.0 {
            match m.iter().find(|(x, _)| *x == a) {
                None => rest.push((m.clone(), k.clone())),
                Some((_, 1)) if m.len() == 1 && c.is_none() => c = Some(k.clone()),
                Some(_) => return None,
            }
        }
        Some((c?, Poly(rest)))
    }

    /// This polynomial with `p` put for the atom `a`.
    pub(crate) fn substitute(&self, a: u16, p: &Poly) -> Option<Poly> {
        let mut acc = Poly::zero();
        let mut powers: Vec<Poly> = vec![Poly::constant(Rat::ONE)];
        for (m, c) in &self.0 {
            let e = m.iter().find(|(x, _)| *x == a).map_or(0, |(_, e)| *e) as usize;
            while powers.len() <= e {
                let next = powers.last().expect("one power").mul(p)?;
                powers.push(next);
            }
            let rest: Mono = m.iter().filter(|(x, _)| *x != a).copied().collect();
            let term = Poly(vec![(rest, c.clone())]).mul(&powers[e])?;
            acc = acc.add(&term)?;
        }
        Some(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(terms: &[(&[(u16, u32)], i128)]) -> Poly {
        Poly::normalize(
            terms
                .iter()
                .map(|(m, c)| (m.to_vec(), Rat::int(*c)))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn division_decides_divisibility() {
        // 9 - x^2 = (3 - x)(3 + x); x - 3 divides it, x - 2 does not.
        let t = p(&[(&[], 9), (&[(0, 2)], -1)]);
        let s = p(&[(&[(0, 1)], 1), (&[], -3)]);
        assert_eq!(t.div_exact(&s), Some(p(&[(&[(0, 1)], -1), (&[], -3)])));
        assert_eq!(t.div_exact(&p(&[(&[(0, 1)], 1), (&[], -2)])), None);
        // x^2 - y^2 over x - y, two atoms.
        let t = p(&[(&[(0, 2)], 1), (&[(1, 2)], -1)]);
        let s = p(&[(&[(0, 1)], 1), (&[(1, 1)], -1)]);
        assert_eq!(t.div_exact(&s), Some(p(&[(&[(0, 1)], 1), (&[(1, 1)], 1)])));
    }

    #[test]
    fn substitution_reads_the_cofactor_on_the_zero() {
        // x + 3 at x = 3 (x - 3 = 0): 6.
        let r = p(&[(&[(0, 1)], 1), (&[], 3)]);
        let (c, rest) = p(&[(&[(0, 1)], 1), (&[], -3)]).linear_in(0).unwrap();
        assert!(c.is_one());
        let at = Poly(
            rest.0
                .iter()
                .map(|(m, k)| (m.clone(), k.checked_neg().unwrap()))
                .collect(),
        );
        assert_eq!(
            r.substitute(0, &at).unwrap().as_constant(),
            Some(Rat::int(6))
        );
    }
}
