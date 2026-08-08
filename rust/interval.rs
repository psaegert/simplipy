//! EXACT value-class of an expression by interval/range analysis (no sampling).
//!
//! This replaces the three SAMPLED ways the miner used to estimate one and the same quantity --
//! "what is this expression's value-class and domain?" -- namely the constant POLE_GRID, the
//! `classify_const_source` probe, and the blanket nan-skip in `allclose_extends`.
//!
//! For an expression whose free `<constant>`s and variables range over all finite R, compute
//! which POSITIVE-MEASURE components its value takes: finite (with range), +-inf, nan.
//! Deterministic, seed-independent, no thresholds, no enumerated pole list (a fixed sampling
//! grid misses narrow pole bands that fall between its points).
//!
//! Port of a Python reference validated against dense mpmath ground truth. Semantics that
//! matter:
//!   * A finite range may be UNBOUNDED yet every value in it FINITE (`atanh` over (-1,1) ->
//!     (-inf, inf), never ATTAINS inf). `lo/hi = +-INF` means unbounded; only `pinf`/`ninf`
//!     mean a literal inf on positive measure.
//!   * OPEN vs CLOSED bounds decide real cases: `exp` over R is (0, inf) OPEN at 0, which is
//!     exactly what makes `pow(0, exp C)` identically +inf.
//!   * The +-INF endpoints of a free constant are LIMITS, not attained values -> never pinf.
//!   * POINT singularities (inv at 0, tan at pi/2, atanh at +-1) are MEASURE-ZERO: unbounded
//!     range, NO positive-measure inf. REGION singularities (asin/acos/acosh/atanh domains,
//!     log/sqrt of negatives, pow(0,neg), pow(.,+-inf)) DO contribute positive measure.
//!   * A float OVERFLOW means the value is finite but unrepresentable -> an unbounded EDGE,
//!     never an inf value (that artifact is the escalation's job, not a pole).

use crate::operators::Operators;

const INF: f64 = f64::INFINITY;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Finite,
    PosInf,
    NegInf,
    Nan,
    Mixed,
    Empty,
}

#[derive(Clone, Copy, Debug)]
pub struct Vs {
    pub lo: f64,
    pub hi: f64,
    pub lo_open: bool,
    pub hi_open: bool,
    pub has_fin: bool,
    /// The finite component exists ONLY on a measure-NULL x-set (e.g. `pow(negative, exponent)`
    /// is finite exactly where the exponent is an integer). `has_fin` keeps REACHABILITY
    /// semantics (`reaches_all_of` must see point-reachable values); this flag carries the
    /// POSITIVE-MEASURE semantics the domain gate's witness needs. Set only where PROVEN null;
    /// false is the conservative default (= the pre-flag behaviour, which blocks the witness).
    pub fin_null: bool,
    pub pinf: bool,
    pub ninf: bool,
    pub nan: bool,
}

impl Vs {
    pub fn empty() -> Self {
        Vs {
            lo: 0.0,
            hi: 0.0,
            lo_open: false,
            hi_open: false,
            has_fin: false,
            fin_null: false,
            pinf: false,
            ninf: false,
            nan: false,
        }
    }
    /// a free `<constant>`/variable: every FINITE real, unbounded both ways
    pub fn reals() -> Self {
        Vs {
            lo: -INF,
            hi: INF,
            lo_open: true,
            hi_open: true,
            has_fin: true,
            fin_null: false,
            pinf: false,
            ninf: false,
            nan: false,
        }
    }
    pub fn interval(lo: f64, hi: f64, lo_open: bool, hi_open: bool) -> Self {
        Vs {
            lo,
            hi,
            lo_open,
            hi_open,
            has_fin: true,
            fin_null: false,
            pinf: false,
            ninf: false,
            nan: false,
        }
    }
    pub fn constant(v: f64) -> Self {
        if v.is_nan() {
            let mut r = Vs::empty();
            r.nan = true;
            return r;
        }
        if v.is_infinite() {
            let mut r = Vs::empty();
            if v > 0.0 {
                r.pinf = true
            } else {
                r.ninf = true
            }
            return r;
        }
        Vs::interval(v, v, false, false)
    }
    pub fn is_const(&self) -> bool {
        self.has_fin && self.lo == self.hi && !self.pinf && !self.ninf && !self.nan
    }
    /// Is the FINITE part a single POINT? Unlike `is_const`, says nothing about the other
    /// components: a set can be the point {0} TOGETHER with +inf (`pow(C, inf)` is 0 for |C|<1
    /// and +inf for C>1), and its finite part is still degenerate. Dividing by such a set is
    /// +-inf on POSITIVE MEASURE -- the denominator is identically 0 across a whole region of
    /// the input, not merely at isolated points -- so the measure-zero pole reasoning that
    /// applies to a denominator sweeping CONTINUOUSLY through 0 does not apply here.
    pub fn fin_is_point(&self) -> bool {
        self.has_fin && self.lo == self.hi
    }
    /// A finite part on POSITIVE measure: `has_fin` that is NOT confined to a null set.
    /// THE support predicate of the certificate layer (negated: witness against finite-a.e.).
    pub fn fin_pm(&self) -> bool {
        self.has_fin && !self.fin_null
    }
    /// Defined on POSITIVE measure: the domain gate's witness semantics. `pinf`/`ninf` keep
    /// their historical reading (their x-support is not tracked; unchanged behaviour).
    pub fn defined_pm(&self) -> bool {
        self.fin_pm() || self.pinf || self.ninf
    }
    pub fn defined(&self) -> bool {
        self.has_fin || self.pinf || self.ninf
    }
    /// The value CLASS: `Mixed` means "more than one behaviour is present", i.e. no single
    /// collapse target can stand for this value set, and the caller must not collapse.
    ///
    /// RATIFIED LICENCE: a source may be simplified to one value only when its
    /// behaviour is unambiguous. Two consequences are encoded here:
    ///   - finite together with an infinity, or both infinities, is Mixed (as before); and
    ///   - **finite together with nan is Mixed too**. That arm was missing, so
    ///     `log <constant>` -- components (has_fin, nan) = (true, true), finite on C>0 and
    ///     undefined on C<0, i.e. TWO positive-measure behaviours -- reported `Finite` and was
    ///     collapsed to `<constant>`, filling a positive-measure undefined region. §4 R3 and
    ///     §9.8.3(b) both put that in the kill set ("KILLED on positive measure"); only a NULL
    ///     hole may be filled. LOSSY mode is unaffected: the fold short-circuits on
    ///     `wildcard_all` before consulting any certificate, so the training-corpus path still
    ///     collapses these.
    pub fn cls(&self) -> Class {
        let inf = self.pinf || self.ninf;
        if self.has_fin && inf {
            return Class::Mixed;
        }
        if self.pinf && self.ninf {
            return Class::Mixed;
        }
        if self.has_fin && self.nan {
            return Class::Mixed;
        }
        // An INFINITY together with nan is ambiguous too (H-019, 2026-08-04): the
        // finite+nan arm above was added when that gap shipped `<constant>` for
        // `log <constant>`, but the inf+nan twin was left behind -- `{+inf, nan}`
        // classified PosInf and the ground fold shipped an inf LITERAL for
        // `atanh(pow(x, nan))`-shaped values whose true behaviour is nan a.e.
        // (surfaced when the outward-rounded arms' null-point components met
        // atanh's pole arm). Same ratified licence: collapse only the unambiguous.
        if (self.pinf || self.ninf) && self.nan {
            return Class::Mixed;
        }
        if self.has_fin {
            return Class::Finite;
        }
        if self.pinf {
            return Class::PosInf;
        }
        if self.ninf {
            return Class::NegInf;
        }
        if self.nan {
            return Class::Nan;
        }
        Class::Empty
    }
    fn merge_pt(&mut self, v: f64) {
        self.fin_null = false; // a merged point has unknown x-support: conservative
        if !self.has_fin {
            self.has_fin = true;
            self.lo = v;
            self.hi = v;
            self.lo_open = false;
            self.hi_open = false;
        } else {
            if v < self.lo {
                self.lo = v;
                self.lo_open = false;
            }
            if v > self.hi {
                self.hi = v;
                self.hi_open = false;
            }
        }
    }
}

// ================= OUTWARD ROUNDING (H-019 option (a), owner-ruled 2026-08-04) ===============
// The arms below evaluate endpoint images in f64 round-to-nearest, and nearest results are
// NOT bounds: each operation can land up to half an ulp on the wrong side, so a computed
// "enclosure" could exclude true values -- and a certificate whose decision boundary sits
// within an ulp of an endpoint could then certify something false (the H-016/H-018 family,
// arm-level). Every COMPUTED endpoint is therefore stepped outward: 1 ulp for IEEE
// correctly-rounded arithmetic (+ - * /), a conservative budget for libm transcendentals
// (not correctly rounded; typically <= 1-2 ulp, 8 is cheap insurance). Amplification through
// ill-conditioned arms (exp of a wide argument) is carried by the INTERVAL ITSELF -- leaves
// are true enclosures and each arm maps true enclosures to true enclosures, so no
// consumption-site margin ever needs a condition-number argument (why option (b) lost).
// EXACT values are never stepped: semantic constants (the one-zero flush, parity minima,
// attained limits like tanh(+-inf) = 1), a-priori mathematical range bounds (exp >= 0,
// |sin| <= 1, cosh >= 1 -- applied as CLAMPS after widening), and computed-zero SUMS
// (IEEE addition is exact when the result is near zero: x + y == 0 iff y == -x).
// Computed-zero PRODUCTS and powers are NOT exact (underflow: 1e-200 * 1e-200 -> 0.0 with a
// true nonzero value whose sign is known) -- those corners bound by the signed minimum
// subnormal instead. Widening is always SOUND (a larger enclosure only makes certificates
// refuse); the care above is recall, not soundness.
// (`f64::next_up`/`next_down` need Rust 1.86; MSRV floor is 1.83 (pyo3) -- bit-stepping,
// as in `leaf_vs::ulp_bracket`.)
const ULP_ARITH: u32 = 1;
const ULP_LIBM: u32 = 8;

fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == -INF {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1); // the negative minimum subnormal
    }
    let b = x.to_bits();
    if x > 0.0 {
        f64::from_bits(b - 1)
    } else {
        f64::from_bits(b + 1)
    }
}

fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == INF {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let b = x.to_bits();
    if x > 0.0 {
        f64::from_bits(b + 1)
    } else {
        f64::from_bits(b - 1)
    }
}

/// Step a COMPUTED lower bound down by `n` ulps (non-finite passes through; stepping past
/// the finite range yields -INF = the unbounded edge, the module's overflow convention).
fn out_lo(x: f64, n: u32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let mut y = x;
    for _ in 0..n {
        y = next_down(y);
    }
    y
}

/// Step a COMPUTED upper bound up by `n` ulps.
fn out_hi(x: f64, n: u32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let mut y = x;
    for _ in 0..n {
        y = next_up(y);
    }
    y
}

/// f(x) tolerating overflow: an overflowed value is FINITE but unrepresentable -> report the
/// unbounded edge (+-INF), never an inf VALUE.
fn safe(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let y = f(x);
    if y.is_finite() || y.is_nan() {
        return y;
    }
    // overflowed: take the sign from a representable proxy in the same direction
    let probe = f(x.abs().min(700.0).copysign(x));
    if probe > 0.0 {
        INF
    } else {
        -INF
    }
}

/// Restrict v's finite part to [dlo,dhi]. Returns (clipped, escaped); `escaped` = a
/// positive-measure part lies OUTSIDE the domain (=> a nan REGION). A surviving overlap that is
/// a single POINT is measure-zero for a continuum input (`asin(cosh C)`) -> no finite part.
fn clip(v: &Vs, dlo: f64, dhi: f64) -> (Option<Vs>, bool) {
    if !v.has_fin {
        return (None, false);
    }
    if v.lo == v.hi {
        let c = v.lo;
        if c >= dlo && c <= dhi {
            return (Some(Vs::interval(c, c, false, false)), false);
        }
        return (None, true);
    }
    let lo = v.lo.max(dlo);
    let hi = v.hi.min(dhi);
    if lo > hi {
        return (None, true);
    }
    if lo == hi {
        // The overlap degenerates to the boundary point. If the graze sits at an OPEN
        // endpoint of v, the point is not attained at all: no finite part, exactly as
        // before (tanh's range is open at 1, so `acosh(tanh(..))` keeps its clean Nan).
        let open_at_graze = (lo == v.lo && v.lo_open) || (hi == v.hi && v.hi_open);
        if open_at_graze {
            return (None, true);
        }
        // Attained graze: under outward-rounded arms (H-019, 2026-08-04) a bracketed
        // POINT value grazes a domain edge in exactly this shape, and DROPPING the point
        // manufactured a Nan CLASS for a defined ground: `acosh(cos(0))` -- cos's honest
        // enclosure [1 - 8ulp, 1] met acosh's domain in {1}, the TRUE attained value,
        // and the fold shipped `nan` for a value that is 0 (caught by the 64k judge as
        // a positive-measure SHRINK). Keep the point, NULL-supported: for a ground the
        // classification then reads Mixed (fold refused, the exact rules take over --
        // `cos 0 -> 1`, `acosh 1 -> 0`); for a variable/Const-parameterized family the
        // measure-aware consumer (`value_class`) may still null-kill it (R3), restoring
        // the continuum verdicts. `defined_pm` witnesses stay blocked either way.
        let mut kept = Vs::interval(lo, hi, false, false);
        kept.fin_null = true;
        return (Some(kept), true);
    }
    let escaped = v.lo < dlo || v.hi > dhi;
    let lo_open = if v.lo >= dlo { v.lo_open } else { false };
    let hi_open = if v.hi <= dhi { v.hi_open } else { false };
    (Some(Vs::interval(lo, hi, lo_open, hi_open)), escaped)
}

/// monotone-increasing continuous op on a domain, with optional REGION nan outside it and
/// optional measure-zero POLES at the domain edges.
#[allow(clippy::too_many_arguments)]
fn domain_op(
    v: &Vs,
    dlo: f64,
    dhi: f64,
    f: impl Fn(f64) -> f64,
    inc: bool,
    lo_lim: Option<f64>,
    hi_lim: Option<f64>,
    pole_lo: bool,
    pole_hi: bool,
    inf_at_lo: Option<f64>,
    inf_at_hi: Option<f64>,
) -> Vs {
    let mut r = Vs::empty();
    r.nan = v.nan;
    let (clipped, escaped) = clip(v, dlo, dhi);
    if escaped {
        r.nan = true;
    }
    // `fin_is_point`, NOT `is_const` (same root as `u_inv`): the finite part can be a single
    // point while the set ALSO carries +-inf -- `pow(0, C)` is {0} u {+inf}. `is_const` is false
    // there, so `log(pow(0, C))` skipped this pole branch and mapped [0,0] through the range path
    // into an "unbounded but FINITE" [-INF,-INF], losing the -inf VALUE that log(0) takes on
    // positive measure. A point that is NOT at a domain edge still falls through to the range
    // mapping below, which is what we want.
    if v.fin_is_point() {
        let c = v.lo;
        if c == dlo {
            if let Some(s) = inf_at_lo {
                if s > 0.0 {
                    r.pinf = true
                } else {
                    r.ninf = true
                }
                return r;
            }
        }
        if c == dhi {
            if let Some(s) = inf_at_hi {
                if s > 0.0 {
                    r.pinf = true
                } else {
                    r.ninf = true
                }
                return r;
            }
        }
    }
    if let Some(cl) = clipped {
        // `computed` marks endpoint images that came from f64 evaluation of `f` -- ONLY
        // those are stepped outward (H-019). Pole reaches and caller limits are exact
        // semantics; an irrational limit is the CALLER's job to pre-step (atan).
        let (mut a, mut a_open, mut a_computed) = if pole_lo && cl.lo == dlo {
            (-INF, true, false)
        } else if cl.lo.is_infinite() {
            (lo_lim.unwrap_or(-INF), true, false)
        } else {
            (safe(&f, cl.lo), cl.lo_open, true)
        };
        let (mut b, mut b_open, mut b_computed) = if pole_hi && cl.hi == dhi {
            (INF, true, false)
        } else if cl.hi.is_infinite() {
            (hi_lim.unwrap_or(INF), true, false)
        } else {
            (safe(&f, cl.hi), cl.hi_open, true)
        };
        if a.is_infinite() {
            a_open = true;
        }
        if b.is_infinite() {
            b_open = true;
        }
        if !inc {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut a_open, &mut b_open);
            std::mem::swap(&mut a_computed, &mut b_computed);
        }
        // EXACT HITS never widen: libm guarantees these identically and they are true
        // (f(0) = 0 for the odd functions, exp(0) = cos(0) = cosh(0) = 1, log(1) =
        // acosh(1) = 0, odd roots fix +-1). Widening them is what cost the mine its
        // strict-bound families (`log(sinh(exp C))` needs sinh's range open at the
        // EXACT 0).
        let exact_hit = |x: f64, y: f64| {
            ((x == 0.0 || x == 1.0) && (y == 0.0 || y == 1.0)) || (x == -1.0 && y == -1.0)
        };
        let (src_a, src_b) = if inc { (cl.lo, cl.hi) } else { (cl.hi, cl.lo) };
        if a_computed && !exact_hit(src_a, a) {
            a = out_lo(a, ULP_LIBM);
        }
        if b_computed && !exact_hit(src_b, b) {
            b = out_hi(b, ULP_LIBM);
        }
        r.has_fin = true;
        r.lo = a;
        r.hi = b;
        r.lo_open = a_open;
        r.hi_open = b_open;
        // Image of a null x-support is null-supported; a degenerate boundary overlap
        // kept by `clip` (H-019) carries its own null marking.
        r.fin_null = v.fin_null || cl.fin_null;
    }
    r
}

/// everywhere-defined monotone-increasing op
fn mono(v: &Vs, f: impl Fn(f64) -> f64, lo_lim: Option<f64>, hi_lim: Option<f64>) -> Vs {
    domain_op(
        v, -INF, INF, f, true, lo_lim, hi_lim, false, false, None, None,
    )
}

// ================= UNARY =====================================================
fn u_neg(v: &Vs) -> Vs {
    let mut r = Vs::empty();
    r.nan = v.nan;
    r.pinf = v.ninf;
    r.ninf = v.pinf;
    if v.has_fin {
        r.has_fin = true;
        r.lo = -v.hi;
        r.hi = -v.lo;
        r.lo_open = v.hi_open;
        r.hi_open = v.lo_open;
        r.fin_null = v.fin_null;
    }
    r
}

fn u_abs(v: &Vs) -> Vs {
    let mut r = Vs::empty();
    r.nan = v.nan;
    r.pinf = v.pinf || v.ninf;
    if v.has_fin {
        r.has_fin = true;
        if v.lo <= 0.0 && 0.0 <= v.hi {
            r.lo = 0.0;
            r.lo_open = false;
            r.hi = v.lo.abs().max(v.hi.abs());
            r.hi_open = r.hi.is_infinite();
        } else {
            let (a, b) = (v.lo.abs(), v.hi.abs());
            r.lo = a.min(b);
            r.hi = a.max(b);
            r.lo_open = r.lo.is_infinite();
            r.hi_open = r.hi.is_infinite();
        }
        r.fin_null = v.fin_null;
    }
    r
}

/// 1/x is EXACT iff |x| is a power of two (finite, zero mantissa): those reciprocals --
/// 1/1, 1/2, 1/(-4) -- never widen (H-019), keeping `inv(cosh C) <= 1` provable.
fn recip_exact(x: f64) -> bool {
    x.is_finite() && x != 0.0 && (x.to_bits() & ((1u64 << 52) - 1)) == 0
}

/// 1/x: the POLE at 0 is a measure-ZERO point unless the input is the constant 0.
fn u_inv(v: &Vs) -> Vs {
    let mut r = Vs::empty();
    r.nan = v.nan;
    if v.pinf || v.ninf {
        r.merge_pt(0.0);
    }
    if v.has_fin {
        // `fin_is_point`, NOT `is_const`: the finite part may be the degenerate point {0} while
        // the set ALSO carries +inf (`inv(pow(neg C, inf))`). `is_const` is false there, so the
        // point fell into the range branch below and `1/[0,0]` came out as an unbounded-but-
        // FINITE range -- silently losing the +inf that division by an identically-zero
        // denominator produces on positive measure.
        if v.fin_is_point() {
            if v.lo == 0.0 {
                // ONE ZERO (contract v2 §9.2, DECIDED 2026-07-18; H-016 2026-08-03): the point
                // zero is THE unsigned zero and its pole is +inf by convention -- the sign of an
                // f64 zero is measurement rendering, never a value. This branch used to read the
                // sign bit (1/(-0) = -inf, referee-era deployment semantics) to protect the
                // gates from under-reporting what IEEE evaluation reaches; §9.2 formally
                // overturned that semantics (the miner-side R2'' signed-zero tie-break was
                // deleted at ratification) and the surviving read here shipped a `-inf` LITERAL
                // for `inv(0/tanh(-2))` while every structurally-collapsing sibling spelling
                // shipped +inf. Deployment still EMITS -inf on the -0 slice at runtime; that is
                // documented measurement error (§9.2), not a value this layer may report.
                r.pinf = true;
            } else {
                // 1/c is generally inexact: bracket the computed reciprocal (H-019);
                // reciprocals of powers of two are exact and stay points.
                let y = 1.0 / v.lo;
                if recip_exact(v.lo) {
                    r.merge_pt(y);
                } else {
                    r.merge_pt(out_lo(y, ULP_ARITH));
                    r.merge_pt(out_hi(y, ULP_ARITH));
                }
            }
        } else if v.lo < 0.0 && 0.0 < v.hi {
            // crosses the pole -> unbounded, but every value finite
            r.has_fin = true;
            r.lo = -INF;
            r.hi = INF;
            r.lo_open = true;
            r.hi_open = true;
        } else {
            // Each endpoint reciprocal becomes its own bracket unless exact -- a
            // power-of-two source, or the exact 0 limit of an infinite edge (widening
            // 1/inf = 0 below zero manufactured a spurious log-domain escape); the
            // hull of the brackets is the enclosure (H-019).
            let br = |src: f64, y: f64| -> (f64, f64) {
                if !y.is_finite() || y == 0.0 || recip_exact(src) {
                    (y, y)
                } else {
                    (out_lo(y, ULP_ARITH), out_hi(y, ULP_ARITH))
                }
            };
            let (a1, a2) = if v.lo == 0.0 {
                if v.hi > 0.0 {
                    (INF, INF)
                } else {
                    (-INF, -INF)
                }
            } else {
                br(v.lo, 1.0 / v.lo)
            };
            let (b1, b2) = if v.hi == 0.0 {
                if v.lo < 0.0 {
                    (-INF, -INF)
                } else {
                    (INF, INF)
                }
            } else {
                br(v.hi, 1.0 / v.hi)
            };
            let (mut nlo, mut nhi) = (a1.min(b1), a2.max(b2));
            if r.has_fin {
                nlo = nlo.min(r.lo);
                nhi = nhi.max(r.hi);
            }
            r.has_fin = true;
            r.lo = nlo;
            r.hi = nhi;
            r.lo_open = nlo.is_infinite();
            r.hi_open = nhi.is_infinite();
        }
        if r.has_fin {
            // fin here derives from v's fin UNLESS the inf-image point 0 also contributed
            r.fin_null = v.fin_null && !(v.pinf || v.ninf);
        }
    }
    r
}

fn u_exp(v: &Vs) -> Vs {
    // exp over FINITE reals is finite and STRICTLY positive: (0, inf), OPEN at 0.
    // The 0 floor is an a-priori mathematical bound: outward stepping (H-019) may push a
    // computed endpoint below it (underflowed exp(-800) steps to a negative subnormal),
    // and the clamp restores the exact floor -- enclosure INTERSECT known range.
    let mut r = mono(v, |x| x.exp(), Some(0.0), Some(INF));
    if r.has_fin && r.lo <= 0.0 {
        r.lo = 0.0;
        r.lo_open = true;
    }
    r.nan = v.nan;
    if v.pinf {
        r.pinf = true;
    }
    if v.ninf {
        r.merge_pt(0.0);
    }
    r
}

fn u_log(v: &Vs) -> Vs {
    // domain (0, inf): nan for input<0 (REGION); -inf at input=0 (measure-zero POINT)
    let mut r = domain_op(
        v,
        0.0,
        INF,
        |x| x.ln(),
        true,
        Some(-INF),
        Some(INF),
        true,
        false,
        Some(-1.0),
        None,
    );
    if v.pinf {
        r.pinf = true;
    }
    if v.ninf {
        r.nan = true;
    }
    r
}

fn u_tanh(v: &Vs) -> Vs {
    // (-1, 1) is the a-priori range: clamp widened endpoints back to the exact bounds.
    let mut r = mono(v, |x| x.tanh(), Some(-1.0), Some(1.0));
    r.nan = v.nan;
    if r.has_fin && r.lo <= -1.0 {
        r.lo = -1.0;
        r.lo_open = true;
    }
    if r.has_fin && r.hi >= 1.0 {
        r.hi = 1.0;
        r.hi_open = true;
    }
    if v.pinf {
        r.merge_pt(1.0);
    }
    if v.ninf {
        r.merge_pt(-1.0);
    }
    r
}

fn u_atanh(v: &Vs) -> Vs {
    // domain (-1,1): nan for |x|>1 (REGION); +-inf at x=+-1 (measure-zero POINTS)
    let mut r = domain_op(
        v,
        -1.0,
        1.0,
        |x| x.atanh(),
        true,
        None,
        None,
        true,
        true,
        Some(-1.0),
        Some(1.0),
    );
    if v.pinf || v.ninf {
        r.nan = true;
    }
    r
}

fn u_asin(v: &Vs) -> Vs {
    let mut r = domain_op(
        v,
        -1.0,
        1.0,
        |x| x.asin(),
        true,
        None,
        None,
        false,
        false,
        None,
        None,
    );
    if v.pinf || v.ninf {
        r.nan = true;
    }
    r
}

fn u_acos(v: &Vs) -> Vs {
    let mut r = domain_op(
        v,
        -1.0,
        1.0,
        |x| x.acos(),
        false,
        None,
        None,
        false,
        false,
        None,
        None,
    );
    // Range floor 0 (attained at x = 1) is exact; the ceiling pi is irrational and the
    // widened computed endpoint is the honest bound there.
    if r.has_fin && r.lo < 0.0 {
        r.lo = 0.0;
        r.lo_open = false;
    }
    if v.pinf || v.ninf {
        r.nan = true;
    }
    r
}

fn u_acosh(v: &Vs) -> Vs {
    let mut r = domain_op(
        v,
        1.0,
        INF,
        |x| x.acosh(),
        true,
        None,
        Some(INF),
        false,
        false,
        None,
        None,
    );
    // Range floor 0 (attained at x = 1) is exact.
    if r.has_fin && r.lo < 0.0 {
        r.lo = 0.0;
        r.lo_open = false;
    }
    if v.pinf {
        r.pinf = true;
    }
    if v.ninf {
        r.nan = true;
    }
    r
}

fn u_even_root(v: &Vs, root: f64) -> Vs {
    let mut r = domain_op(
        v,
        0.0,
        INF,
        |x| x.powf(1.0 / root),
        true,
        None,
        Some(INF),
        false,
        false,
        None,
        None,
    );
    // Range floor 0 (attained at x = 0) is exact.
    if r.has_fin && r.lo < 0.0 {
        r.lo = 0.0;
        r.lo_open = false;
    }
    if v.pinf {
        r.pinf = true;
    }
    if v.ninf {
        r.nan = true;
    }
    r
}

fn u_odd_root(v: &Vs, root: f64) -> Vs {
    let mut r = mono(v, |x| x.abs().powf(1.0 / root).copysign(x), None, None);
    r.nan = v.nan;
    r.pinf = v.pinf;
    r.ninf = v.ninf;
    r
}

fn u_cosh(v: &Vs) -> Vs {
    let mut r = Vs::empty();
    r.nan = v.nan;
    if v.has_fin {
        r.has_fin = true;
        r.fin_null = v.fin_null; // image of a null x-support is null-supported (H-019)
                                 // The minimum 1 is ATTAINED only if 0 itself is in the set: an open-zero
                                 // endpoint ((0, b] -- exp's range) yields the OPEN exact bound (1, cosh b],
                                 // which is what lets `acos(cosh(exp C))` keep its clean Nan (H-019).
        let zero_attained = (v.lo < 0.0 || (v.lo == 0.0 && !v.lo_open))
            && (v.hi > 0.0 || (v.hi == 0.0 && !v.hi_open));
        if zero_attained {
            r.lo = 1.0; // the a-priori minimum, attained at 0: exact, never widened
            r.lo_open = false;
            r.hi = if v.lo.is_infinite() || v.hi.is_infinite() {
                INF
            } else if v.lo == 0.0 && v.hi == 0.0 {
                1.0 // cosh(0) = 1 is an exact libm hit: the POINT image stays a point
                    // (a widened [1, 1+8u] made pow(-1, cosh 0) read a CONTINUUM
                    // exponent and fold nan where the true value is tanh(-1))
            } else {
                out_hi(
                    safe(|x| x.cosh(), v.lo).max(safe(|x| x.cosh(), v.hi)),
                    ULP_LIBM,
                )
            };
            r.hi_open = r.hi.is_infinite();
        } else {
            // Monotone on either side of 0: per-endpoint image brackets carrying the
            // endpoint's OPENNESS; cosh(0) = 1 is an exact libm hit and never widens;
            // cosh >= 1 is the a-priori floor for the widened rest.
            let im = |x: f64, open: bool| -> (f64, f64, bool) {
                if x.is_infinite() {
                    return (INF, INF, true);
                }
                let y = safe(|z| z.cosh(), x);
                if x == 0.0 && y == 1.0 {
                    (y, y, open)
                } else {
                    (out_lo(y, ULP_LIBM).max(1.0), out_hi(y, ULP_LIBM), open)
                }
            };
            let (alo, ahi, ao) = im(v.lo, v.lo_open);
            let (blo, bhi, bo) = im(v.hi, v.hi_open);
            if alo <= blo {
                r.lo = alo;
                r.lo_open = ao || alo.is_infinite();
            } else {
                r.lo = blo;
                r.lo_open = bo || blo.is_infinite();
            }
            if ahi >= bhi {
                r.hi = ahi;
                r.hi_open = ao || ahi.is_infinite();
            } else {
                r.hi = bhi;
                r.hi_open = bo || bhi.is_infinite();
            }
        }
    }
    if v.pinf || v.ninf {
        r.pinf = true;
    }
    r
}

fn u_tan(v: &Vs) -> Vs {
    // poles at pi/2 + k*pi are MEASURE-ZERO -> unbounded finite range, no pinf
    let mut r = Vs::empty();
    r.nan = v.nan;
    if v.pinf || v.ninf {
        r.nan = true;
    }
    if v.has_fin {
        r.has_fin = true;
        r.fin_null = v.fin_null; // image of a null x-support is null-supported (H-019)
                                 // The pole positions pi/2 + k*pi are irrational and computed in f64: a containment
                                 // test on the nearest-rounded position can MISS a true pole by an ulp, and a missed
                                 // pole is an O(1) enclosure hole, not an ulp (H-019). The computed pole is therefore
                                 // BRACKETED and the test errs toward "spans" -- the unbounded verdict is the sound
                                 // direction. Three candidates cover the quotient's own rounding.
        let spans =
            v.lo.is_infinite() || v.hi.is_infinite() || (v.hi - v.lo) >= std::f64::consts::PI || {
                let k = ((v.lo + std::f64::consts::FRAC_PI_2) / std::f64::consts::PI).floor();
                (0..3).any(|i| {
                    let p = -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * (k + i as f64);
                    out_hi(p, 4) > v.lo && out_lo(p, 4) < v.hi
                })
            };
        if spans {
            r.lo = -INF;
            r.hi = INF;
            r.lo_open = true;
            r.hi_open = true;
        } else {
            // tan(0) = 0 is an exact libm hit and never widens (a widened tan(0)
            // bracket broke the x^0 = 1 total fold for `pow(nan, tan 0)`).
            let im = |x: f64| -> (f64, f64) {
                let y = safe(|z| z.tan(), x);
                if x == 0.0 && y == 0.0 {
                    (y, y)
                } else {
                    (out_lo(y, ULP_LIBM), out_hi(y, ULP_LIBM))
                }
            };
            let (a1, a2) = im(v.lo);
            let (b1, b2) = im(v.hi);
            r.lo = a1.min(b1);
            r.hi = a2.max(b2);
        }
    }
    r
}

fn u_trig(v: &Vs, is_sin: bool) -> Vs {
    let mut r = Vs::empty();
    r.nan = v.nan;
    if v.pinf || v.ninf {
        r.nan = true;
    }
    if v.has_fin {
        r.has_fin = true;
        r.fin_null = v.fin_null; // image of a null x-support is null-supported (H-019)
        let two_pi = 2.0 * std::f64::consts::PI;
        if v.lo.is_infinite() || v.hi.is_infinite() || (v.hi - v.lo) >= two_pi {
            r.lo = -1.0;
            r.hi = 1.0;
        } else {
            let f = |x: f64| if is_sin { x.sin() } else { x.cos() };
            let (crit_max, crit_min) = if is_sin {
                (std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2)
            } else {
                (0.0, std::f64::consts::PI)
            };
            // Peak positions are irrational, computed in f64: a missed peak is an O(1)
            // enclosure hole (the hull would exclude +-1). The computed position is
            // BRACKETED and the test errs toward containment (H-019); the +-1 verdict is
            // the a-priori exact bound, so over-claiming a peak is only recall loss.
            let contains = |c: f64| {
                let k0 = ((v.lo - c) / two_pi).floor();
                (0..3).any(|i| {
                    let p = c + two_pi * (k0 + i as f64);
                    out_hi(p, 4) >= v.lo && out_lo(p, 4) <= v.hi
                })
            };
            let (a, b) = (f(v.lo), f(v.hi));
            // Exact libm hits at 0 (sin 0 = 0, cos 0 = 1) never widen; everything else
            // steps outward and clamps back into the exact [-1, 1] range.
            let exact = |x: f64, y: f64| x == 0.0 && (y == 0.0 || y == 1.0);
            r.hi = if contains(crit_max) {
                1.0
            } else {
                let m = a.max(b);
                let src = if m == a { v.lo } else { v.hi };
                if exact(src, m) {
                    m
                } else {
                    out_hi(m, ULP_LIBM).min(1.0)
                }
            };
            r.lo = if contains(crit_min) {
                -1.0
            } else {
                let m = a.min(b);
                let src = if m == a { v.lo } else { v.hi };
                if exact(src, m) {
                    m
                } else {
                    out_lo(m, ULP_LIBM).max(-1.0)
                }
            };
        }
    }
    r
}

/// Integer power of a value set for the RETIRED unary arms (`pow2`..`pow5` only, k > 0):
/// monotone-piece endpoint bounds, kept as cross-generation instrument support (marked
/// legacy at the arm table). Deliberately NOT shared with the live binary path: `b_pow`'s
/// const-integer-exponent branch (audit Tier-2, 2026-08-03) carries the module's
/// open-edge/pole conventions (pole reach as unbounded OPEN edges, `fin_null`
/// propagation, attained base infinities by sign/parity) and negative exponents, none of
/// which this positive-k instrument arm needs -- delegating either way would either lose
/// precision or re-prove the legacy arms for conventions they never exercise.
fn u_pow_int(v: &Vs, k: i32) -> Vs {
    let mut r = Vs::empty();
    r.nan = v.nan;
    if v.has_fin {
        r.fin_null = v.fin_null; // image of a null x-support is null-supported (H-019)
        let ulps = 2 + k.unsigned_abs(); // powi compounds ~one rounding per multiply
        let p = |x: f64| safe(|z| z.powi(k), x);
        let (a, b) = if k % 2 == 0 {
            if v.lo <= 0.0 && 0.0 <= v.hi {
                (
                    0.0, // the parity minimum, attained: exact, never widened
                    if v.lo.is_infinite() || v.hi.is_infinite() {
                        INF
                    } else {
                        out_hi(p(v.lo).max(p(v.hi)), ulps)
                    },
                )
            } else {
                let (x, y) = (
                    if v.lo.is_infinite() { INF } else { p(v.lo) },
                    if v.hi.is_infinite() { INF } else { p(v.hi) },
                );
                (out_lo(x.min(y), ulps), out_hi(x.max(y), ulps))
            }
        } else {
            (
                if v.lo.is_infinite() {
                    -INF
                } else {
                    out_lo(p(v.lo), ulps)
                },
                if v.hi.is_infinite() {
                    INF
                } else {
                    out_hi(p(v.hi), ulps)
                },
            )
        };
        r.has_fin = true;
        r.lo = a;
        r.hi = b;
        r.lo_open = a.is_infinite();
        r.hi_open = b.is_infinite();
    }
    if v.pinf {
        r.pinf = true;
    }
    if v.ninf {
        if k % 2 == 0 {
            r.pinf = true
        } else {
            r.ninf = true
        }
    }
    r
}

fn u_scale(v: &Vs, c: f64) -> Vs {
    let mut r = Vs::empty();
    r.nan = v.nan;
    if v.has_fin {
        r.fin_null = v.fin_null; // image of a null x-support is null-supported (H-019)
        let mut a = if v.lo.is_infinite() {
            -INF
        } else {
            safe(|z| z * c, v.lo)
        };
        let mut b = if v.hi.is_infinite() {
            INF
        } else {
            safe(|z| z * c, v.hi)
        };
        let (mut ao, mut bo) = (v.lo_open, v.hi_open);
        if c < 0.0 {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut ao, &mut bo);
        }
        // AFTER the sign swap, so each bound steps in its own direction. 2 ulps: one for
        // the product rounding, one for `c` itself (div3's 1/3 is an inexact constant).
        a = out_lo(a, 2);
        b = out_hi(b, 2);
        r.has_fin = true;
        r.lo = a;
        r.hi = b;
        r.lo_open = ao || a.is_infinite();
        r.hi_open = bo || b.is_infinite();
    }
    if v.pinf {
        if c > 0.0 {
            r.pinf = true
        } else {
            r.ninf = true
        }
    }
    if v.ninf {
        if c > 0.0 {
            r.ninf = true
        } else {
            r.pinf = true
        }
    }
    r
}

fn apply_unary(op: &str, v: &Vs) -> Option<Vs> {
    Some(match op {
        "neg" => u_neg(v),
        "abs" => u_abs(v),
        "inv" => u_inv(v),
        "exp" => u_exp(v),
        "log" => u_log(v),
        "sin" => u_trig(v, true),
        "cos" => u_trig(v, false),
        "tan" => u_tan(v),
        "tanh" => u_tanh(v),
        "atanh" => u_atanh(v),
        "asin" => u_asin(v),
        "acos" => u_acos(v),
        "acosh" => u_acosh(v),
        "cosh" => u_cosh(v),
        "sinh" => {
            let mut r = mono(v, |x| x.sinh(), None, None);
            r.nan = v.nan;
            r.pinf = v.pinf;
            r.ninf = v.ninf;
            r
        }
        "asinh" => {
            let mut r = mono(v, |x| x.asinh(), None, None);
            r.nan = v.nan;
            r.pinf = v.pinf;
            r.ninf = v.ninf;
            r
        }
        "atan" => {
            // pi/2 is IRRATIONAL: the f64 constant sits BELOW the true limit, so both the
            // range limits and the attained values at +-inf must be stepped OUTWARD to be
            // enclosures (true atan(1e300) exceeds f64(pi/2)) -- the leaf-level H-018
            // doctrine applied to a limit constant.
            let mut r = mono(
                v,
                |x| x.atan(),
                Some(out_lo(-std::f64::consts::FRAC_PI_2, 1)),
                Some(out_hi(std::f64::consts::FRAC_PI_2, 1)),
            );
            r.nan = v.nan;
            if v.pinf {
                r.merge_pt(out_lo(std::f64::consts::FRAC_PI_2, 1));
                r.merge_pt(out_hi(std::f64::consts::FRAC_PI_2, 1));
            }
            if v.ninf {
                r.merge_pt(out_lo(-std::f64::consts::FRAC_PI_2, 1));
                r.merge_pt(out_hi(-std::f64::consts::FRAC_PI_2, 1));
            }
            r
        }
        "pow2" => u_pow_int(v, 2),
        "pow3" => u_pow_int(v, 3),
        "pow4" => u_pow_int(v, 4),
        "pow5" => u_pow_int(v, 5),
        "pow1_2" => u_even_root(v, 2.0),
        "pow1_4" => u_even_root(v, 4.0),
        "pow1_3" => u_odd_root(v, 3.0),
        "pow1_5" => u_odd_root(v, 5.0),
        "mult2" => u_scale(v, 2.0),
        "mult3" => u_scale(v, 3.0),
        "mult4" => u_scale(v, 4.0),
        "mult5" => u_scale(v, 5.0),
        "div2" => u_scale(v, 0.5),
        "div3" => u_scale(v, 1.0 / 3.0),
        "div4" => u_scale(v, 0.25),
        "div5" => u_scale(v, 0.2),
        _ => return None,
    })
}

// ================= BINARY ====================================================
fn abs_range(v: &Vs) -> Option<(f64, f64)> {
    if !v.has_fin {
        return None;
    }
    let amin = if v.lo <= 0.0 && 0.0 <= v.hi {
        0.0
    } else {
        v.lo.abs().min(v.hi.abs())
    };
    Some((amin, v.lo.abs().max(v.hi.abs())))
}

fn b_add(a: &Vs, b: &Vs) -> Vs {
    let mut r = Vs::empty();
    r.nan = a.nan || b.nan || (a.pinf && b.ninf) || (a.ninf && b.pinf);
    if a.has_fin && b.has_fin {
        // A computed-zero SUM is exact (IEEE addition: x + y == 0 iff y == -x, and
        // near-zero sums are exactly representable), so the zero bound stays; every other
        // computed sum steps outward (H-019).
        let lo = if a.lo.is_infinite() || b.lo.is_infinite() {
            -INF
        } else {
            let s = a.lo + b.lo;
            if s == 0.0 {
                s
            } else {
                out_lo(s, ULP_ARITH)
            }
        };
        let hi = if a.hi.is_infinite() || b.hi.is_infinite() {
            INF
        } else {
            let s = a.hi + b.hi;
            if s == 0.0 {
                s
            } else {
                out_hi(s, ULP_ARITH)
            }
        };
        r.has_fin = true;
        r.lo = lo;
        r.hi = hi;
        r.lo_open = lo.is_infinite() || a.lo_open || b.lo_open;
        r.hi_open = hi.is_infinite() || a.hi_open || b.hi_open;
        r.fin_null = a.fin_null || b.fin_null; // x-support of a+b is within both supports
    }
    // The PAIRING matters: +inf survives only when added to something that is NOT -inf.
    // `defined()` includes the opposite infinity, so `inf + (-inf)` reported +inf AND -inf
    // alongside the correct nan -- an over-approximation, and a costly one: the reachability
    // gate would then demand a candidate reach +-inf and reject the true rule
    // `inf + (-inf) -> nan`, whose target reaches neither.
    if (a.pinf && (b.has_fin || b.pinf)) || (b.pinf && (a.has_fin || a.pinf)) {
        r.pinf = true;
    }
    if (a.ninf && (b.has_fin || b.ninf)) || (b.ninf && (a.has_fin || a.ninf)) {
        r.ninf = true;
    }
    r
}

fn b_mul(a: &Vs, b: &Vs) -> Vs {
    let mut r = Vs::empty();
    r.nan = a.nan || b.nan;
    if (a.pinf || a.ninf) && b.has_fin {
        if b.hi > 0.0 {
            if a.pinf {
                r.pinf = true
            } else {
                r.ninf = true
            }
        }
        if b.lo < 0.0 {
            if a.pinf {
                r.ninf = true
            } else {
                r.pinf = true
            }
        }
        if b.is_const() && b.lo == 0.0 {
            r.nan = true;
        } // inf*0 = nan identically
    }
    if (b.pinf || b.ninf) && a.has_fin {
        if a.hi > 0.0 {
            if b.pinf {
                r.pinf = true
            } else {
                r.ninf = true
            }
        }
        if a.lo < 0.0 {
            if b.pinf {
                r.ninf = true
            } else {
                r.pinf = true
            }
        }
        if a.is_const() && a.lo == 0.0 {
            r.nan = true;
        }
    }
    if (a.pinf || a.ninf) && (b.pinf || b.ninf) {
        if (a.pinf && b.pinf) || (a.ninf && b.ninf) {
            r.pinf = true
        } else {
            r.ninf = true
        }
    }
    if a.has_fin && b.has_fin {
        let m = |x: f64, y: f64| -> f64 {
            if (x == 0.0 && y.is_infinite()) || (y == 0.0 && x.is_infinite()) {
                // A zero endpoint against an OPEN infinite endpoint: the values near that corner
                // are 0 * (huge finite), i.e. a zero -- THE zero (§9.2 one-zero: the sign of an
                // f64 zero is measurement rendering, never a value). IEEE `0 * inf = NaN` is the
                // wrong answer for the open corner, hence the case; +0.0 is the only zero.
                return 0.0;
            }
            if x.is_infinite() || y.is_infinite() {
                return if (x > 0.0) == (y > 0.0) { INF } else { -INF };
            }
            let p = safe(|_| x * y, 0.0);
            // A computed-zero PRODUCT with nonzero factors is UNDERFLOW, not zero: the
            // true value is nonzero-tiny with the factors' sign product -- bound it by
            // the signed minimum subnormal so the hull cannot exclude it (H-019). A
            // zero FACTOR gives the exact zero and stays.
            if p == 0.0 && x != 0.0 && y != 0.0 {
                return f64::from_bits(1).copysign(x) * y.signum();
            }
            p
        };
        let ps = [m(a.lo, b.lo), m(a.lo, b.hi), m(a.hi, b.lo), m(a.hi, b.hi)];
        r.fin_null = a.fin_null || b.fin_null; // x-support of a*b is within both supports
        let mut lo = ps.iter().cloned().fold(INF, f64::min);
        let mut hi = ps.iter().cloned().fold(-INF, f64::max);
        // Widen the computed corners outward (H-019); an exact-zero bound (a zero factor)
        // stays -- IEEE multiplication by zero is exact, and the one-zero flush below
        // depends on the bound being genuinely zero.
        if lo != 0.0 {
            lo = out_lo(lo, ULP_ARITH);
        }
        if hi != 0.0 {
            hi = out_hi(hi, ULP_ARITH);
        }
        // ONE ZERO (§9.2; H-016): zero endpoints are normalized to +0.0. This block used to do
        // the OPPOSITE -- restore the IEEE sign of the endpoint products so that u_inv and
        // b_pow's zero-base branch could read the bit and pick a signed pole; §9.2 made that
        // sign non-normative and H-016 caught it shipping `-inf` for `inv(0/tanh(-2))`. No
        // consumer reads the bit anymore; flushing here keeps a `-0.0` that IEEE endpoint
        // arithmetic happens to produce (e.g. `0 * negative`) from ever leaving the fold, so
        // the representation cannot re-grow a reader.
        if lo == 0.0 {
            lo = 0.0;
        }
        if hi == 0.0 {
            hi = 0.0;
        }
        r.has_fin = true;
        r.lo = lo;
        r.hi = hi;
        r.lo_open = lo.is_infinite();
        r.hi_open = hi.is_infinite();
    }
    r
}

fn b_pow(a: &Vs, b: &Vs) -> Vs {
    let mut r = Vs::empty();
    r.nan = a.nan || b.nan;
    let ar = abs_range(a);

    // IEEE 754 (and numpy, the DEPLOYED evaluator): x^0 == 1 for EVERY x -- nan and +-inf
    // included. Must come before every other branch, including the nan propagation above:
    // `pow(nan, 0)` is 1.0, not nan. A `<constant>` exponent that could be 0 is measure-zero and
    // deliberately does NOT qualify (`is_const` is false for a range).
    if b.is_const() && b.lo == 0.0 {
        let mut z = Vs::empty();
        z.merge_pt(1.0);
        return z;
    }

    // H-029 (2026-08-05): the x^0 == 1 TOTALITY also holds on the {exponent == 0} SLICE
    // of a TIGHT exponent range -- an outward-rounded ENCLOSURE of a ground whose exact
    // value IS 0 but which could not fold to the literal (`sin(sin(acos 1))` brackets
    // 0). The value 1 is reachable there for EVERY base including nan and the
    // infinities; without this point a nan-based pow classified nan-ONLY and the
    // determined-ground fold SHIPPED nan for an expression whose true value is 1 (fuzz
    // row 662833). TIGHTNESS GATE (contains 0 and no other integer, i.e. within
    // (-1, 1)): wide ranges keep the ruled null-drop conventions -- the x^0 arm's
    // "`<constant>` does not qualify" note and the continuum nan convention -- because
    // Vs carries no null annotation on its inf flags and a broad hull would poison
    // downstream ranges (the `log(pow(0, <constant>))` pin). Null-supported: for a
    // measured family the slice is measure-zero (value_class's R3 null-kill keeps the
    // a.e. verdicts), while a GROUND has no measure space, the class reads Mixed, and
    // the fold REFUSES -- fail closed, the ground stays symbolic.
    if b.has_fin && b.lo <= 0.0 && 0.0 <= b.hi && b.lo.ceil() == 0.0 && b.hi.floor() == 0.0 {
        r.merge_pt(1.0);
        r.fin_null = true;
    }

    // exponent is the +-inf LITERAL: a STEP function of |a| vs 1 (a REGION behaviour)
    if b.pinf && !b.has_fin && !b.ninf {
        if let Some((amin, amax)) = ar {
            if amax > 1.0 {
                r.pinf = true;
            }
            if amin < 1.0 {
                r.merge_pt(0.0);
            }
            if a.is_const() && a.lo.abs() == 1.0 {
                r.merge_pt(1.0);
            }
        }
        // an inf BASE has no abs_range, so it used to fall straight out as EMPTY:
        // numpy `(+-inf)^inf = inf`
        if a.pinf || a.ninf {
            r.pinf = true;
        }
        return r;
    }
    if b.ninf && !b.has_fin && !b.pinf {
        if let Some((amin, amax)) = ar {
            if amin < 1.0 {
                r.pinf = true;
            }
            if amax > 1.0 {
                r.merge_pt(0.0);
            }
            if a.is_const() && a.lo.abs() == 1.0 {
                r.merge_pt(1.0);
            }
        }
        // numpy `(+-inf)^(-inf) = 0`
        if a.pinf || a.ninf {
            r.merge_pt(0.0);
        }
        return r;
    }
    // base is the constant 0: 0^b = 0 (b>0), +inf (b<0), 1 (b=0, measure-zero)
    if a.is_const() && a.lo == 0.0 {
        if b.has_fin {
            // ONE ZERO (§9.2; H-016): pow(0, y < 0) = +inf for EVERY y -- the base is THE
            // unsigned zero, so the odd-integer null slice that IEEE `pow(-0.0, odd k < 0) =
            // -inf` carved out does not exist in the value model. The `a.lo.is_sign_negative()`
            // ninf-add that lived here was the referee-era deployment read (the "fix direction
            // still flagged" thread in contract §7's R2/R3 row); deployment's -inf on the -0
            // slice is documented measurement error, not a reachable value.
            if b.lo < 0.0 {
                r.pinf = true;
            }
            if b.hi > 0.0 {
                r.merge_pt(0.0); // measured; merge_pt clears the H-029 null annotation
            }
        }
        if b.ninf {
            r.pinf = true;
        }
        if b.pinf {
            r.merge_pt(0.0);
        }
        return r;
    }
    // base is the constant 1: 1^anything = 1 (numpy convention, including 1^inf)
    if a.is_const() && a.lo == 1.0 {
        let mut z = Vs::empty();
        z.merge_pt(1.0);
        return z;
    }
    // base is a +-inf LITERAL. (+inf)^y = +inf (y>0), 0 (y<0). A -inf base follows the
    // aligned real semantics: defined only at integer exponents -- a const integer k gives
    // +-inf by parity (k>0) or 0 (k<0); a const non-integer gives nan; a CONTINUUM exponent
    // is non-integer a.e. -> nan (null-lattice contributions dropped, the same convention as
    // the negative-finite-const-base branch below). Infinite exponents keep the ratified
    // magnitude-step rows handled in the b-infinite branches above.
    if (a.pinf || a.ninf) && !a.has_fin {
        if a.pinf && b.has_fin {
            if b.hi > 0.0 {
                r.pinf = true;
            }
            if b.lo < 0.0 {
                r.merge_pt(0.0); // measured (the y < 0 half)
            }
        }
        if a.ninf {
            if b.is_const() {
                let k = b.lo;
                if k.is_finite() && k.fract() == 0.0 {
                    if k > 0.0 {
                        // Parity by EXACT float mod (the idiom of the H-029 arm below and
                        // the rootn gate): `k as i64` SATURATES at i64::MAX (odd) for
                        // k >= 2^63, flipping the infinity sign -- every integer-valued
                        // f64 with |k| >= 2^53 is even, and `% 2.0` is exact there (H-045).
                        if k % 2.0 == 0.0 {
                            r.pinf = true;
                        } else {
                            r.ninf = true;
                        }
                    } else if k < 0.0 {
                        r.merge_pt(0.0); // measured for a const exponent
                    }
                } else {
                    r.nan = true;
                }
            } else if b.has_fin {
                r.nan = true;
                // H-029 (2026-08-05): a TIGHT exponent range containing EXACTLY ONE
                // integer -- an outward-rounded ENCLOSURE of a ground integer
                // (`atan(0) + 3` brackets 3) -- reaches `(-inf)^k` on that slice.
                // nan-only asserted "the exponent is not an integer" about enclosure
                // slack: the determined-ground fold SHIPPED nan for an expression
                // whose true value is (-inf)^3 = -inf (fuzz row 809604). Union the
                // contained integer's value so the class reads Mixed and the fold
                // REFUSES (the exact rule path then finishes the true fold). Wide
                // ranges keep the documented continuum convention (nan a.e.,
                // null-lattice contributions dropped -- the `[0.1, 5]` pin).
                let (klo, khi) = (b.lo.ceil(), b.hi.floor());
                if klo == khi && klo.is_finite() {
                    if klo >= 1.0 {
                        if klo.abs() % 2.0 == 0.0 {
                            r.pinf = true;
                        } else {
                            r.ninf = true;
                        }
                    } else if klo <= -1.0 {
                        // (-inf)^(-k) = 0 on the slice only: null-supported
                        r.merge_pt(0.0);
                        r.fin_null = true;
                    }
                }
            }
        }
        if b.pinf {
            r.pinf = true;
        }
        if b.ninf {
            r.merge_pt(0.0);
        }
        return r;
    }
    // a NEGATIVE constant base with a CONTINUUM exponent: non-integer a.e. -> nan a.e.
    if a.is_const() && a.lo < 0.0 && !b.is_const() {
        r.nan = true;
        // H-029 (2026-08-05): same single-contained-integer honesty as the -inf-base
        // arm -- `(-2)^k` IS defined (finite) when a TIGHT exponent range encloses one
        // integer (the enclosure-of-a-ground-integer case). The exact point value,
        // null-supported: the class turns Mixed, ground folds refuse (the exact rule
        // path then finishes), witnesses fail conservatively. Wide/continuum ranges
        // keep the documented nan-a.e. convention.
        if b.has_fin && b.lo.ceil() == b.hi.floor() && b.lo.ceil().is_finite() {
            let k = b.lo.ceil();
            if k != 0.0 {
                let v = a.lo.powf(k);
                if v.is_finite() {
                    // libm point: bracket it (H-019); slice-supported only -> null
                    r.merge_pt(out_lo(v, ULP_LIBM));
                    r.merge_pt(out_hi(v, ULP_LIBM));
                    r.fin_null = true;
                } else {
                    // overflowed slice value: unbounded open edge, module convention
                    r.has_fin = true;
                    r.fin_null = true;
                    if v > 0.0 {
                        r.lo = 0.0;
                        r.hi = INF;
                        r.lo_open = false;
                        r.hi_open = true;
                    } else {
                        r.lo = -INF;
                        r.hi = 0.0;
                        r.lo_open = true;
                        r.hi_open = false;
                    }
                }
            }
        }
        // infinite exponent components: the magnitude step (|a| vs 1), same rows as
        // the dedicated pure-inf arms (H-028: these were dropped by the early return)
        if let Some((amin, amax)) = ar {
            if b.pinf {
                if amax > 1.0 {
                    r.pinf = true;
                }
                if amin < 1.0 {
                    r.merge_pt(0.0);
                }
            }
            if b.ninf {
                if amin < 1.0 {
                    r.pinf = true;
                }
                if amax > 1.0 {
                    r.merge_pt(0.0);
                }
            }
        }
        return r;
    }
    // BOTH sides constant: evaluate exactly. The general branch below cannot bound a^b over
    // intervals (pow is not monotone in either argument) and so reports the whole line -- but
    // with two constants there is nothing to approximate, and the coarse range poisons every
    // downstream op: `pow(-1,-1)` came out as (-inf, inf) instead of {-1}, so `sqrt` of it kept
    // a spurious FINITE part where the true value is nan. Reached only after the 0-base, 1-base,
    // inf-base and 0-exponent branches above, so an infinite result here is a genuine OVERFLOW:
    // finite-but-unrepresentable, an unbounded EDGE rather than an inf value (module contract).
    if a.is_const() && b.is_const() {
        let v = a.lo.powf(b.lo);
        let mut z = Vs::empty();
        if v.is_nan() {
            z.nan = true;
        } else if v.is_finite() {
            // powf is libm: bracket the computed point (H-019).
            z.merge_pt(out_lo(v, ULP_LIBM));
            z.merge_pt(out_hi(v, ULP_LIBM));
        } else {
            z.has_fin = true;
            z.lo = if v > 0.0 { 0.0 } else { -INF };
            z.hi = if v > 0.0 { INF } else { 0.0 };
            z.lo_open = z.lo.is_infinite();
            z.hi_open = z.hi.is_infinite();
        }
        return z;
    }
    // CONST INTEGER EXPONENT, finite-capable base: the exact interval power (2026-08-03
    // vocabulary completion). The retired unary pow2..pow5/inv evaluators bounded their
    // ranges by monotonicity; their binary replacement fell to the generic branch below,
    // which reports (-inf, inf) for every negative-capable base -- starving the analytic
    // witness (a sign-definite cell never excluded 0) and every downstream range proof.
    // (Deliberately NOT delegated to the legacy `u_pow_int` instrument arm -- see its
    // doc: this branch alone carries open-edge poles, fin_null, and negative exponents.)
    // The value hull of the endpoint powers is exact on monotone pieces, plus the parity
    // minimum 0 (k > 0, even) resp. the pole reach (k < 0, base interval containing 0)
    // as UNBOUNDED OPEN EDGES -- the module convention for poles (`inv` over [-64,64]
    // reads has_fin, pinf=false, hi=+inf). Over-approximation (the hull) is the sound
    // direction for every consumer: witnesses and containment proofs fail conservatively.
    // Attained base infinities map through sign/parity; NaN propagates; `x^0` and the
    // const-const exact branch returned above.
    if a.has_fin && b.is_const() && b.lo.is_finite() && b.lo.fract() == 0.0 && b.lo.abs() <= 1024.0
    {
        let k = b.lo as i32;
        let mut z = Vs::empty();
        z.nan = a.nan;
        z.fin_null = a.fin_null;
        if a.pinf {
            if k > 0 {
                z.pinf = true;
            } else {
                z.merge_pt(0.0);
            }
        }
        if a.ninf {
            if k > 0 {
                if k % 2 == 0 {
                    z.pinf = true;
                } else {
                    z.ninf = true;
                }
            } else {
                z.merge_pt(0.0);
            }
        }
        // powi compounds ~one rounding per multiply: step the computed hull extremes by an
        // exponent-scaled budget AFTER the min/max fold (each bound steps in its own
        // direction); an underflowed power (computed 0 from a nonzero base) bounds by the
        // signed minimum subnormal, exactly as in `b_mul` (H-019). The parity minimum 0
        // and the pole edges are exact and never widened.
        let ulps = 2 + k.unsigned_abs().min(64);
        let pw = |x: f64| {
            let p = x.powi(k);
            if p == 0.0 && x != 0.0 {
                let neg = x < 0.0 && k % 2 != 0;
                return if neg {
                    -f64::from_bits(1)
                } else {
                    f64::from_bits(1)
                };
            }
            p
        };
        let mut cands: Vec<(f64, bool)> = vec![
            (pw(a.lo), a.lo_open || pw(a.lo).is_infinite()),
            (pw(a.hi), a.hi_open || pw(a.hi).is_infinite()),
        ];
        if a.lo <= 0.0 && 0.0 <= a.hi {
            if k > 0 {
                cands.push((0.0, false)); // the parity minimum resp. the odd zero
            } else {
                cands.push((INF, true)); // pole reach: open edge by convention
                if k % 2 == 1 && a.lo < 0.0 {
                    cands.push((-INF, true));
                }
            }
        }
        let (mut lo, mut lo_open) = (INF, true);
        let (mut hi, mut hi_open) = (-INF, true);
        for (v, open) in cands {
            if v < lo || (v == lo && !open) {
                (lo, lo_open) = (v, open);
            }
            if v > hi || (v == hi && !open) {
                (hi, hi_open) = (v, open);
            }
        }
        if lo != 0.0 {
            lo = out_lo(lo, ulps);
        }
        if hi != 0.0 {
            hi = out_hi(hi, ulps);
        }
        z.has_fin = true;
        z.lo = lo;
        z.hi = hi;
        z.lo_open = lo_open;
        z.hi_open = hi_open;
        return z;
    }
    // numpy `1^y = 1` for EVERY y, nan included: when the base ENCLOSURE contains 1 but is
    // not the exact constant (handled above), the value 1 stays reachable on the null
    // {base == 1} slice. Without this, a nan-capable exponent collapsed the whole set to
    // nan for a base BRACKETING 1 (H-019: `sinh(asinh(1))` encloses [1-u, 1+u]; the true
    // value IS 1, and the exact inverse rule reduces it -- the fold must refuse, not
    // preempt). Null-supported: `cls` reads Mixed, `defined_pm` witnesses stay blocked.
    if b.nan && a.has_fin && a.lo <= 1.0 && 1.0 <= a.hi {
        r.merge_pt(1.0);
        r.fin_null = true;
    }
    if a.has_fin && b.has_fin {
        let b_int = b.is_const() && b.lo.fract() == 0.0;
        // `pow(negative, k)` is defined EXACTLY on the integers, so a negative base reaches finite
        // values whenever the exponent RANGE contains one -- not only when the exponent is a
        // constant that happens to be an integer.
        //
        // Only `b_int` was tested before, which is a point test, so `pow(x0 in [-19,-8], x1 in
        // [-15,15])` reported nan-ONLY even though pow(-10,3) = -1000 and pow(-10,-2) = 0.01. That
        // is an UNDER-report, and under-reporting is the one direction that breaks the gate's
        // safety proof: `!sv.defined()` ("no defined value anywhere on this box") would fire on a
        // source that IS defined, yielding a FALSE witness and rejecting a SOUND rule.
        //
        // Invisible to the variable-free acceptance tests by construction: they bind `<constant>`
        // to POINTS, so the exponent is never an interval. It took the >=2-variable box path to
        // reach it.
        let b_has_int = b.lo.ceil() <= b.hi.floor();
        if a.lo < 0.0 && !b_int {
            r.nan = true;
        }
        if a.hi > 0.0 || b_int || (a.lo < 0.0 && b_has_int) {
            r.has_fin = true;
            r.lo = if a.lo < 0.0 { -INF } else { 0.0 };
            r.hi = INF;
            r.lo_open = r.lo.is_infinite();
            r.hi_open = true;
            // POSITIVE-MEASURE annotation: fin
            // from the integer slice alone is supported on {x : exponent(x) in Z} -- null for a
            // non-constant exponent (the constant case IS the b_int disjunct). has_fin keeps
            // reachability semantics for `reaches_all_of`; the domain gate's witness reads
            // `defined_pm()` and can now fire on nan-a.e. sources -- the ">=2-var negative-base
            // pow family" an earlier fix had silenced. Fin from the other disjuncts inherits
            // the operands' own support flags.
            r.fin_null = a.fin_null || b.fin_null || !(a.hi > 0.0 || b_int);
        }
    }
    // Infinite exponent components on the GENERAL path (mixed with finite exponent
    // values, or both infinities at once -- the dedicated pure-inf arms above require
    // `!b.has_fin` and a single infinity). H-028 (2026-08-05, fuzz row 540516): this
    // fallback read `r.pinf = true` regardless of base magnitude, so
    // `pow(exp(np.e), -inf/x2)` -- exponent attaining BOTH infinities -- classified
    // PosInf-only where the true value set is {0 on one half-line, +inf on the other};
    // its inverse then classified FINITE and the zero-absorption licence folded
    // `0 / pow(exp(np.e), -inf/x2)` to 0 where the true value is nan on a half-line.
    // The step function of |a| vs 1, per component, exactly as the dedicated arms.
    if b.pinf || b.ninf {
        if let Some((amin, amax)) = ar {
            if b.pinf {
                if amax > 1.0 {
                    r.pinf = true;
                }
                if amin < 1.0 {
                    r.merge_pt(0.0);
                }
            }
            if b.ninf {
                if amin < 1.0 {
                    r.pinf = true;
                }
                if amax > 1.0 {
                    r.merge_pt(0.0);
                }
            }
            if a.is_const() && a.lo.abs() == 1.0 {
                r.merge_pt(1.0);
            }
        }
        // attained base infinities: (+-inf)^(+inf) = +inf, (+-inf)^(-inf) = 0
        if a.pinf || a.ninf {
            if b.pinf {
                r.pinf = true;
            }
            if b.ninf {
                r.merge_pt(0.0);
            }
        }
    }
    r
}

fn apply_binary(op: &str, a: &Vs, b: &Vs) -> Option<Vs> {
    Some(match op {
        "+" => b_add(a, b),
        "-" => b_add(a, &u_neg(b)),
        "*" => b_mul(a, b),
        "/" => b_mul(a, &u_inv(b)),
        "pow" => b_pow(a, b),
        // `rootn(x, n)`: IEEE-754 rootn, honest for EVERY integer index:
        // odd = the signed root, even = the principal root, 1 = identity, negative =
        // reciprocal. Sound only when the index is a KNOWN nonzero-integer point (a
        // literal in the serialized form); any uncertainty in the index fails closed like
        // an unknown operator.
        "rootn" => {
            let point = b.has_fin
                && !b.nan
                && !b.pinf
                && !b.ninf
                && b.lo == b.hi
                && !b.lo_open
                && !b.hi_open;
            if !point || b.lo.fract() != 0.0 || b.lo == 0.0 || !b.lo.is_finite() {
                return None;
            }
            let k = b.lo.abs();
            let root = if k == 1.0 {
                *a
            } else if k % 2.0 == 1.0 {
                u_odd_root(a, k)
            } else {
                u_even_root(a, k)
            };
            if b.lo < 0.0 {
                u_inv(&root)
            } else {
                root
            }
        }
        _ => return None,
    })
}

/// Index of a variable leaf (`_0`/`x0` -> 0), or `None` if `t` is not a variable.
fn var_index(t: &str) -> Option<usize> {
    let rest = t.strip_prefix('_').or_else(|| t.strip_prefix('x'))?;
    if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

/// Length a `doms` array needs = 1 + the largest variable index seen (floored at 1, so a
/// variable-free expression still has `doms[0]` for its free `<constant>`s).
fn n_var_slots(exprs: &[&[String]]) -> usize {
    let mut n = 0usize;
    for e in exprs {
        for tok in e.iter() {
            if let Some(i) = var_index(tok) {
                n = n.max(i + 1);
            }
        }
    }
    n.max(1)
}

/// The DISTINCT variable indices an expression family actually references, ascending.
///
/// The box must have one dimension per distinct variable, NOT one per index slot. Sources are built
/// from the dummy variables `x0..x3`, so indices are SPARSE: `exp log x3` references ONE variable
/// but its largest index is 3. Sizing the box by `max index + 1` gives it four dimensions, three of
/// them never referenced -- and an unreferenced dimension is never narrowed, so its full width
/// multiplies into every witness volume (`exp log x3` reported 1.34e8 where `exp log x0` reported
/// the correct 64) and the depth budget is spent bisecting axes the expression cannot see.
fn distinct_vars(exprs: &[&[String]]) -> Vec<usize> {
    let mut seen: Vec<usize> = Vec::new();
    for e in exprs {
        for tok in e.iter() {
            if let Some(i) = var_index(tok) {
                if !seen.contains(&i) {
                    seen.push(i);
                }
            }
        }
    }
    seen.sort_unstable();
    seen
}

/// Number of gate calls whose interesting region fell OUTSIDE the box horizon. Since the
/// undecidable-horizon fix the gate FAILS CLOSED on these (`gate_horizon` returns
/// `decidable = false` and the caller must not accept), so each count is lost RECALL, not a
/// soundness hole -- counted rather than assumed away, recorded in the mine's provenance
/// sidecar (`soundness.interval_undecided.horizon`), and exposed via
/// `interval_horizon_misses()`.
pub static HORIZON_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Largest finite magnitude any VARIABLE-FREE subtree folds to, plus any bound constant.
///
/// The box horizon must reach past the source's own constants or the gate is blind by
/// construction: `pow4(np.pi) = 97.409 > 64` put the NaN boundary of
/// `exp log - pow4 np.pi x0` OUTSIDE the searched region, so the gate reported 0.0 and the rule was
/// accepted 4/4 seeds -- while the control `pow3(np.pi) = 31.006 < 64` was correctly rejected. Same
/// shape; only the constant's magnitude differed.
fn max_const_magnitude(exprs: &[&[String]], params: &[&[f64]], ops: &Operators) -> f64 {
    fn walk(t: &[String], i: usize, ops: &Operators, best: &mut f64) -> Option<usize> {
        if i >= t.len() {
            return None;
        }
        let tok = t[i].as_str();
        let ar = if var_index(tok).is_some() || tok == "<constant>" {
            0
        } else {
            ops.arity_of(tok).unwrap_or(0)
        };
        let mut j = i + 1;
        for _ in 0..ar {
            j = walk(t, j, ops, best)?;
        }
        let sub = &t[i..j];
        if !sub
            .iter()
            .any(|x| var_index(x).is_some() || x == "<constant>")
        {
            if let Some(s) = crate::numeric::evaluate_constant_subtree(sub, ops) {
                if let Ok(v) = s.parse::<f64>() {
                    if v.is_finite() && v.abs() > *best {
                        *best = v.abs();
                    }
                }
            }
        }
        Some(j)
    }
    let mut best = 0.0f64;
    for (k, e) in exprs.iter().enumerate() {
        // Fold with the BOUND constants substituted, so the horizon sees f(C) and not |C|:
        // `pow4(<constant>)` at C = pi must contribute 97.409, not 3.14 (the horizon must be
        // computed from the folded value, not the raw constant magnitude). A `<constant>` with
        // no bound param (the free-constant probes) stays a skip.
        let p = params.get(k).copied().unwrap_or(&[]);
        if !p.is_empty() && e.iter().any(|t| t == "<constant>") {
            let mut ci = 0usize;
            let subst: Vec<String> = e
                .iter()
                .map(|t| {
                    if t == "<constant>" {
                        let s = p
                            .get(ci)
                            .map(|c| format!("{c}"))
                            .unwrap_or_else(|| t.clone());
                        ci += 1;
                        s
                    } else {
                        t.clone()
                    }
                })
                .collect();
            let _ = walk(&subst, 0, ops, &mut best);
        } else {
            let _ = walk(e, 0, ops, &mut best);
        }
    }
    for p in params {
        for &c in p.iter() {
            if c.is_finite() && c.abs() > best {
                best = c.abs();
            }
        }
    }
    best
}

/// The box horizon and the depth that holds RESOLUTION fixed at the baseline `128 / 2^14`.
///
/// Widening R without deepening would silently COARSEN the gate everywhere -- the fix for one hole
/// becoming a regression for every other rule. Returns `(R, depth_per_dim, decidable)`; `decidable`
/// is false when the needed horizon exceeds `R_MAX`, on which every caller FAILS CLOSED
/// (refuse/None) and `gate_horizon` counts the miss (`HORIZON_MISSES`) -- the F0 lesson;
/// the pre-F0 code treated exactly this as a fail-open.
///
/// R_MAX is a real limit, not a formality: constant-foldable subtrees can reach magnitudes
/// ~1e271, and a box that wide would need depth ~909 to hold the resolution -- 2^909 cells. So
/// the horizon is CAPPED and the remainder is counted. The gate is a dyadic witness search over
/// a bounded box, and past the cap it does not know: it says so.
fn horizon(exprs: &[&[String]], params: &[&[f64]], ops: &Operators) -> (f64, u32, bool) {
    const R_MIN: f64 = 64.0;
    const R_MAX: f64 = 1.0e6;
    const RES: f64 = 128.0 / 16384.0; // the baseline cell width, 0.0078125
    let need = (2.0 * max_const_magnitude(exprs, params, ops)).max(R_MIN);
    let r = need.min(R_MAX);
    let depth = ((2.0 * r / RES).log2().ceil().max(14.0)) as u32;
    (r, depth, need <= R_MAX)
}

/// Materialise `doms` (indexed BY VARIABLE INDEX) from the dense box `bx` (indexed by dimension).
///
/// A VARIABLE-FREE expression still gets dimension 0 bound to the box: that is the axis its free
/// `<constant>`s are subdivided along, which is what the pre-box engine did and what the FFI probe
/// depends on. Slots no variable references keep the full range and are never read.
fn doms_from_box(bx: &[(f64, f64)], used: &[usize], width: usize, full: (f64, f64)) -> Vec<Vs> {
    let mut doms = vec![Vs::interval(full.0, full.1, false, false); width];
    if used.is_empty() {
        doms[0] = Vs::interval(bx[0].0, bx[0].1, false, false);
        return doms;
    }
    for (d, &i) in used.iter().enumerate() {
        doms[i] = Vs::interval(bx[d].0, bx[d].1, false, false);
    }
    doms
}

/// `params` BINDS `<constant>` leaves to concrete values (k-th `<constant>` in prefix order ->
/// `params[k]`, exactly the tape's slot numbering, `eval.rs:161`).
///
/// `doms` is ONE INTERVAL PER VARIABLE: variable `k` takes `doms[k]`. This is the whole point of
/// the box subdivision -- binding every variable to a single shared interval explores only the cube
/// family `{[lo,hi]^n}`, on which `x0-x1` is symmetric about 0 and `log` is never all-nan, so no
/// witness can ever form. See `domain_extension_p`.
///
/// An unbound `<constant>` (params exhausted / empty) falls back to `doms[0]` = FREE. Keeping free
/// constants on dimension 0 -- rather than giving them their own -- makes the 0- and 1-variable
/// cases bit-for-bit identical to the pre-box engine, which is what the callers that pass no params
/// (the const-free FFI probe) rely on.
fn leaf_vs(t: &str, doms: &[Vs], params: &[f64], k: &mut usize) -> Option<Vs> {
    if let Some(i) = var_index(t) {
        return Some(*doms.get(i).unwrap_or(&doms[0]));
    }
    let dom = &doms[0];
    if t == "<constant>" {
        let v = params.get(*k).map(|c| Vs::constant(*c)).unwrap_or(*dom);
        *k += 1;
        return Some(v);
    }
    // TRANSCENDENTAL ATOMS ARE ENCLOSURES, NOT POINTS (H-018, 2026-08-03): rendering
    // `np.pi`/`np.e` as their exact f64 rationals let this layer certify facts about the
    // RENDERING that are false of the atom -- `sin(np.pi)` evaluated to the point
    // {1.22e-16}, a value interval excluding 0, and `analytic_nonzero_witness` minted a
    // NONZERO-a.e. certificate for an expression whose ratified exact value IS 0 (the
    // engine's own mined rule folds `sin pi -> 0`). That false certificate licensed the
    // pow-over-mul distribution `inv(sin(pi) * u) -> inv(sin(pi)) * inv(u)`, splitting a
    // true zero factor: +inf identically (one-zero absorption, s9.2) became +-inf by the
    // sign of u -- a positive-measure INF-CHANGE (fuzz row 172346). An outward-rounded
    // bracket is the honest rigorous enclosure of the real pi/e: every exact-zero identity
    // ground (`sin pi`, `log e - 1`, ...) now evaluates to an interval CONTAINING 0, so
    // every nonzero/zero-freeness certificate fails closed and the certified exact rules
    // fold the zero first. Rational and decimal literals stay exact points: they denote
    // rationals in the value model, and their exact-zero combinations are algebraic --
    // folded exactly by the structural layer, never asked of this one.
    // (`f64::next_up`/`next_down` need Rust 1.86; the crate's MSRV floor is 1.83 (pyo3),
    // so bracket by ULP-stepping the bit pattern -- exact for positive normals like pi/e.)
    fn ulp_bracket(c: f64) -> Vs {
        debug_assert!(c.is_finite() && c > 0.0);
        let lo = f64::from_bits(c.to_bits() - 1);
        let hi = f64::from_bits(c.to_bits() + 1);
        Vs::interval(lo, hi, false, false)
    }
    if t == "np.pi" {
        return Some(ulp_bracket(std::f64::consts::PI));
    }
    if t == "np.e" {
        return Some(ulp_bracket(std::f64::consts::E));
    }
    // H-019 amendment to the H-018 literal doctrine: a NON-integer literal's f64 parse is
    // generally NOT the denoted rational (`1/3`, `0.515`) -- a point enclosure at the
    // rounded value is the same rendering-vs-atom lie as the old point pi, one ulp wide.
    // Integer-valued literals below 2^53 are exact and stay points (`0`, `1`, `-7`,
    // `2.0`); above 2^53 a literal stays a point IFF the DENOTED rational certifies as
    // exactly the f64 (H-045: the old blanket bracket was NOT "fail-closed either way" --
    // a 1-ulp bracket at 1e19 spans ~4097 integers, so the `(-inf)^k` arms lost the
    // H-029 single-integer honesty gate and the continuum convention ASSERTED Nan for a
    // ground whose true value is +inf; the ground-classification fold shipped it).
    // (Exactly-representable non-integers like `0.5` are still bracketed: the one-ulp
    // cost to a certificate boundary is nil there -- no integer, so the pow arms stay
    // honest -- and the narrower change keeps the certified behavior envelope.)
    crate::numeric::leaf_value(t).map(|v| {
        // H-052 (RULED 2026-08-06, option A -- exact-real denotation): a leaf's enclosure
        // must bracket what the spelling DENOTES, never merely its f64 image. `1e309`
        // denotes the finite real 10^309 and `1e-400` a positive real; reading them as the
        // POINTS {+inf} and {0} asserts a value the literal does not have, and was the one
        // place an image semantics leaked into a system whose doctrine is exact denotation
        // everywhere else. It is also where the engine CONTRADICTED itself: `1e308 * 2`
        // refuses (the exact product is unrepresentable) while `1e309 * 2` folded to +inf --
        // the same question, answered oppositely by the accident of whether the overflow
        // sits in the leaf or in the result.
        //
        // The honest enclosures are half-open: the infinity is a LIMIT, not an attained
        // value (the convention `reals()` already uses), and the zero is excluded. Every
        // consequence then falls out of machinery that already exists -- the class fold sees
        // no certified +-inf and refuses, `neg 1e309` keeps its identity with no guard of
        // its own, and the judge needs no boundary arm because the engine stops overclaiming.
        // Nothing here introduces a numeric representation: these are f64 endpoints in a
        // type that already distinguishes attained endpoints from limits.
        //
        // SPELLED non-finites keep their point: `float("inf")` IS infinite, and all of its
        // folds are untouched. In-range literals never reach these arms.
        if !v.is_finite() || v == 0.0 {
            match denotation_at_range_boundary(t, v) {
                // Beyond f64 RANGE: |denoted| > f64::MAX (the round-to-nearest overflow
                // threshold sits above f64::MAX, so the closed lower bound is sound and
                // merely loose).
                Some(Boundary::Overflow) if v > 0.0 => {
                    return Vs::interval(f64::MAX, INF, false, true)
                }
                Some(Boundary::Overflow) => return Vs::interval(-INF, -f64::MAX, true, false),
                // Below the subnormal floor: 0 < |denoted| <= the smallest positive
                // subnormal (again sound-and-loose: underflow to zero requires strictly
                // less than half of it).
                Some(Boundary::Underflow) if v.is_sign_negative() => {
                    return Vs::interval(-f64::from_bits(1), 0.0, false, true)
                }
                Some(Boundary::Underflow) => {
                    return Vs::interval(0.0, f64::from_bits(1), true, false)
                }
                // A `p/q` whose COMPONENTS exceed f64 -- `10^400/10^300` images as +inf but
                // denotes 10^100, well inside the range, so neither the point nor a boundary
                // bracket is sound. Its denotation is a finite real and nothing more is
                // known: fail closed to every finite real, which licenses no fold at all.
                // (Beyond-i128 components; `Rat` refuses them, so these survive only as
                // opaque leaves. Registered on H-045 as the beyond-i128 residual.)
                Some(Boundary::UnknownFinite) => return Vs::reals(),
                None => {}
            }
        }
        if !v.is_finite()
            || v == 0.0
            || (v.fract() == 0.0 && (v.abs() < 9.007199254740992e15 || exact_integer_literal(t, v)))
        {
            Vs::constant(v)
        } else {
            Vs::interval(next_down(v), next_up(v), false, false)
        }
    })
}

/// H-052: how a leaf's DENOTATION relates to its degenerate f64 image.
enum Boundary {
    /// A finite real of magnitude beyond `f64::MAX` (image `+-inf`).
    Overflow,
    /// A nonzero real below the subnormal floor (image `+-0`).
    Underflow,
    /// A finite real whose magnitude the spelling does not pin down.
    UnknownFinite,
}

/// H-052: classify a leaf whose f64 image is non-finite or zero. `None` when the image IS
/// the denotation -- the three spelled `float("...")` tokens, and every spelling of zero --
/// in which case the caller's point enclosure is correct and stays.
fn denotation_at_range_boundary(t: &str, v: f64) -> Option<Boundary> {
    let t = t
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(t);
    // The only tokens that SPELL a non-finite value. Everything else in the leaf grammar is
    // a decimal or an exact fraction, and denotes a finite real however large it is written.
    if matches!(t, "float(\"inf\")" | "float(\"-inf\")" | "float(\"nan\")") {
        return None;
    }
    if let Some((p, q)) = t.split_once('/') {
        // A zero denominator denotes nothing; leave the malformed token on the old path
        // rather than invent a semantics for it (`Rat` refuses it at parse).
        if q.chars().all(|c| c == '0') {
            return None;
        }
        if p.chars().all(|c| c == '0' || c == '-') {
            return None; // an exact zero, spelled as a fraction
        }
        return Some(Boundary::UnknownFinite);
    }
    // A decimal/exponent spelling: zero iff its significand is. `1e-400` has significand
    // `1` and denotes a positive real, so it must not share the point {0}.
    let significand = t.split(['e', 'E']).next().unwrap_or(t);
    let mut digits = significand
        .chars()
        .filter(|c| c.is_ascii_digit())
        .peekable();
    if digits.peek().is_some() && digits.all(|c| c == '0') {
        return None; // every digit is 0: the token denotes exactly zero
    }
    if v.is_nan() {
        return None; // not producible by the decimal grammar; no denotation to bracket
    }
    Some(if v.is_infinite() {
        Boundary::Overflow
    } else {
        Boundary::Underflow
    })
}

/// H-045: certify that an integer-valued f64 leaf `v` is EXACTLY the literal's denoted
/// rational, via the same grammar `leaf_value` reads (paren-stripped decimal, `p/q`).
/// Beyond-i128 denotations cannot be certified and keep the bracket (their `(-inf)^k`
/// class residual is registered open on H-045).
fn exact_integer_literal(t: &str, v: f64) -> bool {
    if v.abs() >= 1.7014118346046923e38 {
        return false; // 2^127: the denoted integer cannot fit i128, no certificate
    }
    let t = t
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(t);
    let denoted = crate::ac::rat::Rat::parse_decimal(t).or_else(|| {
        let (p, q) = t.split_once('/')?;
        if p.chars().all(|c| c.is_ascii_digit() || c == '-')
            && !q.is_empty()
            && q.chars().all(|c| c.is_ascii_digit())
        {
            crate::ac::rat::Rat::new(p.parse::<i128>().ok()?, q.parse::<i128>().ok()?)
        } else {
            None
        }
    });
    // `v as i128` is exact: v is an integer-valued f64 with |v| < 2^127.
    denoted.is_some_and(|r| r.is_integer() && r.num() == v as i128)
}

/// Value-set of a prefix expression over `dom` = the domain assigned to every VARIABLE leaf
/// (`Vs::reals()` = all finite reals; a sub-interval is what the domain-gate subdivision passes
/// in). `<constant>` leaves bind to `params` when supplied, else share `dom` as free parameters.
///
/// Binding matters: with `<constant>` free it takes the SAME sub-interval as the variable, which
/// silently conflates the two under subdivision (`pow(<constant>, x)` would be evaluated only on
/// the diagonal C == x). The domain gate therefore always passes the FITTED constants.
pub fn value_set_p(tokens: &[String], ops: &Operators, doms: &[Vs], params: &[f64]) -> Option<Vs> {
    fn go(
        t: &[String],
        i: usize,
        ops: &Operators,
        doms: &[Vs],
        p: &[f64],
        k: &mut usize,
    ) -> Option<(Vs, usize)> {
        if i >= t.len() {
            return None;
        }
        let tok = t[i].as_str();
        if let Some(v) = leaf_vs(tok, doms, p, k) {
            return Some((v, i + 1));
        }
        if ops.is_operator(tok) {
            let ar = ops.arity_of(tok)?;
            if ar == 1 {
                let (v, j) = go(t, i + 1, ops, doms, p, k)?;
                return Some((apply_unary(tok, &v)?, j));
            }
            if ar == 2 {
                let (a, j) = go(t, i + 1, ops, doms, p, k)?;
                let (b, m) = go(t, j, ops, doms, p, k)?;
                return Some((apply_binary(tok, &a, &b)?, m));
            }
        }
        None
    }
    if doms.is_empty() {
        return None;
    }
    let mut k = 0usize;
    go(tokens, 0, ops, doms, params, &mut k).map(|(v, _)| v)
}

/// `value_set_p` with every variable AND every `<constant>` sharing the single interval `dom`.
///
/// Correct for `value_class`, which asks for the union over the WHOLE space: there every variable
/// takes `Vs::reals()` anyway, so per-variable boxes would change nothing. Do NOT use it for a
/// subdivided cell -- that is exactly the conflation `domain_extension_p` exists to avoid.
pub fn value_set(tokens: &[String], ops: &Operators, dom: &Vs) -> Option<Vs> {
    let doms = vec![*dom; n_var_slots(&[tokens])];
    value_set_p(tokens, ops, &doms, &[])
}

pub fn value_class(tokens: &[String], ops: &Operators) -> Option<Class> {
    let v = value_set(tokens, ops, &Vs::reals())?;
    // R3 NULL-KILL at the measure-aware consumer (H-019, 2026-08-04): a finite part
    // supported ONLY on a null x-set (`fin_null` -- e.g. `clip`'s kept boundary graze,
    // or the negative-base pow integer slice) may be dropped for the CLASSIFICATION of
    // a variable/`<constant>`-parameterized family: collapsing kills only a null set,
    // exactly the licence the miner's nan rules always used (`atanh(cosh C) -> nan` is
    // defined nowhere and reaches its pole only at the null set C = 0). A GROUND has no
    // measure space -- its "null-supported" point IS the value (`acosh(cos(0))` = 0) --
    // so the drop applies only when the expression actually carries a continuum.
    if v.has_fin
        && v.fin_null
        && tokens
            .iter()
            .any(|t| var_index(t).is_some() || t == "<constant>")
    {
        let mut w = v;
        w.has_fin = false;
        return Some(w.cls());
    }
    Some(v.cls())
}

/// Does `target` DEFINE a value on a POSITIVE-MEASURE region where `source` is NaN?
///
/// That is the "grossly domain dependent" rewrite to reject: the reverse round-trips
/// (`cos(acos x) -> x`, `exp(log x) -> x`) agree on the inner function's range but INVENT a
/// function off it. A MEASURE-ZERO hole (`x/x -> 1` at 0) is refined away and can never
/// accumulate width, so it is correctly allowed. Returns the witness width (0.0 = no extension).
///
/// `src_params` / `tgt_params` bind the two sides' `<constant>` leaves (see `value_set_p`). Both
/// MUST be the concrete values of ONE instance -- the source's drawn constants and the constants
/// the fit CHOSE for them. A free constant cannot answer this question: the rule-level statement
/// is "for every source constant there EXISTS a target constant that agrees AND preserves the
/// domain", and a single interval evaluation over a free C collapses the exists/forall (the union
/// over C contains both NaN and finite values almost everywhere, so the gate could never fire).
/// The fit already picked the witness; this checks the domain AT it.
pub fn domain_extension_p(
    source: &[String],
    src_params: &[f64],
    target: &[String],
    tgt_params: &[f64],
    ops: &Operators,
) -> Option<f64> {
    let (r, depth_per_dim, decidable) = gate_horizon(source, src_params, target, tgt_params, ops);
    if !decidable {
        return None; // horizon past R_MAX: the question is not answerable in any affordable box
    }
    domain_extension_p_at(
        source,
        src_params,
        target,
        tgt_params,
        ops,
        r,
        depth_per_dim,
    )
}

/// Bounds the n-dimensional subdivision searches. Unreachable at n=1 with the DEFAULT horizon (a
/// 14-deep binary tree visits <= 2^15 nodes); constant-derived horizons deepen the tree, so
/// exhaustion is possible. FAIL-CLOSED: an exhausted search that found no witness returns
/// `None` (undecided) and the gate REJECTS -- it never reads as "no extension". Raise
/// per-stratum via `SIMPLIPY_IVL_NODE_BUDGET` when the counters flag systematic exhaustion.
const DEFAULT_NODES_BUDGET: u32 = 400_000;

fn max_nodes_budget() -> u32 {
    static B: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("SIMPLIPY_IVL_NODE_BUDGET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_NODES_BUDGET)
    })
}

/// Number of subdivision searches that exhausted the node budget WITHOUT finding a witness --
/// i.e. UNDECIDED verdicts, each of which the gate rejects (fail-closed). A search that found
/// its witness before exhaustion is decided and not counted. Read via
/// `interval_node_budget_misses()`.
pub static NODE_BUDGET_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Number of subdivision searches that visited at least one UNANALYZABLE box (`value_set_p`
/// -> `None`: an operator arm that fails closed, e.g. `rootn` with an expression-position
/// index) and found no witness. Such a search has NOT proven "no extension" / "no support":
/// it returns `None` (undecided) and the gate fails closed. Before 2026-07-30 (F0) the
/// abandoned box silently read as clean and the completed search as a DECIDED verdict --
/// operator-level fail-closed laundered into gate-level fail-open, which is exactly how the
/// five NaN-a.e. `rootn <lit> tanh/cosh ?0` rules of the 2026-07-29 acj-4-3 mine shipped.
/// Read via `interval_unanalyzable_misses()`.
pub static UNANALYZABLE_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[allow(clippy::too_many_arguments)]
fn rec_ext(
    src: &[String],
    sp: &[f64],
    tgt: &[String],
    tp: &[f64],
    ops: &Operators,
    r: f64,
    bx: &mut [(f64, f64)],
    used: &[usize],
    width: usize,
    d: u32,
    depth_max: u32,
    best: &mut f64,
    budget: &mut u32,
    undecided: &mut bool,
) {
    if *budget == 0 {
        return;
    }
    *budget -= 1;
    let vol: f64 = bx.iter().map(|(lo, hi)| hi - lo).product();
    if vol <= *best {
        return;
    } // cannot beat the incumbent: prune
    let doms = doms_from_box(bx, used, width, (-r, r));
    let sv = match value_set_p(src, ops, &doms, sp) {
        Some(v) => v,
        None => {
            // UNANALYZABLE source box: `value_set_p` failed closed (e.g. rootn with an
            // expression-position index). An extension witness may hide in this box, so
            // "nothing to prove here" is not available -- the SEARCH degrades to undecided.
            *undecided = true;
            return;
        }
    };
    if !sv.nan {
        return;
    } // no nan anywhere here: nothing to prove
    let tv = match value_set_p(tgt, ops, &doms, tp) {
        Some(v) => v,
        None => {
            // UNANALYZABLE target box: a witness needs the target PROVEN defined, and
            // unprovable is not the same as absent -- undecided, same as the source arm.
            *undecided = true;
            return;
        }
    };
    if !tv.defined() {
        return;
    } // target nan across the whole box: no extension
    if !sv.defined_pm() && !tv.nan {
        // Source defined on NO positive-measure subset of this box (nan a.e. -- a null integer
        // slice does not count, per `fin_null`), target defined across ALL of it: a
        // positive-measure witness. Report its VOLUME (== width at n=1, so the existing
        // expectations stand). `defined_pm` instead of `defined` is what un-silences the
        // >=2-variable negative-base pow family without giving up reachability's point values.
        if vol > *best {
            *best = vol;
        }
        return;
    }
    // EITHER side is mixed here, so this box may still CONTAIN a witness (`acos(exp(log x))`
    // -> `acos x` is all-nan on [-64,0] while the target is mixed -- defined only on [-1,0)).
    // Refining on the source alone would stop at the first all-nan cell and miss it.
    if d >= depth_max {
        return;
    }
    // Split the WIDEST dimension. Bisecting one axis per level (not all n) keeps the branching
    // factor at 2 and lets the pruning above kill whole subtrees.
    let i = (0..bx.len())
        .max_by(|&a, &b| {
            (bx[a].1 - bx[a].0)
                .partial_cmp(&(bx[b].1 - bx[b].0))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    let (lo, hi) = bx[i];
    let mid = 0.5 * (lo + hi);
    bx[i] = (lo, mid);
    rec_ext(
        src,
        sp,
        tgt,
        tp,
        ops,
        r,
        bx,
        used,
        width,
        d + 1,
        depth_max,
        best,
        budget,
        undecided,
    );
    bx[i] = (mid, hi);
    rec_ext(
        src,
        sp,
        tgt,
        tp,
        ops,
        r,
        bx,
        used,
        width,
        d + 1,
        depth_max,
        best,
        budget,
        undecided,
    );
    bx[i] = (lo, hi); // restore for the caller's next split
}

/// The ONE horizon a gate decision runs at, computed over BOTH sides with BOTH param vectors.
///
/// Deadness and the extension witness must be judged at the SAME horizon: `horizon()` over
/// `[source, target]` in the extension call but over `[expr]` alone in `defined_measure_p`
/// lets a source read "dead" off a smaller box than the harm it exempts -- a fail-open in the
/// anti-conservative direction. The caller computes this once and passes it to
/// `domain_extension_p_at` and both `defined_measure_p_at` calls. A horizon miss is counted
/// HERE, once per gate decision. The `decidable` flag is RETURNED: an undecidable horizon means
/// the interesting region lies outside any box we can afford to resolve, and the gate must fail
/// CLOSED on it -- proceeding with the capped box and reading "no extension inside it" as an
/// accept would be a silent-accept channel.
pub fn gate_horizon(
    source: &[String],
    src_params: &[f64],
    target: &[String],
    tgt_params: &[f64],
    ops: &Operators,
) -> (f64, u32, bool) {
    let (r, depth_per_dim, decidable) = horizon(&[source, target], &[src_params, tgt_params], ops);
    if !decidable {
        // We do not know, and the caller must not accept. Counted, not hidden.
        HORIZON_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    (r, depth_per_dim, decidable)
}

/// `domain_extension_p` at a CALLER-SUPPLIED horizon (see `gate_horizon`). Does not count horizon
/// misses. Returns `Some(width)` only when the verdict is DECIDED: either a positive-measure
/// witness was found (width > 0) or the subdivision COMPLETED without one (width == 0). An
/// exhausted budget with no witness is `None` -- undecided, counted, and callers fail closed.
/// So is a search that visited an UNANALYZABLE box (`value_set_p` fail-closed) and found no
/// witness elsewhere (F0): a completion that skipped boxes it could not read has not earned
/// the decided "no extension".
pub fn domain_extension_p_at(
    source: &[String],
    src_params: &[f64],
    target: &[String],
    tgt_params: &[f64],
    ops: &Operators,
    r: f64,
    depth_per_dim: u32,
) -> Option<f64> {
    let used = distinct_vars(&[source, target]);
    let width = n_var_slots(&[source, target]);
    let n = used.len().max(1); // >=1: free `<constant>`s live on doms[0]
    let mut bx = vec![(-r, r); n];
    let mut best = 0.0f64;
    let mut budget = max_nodes_budget();
    let mut undecided = false;
    rec_ext(
        source,
        src_params,
        target,
        tgt_params,
        ops,
        r,
        &mut bx,
        &used,
        width,
        0,
        depth_per_dim * n as u32,
        &mut best,
        &mut budget,
        &mut undecided,
    );
    if budget == 0 && best == 0.0 {
        // Gave up on node count with no witness: "no extension" is a claim this search cannot
        // back. UNDECIDED -- callers fail closed.
        NODE_BUDGET_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    if undecided && best == 0.0 {
        // At least one box was UNANALYZABLE and no witness was found elsewhere: "no
        // extension" cannot be backed either. UNDECIDED -- callers fail closed. (A search
        // that DID find its witness stays decided: the gate rejects either way, and the
        // proven width is the more informative verdict.)
        UNANALYZABLE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    Some(best)
}

/// `domain_extension_p` with both sides' constants FREE. Exact only for const-free rules; kept for
/// the FFI probe and the const-free tests. `None` = undecided (fail-closed).
pub fn domain_extension(source: &[String], target: &[String], ops: &Operators) -> Option<f64> {
    domain_extension_p(source, &[], target, &[], ops)
}

/// The `!`-sort MATCH-TIME certificate: is `tokens` DEFINED AND FINITE almost everywhere over
/// its variables (and free `<constant>`s, which range over finite reals)?
///
/// The third wildcard sort: `!N` binds any subtree whose pushforward puts NO mass on
/// {nan, +-inf} -- exactly what lets `- !0 !0 -> 0` fire on `exp(x) - exp(x)`
/// while never binding `log(x)` (nan on half the line) or `pow(x, inf)` (a.e. infinite).
///
/// Subdivision over the standalone horizon box, budgeted for the HOT PATH (this runs inside
/// pattern matching): a cell that is entirely finite is CLEAN and prunes; a cell with no
/// positive-measure finite part (per `fin_null`) is an a.e.-nonfinite WITNESS and refutes; a
/// mixed cell subdivides. Every undecided path -- horizon past R_MAX, budget exhausted, depth
/// exhausted on a mixed cell, unevaluable cell -- returns FALSE (fail-closed: an uncertified
/// subtree just doesn't bind; no soundness is ever staked on an unfinished search).
///
/// Two sound under-approximations, unioned:
///   1. the STRUCTURAL path (`nonfinite_null`, below): the only path that can certify
///      pole-bearing trees (`1/x`, `tan x`, `x/(x1 - cos x1)`) -- a pole cell never
///      resolves under subdivision, but the null measure of its x-support is a
///      structural/analytic fact the certificate section proves directly;
///   2. the SUBDIVISION path: bounded-clean cells over the horizon box, which certifies
///      range-restricted compositions the structural tables refuse.
///
/// Exp/sinh/polynomial compositions certify at depth 0-2. Like every interval verdict,
/// the claim is scoped to the horizon box.
pub fn finite_ae(tokens: &[String], ops: &Operators) -> bool {
    nonfinite_null(tokens, ops) || finite_ae_subdivision(tokens, ops)
}

/// The subdivision path of `finite_ae` (the pre-certificate-algebra body, unchanged).
fn finite_ae_subdivision(tokens: &[String], ops: &Operators) -> bool {
    const BANG_CERT_BUDGET: u32 = 4_000;
    #[allow(clippy::too_many_arguments)]
    fn rec(
        e: &[String],
        ops: &Operators,
        r: f64,
        bx: &mut [(f64, f64)],
        used: &[usize],
        width: usize,
        d: u32,
        depth_max: u32,
        budget: &mut u32,
    ) -> bool {
        if *budget == 0 {
            return false; // fail-closed
        }
        *budget -= 1;
        let doms = doms_from_box(bx, used, width, (-r, r));
        let v = match value_set_p(e, ops, &doms, &[]) {
            Some(v) => v,
            None => return false,
        };
        if v.has_fin && !v.nan && !v.pinf && !v.ninf && v.lo.is_finite() && v.hi.is_finite() {
            // Clean: finite AND BOUNDED across the whole cell. Boundedness is load-bearing:
            // the domain encodes a pole's reach in the (lo, hi) bounds, not the attained-inf
            // flags (inv over [-64,64] reads has_fin=true, pinf=false, hi=+inf), and an
            // UNBOUNDED fin part may contain the pole this certificate exists to exclude.
            // (Consequence, stated: exp(exp(x)) does not certify -- its interval hi overflows
            // f64 -- so the certificate stays inside f64-representable boundedness.)
            return true;
        }
        if !v.fin_pm() {
            return false; // no positive-measure finite part: witness
        }
        if d >= depth_max {
            return false; // mixed cell unresolved: fail-closed
        }
        let i = (0..bx.len())
            .max_by(|&a, &b| {
                (bx[a].1 - bx[a].0)
                    .partial_cmp(&(bx[b].1 - bx[b].0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let (lo, hi) = bx[i];
        let mid = 0.5 * (lo + hi);
        bx[i] = (lo, mid);
        let left = rec(e, ops, r, bx, used, width, d + 1, depth_max, budget);
        bx[i] = (mid, hi);
        let right = left && rec(e, ops, r, bx, used, width, d + 1, depth_max, budget);
        bx[i] = (lo, hi);
        left && right
    }
    let (r, depth_per_dim, decidable) = horizon(&[tokens], &[&[]], ops);
    if !decidable {
        return false;
    }
    let used = distinct_vars(&[tokens]);
    let width = n_var_slots(&[tokens]);
    let n = used.len().max(1);
    let mut bx = vec![(-r, r); n];
    let mut budget = BANG_CERT_BUDGET;
    rec(
        tokens,
        ops,
        r,
        &mut bx,
        &used,
        width,
        0,
        depth_per_dim * n as u32,
        &mut budget,
    )
}

// ===================== STRUCTURAL NULL-MEASURE CERTIFICATES ==========================
//
// The subdivision certificate above can never resolve a POLE cell: the cell containing a
// pole subdivides forever (its value interval stays unbounded) and fails closed. Proving
// that a pole's x-support is Lebesgue-NULL is not an interval fact -- it is a STRUCTURAL /
// ANALYTIC fact about the expression tree. This section adds that primitive as a small
// predicate algebra (the regular domains of the ring operations):
//
//   `zero_set_null(f)`     {x : f(x) = 0} is null, for EVERY value of any `<constant>`.
//                          Refuses `<constant>`-bearing trees outright: a fitted constant
//                          may sit exactly at 0 (`sin(C*x)` at C = 0 is identically 0).
//   `nonfinite_null(f)`    {x : f(x) not a finite real} is null, for every `<constant>`
//                          value. Constants are allowed: every zero-set-sensitive position
//                          (a denominator) goes through `zero_set_null`, which refuses
//                          them, and range proofs leave `<constant>` free over all reals,
//                          so they hold for every fitted value.
//   `finite_nonzero_ae(f)` = `zero_set_null && finite_ae`: the multiplicative group's
//                          regular domain, the soundness domain of `A/A -> 1`.
//   `positive_ae(f)`       f > 0 a.e. (completes the pow-base algebra; no consumer yet).
//
// Soundness:
//   * Each structural arm reduces a claim to sub-claims whose exceptional sets COVER the
//     parent's (Z(a*b) = Z(a) ∪ Z(b); poles(a/b) ⊆ nonfinite(a) ∪ nonfinite(b) ∪ Z(b);
//     Z(1/g) ⊆ {g = ±inf}); a finite union of null sets is null.
//   * ANALYTIC-WITNESS base case: if every operator in `f` is everywhere-defined and
//     real-analytic on R (`op_entire_analytic`), the composition is real-analytic on the
//     CONNECTED R^n, hence identically zero or null-zero-set (identity theorem). One
//     subdivision cell whose value interval is a positive-measure set of values excluding
//     0 proves f is not identically zero. The interval decorrelation only WIDENS value
//     intervals, so a found witness is real; `x0 - x0` (identically 0) can never witness.
//     abs / odd roots are total but NOT analytic (kink at 0) and are deliberately absent
//     from the analytic set: `abs(x) + x` vanishes on a half-line and is exactly the
//     plateau family the witness path must never certify (and cannot: the '+' arm defers
//     only to the witness path, which refuses `abs`).
//   * Division / inv are excluded from the analytic set (poles split R^n into components;
//     the identity theorem is per-component) and handled structurally instead.
//   * Everything else fails CLOSED, like every certificate in this module.
//
// The operator classification mirrors the ENGINE's real semantics (the u_* tables above,
// corroborated by the promoted artifact family: `* 0 tan !0 -> 0` shipped, `* 0 asin !0`
// was killed at promotion): odd roots are total (real cbrt); even roots, log, asin, acos,
// acosh, atanh and binary pow are region-restricted.

/// Everywhere-defined AND finite on finite real inputs (engine real semantics; overflow
/// is representation, not a value -- see the module header).
fn op_total_finite(op: &str) -> bool {
    matches!(
        op,
        "+" | "-"
            | "*"
            | "neg"
            | "abs"
            | "pow2"
            | "pow3"
            | "pow4"
            | "pow5"
            | "pow1_3"
            | "pow1_5"
            | "mult2"
            | "mult3"
            | "mult4"
            | "mult5"
            | "div2"
            | "div3"
            | "div4"
            | "div5"
            | "exp"
            | "sin"
            | "cos"
            | "sinh"
            | "cosh"
            | "tanh"
            | "atan"
            | "asinh"
    )
}

/// Everywhere-defined and REAL-ANALYTIC on all of R: the identity-theorem class. `abs`,
/// `pow1_3`, `pow1_5` are total but not analytic at 0 and are excluded on purpose.
fn op_entire_analytic(op: &str) -> bool {
    matches!(
        op,
        "+" | "-"
            | "*"
            | "neg"
            | "pow2"
            | "pow3"
            | "pow4"
            | "pow5"
            | "mult2"
            | "mult3"
            | "mult4"
            | "mult5"
            | "div2"
            | "div3"
            | "div4"
            | "div5"
            | "exp"
            | "sin"
            | "cos"
            | "sinh"
            | "cosh"
            | "tanh"
            | "atan"
            | "asinh"
    )
}

/// End index (exclusive) of the prefix subtree starting at `i`.
fn subtree_end(t: &[String], i: usize, ops: &Operators) -> Option<usize> {
    let mut need = 1usize;
    let mut j = i;
    while need > 0 {
        let tok = t.get(j)?;
        need = need - 1 + ops.arity_of(tok).unwrap_or(0) as usize;
        j += 1;
    }
    Some(j)
}

/// An exponent/index subtree as an exact integer, when it is a bare literal or a
/// `neg`-wrapped literal (the only spellings the emitters produce for `pow`/`rootn`
/// second arguments). `None` for anything else -- consumers fail closed.
fn int_literal(t: &[String]) -> Option<i64> {
    let v = match t {
        [tok] => crate::numeric::leaf_value(tok)?,
        [n, tok] if n == "neg" => -crate::numeric::leaf_value(tok)?,
        _ => return None,
    };
    if v.is_finite() && v.fract() == 0.0 && v.abs() <= 9.0e15 {
        Some(v as i64)
    } else {
        None
    }
}

/// Whole-space range of a subexpression: every variable AND every free `<constant>` over
/// all finite reals -- containment proofs made on it are valid for EVERY fitted constant.
fn whole_range(t: &[String], ops: &Operators) -> Option<Vs> {
    value_set(t, ops, &Vs::reals())
}

const CERT_RECURSION_MAX: u32 = 32;

/// {x : f(x) = 0} is Lebesgue-null, for every value of any `<constant>`. Fail-closed.
pub fn zero_set_null(tokens: &[String], ops: &Operators) -> bool {
    if tokens.iter().any(|s| s == "<constant>") {
        return false;
    }
    zsn(tokens, ops, 0)
}

fn zsn(t: &[String], ops: &Operators, d: u32) -> bool {
    if d > CERT_RECURSION_MAX || t.is_empty() {
        return false;
    }
    let tok = t[0].as_str();
    if !ops.is_operator(tok) {
        if var_index(tok).is_some() {
            return true; // {x = 0} is a point
        }
        if tok == "<constant>" {
            return false; // may BE 0 (belt-and-braces; the entry gate already refused)
        }
        // literal: never zero unless it IS zero. (A NaN literal has an EMPTY zero set.)
        return crate::numeric::leaf_value(tok).is_some_and(|v| v != 0.0);
    }
    match (tok, ops.arity_of(tok)) {
        // h(y) = 0 iff y = 0: the zero set equals the argument's
        (
            "neg" | "abs" | "pow2" | "pow3" | "pow4" | "pow5" | "pow1_2" | "pow1_3" | "pow1_4"
            | "pow1_5" | "mult2" | "mult3" | "mult4" | "mult5" | "div2" | "div3" | "div4" | "div5"
            | "sinh" | "tanh" | "asinh" | "atan" | "asin" | "atanh",
            Some(1),
        ) => zsn(&t[1..], ops, d + 1),
        // never zero on their real range
        ("exp" | "cosh", Some(1)) => true,
        // h(y) = 0 iff y = 1: shift and recurse
        ("log" | "acos" | "acosh", Some(1)) => {
            let mut shifted: Vec<String> = Vec::with_capacity(t.len() + 1);
            shifted.push("-".into());
            shifted.extend_from_slice(&t[1..]);
            shifted.push("1".into());
            zsn(&shifted, ops, d + 1)
        }
        // tan(g) = 0 iff sin(g) = 0
        ("tan", Some(1)) => {
            let mut s: Vec<String> = Vec::with_capacity(t.len());
            s.push("sin".into());
            s.extend_from_slice(&t[1..]);
            zsn(&s, ops, d + 1)
        }
        // Z(1/g) = {g = ±inf} EXACTLY (`1/0 = +inf`, `1/nan = nan`: neither is 0) --
        // the INFINITE-set certificate; a fat NaN domain of `g` is harmless here.
        // (Sharpened 2026-08-03 from `nfn`, which conflated NaN with ±inf and refused
        // every acosh/log-bearing denominator -- the division-tower campaign's Hole 2.)
        ("inv", Some(1)) => isn(&t[1..], ops, d + 1),
        ("*", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            zsn(&t[1..ja], ops, d + 1) && zsn(&t[ja..], ops, d + 1)
        }
        // (a/b) = 0 only where a = 0 (b finite nonzero) or b = ±inf (a finite):
        // Z(a/b) ⊆ Z(a) ∪ {b = ±inf} -- the denominator claim is the INFINITE-set
        // certificate (a NaN of `b` makes the quotient NaN, never 0).
        ("/", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            zsn(&t[1..ja], ops, d + 1) && isn(&t[ja..], ops, d + 1)
        }
        // Literal integer exponents have EXACT zero sets (2026-08-03 sharpening):
        //   q > 0: Z(b^q) = Z(b)      (±inf^q = ±inf; nan^q = nan; neither is 0)
        //   q < 0: Z(b^q) = {b = ±inf} (0^q = +inf one-zero; finite nonzero exact)
        //   q = 0: b^0 = 1 everywhere (incl. NaN base, contract §2): never zero.
        // A general exponent keeps the inclusion Z(b^g) ⊆ Z(b) ∪ {b = ±inf} ∪
        // {g = ±inf} (|b| ≷ 1 with g = ∓inf lives inside the last member; a NaN of
        // either side is NaN, never 0).
        ("pow", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            match int_literal(&t[ja..]) {
                Some(q) if q > 0 => zsn(&t[1..ja], ops, d + 1),
                Some(q) if q < 0 => isn(&t[1..ja], ops, d + 1),
                Some(_) => true,
                None => {
                    zsn(&t[1..ja], ops, d + 1)
                        && isn(&t[1..ja], ops, d + 1)
                        && isn(&t[ja..], ops, d + 1)
                }
            }
        }
        // rootn with a literal index n ≥ 2: Z(rootn(a, n)) = Z(a) (odd roots are
        // total and sign-preserving; an even root of a negative argument is NaN,
        // never 0; rootn(±inf) is ±inf/NaN, never 0). Non-literal index: fail-closed.
        ("rootn", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            match int_literal(&t[ja..]) {
                Some(n) if n >= 2 => zsn(&t[1..ja], ops, d + 1),
                _ => false,
            }
        }
        // sums, sin, cos, and any other analytic composition: identity theorem + witness
        _ => analytic_nonzero_witness(t, ops),
    }
}

/// The identity-theorem witness: all ops entire-analytic, and ONE cell whose value
/// interval is a positive-measure set excluding 0 -> f is not identically zero -> Z(f)
/// is null. Fail-closed on budget/horizon like every certificate here.
/// Tree-aware entire-analyticity scan for the witness path. Every operator must be
/// entire-analytic, EXCEPT that `pow` with a bare non-negative integer literal
/// exponent is admitted on an analytic base: an integer power of an entire function
/// is entire -- exactly the class the retired unary `pow2`..`pow5` tables covered
/// before the binary-vocabulary migration (2026-08-03 completion; the token-level
/// scan refused every polynomial spelled with binary `pow`). Negative or non-literal
/// exponents refuse (poles / branch cuts). Leaf tokens pass as before: the witness
/// gate's soundness never rests on leaf analyticity -- a non-real-valued leaf
/// (±inf/NaN literal) can produce no positive-measure finite value interval.
fn entire_analytic_composition(t: &[String], ops: &Operators) -> bool {
    fn rec(t: &[String], i: usize, ops: &Operators) -> Option<usize> {
        let tok = t.get(i)?;
        if !ops.is_operator(tok) {
            return Some(i + 1);
        }
        if tok == "pow" {
            let ja = rec(t, i + 1, ops)?; // base must itself be analytic
            let jb = subtree_end(t, ja, ops)?;
            return match int_literal(t.get(ja..jb)?) {
                Some(q) if q >= 0 => Some(jb),
                _ => None,
            };
        }
        if !op_entire_analytic(tok) {
            return None;
        }
        let mut j = i + 1;
        for _ in 0..ops.arity_of(tok).unwrap_or(0) as usize {
            j = rec(t, j, ops)?;
        }
        Some(j)
    }
    rec(t, 0, ops).is_some_and(|j| j == t.len())
}

fn analytic_nonzero_witness(t: &[String], ops: &Operators) -> bool {
    const WITNESS_BUDGET: u32 = 2_000;
    if !entire_analytic_composition(t, ops) {
        return false;
    }
    let (r, depth_per_dim, decidable) = horizon(&[t], &[&[]], ops);
    if !decidable {
        return false;
    }
    let used = distinct_vars(&[t]);
    let width = n_var_slots(&[t]);
    let n = used.len().max(1);
    let mut bx = vec![(-r, r); n];
    let mut budget = WITNESS_BUDGET;
    let depth_max = depth_per_dim * n as u32;
    #[allow(clippy::too_many_arguments)]
    fn rec(
        e: &[String],
        ops: &Operators,
        r: f64,
        bx: &mut [(f64, f64)],
        used: &[usize],
        width: usize,
        d: u32,
        depth_max: u32,
        budget: &mut u32,
    ) -> bool {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        let doms = doms_from_box(bx, used, width, (-r, r));
        if let Some(v) = value_set_p(e, ops, &doms, &[]) {
            if v.fin_pm() {
                let zero_in = (v.lo < 0.0 || (v.lo == 0.0 && !v.lo_open))
                    && (v.hi > 0.0 || (v.hi == 0.0 && !v.hi_open));
                if !zero_in {
                    return true; // positive-measure values, all nonzero
                }
            }
        } else {
            return false;
        }
        if d >= depth_max {
            return false;
        }
        let i = (0..bx.len())
            .max_by(|&a, &b| {
                (bx[a].1 - bx[a].0)
                    .partial_cmp(&(bx[b].1 - bx[b].0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let (lo, hi) = bx[i];
        let mid = 0.5 * (lo + hi);
        bx[i] = (lo, mid);
        let left = rec(e, ops, r, bx, used, width, d + 1, depth_max, budget);
        bx[i] = (mid, hi);
        let right = !left && rec(e, ops, r, bx, used, width, d + 1, depth_max, budget);
        bx[i] = (lo, hi);
        left || right
    }
    rec(t, ops, r, &mut bx, &used, width, 0, depth_max, &mut budget)
}

/// {x : f(x) not a finite real} is Lebesgue-null, for every `<constant>` value.
/// Fail-closed.
pub fn nonfinite_null(tokens: &[String], ops: &Operators) -> bool {
    nfn(tokens, ops, 0)
}

fn nfn(t: &[String], ops: &Operators, d: u32) -> bool {
    if d > CERT_RECURSION_MAX || t.is_empty() {
        return false;
    }
    let tok = t[0].as_str();
    if !ops.is_operator(tok) {
        if var_index(tok).is_some() || tok == "<constant>" {
            return true; // finite reals by contract
        }
        return crate::numeric::leaf_value(tok).is_some_and(f64::is_finite);
    }
    match (tok, ops.arity_of(tok)) {
        ("+" | "-" | "*", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            nfn(&t[1..ja], ops, d + 1) && nfn(&t[ja..], ops, d + 1)
        }
        // poles(a/b) ⊆ nonfinite(a) ∪ nonfinite(b) ∪ Z(b); `zero_set_null` refuses a
        // `<constant>`-bearing denominator, which keeps the claim ∀C-sound.
        ("/", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            nfn(&t[1..ja], ops, d + 1)
                && nfn(&t[ja..], ops, d + 1)
                && (!t[ja..].iter().any(|s| s == "<constant>") && zsn(&t[ja..], ops, d + 1))
        }
        ("inv", Some(1)) => {
            nfn(&t[1..], ops, d + 1)
                && (!t[1..].iter().any(|s| s == "<constant>") && zsn(&t[1..], ops, d + 1))
        }
        // tan poles sit on {cos(g) = 0}
        ("tan", Some(1)) => {
            let mut c: Vec<String> = Vec::with_capacity(t.len());
            c.push("cos".into());
            c.extend_from_slice(&t[1..]);
            nfn(&t[1..], ops, d + 1)
                && (!t[1..].iter().any(|s| s == "<constant>") && zsn(&c, ops, d + 1))
        }
        // total-finite unaries pass the claim through
        (op, Some(1)) if op_total_finite(op) => nfn(&t[1..], ops, d + 1),
        // domain-restricted unaries: prove the argument's whole-space range inside the
        // domain (with `<constant>` free over all reals, the proof is ∀C-valid)
        ("log", Some(1)) => {
            nfn(&t[1..], ops, d + 1)
                && whole_range(&t[1..], ops).is_some_and(|v| {
                    v.has_fin
                        && !v.nan
                        && !v.pinf
                        && !v.ninf
                        && (v.lo > 0.0 || (v.lo == 0.0 && v.lo_open))
                })
        }
        ("pow1_2" | "pow1_4", Some(1)) => {
            nfn(&t[1..], ops, d + 1)
                && whole_range(&t[1..], ops)
                    .is_some_and(|v| v.has_fin && !v.nan && !v.pinf && !v.ninf && v.lo >= 0.0)
        }
        ("asin" | "acos", Some(1)) => {
            nfn(&t[1..], ops, d + 1)
                && whole_range(&t[1..], ops).is_some_and(|v| {
                    v.has_fin && !v.nan && !v.pinf && !v.ninf && v.lo >= -1.0 && v.hi <= 1.0
                })
        }
        ("acosh", Some(1)) => {
            nfn(&t[1..], ops, d + 1)
                && whole_range(&t[1..], ops)
                    .is_some_and(|v| v.has_fin && !v.nan && !v.pinf && !v.ninf && v.lo >= 1.0)
        }
        ("atanh", Some(1)) => {
            nfn(&t[1..], ops, d + 1)
                && whole_range(&t[1..], ops).is_some_and(|v| {
                    v.has_fin
                        && !v.nan
                        && !v.pinf
                        && !v.ninf
                        && (v.lo > -1.0 || (v.lo == -1.0 && v.lo_open))
                        && (v.hi < 1.0 || (v.hi == 1.0 && v.hi_open))
                })
        }
        // Binary `pow` with a literal integer exponent (2026-08-03 vocabulary
        // completion -- the retired unary `pow2`..`pow5` had arms, their binary
        // replacement did not):
        //   k ≥ 0: nonfinite(b^k) = nonfinite(b) (finite^k is finite -- overflow is
        //          representation, not a value; ±inf^k infinite; nan^k nan; b^0 = 1
        //          everywhere, a subset claim a fortiori).
        //   k < 0: nonfinite(b^k) = Z(b) ∪ nan(b) ⊆ Z(b) ∪ nonfinite(b)
        //          (0^k = +inf one-zero; ±inf^k = 0; finite nonzero exact).
        // Non-integer or non-literal exponents: fail-closed (fat NaN on b < 0 resp.
        // no structural handle).
        ("pow", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            match int_literal(&t[ja..]) {
                Some(k) if k >= 0 => nfn(&t[1..ja], ops, d + 1),
                Some(_) => {
                    nfn(&t[1..ja], ops, d + 1)
                        && !t[1..ja].iter().any(|s| s == "<constant>")
                        && zsn(&t[1..ja], ops, d + 1)
                }
                None => false,
            }
        }
        // rootn with a literal index: an odd root is TOTAL (sign-preserving real
        // root -- nonfiniteness passes through); an even root is NaN on {arg < 0}
        // and needs the same range proof as `pow1_2`.
        ("rootn", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            match int_literal(&t[ja..]) {
                Some(n) if n >= 3 && n % 2 == 1 => nfn(&t[1..ja], ops, d + 1),
                Some(n) if n >= 2 && n % 2 == 0 => {
                    nfn(&t[1..ja], ops, d + 1)
                        && whole_range(&t[1..ja], ops).is_some_and(|v| {
                            v.has_fin && !v.nan && !v.pinf && !v.ninf && v.lo >= 0.0
                        })
                }
                _ => false,
            }
        }
        _ => false, // unknown ops: fail-closed
    }
}

/// {x : f(x) = ±inf} is Lebesgue-null, for every `<constant>` value -- the
/// INFINITE-set certificate (2026-08-03, division-tower campaign). Strictly weaker
/// than `nonfinite_null`: a NaN is NOT an infinity, and the domain-restricted
/// operators (`acosh`, `asin`, `pow1_2`, ...) mint NaN, never new infinities --
/// their fat NaN domains are harmless to consumers whose disagreement sets involve
/// only ±inf (`Z(1/A) = {A = ±inf}`, the odd-negative-power distribution's
/// reciprocal factors). Fail-closed like every certificate in this module; the
/// zero-set sub-claims inherit `zero_set_null`'s `<constant>` refusal.
#[cfg_attr(not(test), allow(dead_code))] // standalone tokens-interface (certified + poison batteries); in-crate consumers reach `isn` through the zsn/nfn arms
pub fn infinite_set_null(tokens: &[String], ops: &Operators) -> bool {
    isn(tokens, ops, 0)
}

fn isn(t: &[String], ops: &Operators, d: u32) -> bool {
    if d > CERT_RECURSION_MAX || t.is_empty() {
        return false;
    }
    let tok = t[0].as_str();
    if !ops.is_operator(tok) {
        if var_index(tok).is_some() || tok == "<constant>" {
            return true; // finite reals by contract
        }
        // A ±inf literal is infinite on full measure; a NaN literal has an EMPTY
        // infinite set (NaN is not ±inf).
        return crate::numeric::leaf_value(tok).is_some_and(|v| !v.is_infinite());
    }
    match (tok, ops.arity_of(tok)) {
        // Bounded real range: never ±inf anywhere (sin/cos/asin/acos are NaN
        // outside their domains resp. at infinite arguments -- NaN, not ±inf).
        ("sin" | "cos" | "tanh" | "atan" | "asin" | "acos", Some(1)) => true,
        // {op(u) = ±inf} ⊆ {u = ±inf}: every op here maps finite reals to
        // finite-or-NaN and propagates NaN -- the total-finite class plus the
        // fat-NaN domain-restricted ops whose only infinities are inherited
        // (`pow1_2(+inf) = +inf` but `pow1_2(finite)` is finite or NaN, etc.).
        (op, Some(1)) if op_total_finite(op) || matches!(op, "pow1_2" | "pow1_4" | "acosh") => {
            isn(&t[1..], ops, d + 1)
        }
        // {log u = ±inf} = {u = 0} ∪ {u = +inf} (one-zero: log 0 = -inf).
        ("log", Some(1)) => {
            isn(&t[1..], ops, d + 1)
                && !t[1..].iter().any(|s| s == "<constant>")
                && zsn(&t[1..], ops, d + 1)
        }
        // {atanh u = ±inf} = {u = ±1}: both level sets must be null (shift trick).
        ("atanh", Some(1)) => {
            if t[1..].iter().any(|s| s == "<constant>") {
                return false;
            }
            let mut m: Vec<String> = Vec::with_capacity(t.len() + 1);
            m.push("-".into());
            m.extend_from_slice(&t[1..]);
            m.push("1".into());
            let mut p: Vec<String> = Vec::with_capacity(t.len() + 1);
            p.push("+".into());
            p.extend_from_slice(&t[1..]);
            p.push("1".into());
            zsn(&m, ops, d + 1) && zsn(&p, ops, d + 1)
        }
        // tan's poles sit on {cos(g) = 0}.
        ("tan", Some(1)) => {
            if t[1..].iter().any(|s| s == "<constant>") {
                return false;
            }
            let mut c: Vec<String> = Vec::with_capacity(t.len());
            c.push("cos".into());
            c.extend_from_slice(&t[1..]);
            zsn(&c, ops, d + 1)
        }
        // {1/g = ±inf} = {g = 0} (1/±inf = 0, 1/nan = nan).
        ("inv", Some(1)) => !t[1..].iter().any(|s| s == "<constant>") && zsn(&t[1..], ops, d + 1),
        // An infinite sum/product needs an infinite operand (finite ∘ finite is
        // finite; overflow is representation; NaN propagates).
        ("+" | "-" | "*", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            isn(&t[1..ja], ops, d + 1) && isn(&t[ja..], ops, d + 1)
        }
        // {a/b = ±inf} ⊆ {a = ±inf} ∪ {b = 0} (a/±inf = 0; NaN propagates).
        ("/", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            isn(&t[1..ja], ops, d + 1)
                && !t[ja..].iter().any(|s| s == "<constant>")
                && zsn(&t[ja..], ops, d + 1)
        }
        // pow with a literal integer exponent: k > 0 → {b = ±inf}; k < 0 → {b = 0}
        // (0^-k = +inf one-zero; ±inf^-k = 0); k = 0 → b^0 = 1, never infinite.
        ("pow", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            match int_literal(&t[ja..]) {
                Some(k) if k > 0 => isn(&t[1..ja], ops, d + 1),
                Some(k) if k < 0 => {
                    !t[1..ja].iter().any(|s| s == "<constant>") && zsn(&t[1..ja], ops, d + 1)
                }
                Some(_) => true,
                None => false,
            }
        }
        // rootn with a literal index n ≥ 2: infinities are inherited from the
        // argument (odd: ±inf → ±inf; even: +inf → +inf, negatives → NaN).
        ("rootn", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            match int_literal(&t[ja..]) {
                Some(n) if n >= 2 => isn(&t[1..ja], ops, d + 1),
                _ => false,
            }
        }
        _ => false, // unknown ops: fail-closed
    }
}

/// The multiplicative group's regular domain: defined, finite AND nonzero a.e. -- the
/// exact soundness domain of `A/A -> 1` (SELFCANCEL T1). Refuses `<constant>`-bearing
/// trees via `zero_set_null`.
pub fn finite_nonzero_ae(tokens: &[String], ops: &Operators) -> bool {
    zero_set_null(tokens, ops) && finite_ae(tokens, ops)
}

/// Is `f` an ENTIRE-analytic composition (no branch cuts, poles, or abs) that is
/// certainly NOT identically constant? Certified by a PAIR of boxes whose rigorous
/// value intervals are DISJOINT. Consumed by the symbolic-exponent occurrence merge
/// (`ac::expr::mul`): for a nonconstant entire exponent y every level set {y = q} is
/// null (identity theorem), so {y in (1/n)Z \ Z} is null and merging n occurrences of
/// base^y into base^(n*y) changes nothing off a null set. The traps are refused by
/// construction: an identically-constant-but-unfolded y (sin^2 + cos^2) admits no
/// disjoint pair; an abs-bearing y is not entire; a ground y is constant.
pub fn nonconstant_entire(t: &[String], ops: &Operators) -> bool {
    if t.iter()
        .any(|s| ops.is_operator(s) && !op_entire_analytic(s))
    {
        return false;
    }
    let (r, _depth, decidable) = horizon(&[t], &[&[]], ops);
    if !decidable {
        return false;
    }
    let used = distinct_vars(&[t]);
    if used.is_empty() {
        return false;
    }
    let width = n_var_slots(&[t]);
    let n = used.len();
    let center = (0.05 * r, 0.15 * r);
    let lo = (-0.95 * r, -0.8 * r);
    let hi = (0.8 * r, 0.95 * r);
    let disjoint = |a: &Vs, b: &Vs| {
        a.has_fin
            && b.has_fin
            && a.lo.is_finite()
            && a.hi.is_finite()
            && b.lo.is_finite()
            && b.hi.is_finite()
            && (a.hi < b.lo || b.hi < a.lo)
    };
    for i in 0..n {
        for (wa, wb) in [(lo, hi), (center, hi), (lo, center)] {
            let mut ba = vec![center; n];
            let mut bb = vec![center; n];
            ba[i] = wa;
            bb[i] = wb;
            let va = value_set_p(t, ops, &doms_from_box(&ba, &used, width, (-r, r)), &[]);
            let vb = value_set_p(t, ops, &doms_from_box(&bb, &used, width, (-r, r)), &[]);
            if let (Some(a), Some(b)) = (va, vb) {
                if disjoint(&a, &b) {
                    return true;
                }
            }
        }
    }
    false
}

/// f > 0 a.e.: the pow-base predicate completing the algebra. Conservative v1 (no
/// production consumer yet -- the flatten-and-collect pass (proposal P4) is the intended
/// one; the certified/poison batteries below pin its behavior meanwhile).
#[cfg_attr(not(test), allow(dead_code))]
pub fn positive_ae(tokens: &[String], ops: &Operators) -> bool {
    if tokens.is_empty() {
        return false;
    }
    match tokens[0].as_str() {
        "exp" | "cosh" => nonfinite_null(&tokens[1..], ops),
        // squares / abs of a null-zero-set argument: >= 0 with a null zero set
        "pow2" | "pow4" | "abs" => {
            zero_set_null(&tokens[1..], ops) && nonfinite_null(&tokens[1..], ops)
        }
        // the BINARY spelling of the even powers (generation-2 vocabulary completion,
        // audit Tier-1 #3 -- the retired unary arm above never fires on a live engine):
        // pow(b, k) with an EVEN positive literal integer exponent is b^k >= 0 with a
        // null zero set, under the same operand certificates as the unary arm.
        "pow" => {
            let Some(ja) = subtree_end(tokens, 1, ops) else {
                return false;
            };
            matches!(int_literal(&tokens[ja..]), Some(k) if k > 0 && k % 2 == 0)
                && zero_set_null(&tokens[1..ja], ops)
                && nonfinite_null(&tokens[1..ja], ops)
        }
        "inv" => positive_ae(&tokens[1..], ops),
        _ => whole_range(tokens, ops).is_some_and(|v| {
            v.has_fin && !v.nan && !v.ninf && (v.lo > 0.0 || (v.lo == 0.0 && v.lo_open))
        }),
    }
}

/// Can `cand` reach every value COMPONENT the `src` reaches on positive measure?
///
/// The two sides' constants are quantified DIFFERENTLY: a rule must hold for EVERY source
/// constant (∀), while the candidate's constants are CHOSEN by the fit (∃). So the source's
/// achievable components -- the union over its constants -- must all be reachable by the
/// candidate over its own. This is an exact NECESSARY condition, decided with no sampling.
///
/// It is what makes the sampled `POLE_GRID` unnecessary for the artifact family it was added for.
/// `pow(asin(C/2), inf)` is 0 for |C| < 2·sin(1) ≈ 1.683, **+inf** on the narrow band
/// C ∈ (1.683, 2), and NaN past |C| > 2. The band is only 0.317 wide and falls BETWEEN the grid
/// points 1.6 and 2.5, so a sampled sweep can straddle it seed-dependently -- admitting
/// `pow(asin(C/2), inf) -> 0`. Here the source's `pinf` is simply set and the literal `0`
/// cannot reach it: rejected on structure, at every seed, for free.
///
/// NaN is deliberately EXCLUDED: a NaN source row is extendable (`x/x -> 1`), and how far that
/// may go is the domain gate's question, decided by measure rather than by reachability.
pub fn reaches_all_of(cand: &Vs, src: &Vs) -> bool {
    (!src.has_fin || cand.has_fin) && (!src.pinf || cand.pinf) && (!src.ninf || cand.ninf)
}

/// Is `expr` (at these constants) defined on a POSITIVE-MEASURE set -- i.e. is it a function at
/// all, or is it NaN almost everywhere?
///
/// This separates the two ways a source can be NaN-heavy, which sampling cannot tell apart:
/// `exp(log x)` is nan on x<0 but a real function on x>0 (GENERIC support -- the domain gate
/// applies), while `sqrt((-2)^x)` is nan a.e., finite only on the even integers (MEASURE-ZERO
/// support -- it binds nothing generic, and the constants any fit "chose" for it are arbitrary).
/// Whether the mining X happens to contain the atom `x = 2` must not decide a rule's soundness.
///
/// Returns the width of the widest witness cell on which `expr` is defined everywhere (0.0 = no
/// generic support). Deliberately the same subdivision as `domain_extension_p`, so the two agree
/// on what "positive measure" means.
#[allow(dead_code)] // standalone API: the in-crate gate deliberately uses `_at` (see below)
pub fn defined_measure_p(expr: &[String], params: &[f64], ops: &Operators) -> Option<f64> {
    // Standalone use only. A GATE decision must NOT call this: it derives its horizon from `expr`
    // alone, and deadness judged at a different horizon than its extension witness is a
    // fail-open. Gates compute `gate_horizon(..)` once and call `defined_measure_p_at`.
    let (r, depth_per_dim, decidable) = horizon(&[expr], &[params], ops);
    if !decidable {
        HORIZON_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None; // liveness not provable in any affordable box
    }
    defined_measure_p_at(expr, params, ops, r, depth_per_dim)
}

/// `defined_measure_p` at a CALLER-SUPPLIED horizon (see `gate_horizon`). Does not count horizon
/// misses. `Some(width)` only when DECIDED (witness found, or search completed with none);
/// budget-exhausted-with-no-witness is `None` -- NOT proven dead, so the deadness exemption
/// must not apply (an unresolved search flipping a live source to "dead" widened the gate's
/// accept exemption, the anti-conservative direction). An UNANALYZABLE box degrades the
/// search the same way (F0): support may hide in a box `value_set_p` cannot read, so a
/// completion that skipped one has not PROVEN deadness -- `None`, and the gate treats the
/// side as live.
pub fn defined_measure_p_at(
    expr: &[String],
    params: &[f64],
    ops: &Operators,
    r: f64,
    depth_per_dim: u32,
) -> Option<f64> {
    #[allow(clippy::too_many_arguments)]
    fn rec(
        e: &[String],
        p: &[f64],
        ops: &Operators,
        r: f64,
        bx: &mut [(f64, f64)],
        used: &[usize],
        width: usize,
        d: u32,
        depth_max: u32,
        best: &mut f64,
        budget: &mut u32,
        undecided: &mut bool,
    ) {
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        let vol: f64 = bx.iter().map(|(lo, hi)| hi - lo).product();
        if vol <= *best {
            return;
        }
        let doms = doms_from_box(bx, used, width, (-r, r));
        let v = match value_set_p(e, ops, &doms, p) {
            Some(v) => v,
            None => {
                // UNANALYZABLE box: support may hide here, so "proven dead" is off the
                // table -- the search degrades to undecided (unresolved deadness reads
                // as LIVE at the gate, the conservative direction).
                *undecided = true;
                return;
            }
        };
        if !v.defined() {
            return;
        } // nan across the whole box: no support here
        if !v.nan {
            if vol > *best {
                *best = vol;
            } // defined across ALL of it: a witness
            return;
        }
        if d >= depth_max {
            return;
        }
        let i = (0..bx.len())
            .max_by(|&a, &b| {
                (bx[a].1 - bx[a].0)
                    .partial_cmp(&(bx[b].1 - bx[b].0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let (lo, hi) = bx[i];
        let mid = 0.5 * (lo + hi);
        bx[i] = (lo, mid);
        rec(
            e,
            p,
            ops,
            r,
            bx,
            used,
            width,
            d + 1,
            depth_max,
            best,
            budget,
            undecided,
        );
        bx[i] = (mid, hi);
        rec(
            e,
            p,
            ops,
            r,
            bx,
            used,
            width,
            d + 1,
            depth_max,
            best,
            budget,
            undecided,
        );
        bx[i] = (lo, hi);
    }
    let used = distinct_vars(&[expr]);
    let width = n_var_slots(&[expr]);
    let n = used.len().max(1);
    let mut bx = vec![(-r, r); n];
    let mut best = 0.0f64;
    let mut budget = max_nodes_budget();
    let mut undecided = false;
    rec(
        expr,
        params,
        ops,
        r,
        &mut bx,
        &used,
        width,
        0,
        depth_per_dim * n as u32,
        &mut best,
        &mut budget,
        &mut undecided,
    );
    if budget == 0 && best == 0.0 {
        NODE_BUDGET_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    if undecided && best == 0.0 {
        UNANALYZABLE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// The domain gate splits the inverse round-trips exactly where mathematics does: composing
    /// The horizon must reach past the expression's OWN constants.
    ///
    /// With a hard `R = 64`, `pow4(np.pi) = 97.409` put the NaN boundary of
    /// `exp log - pow4 np.pi x0` outside the searched box, so the gate reported 0.0 and the rule
    /// was mined -- while the control `pow3(np.pi) = 31.006` sat inside and was rejected. Same
    /// shape; only the constant's magnitude differed.
    #[test]
    fn horizon_reaches_past_the_expression_s_own_constants() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        for k in ["3", "4"] {
            let src = s(&["exp", "log", "-", "pow", "np.pi", k, "x0"]);
            let tgt = s(&["-", "pow", "np.pi", k, "x0"]);
            assert!(
                domain_extension(&src, &tgt, ops).unwrap() > 0.0,
                "gate blind past its horizon for pow(np.pi, {k})"
            );
        }
        // ... and widening the box must not coarsen the 1-variable verdicts it already had right.
        assert_eq!(
            domain_extension(&s(&["exp", "log", "x0"]), &s(&["x0"]), ops),
            Some(64.0)
        );
        assert_eq!(
            domain_extension(&s(&["log", "exp", "x0"]), &s(&["x0"]), ops),
            Some(0.0)
        );
    }

    /// The PARAMETRIC spelling of the same hole: the miner mines `pow4(<constant>)`, not
    /// `pow4(np.pi)`, and `max_const_magnitude` used to skip every `<constant>`-bearing subtree
    /// and add the BOUND value raw -- deriving the horizon from |pi| = 3.14 instead of
    /// pow4(pi) = 97.409.
    #[test]
    fn horizon_folds_the_bound_constants() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let pi = std::f64::consts::PI;
        for k in ["3", "4"] {
            let src = s(&["exp", "log", "-", "pow", "<constant>", k, "x0"]);
            let tgt = s(&["-", "pow", "<constant>", k, "x0"]);
            assert!(
                domain_extension_p(&src, &[pi], &tgt, &[pi], ops).unwrap() > 0.0,
                "gate blind past its horizon for pow(<constant> = pi, {k})"
            );
        }
    }

    /// The b_pow direction flag, RESOLVED: a negative base with a continuum exponent is finite
    /// exactly on the integer slice -- REACHABLE (has_fin, for `reaches_all_of`) but measure-NULL
    /// (`fin_null`, for the domain gate). Pre-fix, the phantom positive-measure fin blocked the
    /// witness on nan-a.e. sources: the ">=2-variable negative-base pow family" was silenced.
    #[test]
    fn integer_slice_fin_does_not_block_the_witness() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // base = -exp(x0) < 0 everywhere; exponent = x1 a continuum: source is nan a.e., its
        // only finite values on the null slice {x1 in Z}. A target defined everywhere is a
        // positive-measure extension and the gate must SEE it.
        let src = s(&["pow", "neg", "exp", "x0", "x1"]);
        let tgt = s(&["+", "x0", "x1"]);
        assert!(
            domain_extension(&src, &tgt, ops).unwrap() > 0.0,
            "witness blocked by measure-null integer-slice fin"
        );
        // ... through a unary chain (the carry): sqrt of the same family
        let src2 = s(&["rootn", "pow", "neg", "exp", "x0", "x1", "2"]);
        assert!(
            domain_extension(&src2, &tgt, ops).unwrap() > 0.0,
            "fin_null lost through the unary carry"
        );
        // control: a POSITIVE-measure-defined source must NOT produce a witness spuriously
        let src3 = s(&["pow", "abs", "x0", "x1"]); // |x0|^x1 defined a.e.
        assert_eq!(
            domain_extension(&src3, &s(&["+", "x0", "x1"]), ops),
            Some(0.0),
            "spurious witness on a live source"
        );
    }

    /// ONE ZERO (contract v2 §9.2, DECIDED 2026-07-18; H-016 2026-08-03): a point zero is THE
    /// unsigned zero, its inverse is +inf, and pow(0, y<0) = +inf for EVERY y -- there is no
    /// odd-integer -inf slice because there is no -0. This test used to pin the OPPOSITE
    /// (referee-era deployment semantics: b_mul minted -0.0 and u_inv/b_pow read the sign bit
    /// to pick signed poles); §9.2 formally overturned that semantics, and H-016 caught the
    /// surviving device shipping a `-inf` LITERAL for `inv(0/tanh(-2))` while every
    /// structurally-collapsing sibling spelling shipped +inf. Deployment's -inf on the -0
    /// slice at runtime is documented measurement error (§9.2), never a reachable value.
    #[test]
    fn inverse_of_a_point_zero_is_the_positive_pole() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let v = value_set(&s(&["inv", "*", "0", "x0"]), ops, &Vs::reals()).unwrap();
        assert!(
            v.pinf && !v.ninf,
            "1/(point zero) is the +inf pole and nothing else: {v:?}"
        );
        // pow with a constant-zero base and a negative-capable exponent: +inf on the y<0 side,
        // the point 0 on the y>0 side, no -inf anywhere.
        let w = value_set(&s(&["pow", "*", "0", "x0", "x0"]), ops, &Vs::reals()).unwrap();
        assert!(
            w.pinf && !w.ninf,
            "pow(point zero, exponent<0) is +inf only: {w:?}"
        );
        // The documented -0.0 minter is flushed: `x1*0` on x1 < 0 -- the exact fold that used
        // to preserve the endpoint products' IEEE sign -- now hands downstream THE zero.
        let neg = Vs::interval(-60.0, -5.9, false, false);
        let e01 = value_set_p(
            &s(&["*", "x1", "0"]),
            ops,
            &[Vs::interval(-9.5, 2.5, false, false), neg],
            &[],
        )
        .unwrap();
        assert!(
            e01.fin_is_point() && e01.lo == 0.0 && e01.lo.is_sign_positive(),
            "x1*0 on x1<0 must be the unsigned zero point: {e01:?}"
        );
        let g3 = value_set_p(
            &s(&["pow", "*", "x1", "0", "x0"]),
            ops,
            &[Vs::interval(-9.5, 2.5, false, false), neg],
            &[],
        )
        .unwrap();
        assert!(
            g3.pinf && !g3.ninf,
            "pow(point zero, exponent spanning <0) is +inf only: {g3:?}"
        );
        // The H-016 observable at the classification oracle: the surviving zero-times-
        // reciprocal bag classifies POSINF (it used to classify NEGINF and the ground fold
        // shipped the literal).
        let h = value_set(&s(&["inv", "/", "0", "-2"]), ops, &Vs::reals()).unwrap();
        assert!(
            h.pinf && !h.ninf,
            "inv(0/(-2)) must classify the +inf pole: {h:?}"
        );
    }

    /// H-018 (2026-08-03): `np.pi`/`np.e` are outward-rounded ENCLOSURES inside this layer,
    /// never exact f64 points -- a point rendering certified `sin(np.pi) != 0` (the value of
    /// the RENDERING, {1.22e-16}) while the ratified exact value is 0, and that false
    /// NONZERO-a.e. certificate licensed a pow-over-mul distribution that split a true zero
    /// factor (`inv(sin(pi)*u)` shipped +-inf for a +inf truth; fuzz row 172346). The
    /// enclosure makes every exact-zero identity ground contain 0 (certificates fail closed)
    /// while grounds rigorously bounded away from 0 keep certifying.
    #[test]
    fn transcendental_atoms_are_enclosures_not_points() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // Exact-zero identity grounds: the nonzero certificate must REFUSE.
        assert!(
            !zero_set_null(&s(&["sin", "np.pi"]), ops),
            "sin(pi) is exactly 0: a nonzero-a.e. certificate must not exist"
        );
        assert!(
            !zero_set_null(&s(&["abs", "sin", "np.pi"]), ops),
            "abs(sin(pi)) is exactly 0: certificate must refuse"
        );
        // Grounds bounded away from 0 still certify (no blanket refusal).
        assert!(
            zero_set_null(&s(&["np.pi"]), ops),
            "pi itself is rigorously nonzero"
        );
        assert!(
            zero_set_null(&s(&["+", "sin", "np.pi", "2"]), ops),
            "sin(pi) + 2 is rigorously bounded away from 0"
        );
        // The atom's value set really is a bracket containing the true constant.
        let v = value_set(&s(&["sin", "np.pi"]), ops, &Vs::reals()).unwrap();
        assert!(
            v.lo <= 0.0 && 0.0 <= v.hi,
            "the enclosure of sin(pi) must contain 0: {v:?}"
        );
    }

    /// A rule's verdict must not depend on WHICH dummy variable it is spelled with.
    ///
    /// The box has one dimension per DISTINCT variable, not per index slot. Sizing it by
    /// `max index + 1` gave `exp log x3` four dimensions, three never referenced and therefore
    /// never narrowed, so their full width multiplied into the witness: 1.34e8 instead of 64.
    #[test]
    fn witness_is_invariant_to_which_dummy_variable() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let w0 = domain_extension(&s(&["exp", "log", "x0"]), &s(&["x0"]), ops);
        assert_eq!(w0, Some(64.0));
        for v in ["x1", "x2", "x3", "_0", "_2"] {
            assert_eq!(
                domain_extension(&s(&["exp", "log", v]), &s(&[v]), ops),
                w0,
                "witness changed when the variable was spelled {v}"
            );
        }
    }

    /// The gate must decide on the MATHEMATICS, not on whether the source's nan region happens to
    /// contain a cube of the diagonal family. Every rule here is ~50% unsound; before product
    /// boxes the gate fired on `+` and `neg *` and was silent on `-`, `*`, `/`.
    #[test]
    fn domain_gate_fires_on_multivariable_nan_regions() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        for op in ["+", "-", "*", "/"] {
            let src = s(&["exp", "log", op, "x0", "x1"]);
            let tgt = s(&[op, "x0", "x1"]);
            assert!(
                domain_extension(&src, &tgt, ops).unwrap() > 0.0,
                "gate silent on the 50%-unsound `exp log {op} x0 x1`"
            );
        }
        // ... and the same rule plus a negation must not flip the verdict.
        assert!(
            domain_extension(
                &s(&["exp", "log", "neg", "*", "x0", "x1"]),
                &s(&["neg", "*", "x0", "x1"]),
                ops
            )
            .unwrap()
                > 0.0
        );
    }

    /// `outer(inner(x))` is a real identity iff `inner`'s RANGE lies inside `outer`'s DOMAIN.
    /// `log(exp x)` qualifies (exp's range (0,inf) is inside log's domain); `exp(log x)` does not
    /// (log is undefined on x<0, so rewriting to `x` INVENTS a function there).
    #[test]
    fn domain_gate_inverse_roundtrips() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // sound: inner range inside outer domain -> no witness
        for inner_outer in [["log", "exp"], ["atanh", "tanh"], ["asinh", "sinh"]] {
            let src = s(&[inner_outer[0], inner_outer[1], "x0"]);
            assert_eq!(
                domain_extension(&src, &s(&["x0"]), ops),
                Some(0.0),
                "{:?} is a real identity and must not be gated",
                src
            );
        }
        // unsound: the inner function's domain is restricted -> positive-measure witness
        for outer_inner in [
            ["exp", "log"],    // log undefined on x<0
            ["tanh", "atanh"], // atanh undefined off (-1,1)
            ["cos", "acos"],   // acos undefined off [-1,1]
            ["sin", "asin"],
            ["cosh", "acosh"], // acosh undefined on x<1
        ] {
            let src = s(&[outer_inner[0], outer_inner[1], "x0"]);
            assert!(
                domain_extension(&src, &s(&["x0"]), ops).unwrap() > 0.0,
                "{:?} -> x invents a function off the inner domain and must be gated",
                src
            );
        }
        // the even-root pair, spelled generation-2: sqrt undefined on x<0
        let src = s(&["pow", "rootn", "x0", "2", "2"]);
        assert!(
            domain_extension(&src, &s(&["x0"]), ops).unwrap() > 0.0,
            "pow(rootn(x,2),2) -> x invents a function off the inner domain and must be gated"
        );
    }

    /// A MEASURE-ZERO hole is a removable singularity and must stay allowed -- the subdivision
    /// refines it away and it can never accumulate width. This is what separates `x/x -> 1` (fill
    /// the point at 0) from `exp(log x) -> x` (invent the whole half-axis).
    #[test]
    fn domain_gate_allows_removable_singularity() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        assert_eq!(
            domain_extension(&s(&["/", "x0", "x0"]), &s(&["1"]), ops),
            Some(0.0)
        );
        // exp(log|x|) = |x| everywhere except the single point x=0
        assert_eq!(
            domain_extension(&s(&["exp", "log", "abs", "x0"]), &s(&["abs", "x0"]), ops),
            Some(0.0)
        );
    }

    /// F0: an UNANALYZABLE box must read UNDECIDED, never "no extension". `value_set_p`
    /// fails closed (`None`) on e.g. `rootn` with an expression-position index -- but
    /// `rec_ext` used to treat that `None` as "nothing to prove in this box" and return, so
    /// the subdivision COMPLETED with no witness and `domain_extension` reported the
    /// DECIDED `Some(0.0)`: operator-level fail-closed laundered into gate-level fail-open.
    /// That is exactly how the five `rootn <lit> tanh/cosh ?0` rules of the 2026-07-29
    /// acj-4-3 mine shipped (LHS NaN a.e. -- a total domain extension -- killed post hoc by
    /// `verify_ruleset` as bc-positive-measure). The 0.12 vocabulary comes from the staged
    /// acj-4-3 asset via `test_engine()` (skip-if-absent; set SIMPLIPY_TEST_REQUIRE_ASSETS
    /// in CI to make an absent asset fail loudly instead).
    #[test]
    fn unanalyzable_source_or_target_is_undecided_not_clean() {
        let Some(e) = crate::test_engine() else { return };
        let ops = e.operators_ref();
        // The minted poison class: source NaN a.e., target defined. UNDECIDED (the caller
        // fails closed), never the decided "no extension".
        assert_eq!(
            domain_extension(&s(&["rootn", "(-1)", "tanh", "x0"]), &s(&["(-1)"]), ops),
            None
        );
        assert_eq!(
            domain_extension(
                &s(&["rootn", "np.e", "tanh", "x0"]),
                &s(&["exp", "tanh", "x0"]),
                ops
            ),
            None
        );
        // The same laundering through the TARGET arm: analyzable NaN-a.e. source,
        // unanalyzable target.
        assert_eq!(
            domain_extension(
                &s(&["asin", "cosh", "x0"]),
                &s(&["rootn", "1", "cosh", "x0"]),
                ops
            ),
            None
        );
        // LITERAL-index rootn stays fully analyzable: a genuine positive-measure extension
        // is still PROVEN, not blanket-undecided ...
        assert!(
            domain_extension(&s(&["pow", "rootn", "x0", "2", "2"]), &s(&["x0"]), ops).unwrap()
                > 0.0
        );
        // ... and the golden removable singularity stays DECIDED-clean in this vocabulary.
        assert_eq!(
            domain_extension(&s(&["/", "x0", "x0"]), &s(&["1"]), ops),
            Some(0.0)
        );
    }

    /// A FREE `<constant>` cannot answer the domain question: the union over C contains both NaN
    /// and finite values almost everywhere, so the gate can never fire. Binding the constants to
    /// the fit's actual witness is what makes it decidable -- and the verdict then depends on the
    /// witness, which is the whole point (`+sqrt(2)^x` is total, `-sqrt(2)^x` is NaN a.e.).
    #[test]
    fn domain_gate_binds_constants() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let src = s(&["exp", "exp", "log", "x0"]); // = e^x on x>0, NaN on x<0
        let tgt = s(&["pow", "<constant>", "x0"]);
        // free constant: blind
        assert_eq!(domain_extension(&src, &tgt, ops), Some(0.0));
        // at the witness the fit actually returns (C = e), C^x is total -> caught
        assert!(domain_extension_p(&src, &[], &tgt, &[std::f64::consts::E], ops).unwrap() > 0.0);

        // sqrt(C^x) -> C'^x: the true power law. Matching-sign witnesses preserve the domain;
        // the mismatched one (C<0 but C'>0) invents a total function and is caught.
        let p = s(&["rootn", "pow", "<constant>", "x0", "2"]);
        let q = s(&["pow", "<constant>", "x0"]);
        assert_eq!(
            domain_extension_p(&p, &[2.0], &q, &[2f64.sqrt()], ops),
            Some(0.0)
        );
        assert_eq!(
            domain_extension_p(&p, &[-2.0], &q, &[-(2f64.sqrt())], ops),
            Some(0.0)
        );
        assert!(domain_extension_p(&p, &[-2.0], &q, &[2f64.sqrt()], ops).unwrap() > 0.0);
    }

    /// The POLE_GRID's reason for existing, decided structurally instead. `pow(asin(C/2), inf)`
    /// is 0 for |C| < 1.683, +inf on the band (1.683, 2), NaN past |C| > 2 -- a 0.317-wide band
    /// that falls BETWEEN the grid points 1.6 and 2.5, which a sampled sweep can miss
    /// seed-dependently, admitting `-> 0`. Reachability rejects it at every seed, no sampling.
    #[test]
    fn reachability_rejects_the_pole_band_artifact() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let src = s(&["pow", "asin", "/", "<constant>", "2", "float(\"inf\")"]);
        let sv = value_set(&src, ops, &Vs::reals()).expect("evaluable");
        // the source really does take BOTH 0 (|C|<1.683) and +inf (the band) on positive measure
        assert!(sv.has_fin, "source is 0 for |C| < 2*sin(1)");
        assert!(
            sv.pinf,
            "source is +inf on the band C in (1.683, 2) -- the grid straddles it"
        );

        // the mined artifact: a literal 0 cannot reach the +inf the source takes
        let zero = value_set(&s(&["0"]), ops, &Vs::reals()).unwrap();
        assert!(
            !reaches_all_of(&zero, &sv),
            "`-> 0` must be rejected: it cannot be +inf"
        );
        // ... nor can a fitted <constant>, which is finite by construction
        let c = value_set(&s(&["<constant>"]), ops, &Vs::reals()).unwrap();
        assert!(
            !reaches_all_of(&c, &sv),
            "`-> <constant>` must be rejected: a fit is finite"
        );
        // ... but the source itself trivially can (the gate is necessary, not sufficient)
        assert!(reaches_all_of(&sv, &sv));
    }

    /// Reachability is a ONE-WAY test: the candidate may reach MORE than the source (its extra
    /// components are unreachable at the constants the fit will choose), it may just never reach
    /// LESS. Getting the direction wrong would reject ordinary finite rules wholesale.
    #[test]
    fn reachability_is_one_way() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let fin = value_set(&s(&["<constant>"]), ops, &Vs::reals()).unwrap();
        // `pow(0, C)` = 0 for C>0, +inf for C<0: reaches strictly more than a finite source
        let wide = value_set(&s(&["pow", "0", "<constant>"]), ops, &Vs::reals()).unwrap();
        assert!(wide.has_fin && wide.pinf);
        assert!(
            reaches_all_of(&wide, &fin),
            "a wider candidate must stay allowed"
        );
        assert!(
            !reaches_all_of(&fin, &wide),
            "a narrower candidate must be rejected"
        );
    }

    /// Regressions found by sweeping the interval engine against brute-force numeric evaluation
    /// at full production scope (source lengths up to 7, multiple constants). Each was invisible
    /// at a narrower length<=4 / 1-constant validation scope, and each matters at full scale:
    /// the engine is the authority for the short-circuit, the domain gate and the reachability
    /// gate, so an under-reported component ships a defective rule.
    #[test]
    fn production_scope_regressions() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let comps = |t: &[&str]| {
            let v = value_set(&s(t), ops, &Vs::reals()).expect("evaluable");
            (v.has_fin, v.pinf, v.ninf, v.nan)
        };
        // IEEE: x^0 == 1 for EVERY x. `pow(nan, 0)` is 1.0 in numpy, not nan.
        assert_eq!(
            comps(&["pow", "float(\"nan\")", "0"]),
            (true, false, false, false)
        );
        assert_eq!(
            comps(&["pow", "float(\"inf\")", "0"]),
            (true, false, false, false)
        );

        // an inf BASE under an inf EXPONENT: numpy (+-inf)^inf = inf, (+-inf)^-inf = 0.
        // Both used to fall out EMPTY (an inf base has no abs_range to test against 1).
        assert_eq!(
            comps(&["pow", "float(\"inf\")", "float(\"inf\")"]),
            (false, true, false, false)
        );
        assert_eq!(
            comps(&["pow", "float(\"-inf\")", "float(\"inf\")"]),
            (false, true, false, false)
        );
        assert_eq!(
            comps(&["pow", "float(\"-inf\")", "float(\"-inf\")"]),
            (true, false, false, false)
        );

        // `pow(0, C)` is {0} u {+inf}, so log of it is -inf u +inf and NEVER finite. The finite
        // part is the degenerate point {0} AT log's domain edge -- `is_const` missed that because
        // the set also carries +inf, and log(0) came out as an "unbounded but finite" range.
        assert_eq!(
            comps(&["log", "pow", "0", "<constant>"]),
            (false, true, true, false)
        );

        // inf + (-inf) is nan, ONLY nan: the pairing matters, and `defined()` included the
        // opposite infinity. Over-reporting here would make the reachability gate demand +-inf of
        // the candidate and reject the true rule `inf + (-inf) -> nan`.
        assert_eq!(
            comps(&["+", "float(\"inf\")", "float(\"-inf\")"]),
            (false, false, false, true)
        );
        assert_eq!(
            comps(&["-", "float(\"inf\")", "float(\"inf\")"]),
            (false, false, false, true)
        );
        // ... while the ordinary infinite sums survive
        assert_eq!(
            comps(&["+", "float(\"inf\")", "float(\"inf\")"]),
            (false, true, false, false)
        );
        assert_eq!(
            comps(&["+", "float(\"inf\")", "<constant>"]),
            (false, true, false, false)
        );

        // const^const is evaluated exactly; the general branch reports the whole line, which
        // poisons every downstream op (`sqrt(pow(-1,-1))` is nan, not nan-or-finite).
        assert_eq!(
            comps(&[
                "rootn",
                "pow",
                "neg",
                "abs",
                "tanh",
                "float(\"inf\")",
                "(-1)",
                "2"
            ]),
            (false, false, false, true)
        );
    }

    /// `defined_measure_p` separates "NaN on part of the line" from "NaN almost everywhere":
    /// only the former makes a domain claim the gate should act on.
    /// FAIL-CLOSED: a subdivision that exhausts its node budget with no witness is UNDECIDED
    /// (`None`), never "no extension" -- past bound constants of a few hundred, the derived
    /// horizon deepens the tree beyond the budget, and reading exhaustion as an accept would
    /// silently pass every such decision. Same contract on the deadness side: an unresolved
    /// search must not read as proven-dead, or it widens the gate's accept exemption (the
    /// anti-conservative direction).
    #[test]
    fn budget_exhaustion_is_undecided_not_an_accept() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let src = s(&["log", "sin", "*", "<constant>", "x0"]);
        let n0 = NODE_BUDGET_MISSES.load(std::sync::atomic::Ordering::Relaxed);
        let v = domain_extension_p(&src, &[1.0e4], &s(&["x0"]), &[], ops);
        assert_eq!(v, None, "exhausted no-witness search must be undecided");
        assert!(
            NODE_BUDGET_MISSES.load(std::sync::atomic::Ordering::Relaxed) > n0,
            "the undecided verdict must be counted"
        );
        assert_eq!(
            defined_measure_p(&src, &[1.0e4], ops),
            None,
            "unresolved deadness must not read as proven-dead"
        );
        // the same shape at a small constant stays DECIDED (whichever way it decides)
        assert!(domain_extension_p(&src, &[2.0], &s(&["x0"]), &[], ops).is_some());
    }

    /// H-045 (D2' closing re-sample, 2026-08-05): a huge INTEGER literal leaf is a
    /// certified-exact point, not a 1-ulp bracket. The bracket at 1e19 (ulp = 2048)
    /// spans thousands of integers, so the `(-inf)^k` arm lost the H-029 single-integer
    /// honesty gate and ASSERTED Nan for a ground whose true value is +inf -- and the
    /// ground-classification fold shipped `float("nan")` from live simplify. Parity now
    /// reads by exact float mod: `k as i64` saturated at i64::MAX (odd) for k >= 2^63,
    /// which would have flipped the fold to the wrong INFINITY SIGN once the point
    /// certification landed.
    #[test]
    fn huge_integer_literal_exponents_classify_exactly() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // 10^19 (even, exactly representable): +inf in both live spellings.
        assert_eq!(
            value_class(&s(&["pow", "float(\"-inf\")", "1e19"]), ops),
            Some(Class::PosInf)
        );
        assert_eq!(
            value_class(&s(&["pow", "float(\"-inf\")", "10000000000000000000"]), ops),
            Some(Class::PosInf)
        );
        // Sub-2^53 sanity: the odd small integer keeps its sign.
        assert_eq!(
            value_class(&s(&["pow", "float(\"-inf\")", "3"]), ops),
            Some(Class::NegInf)
        );
        // A near-miss literal (denoted 10^19 + 100, NOT the f64 1e19 it rounds to) must
        // NOT certify: bracket kept, class refuses to a Nan-a.e. claim it cannot better.
        assert!(!exact_integer_literal("1.00000000000000001e19", 1e19));
        // INTERVAL-LAYER CONVENTION, documented (H-045-R CLOSED 2026-08-05, owner
        // Option B): a beyond-i128 integer literal cannot certify here, keeps the
        // bracket, and THIS layer's continuum convention still reads Nan for
        // `(-inf)^1e40` -- but the ENGINE's ground fold now classifies the shape
        // exactly from the spelling's sign and parity BEFORE consulting this class
        // (engine/ac.rs::h045_exact_pow_class; live simplify answers +inf, pinned
        // there). Vs point-support provenance stays the RECORDED design direction
        // if more point-derived-enclosure consumers appear (register H-045).
        assert_eq!(
            value_class(&s(&["pow", "float(\"-inf\")", "1e40"]), ops),
            Some(Class::Nan)
        );
    }

    /// The `!`-sort certificate: exp/sinh/polynomial compositions certify, the unsound binders
    /// never do, and the stated-scope exclusions (pole-bearing but truly finite-a.e.) fail CLOSED.
    #[test]
    fn finite_ae_certificate() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // certified: clean after 0-2 refinements
        for expr in [
            vec!["exp", "x0"],
            vec!["sinh", "x0"],
            vec!["pow", "x0", "2"],
            vec!["+", "x0", "x1"],
            vec!["tanh", "*", "x0", "x1"],
            vec!["3.0"],
            // pole-bearing but finite a.e.: certified by the STRUCTURAL path (the former
            // stated-scope exclusion, closed by the certificate algebra)
            vec!["inv", "x0"],
            vec!["tan", "x0"],
            vec!["/", "x0", "-", "x1", "cos", "x1"],
        ] {
            assert!(finite_ae(&s(&expr), ops), "{expr:?} must certify");
        }
        // refuted / fail-closed: never bindable by `!`
        for expr in [
            vec!["log", "x0"],                   // nan on half the line
            vec!["pow", "x0", "float(\"inf\")"], // a.e. in {0, inf}
            vec!["float(\"inf\")"],
            vec!["float(\"nan\")"],
            vec!["/", "x0", "-", "x1", "x1"], // x/0: nonfinite a.e.
            vec!["/", "x0", "*", "<constant>", "x1"], // C = 0 makes it nonfinite a.e.
        ] {
            assert!(!finite_ae(&s(&expr), ops), "{expr:?} must NOT certify");
        }
    }

    /// The certificate algebra: `zero_set_null` / `nonfinite_null` / `finite_nonzero_ae`
    /// / `positive_ae`, including the poison battery (identically-zero composites,
    /// abs-plateaus, `<constant>`-bearing trees).
    #[test]
    fn certificate_algebra() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let v = |expr: &[&str]| s(expr);

        // zero_set_null: certified
        for expr in [
            vec!["x0"],
            vec!["sin", "x0"],
            vec!["-", "x0", "cos", "x0"],
            vec!["cosh", "x0"],
            vec!["*", "x0", "x1"],
            vec!["inv", "x0"],
            vec!["/", "x0", "-", "x1", "cos", "x1"],
            vec!["tan", "x0"],
            vec!["log", "cosh", "x0"], // zero only at x = 0
        ] {
            assert!(
                zero_set_null(&v(&expr), ops),
                "{expr:?} must have null zero set"
            );
        }
        // zero_set_null: refused (poison battery)
        for expr in [
            vec!["0"],
            vec!["-", "x0", "x0"], // identically zero
            // sin^2 + cos^2 - 1: identically zero, no witness cell can exist
            vec![
                "-", "+", "pow", "sin", "x0", "2", "pow", "cos", "x0", "2", "1",
            ],
            vec!["+", "abs", "x0", "x0"], // plateau: zero on the whole half-line
            vec!["*", "<constant>", "x0"], // C = 0 zeroes it identically
            vec!["sin", "*", "<constant>", "x0"],
        ] {
            assert!(!zero_set_null(&v(&expr), ops), "{expr:?} must NOT certify");
        }

        // nonfinite_null: certified
        for expr in [
            vec!["inv", "x0"],
            vec!["tan", "x0"],
            vec!["/", "x0", "-", "x1", "cos", "x1"],
            vec!["log", "cosh", "x0"],
            vec!["rootn", "pow", "x0", "2", "2"], // sqrt(x^2): range proof [0, inf)
            vec!["+", "<constant>", "x0"],
        ] {
            assert!(
                nonfinite_null(&v(&expr), ops),
                "{expr:?} must be finite a.e."
            );
        }
        // nonfinite_null: refused
        for expr in [
            vec!["asin", "x0"], // region nan
            vec!["log", "x0"],  // nan on half the line
            vec!["rootn", "x0", "2"],
            vec!["atanh", "x0"],
            vec!["/", "x0", "-", "x1", "x1"],         // x/0
            vec!["/", "x0", "*", "<constant>", "x1"], // ∀C fails at C = 0
            vec!["/", "x0", "+", "abs", "x1", "x1"],  // plateau denominator: 0/0 on half-line
            // 1 / (sin^2 + cos^2 - 1): identically-zero denominator
            vec![
                "inv", "-", "+", "pow", "sin", "x0", "2", "pow", "cos", "x0", "2", "1",
            ],
            vec!["float(\"inf\")"],
        ] {
            assert!(!nonfinite_null(&v(&expr), ops), "{expr:?} must NOT certify");
        }

        // finite_nonzero_ae: the A/A -> 1 soundness domain
        for expr in [
            vec!["cosh", "x0"],
            vec!["sin", "x0"],
            vec!["x0"],
            vec!["inv", "x0"],
            vec!["-", "x0", "cos", "x0"],
        ] {
            assert!(finite_nonzero_ae(&v(&expr), ops), "{expr:?} must certify");
        }
        for expr in [
            vec!["0"],
            vec!["-", "x0", "x0"],
            vec!["+", "abs", "x0", "x0"],
            vec!["*", "<constant>", "x0"],
            vec!["<constant>"],
            vec!["asin", "x0"], // nonzero-a.e. but NOT finite a.e. -> refused
        ] {
            assert!(
                !finite_nonzero_ae(&v(&expr), ops),
                "{expr:?} must NOT certify"
            );
        }

        // positive_ae v1
        for expr in [
            vec!["exp", "x0"],
            vec!["cosh", "x0"],
            vec!["pow", "x0", "2"],
        ] {
            assert!(positive_ae(&v(&expr), ops), "{expr:?} must certify");
        }
        for expr in [
            vec!["x0"],
            vec!["sin", "x0"],
            vec!["pow", "-", "x0", "x0", "2"],
        ] {
            assert!(!positive_ae(&v(&expr), ops), "{expr:?} must NOT certify");
        }
    }

    /// 2026-08-03 division-tower campaign: the certificate algebra on the BINARY
    /// 0.12 vocabulary (`pow`/`rootn` arms) and the new infinite-set certificate,
    /// with the poison batteries that pin the sharpened arms to their exact sets.
    #[test]
    fn certificate_algebra_binary_vocabulary() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let v = |expr: &[&str]| s(expr);

        // nonfinite_null: the pow/rootn arms (all REFUSED before the completion)
        for expr in [
            vec!["pow", "x0", "2"],               // finite^int is finite
            vec!["rootn", "x0", "3"],             // odd root is total
            vec!["pow", "x0", "-2"],              // pole only on {x0 = 0}
            vec!["/", "pow", "x0", "2", "x1"],    // pole on {x1 = 0}
            vec!["rootn", "pow", "x0", "2", "2"], // even root of a square
            // the row-481 argument: pole only on the null {x1 = 0}
            vec![
                "sinh", "/", "*", "pow", "x0", "2", "+", "x2", "cos", "x2", "x1",
            ],
        ] {
            assert!(
                nonfinite_null(&v(&expr), ops),
                "{expr:?} must be finite a.e."
            );
        }
        for expr in [
            vec!["pow", "x0", "0.5"],                  // fat NaN on x0 < 0
            vec!["rootn", "x0", "2"],                  // fat NaN on x0 < 0
            vec!["pow", "x0", "x1"],                   // symbolic exponent
            vec!["pow", "-", "x0", "abs", "x0", "-1"], // fat zero set under a pole
        ] {
            assert!(!nonfinite_null(&v(&expr), ops), "{expr:?} must NOT certify");
        }

        // infinite_set_null: certified
        for expr in [
            vec!["x0"],
            vec!["<constant>"],
            vec!["float(\"nan\")"], // empty infinite set: NaN is not ±inf
            vec!["+", "x0", "pow", "x0", "2"],
            vec!["-", "x0", "acosh", "*", "2", "x0"], // fat NaN, never infinite
            vec!["-", "x0", "log", "x0"],             // log's infinities on {0, +inf}
            vec!["inv", "x0"],                        // +inf only on {x0 = 0}
            vec!["tan", "x0"],                        // poles on the null {cos = 0}
            vec!["sin", "inv", "-", "x0", "abs", "x0"], // bounded op: NEVER ±inf
            vec!["exp", "x0"],
            vec!["atanh", "*", "2", "x0"], // level sets {2x0 = ±1} are points
            vec!["rootn", "x0", "2"],      // sqrt: NaN fat, infinities inherited
        ] {
            assert!(
                infinite_set_null(&v(&expr), ops),
                "{expr:?} must have null infinite set"
            );
        }
        // infinite_set_null: poison battery
        for expr in [
            vec!["float(\"inf\")"],
            vec!["*", "float(\"inf\")", "x0"],
            vec!["inv", "-", "x0", "abs", "x0"], // +inf on the whole half-line
            vec!["log", "-", "x0", "abs", "x0"], // -inf on the fat plateau zero set
            vec!["exp", "inv", "-", "x0", "abs", "x0"], // +inf pushed through exp
            vec!["/", "x0", "-", "x1", "abs", "x1"], // fat-zero denominator
            vec!["inv", "*", "<constant>", "x0"], // C = 0: identically ±inf
        ] {
            assert!(
                !infinite_set_null(&v(&expr), ops),
                "{expr:?} must NOT certify"
            );
        }

        // zero_set_null, sharpened arms: reciprocals of never-infinite trees
        // (REFUSED before: nfn conflated the fat NaN domains with infinities)
        for expr in [
            vec!["inv", "-", "pow", "x0", "2", "x0"], // the P1 tower factor
            vec!["inv", "-", "x0", "acosh", "*", "2", "x0"],
            vec!["/", "x0", "-", "x1", "log", "x1"],
            vec!["inv", "tan", "x0"], // zero only where tan is infinite: null
            vec!["+", "pow", "x0", "3", "1"], // polynomial witness via binary pow
        ] {
            assert!(zero_set_null(&v(&expr), ops), "{expr:?} must certify");
        }
        // zero_set_null: sharpened arms must keep refusing the fat sets
        for expr in [
            vec!["inv", "inv", "-", "x0", "abs", "x0"], // zero on the fat half-line
            vec!["pow", "-", "x0", "x0", "2"],          // identically zero, no witness
            vec!["+", "abs", "x0", "x0"],               // plateau (unchanged refusal)
        ] {
            assert!(!zero_set_null(&v(&expr), ops), "{expr:?} must NOT certify");
        }
    }

    #[test]
    fn defined_measure_separates_generic_from_atoms() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // generic support: exp(log x) is a real function on x>0
        assert!(defined_measure_p(&s(&["exp", "log", "x0"]), &[], ops).unwrap() > 0.0);
        // measure-zero support: sqrt((-2)^x) is finite only on the even integers.
        // Some(0.0) = PROVEN dead (a completed search) -- None would be merely unresolved.
        assert_eq!(
            defined_measure_p(&s(&["rootn", "pow", "<constant>", "x0", "2"]), &[-2.0], ops),
            Some(0.0)
        );
        // ... but the same source at C>0 is total
        assert!(
            defined_measure_p(&s(&["rootn", "pow", "<constant>", "x0", "2"]), &[2.0], ops).unwrap()
                > 0.0
        );
    }
}

#[cfg(test)]
mod aligned_pow_interval_tests {
    use super::*;

    fn ninf_base() -> Vs {
        let mut v = Vs::empty();
        v.ninf = true;
        v
    }

    fn pt(x: f64) -> Vs {
        let mut v = Vs::empty();
        v.merge_pt(x);
        v
    }

    fn range(lo: f64, hi: f64) -> Vs {
        let mut v = Vs::empty();
        v.has_fin = true;
        v.lo = lo;
        v.hi = hi;
        v
    }

    /// The aligned -inf-base rows: nan for non-integer / continuum exponents, parity for
    /// const integers, magnitude-step for infinite exponents; +inf base unchanged.
    #[test]
    fn neg_inf_base_rows() {
        let r = b_pow(&ninf_base(), &pt(0.25));
        assert!(r.nan && !r.pinf && !r.ninf && !r.has_fin);
        let r = b_pow(&ninf_base(), &pt(2.0));
        assert!(r.pinf && !r.nan && !r.ninf);
        let r = b_pow(&ninf_base(), &pt(3.0));
        assert!(r.ninf && !r.nan && !r.pinf);
        let r = b_pow(&ninf_base(), &pt(-2.0));
        assert!(r.has_fin && r.lo == 0.0 && r.hi == 0.0 && !r.nan);
        let r = b_pow(&ninf_base(), &range(0.1, 5.0));
        assert!(r.nan && !r.pinf && !r.ninf);
        // magnitude-step at infinite exponents (ratified): |t| > 1
        let mut binf = Vs::empty();
        binf.pinf = true;
        let r = b_pow(&ninf_base(), &binf);
        assert!(r.pinf && !r.nan);
        let mut bninf = Vs::empty();
        bninf.ninf = true;
        let r = b_pow(&ninf_base(), &bninf);
        assert!(r.has_fin && r.lo == 0.0 && r.hi == 0.0);
        // +inf base: unchanged convention
        let mut pinf_base = Vs::empty();
        pinf_base.pinf = true;
        let r = b_pow(&pinf_base, &pt(0.25));
        assert!(r.pinf && !r.nan);
    }
}
