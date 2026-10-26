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

// ===== f64 SATURATION THRESHOLDS =============================================
// DERIVED, never declared: the image-derivation probe probes the deployed evaluator
// and bisects to the ulp. These are the points where the f64 implementation ATTAINS a
// value the idealized function only approaches -- so an interval that reaches them has a
// CLOSED bound, not the open one mathematics prescribes.
//
// Modelling the deployed evaluator rather than the textbook function is the point: rules
// certified against exact-real images are applied by an f64 evaluator, and every place the
// two differ is a place a certified rule can be false in production.
//
// HOST-DEPENDENT by nature (libm differs; D20 already routes minting to the reference mint host for that
// reason). The deployed evaluator IS the engine's (`simplipy.operators` calls `numeric::kernels`),
// so one measurement serves both. Derived on glibc 2.43 and pinned against the kernels by
// `saturation_thresholds_are_the_deployed_kernels`; re-derive per host and stamp with the libm.
/// Smallest x with `tanh(x) == 1.0` exactly; just below it tanh gives 0.9999999999999999.
const TANH_SAT: f64 = 19.061547465398498;
/// Largest x with `exp(x) == 0.0` exactly; just above it exp gives 5e-324, the smallest
/// subnormal.
const EXP_UNDERFLOW: f64 = -745.1332191019412;
/// Smallest x with `atan(x) == fl(pi/2)` exactly (the deployed evaluator and the engine's agree).
const ATAN_SAT: f64 = 5805358775541310.0;
/// Smallest x with `erf(x) == 1.0` exactly; just below it erf gives 0.9999999999999999.
const ERF_SAT: f64 = 5.921587195794507;

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
    /// The `lo` / `hi` bound is CLOSED **and attained on a POSITIVE-MEASURE x-set**, because a
    /// modelled f64 SATURATION pins a whole ray of the input to it (`attain_pt`): `exp` is
    /// exactly +0 on the entire ray below `EXP_UNDERFLOW`, `tanh` exactly +-1 past +-`TANH_SAT`.
    ///
    /// This is the bit that separates the two shapes a CLOSED bound can have, which are
    /// otherwise identical in this struct:
    ///   * `cosh(exp C)`'s `lo = 1` -- attained on the whole ray `C <= EXP_UNDERFLOW`; and
    ///   * `cosh(inv C)`'s `lo = 1` -- the mathematical minimum, attained (in exact arithmetic)
    ///     only where `inv C = 0`, which is a NULL set and in fact never.
    ///
    /// A domain edge grazed by the first is real support; grazed by the second it is an
    /// over-approximation artifact. PROVEN-only: `false` is the conservative default and
    /// reproduces the pre-flag behaviour everywhere, so under-propagation is fail-CLOSED.
    pub sat_lo: bool,
    pub sat_hi: bool,
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
            sat_lo: false,
            sat_hi: false,
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
            sat_lo: false,
            sat_hi: false,
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
            sat_lo: false,
            sat_hi: false,
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
    /// Mark `v` as ATTAINED. `merge_pt` only WIDENS, so it cannot close a bound that
    /// already sits at `v` but was recorded open -- and that is exactly the case the f64
    /// saturations produce (`tanh`'s hi is already 1.0, open, when f64 attains 1.0).
    fn attain_pt(&mut self, v: f64) {
        // `merge_pt` clears `fin_null` -- conservative for a point of UNKNOWN support, but
        // wrong here: attaining a value does not change WHERE the finite part is supported,
        // only which value that support maps to. `mono` has already propagated the support
        // character from the operand, so a PROVEN-null finite part must survive.
        //
        // Dropping it collapses the positive-measure / measure-zero separation this layer
        // exists to maintain, and the domain gate reads exactly that predicate: with the
        // marking lost, `exp(log x)` on [-64, 0] reported `defined_pm` and suppressed its
        // own witness, halving the measured extension of `exp log x0 -> x0` from 64 to 32.
        // Suppressed witnesses are unsound rules NOT rejected, so the failure was on the
        // fail-open side.
        let was_null = self.has_fin && self.fin_null;
        self.merge_pt(v);
        if was_null {
            self.fin_null = true;
        }
        if self.has_fin {
            if self.lo == v {
                self.lo_open = false;
                // The saturation attains `v` on a whole RAY of the operand, not at an
                // isolated point: record it, so a later domain graze at this bound can be
                // told apart from the mathematical-extremum graze that only LOOKS the same.
                self.sat_lo = true;
            }
            if self.hi == v {
                self.hi_open = false;
                self.sat_hi = true;
            }
        }
    }

    /// Is `t` attained on a POSITIVE-MEASURE x-set? True only where PROVEN (`attain_pt`
    /// provenance survived to this bound); `false` means "not proven", never "proven not".
    fn attains_pm(&self, t: f64) -> bool {
        self.has_fin
            && ((self.lo == t && !self.lo_open && self.sat_lo)
                || (self.hi == t && !self.hi_open && self.sat_hi))
    }

    /// Does this value set reach `t` or beyond, on the finite side?
    fn reaches_up_to(&self, t: f64) -> bool {
        self.pinf || (self.has_fin && (self.hi > t || (self.hi == t && !self.hi_open)))
    }

    fn reaches_down_to(&self, t: f64) -> bool {
        self.ninf || (self.has_fin && (self.lo < t || (self.lo == t && !self.lo_open)))
    }

    fn merge_pt(&mut self, v: f64) {
        self.fin_null = false; // a merged point has unknown x-support: conservative
        if !self.has_fin {
            self.has_fin = true;
            self.lo = v;
            self.hi = v;
            self.lo_open = false;
            self.hi_open = false;
            self.sat_lo = false;
            self.sat_hi = false;
        } else {
            // Only a bound the merge MOVES loses its provenance -- the merged point has
            // unknown support, so the new bound is unproven. A bound left where it was keeps
            // what it had proven (`u_tanh` attains +1 and then -1: the second call must not
            // erase the first).
            if v < self.lo {
                self.lo = v;
                self.lo_open = false;
                self.sat_lo = false;
            }
            if v > self.hi {
                self.hi = v;
                self.hi_open = false;
                self.sat_hi = false;
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

/// The float below `x` (`-inf` and NaN unchanged; below 0 is the negative minimum subnormal).
pub(crate) fn step_down(x: f64) -> f64 {
    next_down(x)
}

/// The float above `x` (`inf` and NaN unchanged).
pub(crate) fn step_up(x: f64) -> f64 {
    next_up(x)
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
            let mut kept = Vs::interval(c, c, false, false);
            kept.sat_lo = v.sat_lo;
            kept.sat_hi = v.sat_hi;
            return (Some(kept), false);
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
        //
        // NULL-supported is the DEFAULT, not the verdict. It is right for the artifact shape
        // above and for a mathematical extremum that grazes the edge (`cosh(inv C)`'s `lo = 1`
        // is reached only where `inv C = 0`, i.e. nowhere). It is WRONG when the operand
        // SATURATES onto the edge: `cosh(exp C) = 1` on the entire ray `C <= EXP_UNDERFLOW`,
        // so `acos(cosh(exp C))` really is 0 on positive measure (measured: 2005 of 4177
        // samples), and calling that null is a finite part silently deleted. `sat_lo`/`sat_hi`
        // carry exactly that proof; absent it we keep the conservative null marking.
        let pm = v.attains_pm(lo);
        let mut kept = Vs::interval(lo, hi, false, false);
        kept.fin_null = !pm;
        kept.sat_lo = pm;
        kept.sat_hi = pm;
        return (Some(kept), true);
    }
    let escaped = v.lo < dlo || v.hi > dhi;
    let lo_open = if v.lo >= dlo { v.lo_open } else { false };
    let hi_open = if v.hi <= dhi { v.hi_open } else { false };
    let mut kept = Vs::interval(lo, hi, lo_open, hi_open);
    // A bound the clip did not move keeps its provenance; a clipped one is a domain edge,
    // with no claim to attainment.
    kept.sat_lo = v.sat_lo && lo == v.lo;
    kept.sat_hi = v.sat_hi && hi == v.hi;
    (Some(kept), escaped)
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
    // ATTAINED POLE (not an overflow). The module's standing doctrine -- "a float OVERFLOW
    // means the value is finite but unrepresentable -> an unbounded EDGE, never an inf value"
    // -- is about `safe()`, where `f` of a REPRESENTABLE argument leaves f64's range. It says
    // nothing about this case, which is the opposite one: the argument itself lands exactly on
    // a POLE of `f`, where the true value IS infinite and the evaluator returns that infinity.
    //
    // The pole is normally measure-zero (a continuum sweeping through `atanh`'s +-1 hits it on
    // a null set) and the EDGE encoding is then right: unbounded range, no inf component. But
    // when the operand SATURATES onto the domain edge, the whole clipped overlap degenerates to
    // that one point AND it is reached on positive measure -- `cosh(exp C)` is exactly 1 on the
    // ray `C <= EXP_UNDERFLOW`, so `atanh(cosh(exp C))` is +inf on 2005 of 4177 sampled C and
    // FINITE on none of them. Reporting an unbounded finite EDGE there claims a finite value the
    // evaluator never produces; the honest components are `pinf`/`ninf` with `has_fin = false`.
    //
    // Gated on the PROOF (`!fin_null`, set by `clip` only from `attain_pt` provenance), so an
    // unproven graze keeps the previous EDGE encoding untouched.
    if let Some(cl) = clipped.as_ref() {
        if cl.lo == cl.hi && !cl.fin_null {
            let pole = if cl.lo == dlo {
                inf_at_lo
            } else if cl.hi == dhi {
                inf_at_hi
            } else {
                None
            };
            if let Some(sign) = pole {
                if sign > 0.0 {
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
        // `sat` rides along: only an EXACT image of a proven-attained bound stays proven.
        let (mut a, mut a_open, mut a_computed, mut a_sat) = if pole_lo && cl.lo == dlo {
            (-INF, true, false, false)
        } else if cl.lo.is_infinite() {
            (lo_lim.unwrap_or(-INF), true, false, false)
        } else {
            (safe(&f, cl.lo), cl.lo_open, true, cl.sat_lo)
        };
        let (mut b, mut b_open, mut b_computed, mut b_sat) = if pole_hi && cl.hi == dhi {
            (INF, true, false, false)
        } else if cl.hi.is_infinite() {
            (hi_lim.unwrap_or(INF), true, false, false)
        } else {
            (safe(&f, cl.hi), cl.hi_open, true, cl.sat_hi)
        };
        if a.is_infinite() {
            a_open = true;
            a_sat = false;
        }
        if b.is_infinite() {
            b_open = true;
            b_sat = false;
        }
        if !inc {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut a_open, &mut b_open);
            std::mem::swap(&mut a_computed, &mut b_computed);
            std::mem::swap(&mut a_sat, &mut b_sat);
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
            a_sat = false; // a stepped endpoint is no longer the attained value
        }
        if b_computed && !exact_hit(src_b, b) {
            b = out_hi(b, ULP_LIBM);
            b_sat = false;
        }
        r.has_fin = true;
        r.lo = a;
        r.hi = b;
        r.lo_open = a_open;
        r.hi_open = b_open;
        r.sat_lo = a_sat && !a_open;
        r.sat_hi = b_sat && !b_open;
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
        r.sat_lo = v.sat_hi;
        r.sat_hi = v.sat_lo;
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
        r.sat_lo = false;
    }
    r.nan = v.nan;
    if v.pinf {
        r.pinf = true;
    }
    // f64 UNDERFLOWS to exactly +0 below EXP_UNDERFLOW (and at -inf). The clamp above
    // restored the mathematical OPEN floor; the evaluator attains it, so close it.
    if v.reaches_down_to(EXP_UNDERFLOW) {
        r.attain_pt(0.0);
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
        r.sat_lo = false;
    }
    if r.has_fin && r.hi >= 1.0 {
        r.hi = 1.0;
        r.hi_open = true;
        r.sat_hi = false;
    }
    // f64 ATTAINS +-1 past +-TANH_SAT (and at +-inf). `merge_pt` cannot express that --
    // hi is already 1.0 and open -- so the attainment must close the bound explicitly.
    if v.reaches_up_to(TANH_SAT) {
        r.attain_pt(1.0);
    }
    if v.reaches_down_to(-TANH_SAT) {
        r.attain_pt(-1.0);
    }
    r
}

fn u_atanh(v: &Vs) -> Vs {
    // domain (-1,1): nan for |x|>1 (REGION); +-inf at x=+-1 (measure-zero POINTS)
    // Endpoint images through the LIBM atanh (the fold's function), NOT Rust std's own
    // composition: their gap grows unbounded in ulps toward +-1 (161 ulps at -0.999,
    // 4th-significant-digit at 1 - 1e-15), and the 8-ulp outward budget only covers
    // libm's rounding -- with std here the box EXCLUDED the fold's value (audit B6).
    let mut r = domain_op(
        v,
        -1.0,
        1.0,
        crate::numeric::libm_atanh,
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
            // ...but attained on POSITIVE MEASURE only if the operand attains 0 there.
            // `exp C` does (saturation ray); `inv C` merely has 0 inside its over-approximated
            // range and in truth never reaches it.
            r.sat_lo = v.attains_pm(0.0);
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

// ================= THE 0.15 HEADS ============================================
// Simple enclosures, each sound for the deployed kernel (`numeric::kernels`) and for the real
// function. Where one is not simple, the arm refuses (`None`), and every consumer fails closed.
//
// The PLATEAU heads (`floor`, `ceil`, `max`, `min`, `mod`) attain their bounds on whole
// intervals, which the provenance bits (`sat_lo`/`sat_hi`) cannot record. A non-point image is
// therefore widened one ulp past its bounds, so that a domain edge downstream never grazes one
// (`clip` reads an unproven graze as a NULL point, which is false for a plateau). A point image
// is exact: the head is constant on the operand's whole support.

/// `r` with its finite part set to the box `[lo, hi]` (a plateau image, see above).
fn with_plateau(mut r: Vs, lo: f64, hi: f64, fin_null: bool) -> Vs {
    let (lo, hi) = if lo == hi {
        (lo, hi)
    } else {
        (out_lo(lo, ULP_ARITH), out_hi(hi, ULP_ARITH))
    };
    r.has_fin = true;
    r.lo = lo;
    r.hi = hi;
    r.lo_open = lo == -INF;
    r.hi_open = hi == INF;
    r.sat_lo = false;
    r.sat_hi = false;
    r.fin_null = fin_null;
    r
}

/// `floor` / `ceil`: monotone, exact, finite on finite values; +-inf and NaN pass through.
fn u_step(v: &Vs, f: fn(f64) -> f64) -> Vs {
    let mut r = Vs::empty();
    r.nan = v.nan;
    r.pinf = v.pinf;
    r.ninf = v.ninf;
    if v.has_fin {
        r = with_plateau(r, f(v.lo), f(v.hi), v.fin_null);
    }
    r
}

/// `erf`: `tanh`'s shape -- increasing, within (-1, 1), with float64 attaining +-1 past
/// +-`ERF_SAT`.
fn u_erf(v: &Vs) -> Vs {
    let mut r = mono(v, crate::numeric::kernels::erf, Some(-1.0), Some(1.0));
    r.nan = v.nan;
    if r.has_fin && r.lo <= -1.0 {
        r.lo = -1.0;
        r.lo_open = true;
        r.sat_lo = false;
    }
    if r.has_fin && r.hi >= 1.0 {
        r.hi = 1.0;
        r.hi_open = true;
        r.sat_hi = false;
    }
    if v.reaches_up_to(ERF_SAT) {
        r.attain_pt(1.0);
    }
    if v.reaches_down_to(-ERF_SAT) {
        r.attain_pt(-1.0);
    }
    r
}

/// `harmonic`: increasing on (-1, inf) and between any two of its poles, +inf at +inf, NaN at
/// -inf and at every negative integer (D6). A finite operand on (-1, inf), the point of a
/// negative integer, or a box strictly between two poles is enclosed; one that may reach a pole
/// among other values is refused.
fn u_harmonic(v: &Vs) -> Option<Vs> {
    let mut r = Vs::empty();
    r.nan = v.nan || v.ninf;
    r.pinf = v.pinf;
    if v.has_fin && v.lo <= -1.0 {
        if v.is_const() && v.lo.fract() == 0.0 {
            r.nan = true; // a pole: the one-sided limits differ
            return Some(r);
        }
        // Strictly between the poles -n-1 and -n (n >= 1), where H increases. The kernel's
        // error there is absolute (`numeric::harmonic`): at most ulp(H) + 2^-58 max(|H(-x-1)|,
        // |pi cot(pi x)|), where 0 <= H(-x-1) <= 1 + ln(-x) and |pi cot(pi x)| <= H(-x-1) + |H(x)|.
        let k = v.lo.floor();
        if !(v.lo > k && v.hi < k + 1.0 && v.hi < -1.0 && v.lo.is_finite()) {
            return None;
        }
        let h = crate::numeric::kernels::harmonic;
        let (a, b) = (h(v.lo), h(v.hi));
        if !(a.is_finite() && b.is_finite()) {
            return None;
        }
        let m = a.abs().max(b.abs());
        let reflected = 1.0 + (-v.lo).ln() + 1.0;
        let margin = 2.0 * m * f64::EPSILON + (2.0 * reflected + 2.0 * m) * 2f64.powi(-55);
        r.has_fin = true;
        r.lo = out_lo(a - margin, ULP_LIBM);
        r.hi = out_hi(b + margin, ULP_LIBM);
        r.fin_null = v.fin_null;
        return Some(r);
    }
    if v.has_fin {
        let h = crate::numeric::kernels::harmonic;
        r.has_fin = true;
        r.lo = out_lo(h(v.lo), ULP_LIBM);
        r.hi = if v.hi == INF {
            INF
        } else {
            out_hi(h(v.hi), ULP_LIBM)
        };
        r.lo_open = false;
        r.hi_open = r.hi == INF;
        r.fin_null = v.fin_null;
    }
    Some(r)
}

/// `max` / `min`: monotone in both operands, NaN-propagating. The finite part is the lattice
/// of the two finite boxes, plus either operand's box against the infinity it absorbs
/// (`max(a, -inf) = a`).
fn b_lattice(a: &Vs, b: &Vs, is_max: bool) -> Vs {
    let mut r = Vs::empty();
    r.nan = a.nan || b.nan;
    let def = |v: &Vs| v.has_fin || v.pinf || v.ninf;
    let (top_a, top_b, bot_a, bot_b) = if is_max {
        (a.pinf, b.pinf, a.ninf, b.ninf)
    } else {
        (a.ninf, b.ninf, a.pinf, b.pinf)
    };
    // the infinity that wins against anything defined, and the one that survives only
    // against itself
    let wins = (top_a && def(b)) || (top_b && def(a));
    let loses = bot_a && bot_b;
    if is_max {
        r.pinf = wins;
        r.ninf = loses;
    } else {
        r.ninf = wins;
        r.pinf = loses;
    }
    let (mut lo, mut hi, mut any) = (INF, -INF, false);
    if a.has_fin && b.has_fin {
        if is_max {
            lo = a.lo.max(b.lo);
            hi = a.hi.max(b.hi);
        } else {
            lo = a.lo.min(b.lo);
            hi = a.hi.min(b.hi);
        }
        any = true;
    }
    for (v, absorbed) in [(a, bot_b), (b, bot_a)] {
        if v.has_fin && absorbed {
            lo = lo.min(v.lo);
            hi = hi.max(v.hi);
            any = true;
        }
    }
    if any {
        r = with_plateau(r, lo, hi, false);
    }
    r
}

/// `atan2(y, x)`: every value lies in [-pi, pi]. Two nonzero points are evaluated (the sign
/// of a zero decides the branch cut, and a point box does not carry it).
fn b_atan2(y: &Vs, x: &Vs) -> Vs {
    let mut r = Vs::empty();
    r.nan = y.nan || x.nan;
    let def = |v: &Vs| v.has_fin || v.pinf || v.ninf;
    if !(def(y) && def(x)) {
        return r;
    }
    r.has_fin = true;
    // On the positive x axis, and at x = +inf for a finite y, the angle is a zero (of y's sign).
    let y_finite = y.has_fin && !y.pinf && !y.ninf;
    let x_positive = !x.ninf && (x.has_fin || x.pinf) && (!x.has_fin || x.lo > 0.0);
    if y_finite && ((y.lo == 0.0 && y.hi == 0.0 && x_positive) || (x.pinf && !x.has_fin && !x.ninf))
    {
        r.lo = 0.0;
        r.hi = 0.0;
        return r;
    }
    // On the y axis (x = 0 of either sign), at a finite y certainly nonzero: +-pi/2 by y's sign.
    if y_finite && x.is_const() && x.lo == 0.0 && (y.lo > 0.0 || y.hi < 0.0) {
        let half = std::f64::consts::FRAC_PI_2;
        let (lo, hi) = (out_lo(half, ULP_LIBM), out_hi(half, ULP_LIBM));
        (r.lo, r.hi) = if y.lo > 0.0 { (lo, hi) } else { (-hi, -lo) };
        return r;
    }
    if y.is_const() && x.is_const() && y.lo != 0.0 && x.lo != 0.0 && !r.nan {
        let t = crate::numeric::kernels::atan2(y.lo, x.lo);
        r.lo = out_lo(t, ULP_LIBM);
        r.hi = out_hi(t, ULP_LIBM);
    } else {
        // fl(pi) lies below pi: the bound is stepped outward
        let p = out_hi(std::f64::consts::PI, ULP_LIBM);
        r.lo = -p;
        r.hi = p;
    }
    r
}

/// `mod(a, b)` (numpy's: the sign of the divisor). Two points are the kernel's value, one ulp
/// wide (the remainder is exact, its adjustment one rounded addition). Otherwise the value
/// lies between 0 and the divisor, both closed (a tiny negative dividend gives `b` itself),
/// and a zero divisor is NaN. A divisor that may be zero among other values is refused.
///
/// The infinite cells are numpy's (D7): an infinite dividend is NaN; under an infinite divisor
/// the dividend is its own remainder where it is zero or has the divisor's sign, and the
/// divisor's infinity otherwise (`mod(3, inf) = 3`, `mod(-3, inf) = inf`, `mod(3, -inf) = -inf`,
/// `mod(-3, -inf) = -3`).
fn b_mod(a: &Vs, b: &Vs) -> Option<Vs> {
    let mut r = Vs::empty();
    r.nan = a.nan || b.nan || ((a.pinf || a.ninf) && b.defined());
    // The finite remainders, as boxes (lo, hi), joined at the end.
    let mut boxes: Vec<(f64, f64)> = Vec::new();
    if a.has_fin && b.pinf {
        if a.hi >= 0.0 {
            boxes.push((a.lo.max(0.0), a.hi));
        }
        r.pinf |= a.lo < 0.0;
    }
    if a.has_fin && b.ninf {
        if a.lo <= 0.0 {
            boxes.push((a.lo, a.hi.min(0.0)));
        }
        r.ninf |= a.hi > 0.0;
    }
    if a.has_fin && b.has_fin {
        if b.lo == 0.0 && b.hi == 0.0 {
            r.nan = true;
        } else if b.lo <= 0.0 && 0.0 <= b.hi {
            return None;
        } else if a.is_const() && b.is_const() {
            let m = crate::numeric::kernels::modulo(a.lo, b.lo);
            boxes.push((out_lo(m, ULP_ARITH), out_hi(m, ULP_ARITH)));
        } else {
            let (lo, hi) = if b.lo > 0.0 { (0.0, b.hi) } else { (b.lo, 0.0) };
            let w = with_plateau(Vs::empty(), lo, hi, false);
            boxes.push((w.lo, w.hi));
        }
    }
    if let Some(lo) = boxes.iter().map(|b| b.0).reduce(f64::min) {
        let hi = boxes.iter().map(|b| b.1).fold(lo, f64::max);
        r.has_fin = true;
        r.lo = lo;
        r.hi = hi;
        r.lo_open = lo == -INF;
        r.hi_open = hi == INF;
    }
    Some(r)
}

/// `besselj(nu, x)`, at every real order.
///
/// * An order nu >= 0, or a negative integer one (`J_{-n} = (-1)^n J_n`, which the kernel reads):
///   |J_nu(x)| <= 1 at every real x, so the box is [-1, 1], widened one ulp (J_0 is exactly 1.0
///   on |x| < 1e-8 in float64, a plateau).
/// * An order that is no integer on an argument below 0: NaN alone; at the point x = 0, a
///   negative such order's pole, the infinity of the sign of 1/Gamma(nu + 1).
/// * An order that may be negative and no integer: unbounded near x = 0 (a pole) and NaN below,
///   so an argument that may reach 0 among other values is refused; at x >= x_lo > 0 the recurrence
///   `J_nu = (2 (nu + 1)/x) J_{nu+1} - J_{nu+2}` bounds |J_nu| from the orders nu + j >= 0
///   (`negative_order_bound`), doubled for the kernel's error, refused where it nears the float
///   range's end (the kernel overflows there).
///
/// NaN at +-inf (scipy's cell), under an order that may be a non-integer at a negative argument,
/// under an order beyond `BESSELJ_NU_MEASURED` in magnitude, where the kernel's own routines give
/// up (it is NaN at order 3.8e16, x = 2.05), and under a negative non-integer order beyond
/// `BESSELJ_NEG_NU_MEASURED` (NaN at order -470.8, x = 78.7, where the value nears the float
/// range's end). A NaN order is NaN, an infinite one 0 at x >= 0 and NaN below (scipy's cells);
/// a range of orders holding both finite and infinite ones is refused.
/// The orders up to which the besselj kernel is read as NaN-free at every finite argument
/// (measured: no NaN at orders up to 1.1e9 over 3,200 arguments out to the largest float, so
/// this keeps a wide margin), integer orders of both signs alike.
const BESSELJ_NU_MEASURED: f64 = 1e3;
/// The magnitude of a negative non-integer order up to which the kernel is read as NaN-free at
/// every finite x > 0 (measured: the first NaN at -470.8 over 5,600 orders and 1,100 arguments).
const BESSELJ_NEG_NU_MEASURED: f64 = 400.0;

fn b_besselj(nu: &Vs, x: &Vs) -> Option<Vs> {
    if !nu.has_fin {
        // A NaN order gives NaN; an infinite one 0 at x >= 0 and NaN below (scipy's cells).
        let mut r = Vs::empty();
        r.nan = nu.nan
            || ((nu.pinf || nu.ninf) && (x.nan || x.pinf || x.ninf || (x.has_fin && x.lo < 0.0)));
        if (nu.pinf || nu.ninf) && x.has_fin && x.hi >= 0.0 {
            r.has_fin = true;
            r.lo = 0.0;
            r.hi = 0.0;
        }
        return Some(r);
    }
    if nu.pinf || nu.ninf {
        return None;
    }
    let integer_order = nu.is_const() && nu.lo.fract() == 0.0;
    // May the order be an integer: a point that is one, or a range that holds one?
    let may_be_integer = integer_order
        || (!nu.is_const() && (nu.lo.floor() != nu.hi.floor() || nu.lo.fract() == 0.0));
    let mut r = Vs::empty();
    r.nan = nu.nan
        || x.nan
        || x.pinf
        || x.ninf
        || (!integer_order && x.has_fin && x.lo < 0.0)
        || nu.lo.abs().max(nu.hi.abs()) > BESSELJ_NU_MEASURED;
    if !x.has_fin {
        return Some(r);
    }
    if !may_be_integer && x.hi < 0.0 {
        return Some(r); // a non-integer order on the negative axis: NaN alone
    }
    if !may_be_integer && nu.hi < 0.0 && x.lo == 0.0 && x.hi == 0.0 {
        // the pole at x = 0 of an order in (-k-1, -k): the infinity of the sign of
        // 1/Gamma(nu + 1), (-1)^k (nu + 1 lies in (-k, -k+1))
        if (-nu.hi).floor() % 2.0 == 0.0 {
            r.pinf = true;
        } else {
            r.ninf = true;
        }
        return Some(r);
    }
    // At the point x = 0: J_0(0) = 1, and J_nu(0) = 0 for nu > 0 and at a negative integer nu
    // (the kernel writes a zero of either sign, which the point holds).
    if x.is_const() && x.lo == 0.0 && (nu.lo > 0.0 || integer_order) {
        r.has_fin = true;
        let one = nu.is_const() && nu.lo == 0.0;
        r.lo = if one { 1.0 } else { 0.0 };
        r.hi = r.lo;
        return Some(r);
    }
    if integer_order {
        // An integer order on 0 < |x| <= 2: the alternating series' first two partial sums;
        // elsewhere |J_n| <= 1/sqrt(2) at every n != 0 (Abramowitz and Stegun 9.1.60).
        let (lo, hi) = besselj_integer_small_argument(nu.lo, x).unwrap_or(if nu.lo == 0.0 {
            (-1.0, 1.0)
        } else {
            (
                -std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            )
        });
        r.has_fin = true;
        r.lo = out_lo(lo, ULP_ARITH);
        r.hi = out_hi(hi, ULP_ARITH);
        return Some(r);
    }
    if nu.lo >= 0.0 {
        r.has_fin = true;
        r.lo = out_lo(-1.0, ULP_ARITH);
        r.hi = out_hi(1.0, ULP_ARITH);
        return Some(r);
    }
    if x.lo <= 0.0 {
        return None;
    }
    r.nan |= nu.lo < -BESSELJ_NEG_NU_MEASURED;
    let b = negative_order_bound(nu.lo, nu.hi, x.lo)?;
    r.has_fin = true;
    r.lo = -b;
    r.hi = b;
    Some(r)
}

/// A bound on |J_nu(x)| over every order nu in [lo, hi] (lo < 0) and every x >= `x` > 0: with
/// B(j) bounding |J_{nu+j}|, B(j) = 1 once lo + j >= 0 (|J_mu| <= 1 for mu >= 0), and below,
/// by the recurrence, B(j) = 2 max|nu + j + 1| / x B(j+1) + B(j+2), each step rounded up. Every
/// B(j) falls as x grows, so the bound at the least x holds on the whole range. Doubled for the
/// kernel's error; `None` where it nears the float range's end, or past 1,000 steps.
fn negative_order_bound(lo: f64, hi: f64, x: f64) -> Option<f64> {
    let m = (-lo).ceil();
    if m.is_nan() || m > 1000.0 {
        return None;
    }
    let up = 1.0 + 4.0 * f64::EPSILON;
    let (mut b, mut b_next) = (1.0f64, 1.0f64);
    for j in (0..m as usize).rev() {
        let j = j as f64;
        let s = (lo + j + 1.0).abs().max((hi + j + 1.0).abs());
        let bj = (2.0 * s / x * up * b * up + b_next) * up;
        b_next = b;
        b = bj;
    }
    let b = 2.0 * b;
    (b.is_finite() && b < 1e300).then_some(b)
}

/// Bounds on J_n(x) at an integer order n, |n| <= 50, over a finite box x of one sign with
/// 0 < |x| <= 2: there the series J_m(t) = sum_j (-1)^j (t/2)^(2j+m) / (j! (m+j)!) alternates
/// with decreasing terms, so t0 - t1 <= J_m(t) <= t0 - t1 + t2, with t0 = (t/2)^m / m!
/// increasing in t, t1 / t0 = (t/2)^2 / (m+1) and t2 / t0 = (t/2)^4 / (2 (m+1) (m+2)). Widened
/// by 2^-36 relative (the kernel's error, measured within
/// about 2,600 ulp, and this arithmetic's); the sign follows `J_-m = (-1)^m J_m` and
/// `J_m(-t) = (-1)^m J_m(t)`. `None` outside that region, or where the bound nears underflow.
fn besselj_integer_small_argument(n: f64, x: &Vs) -> Option<(f64, f64)> {
    if !x.has_fin || x.pinf || x.ninf {
        return None;
    }
    let (lo, hi, mirrored) = if x.lo > 0.0 && x.hi <= 2.0 {
        (x.lo, x.hi, false)
    } else if x.hi < 0.0 && x.lo >= -2.0 {
        (-x.hi, -x.lo, true)
    } else {
        return None;
    };
    let m = n.abs();
    if m > 50.0 {
        return None;
    }
    let k = m as i32;
    let factorial: f64 = (1..=k).map(f64::from).product();
    let t0 = |t: f64| (t / 2.0).powi(k) / factorial;
    let (qlo, qhi) = ((lo / 2.0).powi(2), (hi / 2.0).powi(2));
    let upper = t0(hi) * (1.0 - qlo / (m + 1.0) + qhi * qhi / (2.0 * (m + 1.0) * (m + 2.0)));
    if upper.is_nan() || upper < 1e-290 {
        return None;
    }
    let lower = t0(lo) * (1.0 - qhi / (m + 1.0));
    let w = 2f64.powi(-36);
    let a = if lower > 0.0 {
        lower * (1.0 - w)
    } else {
        lower * (1.0 + w) - 1e-300
    };
    let b = (upper * (1.0 + w)).min(1.0);
    let odd = k % 2 == 1;
    Some(if odd && ((n < 0.0) != mirrored) {
        (-b, -a)
    } else {
        (a, b)
    })
}

fn apply_unary(op: &str, v: &Vs) -> Option<Vs> {
    Some(match op {
        "floor" => u_step(v, f64::floor),
        "ceil" => u_step(v, f64::ceil),
        "erf" => u_erf(v),
        "harmonic" => return u_harmonic(v),
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
    // F82 (2026-08-11, extreme row 433087): these three infinite blocks used to
    // decide with `if x.pinf { .. } else { .. }` -- written for a SINGLE-signed
    // infinite operand. A MIXED operand (pinf AND ninf, e.g. `inf * sign-unknown`)
    // collapsed to one sign, DROPPING the other flag; `cls()` then read the
    // under-set as a DEFINITE PosInf/NegInf and the classification fold shipped a
    // fabricated literal (`(A * -inf)/pi -> +inf` for a -inf truth). The flags are
    // reachability claims and every fold/witness consumer reads Vs as an
    // OVER-approximation, so each sign contribution must propagate independently:
    // dropping a true flag is exactly as unsound as inventing a false one.
    if (a.pinf || a.ninf) && b.has_fin {
        if b.hi > 0.0 {
            r.pinf |= a.pinf;
            r.ninf |= a.ninf;
        }
        if b.lo < 0.0 {
            r.ninf |= a.pinf;
            r.pinf |= a.ninf;
        }
        if b.is_const() && b.lo == 0.0 {
            r.nan = true;
        } // inf*0 = nan identically
    }
    if (b.pinf || b.ninf) && a.has_fin {
        if a.hi > 0.0 {
            r.pinf |= b.pinf;
            r.ninf |= b.ninf;
        }
        if a.lo < 0.0 {
            r.ninf |= b.pinf;
            r.pinf |= b.ninf;
        }
        if a.is_const() && a.lo == 0.0 {
            r.nan = true;
        }
    }
    if (a.pinf || a.ninf) && (b.pinf || b.ninf) {
        r.pinf |= (a.pinf && b.pinf) || (a.ninf && b.ninf);
        r.ninf |= (a.pinf && b.ninf) || (a.ninf && b.pinf);
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
    //
    // THE |t| = 1 ROW, in every magnitude-step arm (audit B1, D8(iii)): `pow(t, +-inf)`
    // is 1 at |t| = 1 (the ratified Sec 9.8.2 magnitude-step exception, negative t
    // included), and that value is REACHABLE whenever the magnitude range CONTAINS 1 --
    // not only for an exact +-1 point base. The old `is_const && |lo| == 1` read missed
    // every BRACKETED base (H-019 outward rounding: `cos(np.pi)` encloses
    // [-1, -0.9999999999999991]), so this arm asserted a DEFINITE infinity where the
    // contract value is 1 -- the 46-row fabricated-Inf family, escaping as
    // `x0 + pow(cos(np.pi), -inf) -> inf` -- and it under-reported every genuinely
    // straddling range (x0 in [1/2, 2]: the true set is {0, 1, +inf} and 1 was
    // absent), the under-approximation direction `lib.rs` calls fatal. Merging 1 only
    // WIDENS; for a bracketed +-1 ground the class turns Mixed and the fold REFUSES
    // (fail closed, D8(i): refusal is the fix).
    if b.pinf && !b.has_fin && !b.ninf {
        if let Some((amin, amax)) = ar {
            if amax > 1.0 {
                r.pinf = true;
            }
            if amin < 1.0 {
                r.merge_pt(0.0);
            }
            if amin <= 1.0 && 1.0 <= amax {
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
            if amin <= 1.0 && 1.0 <= amax {
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
        // the dedicated pure-inf arms (H-028: these were dropped by the early return).
        // The |t| = 1 row was MISSING from this copy alone (audit B1 part (a): copies
        // 1-3 carried at least the exact-point form under a comment here claiming
        // parity; the F1 sweep family -- negative const base, mixed-inf exponent --
        // shipped nan for a contract value of 1 through exactly this gap). Doc at the
        // dedicated arms; `r.nan` is already true on this path, so the merged 1 turns
        // the class Mixed and the ground fold refuses.
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
            if amin <= 1.0 && 1.0 <= amax {
                r.merge_pt(1.0);
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
    // A POSITIVE base under a finite exponent range (all but a constant integer exponent, the
    // branch below): `a^b = exp(b ln a)` is monotone in each argument on (0, inf) x R, so its
    // image over a box is the hull of the four corner powers, bracketed (libm, H-019); a closed
    // hull over an open end only widens. The general branch below reports the whole half-line
    // here, so a ground `1/pi^(1/6)` (a root of even index, as the canon spells it, its exponent
    // the enclosure of 1/6) could not be read as the number below 1 it is, and `acos` of its
    // reciprocal never classified as NaN.
    let integer_point = b.is_const() && b.lo.fract() == 0.0;
    if a.has_fin
        && !a.nan
        && !a.pinf
        && !a.ninf
        && a.lo > 0.0
        && a.hi.is_finite()
        && b.has_fin
        && !b.nan
        && !b.pinf
        && !b.ninf
        && b.lo.is_finite()
        && b.hi.is_finite()
        && !integer_point
    {
        let corners = [
            a.lo.powf(b.lo),
            a.lo.powf(b.hi),
            a.hi.powf(b.lo),
            a.hi.powf(b.hi),
        ];
        let lo = corners.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if lo.is_normal() && hi.is_finite() {
            let mut z = Vs::empty();
            z.merge_pt(out_lo(lo, ULP_LIBM));
            z.merge_pt(out_hi(hi, ULP_LIBM));
            z.fin_null = a.fin_null;
            return z;
        }
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
                // PARITY, NOT REMAINDER (audit B4). Rust's `%` truncates toward zero, so
                // `(-1) % 2 == -1` and `k % 2 == 1` is FALSE for every negative odd k --
                // exactly the k this branch handles. The -INF candidate was therefore never
                // pushed, the two-sided pole came back as a one-sided reflected box, and the
                // under-approximation the `interval_value_set_box` contract (rust/lib.rs)
                // calls fatal licensed `finite_ae` to
                // certify expressions that are nan on positive measure: measured at runtime,
                // `A - A -> '0'` in SOUND mode for `A = log(x0**K + 1)`, nan across all of
                // (-1, 0). The correct idiom is the one 18 lines above at the pw() closure.
                cands.push((INF, true)); // pole reach: open edge by convention
                if k % 2 != 0 && a.lo < 0.0 {
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
            // the |t| = 1 row, contains-1 form (B1/D8(iii); doc at the dedicated arms)
            if amin <= 1.0 && 1.0 <= amax {
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

/// The floor of the quotient `a / b` where it is one integer at every value of the two sets: a
/// certainly finite `a` over a divisor certainly finite and nonzero (the `floor(a/b)` of the
/// real `mod`); `None` where it is not decided.
pub(crate) fn quotient_floor(a: &Vs, b: &Vs) -> Option<f64> {
    let finite = |v: &Vs| v.has_fin && !(v.nan || v.pinf || v.ninf);
    if !finite(a) || !finite(b) || (b.lo <= 0.0 && 0.0 <= b.hi) {
        return None;
    }
    let q = u_step(&b_mul(a, &u_inv(b)), f64::floor);
    q.is_const().then_some(q.lo)
}

fn apply_binary(op: &str, a: &Vs, b: &Vs) -> Option<Vs> {
    Some(match op {
        "max" => b_lattice(a, b, true),
        "min" => b_lattice(a, b, false),
        "atan2" => b_atan2(a, b),
        "mod" => return b_mod(a, b),
        "besselj" => return b_besselj(a, b),
        "+" => b_add(a, b),
        "-" => b_add(a, &u_neg(b)),
        "*" => b_mul(a, b),
        "/" => b_mul(a, &u_inv(b)),
        "pow" => b_pow(a, b),
        // `rootn(x, n)`: IEEE-754 rootn, honest for EVERY integer index:
        // odd = the signed root, even = the principal root, 1 = identity, negative =
        // reciprocal. Sound only when the index is a KNOWN nonzero-integer point (a
        // literal in the serialized form); any other uncertainty in the index fails closed
        // like an unknown operator. An index that can reach NO nonzero integer (a range
        // strictly between two integers, 0, an infinity, NaN) is the invalid operation at
        // every value: the root is NaN whatever the radicand (`rootn(-5, asinh(tanh 4))`,
        // whose index is 0.8807...).
        "rootn" => {
            if !index_reaches_a_nonzero_integer(b) {
                let mut r = Vs::empty();
                r.nan = true;
                return Some(r);
            }
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

/// Does `tokens` hold an operator the interval layer may not analyze: one whose arm can refuse
/// (`harmonic`, `besselj`, `mod`) or one with no arm at all? The vocabulary before 0.15 and the
/// total 0.15 arms are not such heads, so a consumer that fails closed on them changes nothing
/// for an engine without them.
pub(crate) fn holds_partial_head(tokens: &[String], ops: &Operators) -> bool {
    const TOTAL: [&str; 45] = [
        "neg", "abs", "inv", "exp", "log", "sin", "cos", "tan", "tanh", "atanh", "asin", "acos",
        "acosh", "cosh", "sinh", "asinh", "atan", "pow2", "pow3", "pow4", "pow5", "pow1_2",
        "pow1_3", "pow1_4", "pow1_5", "mult2", "mult3", "mult4", "mult5", "div2", "div3", "div4",
        "div5", "+", "-", "*", "/", "pow", "rootn", "floor", "ceil", "erf", "max", "min", "atan2",
    ];
    tokens
        .iter()
        .any(|t| ops.is_operator(t) && !TOTAL.contains(&t.as_str()))
}

/// May a value of the set `b` be a NONZERO INTEGER (the indices at which `rootn` is a root)? Only
/// the finite part can: an infinity and NaN are no integers. The finite part reaches one unless it
/// lies strictly between two consecutive integers (an open end at an integer excludes it) or is
/// the point 0. Outward-rounded bounds only widen the range, so a `false` is a proof.
pub(crate) fn index_reaches_a_nonzero_integer(b: &Vs) -> bool {
    if !b.has_fin {
        return false;
    }
    if !b.lo.is_finite() || !b.hi.is_finite() {
        return true; // an unbounded finite part holds every integer beyond some point
    }
    // The least integer the range holds (the bound itself where it is a closed integer).
    let first = if b.lo.fract() == 0.0 && !b.lo_open {
        b.lo
    } else {
        b.lo.floor() + 1.0
    };
    let holds = |n: f64| n < b.hi || (n == b.hi && !b.hi_open);
    if !holds(first) {
        return false;
    }
    // `first` is the only integer when the next one is out of range; it must then be nonzero.
    first != 0.0 || holds(first + 1.0)
}

/// The dimension of a DUMMY variable (`x0`/`_0` -> 0): the miner's sources are spelled over
/// `x<k>` and a pattern instance binds `_<k>`, and the layer numbers its dimensions by them. Every
/// other variable is spelled as a dummy at the public entry points (`dummy_spelling`), so inside
/// the layer this is the one variable test. `None` for anything else.
fn var_index(t: &str) -> Option<usize> {
    let rest = t.strip_prefix('_').or_else(|| t.strip_prefix('x'))?;
    if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

/// Is `t` a variable? What the engine's vocabulary says one is (as the AC core reads a leaf, and
/// `SimpliPyEngine.expression_variables`): a token that is no operator, no literal (a numeral, a
/// parenthesized one, `np.pi`, `np.e`, a spelled non-finite), no `<constant>` and no placeholder
/// of the rule or tagged language (`_`, `?`, `!`, `$`, `<` ...), whatever it is called; a dummy
/// (`var_index`) is one too.
fn is_variable(t: &str, ops: &Operators) -> bool {
    if var_index(t).is_some() {
        return true;
    }
    !t.is_empty()
        && !t.starts_with(['_', '?', '!', '$', '<', '('])
        && ops.arity_of(t).is_none()
        && crate::numeric::leaf_value(t).is_none()
        && !crate::utils::looks_numeric(t)
}

/// A FAMILY'S VARIABLES SPELLED AS DUMMIES, so that what a variable is called decides nothing:
/// `sin(x_1)` and `sin(v1)` are analysed as `sin(x1)` is. Each variable that is no dummy
/// (`is_variable`, `var_index`) takes, in order of first appearance across the family, the dummy
/// past every dummy the family holds, and never dimension 0, the axis an unbound `<constant>`
/// shares (`leaf_vs`): over `x1, x2` and over `x_1, x_2` a family is the same box. `None` where
/// every variable already is a dummy. Applied at every public entry point that reads its
/// variables from the tokens; `value_set_p` takes its domains by dummy index and is not one.
fn dummy_spelling(exprs: &[&[String]], ops: &Operators) -> Option<Vec<Vec<String>>> {
    let mut named: Vec<&str> = Vec::new();
    let mut top = 0usize;
    for e in exprs {
        for t in e.iter() {
            if let Some(i) = var_index(t) {
                top = top.max(i);
            } else if !named.contains(&t.as_str()) && is_variable(t, ops) {
                named.push(t);
            }
        }
    }
    if named.is_empty() {
        return None;
    }
    let base = top + 1;
    Some(
        exprs
            .iter()
            .map(|e| {
                e.iter()
                    .map(|t| match named.iter().position(|n| *n == t.as_str()) {
                        Some(j) => format!("x{}", base + j),
                        None => t.clone(),
                    })
                    .collect()
            })
            .collect(),
    )
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
                // In the f64 number domain the deployed evaluator READS such a literal as
                // +-inf (design 2c, review M4), so the infinity joins the set; the exact domain
                // keeps the finite denotation alone.
                Some(Boundary::Overflow) if v > 0.0 => {
                    let mut vs = Vs::interval(f64::MAX, INF, false, true);
                    vs.pinf = crate::ac::rat::f64_numbers();
                    return vs;
                }
                Some(Boundary::Overflow) => {
                    let mut vs = Vs::interval(-INF, -f64::MAX, true, false);
                    vs.ninf = crate::ac::rat::f64_numbers();
                    return vs;
                }
                // Below the subnormal floor: 0 < |denoted| <= the smallest positive
                // subnormal (again sound-and-loose: underflow to zero requires strictly
                // less than half of it).
                Some(Boundary::Underflow) if v.is_sign_negative() => {
                    return Vs::interval(-f64::from_bits(1), 0.0, false, true)
                }
                Some(Boundary::Underflow) => {
                    return Vs::interval(0.0, f64::from_bits(1), true, false)
                }
                // A `p/q` beyond the exact reader's digit limit, read as
                // `float(p) / float(q)`: `10^5000/10^4900` images as NaN but denotes a finite
                // real, so neither the point nor a boundary bracket is sound. Its denotation
                // is a finite real and nothing more is known: fail closed to every finite
                // real, which licenses no fold at all. (Within the limit a fraction reads as
                // its correctly rounded f64 and classifies like a decimal above.)
                Some(Boundary::UnknownFinite) => return Vs::reals(),
                None => {}
            }
        }
        // A POINT only where the literal DENOTES an integer: an integer-valued f64 below 2^53
        // can be the rounding of a non-integer (`2311319199101329.25` reads ...329.0, and since
        // fractions read as their nearest f64, `10182333997706870151/10000` reads ...687.0), and
        // a point there misses the denoted value. A plain integer spelling below 2^53 is exact
        // and keeps the cheap test; everything else is certified on its spelling.
        let integer_spelling = {
            let d = t.strip_prefix('-').unwrap_or(t);
            !d.is_empty() && d.bytes().all(|c| c.is_ascii_digit())
        };
        if !v.is_finite()
            || v == 0.0
            || (v.fract() == 0.0
                && ((integer_spelling && v.abs() < 9.007199254740992e15)
                    || exact_integer_literal(t, v)))
        {
            Vs::constant(v)
        } else if crate::numeric::leaf_value_is_nearest(t) {
            Vs::interval(next_down(v), next_up(v), false, false)
        } else {
            // An exact fraction whose correctly rounded f64 could not be certified (a component
            // beyond `i128`): `v` is `float(p) / float(q)`, within three ulps of the fraction,
            // so bracket four.
            let (mut lo, mut hi) = (v, v);
            for _ in 0..4 {
                lo = next_down(lo);
                hi = next_up(hi);
            }
            Vs::interval(lo, hi, false, false)
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
        if p.trim_start_matches(['+', '-']).chars().all(|c| c == '0') {
            return None; // an exact zero, spelled as a fraction (`-0/5`, `+0/5`, `+00/7`)
        }
        // Read exactly, `v` is the correctly rounded value: an infinite image is an overflow
        // and a zero image an underflow, exactly as for a decimal.
        if crate::ac::rat::token_rational(t).is_some() {
            return Some(if v.is_infinite() {
                Boundary::Overflow
            } else {
                Boundary::Underflow
            });
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
/// rational, via the same grammar `leaf_value` reads (paren-stripped decimal, `p/q`), at any
/// size: the exact reader compares the denoted rational with `v` (this closes the beyond-i128
/// residual registered on H-045; `2^200` written out is a point, `1e40` is not -- its nearest
/// f64 is not 10^40).
fn exact_integer_literal(t: &str, v: f64) -> bool {
    let t = t
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(t);
    crate::ac::rat::token_denotes_integer(t, v)
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
    if let Some(d) = dummy_spelling(&[tokens], ops) {
        return value_set(&d[0], ops, dom);
    }
    let doms = vec![*dom; n_var_slots(&[tokens])];
    value_set_p(tokens, ops, &doms, &[])
}

pub fn value_class(tokens: &[String], ops: &Operators) -> Option<Class> {
    if let Some(d) = dummy_spelling(&[tokens], ops) {
        return value_class(&d[0], ops);
    }
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
    if let Some(d) = dummy_spelling(&[source, target], ops) {
        return domain_extension_p(&d[0], src_params, &d[1], tgt_params, ops);
    }
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
/// `None` (undecided) and the gate REJECTS -- it never reads as "no extension". A mine raises
/// it per stratum through `find_rules(interval_node_budget=...)`, which records the value in
/// the artifact's parameters, when the counters flag systematic exhaustion.
pub const DEFAULT_NODES_BUDGET: u32 = 400_000;

static NODES_BUDGET: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(DEFAULT_NODES_BUDGET);

fn max_nodes_budget() -> u32 {
    NODES_BUDGET.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set the subdivision node budget (`None`: the default) and return the previous one. The
/// miner sets it for the duration of a mine and restores it after.
pub fn set_max_nodes_budget(budget: Option<u32>) -> u32 {
    NODES_BUDGET.swap(
        budget.unwrap_or(DEFAULT_NODES_BUDGET),
        std::sync::atomic::Ordering::Relaxed,
    )
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
    if let Some(d) = dummy_spelling(&[source, target], ops) {
        return gate_horizon(&d[0], src_params, &d[1], tgt_params, ops);
    }
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
    if let Some(d) = dummy_spelling(&[source, target], ops) {
        return domain_extension_p_at(&d[0], src_params, &d[1], tgt_params, ops, r, depth_per_dim);
    }
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
///   1. the STRUCTURAL path (`nonfinite_null`, below): GLOBAL -- its per-operator
///      table discharges each obligation over all of R (`whole_range`), and it is the
///      only path that can certify pole-bearing trees (`1/x`, `tan x`,
///      `x/(x1 - cos x1)`) -- a pole cell never resolves under subdivision, but the
///      null measure of its x-support is a structural/analytic fact the certificate
///      section proves directly;
///   2. the SUBDIVISION path: bounded-clean cells over the horizon box, CONJOINED
///      (audit B2) with the TAIL ARM below -- a bounded subdivision cannot discharge
///      an unbounded a.e. property, and before the conjunct nothing looked outside
///      `[-R, R]^n`, so `log(1 - 0.001*x1)` certified and `- !0 !0 -> 0` cancelled an
///      expression that is nan for every x1 > 1000 (the nan onset merely had to sit
///      past the horizon, R >= 64).
///
/// Exp/sinh/polynomial compositions certify at depth 0-2.
///
/// SCOPE (D9 wording obligation): the tail arm removes the HORIZON failure class
/// only; never read this as "`finite_ae` is sound". Certificates lost to interval
/// DEPENDENCY LOSS remain unsound after it (`x3 - x3` widens to (-inf, inf) and the
/// verdict fails closed the wrong way around), and the arm decides only the shipped
/// vocabulary -- see the hazard note on `tail_finite_ae`.
///
/// `f64` (the `f64` mode) reads the expression in float64, where `tanh` and `atan` are constant
/// on whole tails (`on_float64_tails`): `1/(tanh(x1) - 1)` is `+inf` wherever x1 >= 19.06.
/// `real` and `permissive` read it over the reals.
pub fn finite_ae(tokens: &[String], ops: &Operators, f64: bool) -> bool {
    if let Some(d) = dummy_spelling(&[tokens], ops) {
        return finite_ae(&d[0], ops, f64);
    }
    let real = |t: &[String]| {
        nonfinite_null(t, ops) || (finite_ae_subdivision(t, ops) && tail_finite_ae(t, ops))
    };
    if f64 {
        !harmonic_nan_tail(tokens, ops) && on_float64_tails(tokens, ops, 0, &real, &real)
    } else {
        real(tokens)
    }
}

/// The TAIL ARM (audit B2, ruling D9): discharge the a.e.-finiteness obligation on the
/// COMPLEMENT of the subdivision box. That complement is exactly the union over i of
/// the slabs `{|x_i| > R}` with every other coordinate free over all of R, so
/// discharging every slab plus the box covers R^n -- a global claim, not a wider box.
/// Each slab is one interval evaluation over the unbounded ray (`value_set_p` already
/// computes correct tail enclosures; X3 measured 0 containment violations across the
/// shipped operators x 4 unbounded domains).
///
/// The predicate is T3 (D9): a slab passes when no nan and no attained infinity is
/// possible AND (the enclosure is BOUNDED or the tree is POLE-FREE). Unboundedness
/// alone cannot be tolerated -- the module encodes a pole's reach in the (lo, hi)
/// BOUNDS, not the pinf/ninf flags (see the boundedness note in
/// `finite_ae_subdivision`), so `hi = +inf` does not distinguish growth (exp, sinh:
/// finite everywhere, sup infinite) from an attained pole. For a pole-free tree every
/// shipped operator maps finite -> finite-or-nan, so "no nan on the slab" already
/// proves every point finite and the unbounded sup is irrelevant (T3's soundness
/// argument); for pole-bearing trees the slab must be bounded. Declined: T1 (tolerate
/// unbounded always -- loses nothing today and is unsound in principle, the exact
/// claim-shape this audit exists to kill) and T2 (require bounded always -- kills
/// exp(x), the flagship binder). Measured price on the real mined corpus: 18 sound
/// certificates (verify_X3), owner-accepted in D9.
///
/// HAZARD (finite_ae tail scope): the arm decides the operators of the SHIPPED
/// vocabulary only; it is not a general procedure for discharging an unbounded a.e.
/// property. `POLE_OPS` is a correctness-critical syntactic table: it must list every
/// operator that can map a FINITE input to +-inf, and a new pole-bearing operator
/// that is not listed makes the arm unsound for unbounded trees (the clean fix -- a
/// faithful attains-infinity flag on `Vs` -- is the budgeted contingency, D9). The
/// verify_X3 corrections are in: `atanh` (+-1 -> -+inf) and `rootn` (negative index:
/// rootn(0, -2) = +inf) were missing from the prototype's table; `rootn` and `pow`
/// are listed wholesale because the check is syntactic and an index's sign may not be
/// a literal -- conservative, fail-closed (the benign families certify structurally
/// through `nonfinite_null`, so the recall cost lands only on subdivision-exclusive
/// trees).
fn tail_finite_ae(tokens: &[String], ops: &Operators) -> bool {
    // Keyed the SAFE way round: the operators known to map no finite input to +-inf. Any
    // other operator (`inv`, `/`, `tan`, `log`, `atanh`, `rootn`, `pow`, `harmonic`,
    // `besselj`, and every head this table does not name) makes the tree pole-bearing, which
    // only asks for a bounded slab.
    const POLE_FREE: [&str; 39] = [
        "+", "-", "*", "neg", "abs", "exp", "sin", "cos", "sinh", "cosh", "tanh", "asinh", "atan",
        "asin", "acos", "acosh", "pow2", "pow3", "pow4", "pow5", "pow1_2", "pow1_3", "pow1_4",
        "pow1_5", "mult2", "mult3", "mult4", "mult5", "div2", "div3", "div4", "div5", "floor",
        "ceil", "erf", "max", "min", "mod", "atan2",
    ];
    let (r, _d, decidable) = horizon(&[tokens], &[&[]], ops);
    if !decidable {
        return false; // horizon past R_MAX: fail closed, as in the subdivision arm
    }
    let used = distinct_vars(&[tokens]);
    let width = n_var_slots(&[tokens]);
    let n = used.len().max(1);
    let pole_free = tokens
        .iter()
        .all(|s| !ops.is_operator(s) || POLE_FREE.contains(&s.as_str()));
    for dim in 0..n {
        for &(lo, hi) in &[(r, INF), (-INF, -r)] {
            let mut bx = vec![(-INF, INF); n];
            bx[dim] = (lo, hi);
            let doms = doms_from_box(&bx, &used, width, (-INF, INF));
            let Some(v) = value_set_p(tokens, ops, &doms, &[]) else {
                return false; // unevaluable slab: fail closed
            };
            if !(v.has_fin && !v.nan && !v.pinf && !v.ninf) {
                return false;
            }
            if !(v.lo.is_finite() && v.hi.is_finite()) && !pole_free {
                return false; // T3: an unbounded slab is tolerable only pole-free
            }
        }
    }
    true
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
            // the 0.15 heads that map every finite real to a finite real
            | "floor"
            | "ceil"
            | "erf"
    )
}

/// Everywhere-defined and REAL-ANALYTIC on all of R: the identity-theorem class. `abs`,
/// `pow1_3`, `pow1_5` are total but not analytic at 0 and are excluded on purpose. `erf` is
/// entire (its float64 saturation at +-1 is a tail, `tail_of`); the other 0.15 heads are not
/// analytic (steps, kinks, poles, a branch cut) and are read by their own arms.
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
            | "erf"
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

/// Is every level set {g = c} of `g` null, for every real c at once? Where `g` is an
/// entire-analytic composition (`entire_analytic_composition`: entire heads, integer powers,
/// ground subtrees; a variable is one) certified nonconstant (`disjoint_value_pair`), the
/// identity theorem gives it. A `<constant>` fails closed: at C = 0, `C*x1` is constant. The
/// consumers take a countable union of level sets (the zeros of `harmonic`, of `besselj` at a
/// literal order, the multiples of a `mod` divisor).
fn levels_null(g: &[String], ops: &Operators) -> bool {
    !g.iter().any(|s| s == "<constant>")
        && entire_analytic_composition(g, ops)
        && disjoint_value_pair(g, ops, true)
}

/// The class of a `besselj` order the certificates read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BesseljOrder {
    /// An integer: J_n is entire in x.
    Integer,
    /// Certainly positive and no integer: NaN at every x < 0, 0 at x = 0.
    PositiveFraction,
    /// Certainly negative and no integer: NaN at every x < 0, a pole at x = 0.
    NegativeFraction,
}

/// The class of the order of `besselj(nu, x)` where it is a ground subtree whose enclosure
/// decides it in every reading: one integer point, or an interval holding no integer and of one
/// sign (a decimal literal such as `1.5` encloses its value and its float's), inside the range
/// the kernel is read as NaN-free at every finite argument (`BESSELJ_NU_MEASURED`, and
/// `BESSELJ_NEG_NU_MEASURED` for a negative non-integer order). `None` for any other order: the
/// certificates fail closed.
fn besselj_order(nu: &[String], ops: &Operators) -> Option<BesseljOrder> {
    if nu
        .iter()
        .any(|s| var_index(s).is_some() || s == "<constant>")
    {
        return None;
    }
    let v = whole_range(nu, ops)?;
    if !v.has_fin || v.nan || v.pinf || v.ninf || !(v.lo.is_finite() && v.hi.is_finite()) {
        return None;
    }
    let order = if v.is_const() && v.lo.fract() == 0.0 {
        BesseljOrder::Integer
    } else if v.lo.floor() == v.hi.floor() && v.lo.fract() != 0.0 && v.lo > 0.0 {
        BesseljOrder::PositiveFraction
    } else if v.lo.floor() == v.hi.floor() && v.lo.fract() != 0.0 && v.hi < 0.0 {
        BesseljOrder::NegativeFraction
    } else {
        return None;
    };
    let bound = if order == BesseljOrder::NegativeFraction {
        BESSELJ_NEG_NU_MEASURED
    } else {
        BESSELJ_NU_MEASURED
    };
    (v.lo.abs().max(v.hi.abs()) <= bound).then_some(order)
}

/// float64's `harmonic` is NaN at every float at or below -2^52 (each is a negative integer, a
/// pole): a NaN tail on a whole ray, which the real reading does not have. In `f64` a tree
/// holding `harmonic` of an argument that may reach that ray is not finite almost everywhere.
fn harmonic_nan_tail(t: &[String], ops: &Operators) -> bool {
    const NAN_TAIL: f64 = -4503599627370496.0; // -2^52
    t.iter().enumerate().any(|(i, tok)| {
        tok == "harmonic"
            && ops.arity_of(tok) == Some(1)
            && subtree_end(t, i + 1, ops).is_none_or(|j| {
                whole_range(&t[i + 1..j], ops)
                    .is_none_or(|v| v.ninf || (v.has_fin && v.lo <= NAN_TAIL))
            })
    })
}

const CERT_RECURSION_MAX: u32 = 32;

/// {x : f(x) = 0} is Lebesgue-null, for every value of any `<constant>`. Fail-closed. `f64` reads
/// the expression in float64 (`on_float64_tails`), `real` and `permissive` over the reals.
pub fn zero_set_null(tokens: &[String], ops: &Operators, f64: bool) -> bool {
    if let Some(d) = dummy_spelling(&[tokens], ops) {
        return zero_set_null(&d[0], ops, f64);
    }
    if tokens.iter().any(|s| s == "<constant>") {
        return false;
    }
    let real = |t: &[String]| zsn(t, ops, 0, false);
    if f64 {
        on_float64_tails(tokens, ops, 0, &real, &real)
    } else {
        real(tokens)
    }
}

/// E1 (F80, owner-ruled 2026-08-10): `zero_set_null` with `<constant>` as a GENERIC
/// PARAMETER. Each occurrence is an INDEPENDENT existentially quantified constant
/// (the ratified `Const` semantics), so the question moves to the JOINT (x, c)-space:
/// occurrence j is renamed to the fresh variable `x<K+j>` and the ordinary analysis
/// runs there. The lemma (via Fubini): a null zero set in the joint space yields, for
/// almost every parameter draw, a null x-zero-set -- and a null exceptional set in
/// c-space is exactly the tolerance the ruled bare-`Const` doctrine already grants at
/// the licence level ("{c = 0} is a null set of the mask's own parameter space",
/// `Cx::nz_ae_licensed`). SCOPE: consumed ONLY by the odd-negative power-distribution
/// licence (`engine::ac::ac_zsn`). Deliberately NOT wired into `finite_nonzero_ae`:
/// the `$`-cancel licence keeps its SELFCANCEL asymmetry (the c = 0 atom stays
/// protected there), pinned by test. `f64` (the `f64` mode) reads the expression in float64,
/// where `tanh` and `atan` are constant on whole tails (`on_float64_tails`); `real` and
/// `permissive` read it over the reals.
pub fn zero_set_null_generic_const(tokens: &[String], ops: &Operators, f64: bool) -> bool {
    if let Some(d) = dummy_spelling(&[tokens], ops) {
        return zero_set_null_generic_const(&d[0], ops, f64);
    }
    let _memo = ZsnScope::open();
    let real = |t: &[String]| zsn(t, ops, 0, true);
    let read = |t: &[String]| {
        if f64 {
            on_float64_tails(t, ops, 0, &real, &real)
        } else {
            real(t)
        }
    };
    if !tokens.iter().any(|s| s == "<constant>") {
        return read(tokens);
    }
    let base = 1 + tokens
        .iter()
        .filter_map(|t| var_index(t))
        .max()
        .unwrap_or(0);
    let mut j = 0usize;
    let renamed: Vec<String> = tokens
        .iter()
        .map(|t| {
            if t == "<constant>" {
                let s = format!("x{}", base + j);
                j += 1;
                s
            } else {
                t.clone()
            }
        })
        .collect();
    read(&renamed)
}

/// float64's tails: `(threshold, limit)` where the function is EXACTLY `limit` for every argument
/// at or past `threshold`, and `-limit` at or below `-threshold`. Measured on every operator of the
/// vocabulary with the deployed evaluator (the engine's own kernels), out to the largest float:
/// only `tanh` (+-1, from `TANH_SAT`) and
/// `atan` (+-fl(pi/2)) have one. Overflow and underflow to an infinity or 0 (`exp`, `sinh`,
/// `cosh`, `pow`) are the overflow reading's; `asin`, `acos`, `atanh`, `acosh` and `log` are NaN
/// off their domains.
fn tail_of(op: &str) -> Option<(f64, f64)> {
    match op {
        "tanh" => Some((TANH_SAT, 1.0)),
        "atan" => Some((ATAN_SAT, std::f64::consts::FRAC_PI_2)),
        "erf" => Some((ERF_SAT, 1.0)),
        _ => None,
    }
}

/// A certificate of an almost-everywhere property (a null zero set, finiteness), read in
/// float64. Off the tails of its `tanh` and `atan` (`tail_of`) an expression is the real one, so
/// there the property holds almost everywhere when the real certificate `real` says so. On a tail
/// of one of them it is EXACTLY the expression with that function replaced by its limit, which
/// must then have the property in float64 too, on the whole space (a stronger claim); `ground`
/// decides one that is a constant there. So the zero set of `x2*tanh(x1) - x1` is null (its tails
/// read `x2 - x1` and `-x2 - x1`), and that of `x2*tanh(x1) - x2`, `+0` wherever x1 >= 19.06, is
/// not; `1/(tanh(x1) - 1)` is not finite almost everywhere. A function whose argument is ground,
/// or never leaves (-threshold, threshold) (`tanh(sin(x1))`), has no tail to read; more than
/// `MAX_TAILS` of them fail closed.
fn on_float64_tails(
    t: &[String],
    ops: &Operators,
    d: u32,
    real: &dyn Fn(&[String]) -> bool,
    ground: &dyn Fn(&[String]) -> bool,
) -> bool {
    const MAX_TAILS: usize = 3;
    if d > CERT_RECURSION_MAX || !real(t) {
        return false;
    }
    let Some(sites) = tail_sites(t, ops) else {
        return false;
    };
    sites.len() <= MAX_TAILS
        && sites.iter().all(|&(site, limit)| {
            [limit, -limit].iter().all(|&c| {
                let f = replace_subtree(t, site, c, ops);
                if f.iter()
                    .all(|x| var_index(x).is_none() && x != "<constant>")
                {
                    ground(&f)
                } else {
                    on_float64_tails(&f, ops, d + 1, real, ground)
                }
            })
        })
}

/// The functions of `t` that reach a float64 tail: each distinct subtree once, with its limit.
fn tail_sites<'a>(t: &'a [String], ops: &Operators) -> Option<Vec<(&'a [String], f64)>> {
    let mut sites: Vec<(&[String], f64)> = Vec::new();
    for (i, tok) in t.iter().enumerate() {
        let Some((threshold, limit)) = tail_of(tok) else {
            continue;
        };
        if ops.arity_of(tok) != Some(1) {
            continue;
        }
        let j = subtree_end(t, i, ops)?;
        if ground_leaf(t, i + 1, ops).is_some() {
            continue;
        }
        let inside = |v: Vs| {
            !v.pinf
                && !v.ninf
                && (!v.has_fin
                    || ((v.lo > -threshold || (v.lo == -threshold && v.lo_open))
                        && (v.hi < threshold || (v.hi == threshold && v.hi_open))))
        };
        if whole_range(&t[i + 1..j], ops).is_some_and(inside) {
            continue;
        }
        if !sites.iter().any(|&(s, _)| s == &t[i..j]) {
            sites.push((&t[i..j], limit));
        }
    }
    Some(sites)
}

/// `t` with every occurrence of the subtree `site` replaced by the literal `c`, spelled as the
/// shortest decimal that reads back as `c` (an integer limit as an integer, which reads as a
/// point).
fn replace_subtree(t: &[String], site: &[String], c: f64, ops: &Operators) -> Vec<String> {
    let lit = format!("{c}");
    let mut out = Vec::with_capacity(t.len());
    let mut i = 0;
    while i < t.len() {
        if t[i..].starts_with(site) && subtree_end(t, i, ops) == Some(i + site.len()) {
            out.push(lit.clone());
            i += site.len();
        } else {
            out.push(t[i].clone());
            i += 1;
        }
    }
    out
}

/// `e2` (owner-ratified 2026-08-11, audit F81; design in the research harness):
/// the denominator-clearing fallback (`cleared_witness`) fires only from the
/// `zero_set_null_generic_const` entry -- the odd-negative distribution
/// licence's scoped certificate, mirroring E1's deliberate scoping. The plain
/// `zero_set_null` (and through it `finite_nonzero_ae`, the `$`-cancel) keeps
/// its current strength, pinned by test; widening is a separate future ruling.
/// The identity theorem on the expression's domain (`domain_witness`), a factor
/// the terms of a sum share (`common_factor_witness`) and a slope in one
/// variable (`slice_witness`) are scoped the same way: they serve the zero-set
/// certificate the core reads (`engine::ac::ac_zsn`: reading A, the
/// distribution licences, orientation).
fn zsn(t: &[String], ops: &Operators, d: u32, e2: bool) -> bool {
    if d > CERT_RECURSION_MAX || t.is_empty() {
        return false;
    }
    let key = (t.to_vec(), e2);
    if let Some(hit) = ZSN_MEMO.with(|m| m.borrow().as_ref().and_then(|m| m.get(&key).copied())) {
        return hit;
    }
    let r = zsn_read(t, ops, d, e2);
    ZSN_MEMO.with(|m| {
        if let Some(m) = m.borrow_mut().as_mut() {
            m.insert(key, r);
        }
    });
    r
}

/// `zsn`'s verdicts within one reading (`ZsnScope`), by expression and scope: a verdict that holds
/// is a proof wherever it is read again, and one that does not, read again with more of the
/// recursion budget left, only fails closed where it might have held.
type ZsnMemo = rustc_hash::FxHashMap<(Vec<String>, bool), bool>;

thread_local! {
    static ZSN_MEMO: std::cell::RefCell<Option<ZsnMemo>> = const { std::cell::RefCell::new(None) };
}

/// Holds `zsn`'s memo open for one reading (the outermost opens it) and restores the caller's.
struct ZsnScope(bool);

impl ZsnScope {
    fn open() -> Self {
        ZsnScope(ZSN_MEMO.with(|m| {
            let mut m = m.borrow_mut();
            let fresh = m.is_none();
            if fresh {
                *m = Some(Default::default());
            }
            fresh
        }))
    }
}

impl Drop for ZsnScope {
    fn drop(&mut self) {
        if self.0 {
            ZSN_MEMO.with(|m| *m.borrow_mut() = None);
        }
    }
}

fn zsn_read(t: &[String], ops: &Operators, d: u32, e2: bool) -> bool {
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
            | "sinh" | "tanh" | "asinh" | "atan" | "asin" | "atanh" | "erf",
            Some(1),
        ) => zsn(&t[1..], ops, d + 1, e2),
        // never zero on their real range: `cosh` at every argument, `exp` at every finite one --
        // `exp(-inf)` is 0, so `exp(g)` vanishes where `g` is `-inf`, a whole region where `g`
        // inverts a zero that holds one (`exp(1/ceil(x1))` is 0 on (-1, 0) in float64, and
        // `exp(-1/(x1 - |x1|))` on x1 >= 0 in any reading): Z(exp g) ⊆ {g = ±inf}.
        ("cosh", Some(1)) => true,
        ("exp", Some(1)) => isn_e(&t[1..], ops, d + 1, e2),
        // h(y) = 0 iff y = 1: shift and recurse
        ("log" | "acos" | "acosh", Some(1)) => {
            let mut shifted: Vec<String> = Vec::with_capacity(t.len() + 1);
            shifted.push("-".into());
            shifted.extend_from_slice(&t[1..]);
            shifted.push("1".into());
            zsn(&shifted, ops, d + 1, e2)
        }
        // tan(g) = 0 iff sin(g) = 0
        ("tan", Some(1)) => {
            let mut s: Vec<String> = Vec::with_capacity(t.len());
            s.push("sin".into());
            s.extend_from_slice(&t[1..]);
            zsn(&s, ops, d + 1, e2)
        }
        // Z(1/g) = {g = ±inf} EXACTLY (`1/0 = +inf`, `1/nan = nan`: neither is 0) --
        // the INFINITE-set certificate; a fat NaN domain of `g` is harmless here.
        // (Sharpened 2026-08-03 from `nfn`, which conflated NaN with ±inf and refused
        // every acosh/log-bearing denominator -- the division-tower campaign's Hole 2.)
        ("inv", Some(1)) => isn_e(&t[1..], ops, d + 1, e2) || (e2 && cleared_witness(t, ops)),
        ("*", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            zsn(&t[1..ja], ops, d + 1, e2) && zsn(&t[ja..], ops, d + 1, e2)
        }
        // (a/b) = 0 only where a = 0 (b finite nonzero) or b = ±inf (a finite):
        // Z(a/b) ⊆ Z(a) ∪ {b = ±inf} -- the denominator claim is the INFINITE-set
        // certificate (a NaN of `b` makes the quotient NaN, never 0).
        ("/", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            (zsn(&t[1..ja], ops, d + 1, e2) && isn_e(&t[ja..], ops, d + 1, e2))
                || (e2 && cleared_witness(t, ops))
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
                Some(q) if q > 0 => zsn(&t[1..ja], ops, d + 1, e2),
                Some(q) if q < 0 => {
                    isn_e(&t[1..ja], ops, d + 1, e2) || (e2 && cleared_witness(t, ops))
                }
                Some(_) => true,
                None => {
                    zsn(&t[1..ja], ops, d + 1, e2)
                        && isn_e(&t[1..ja], ops, d + 1, e2)
                        && isn_e(&t[ja..], ops, d + 1, e2)
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
                Some(n) if n >= 2 => zsn(&t[1..ja], ops, d + 1, e2),
                _ => false,
            }
        }
        // The 0.15 heads (`erf` is above: it vanishes only at 0, `erf(y) = 0` iff `y = 0`, and
        // float64's `erf` keeps every nonzero argument nonzero, `erf(5e-324) = 5e-324`; its
        // saturation at +-1 is a float64 tail, `tail_of`).
        //
        // `floor(g)` is 0 exactly on {0 <= g < 1} and `ceil(g)` on {-1 < g <= 0}: whole
        // intervals, so a step head is nonzero almost everywhere only where `g`'s whole-space
        // range misses that interval, and then it is never zero (NaN and +-inf pass through,
        // and are no zero). Exact in float64 too (the steps are exact).
        // A range that meets the interval only at `g = 0` leaves `g`'s own zero set (`ceil(|x1|)`
        // is 0 only where `x1` is).
        ("floor", Some(1)) => whole_range(&t[1..], ops).is_some_and(|v| {
            !v.has_fin
                || v.hi < 0.0
                || (v.hi == 0.0 && v.hi_open)
                || v.lo >= 1.0
                || (v.hi <= 0.0 && zsn(&t[1..], ops, d + 1, e2))
        }),
        ("ceil", Some(1)) => whole_range(&t[1..], ops).is_some_and(|v| {
            !v.has_fin
                || v.lo > 0.0
                || (v.lo == 0.0 && v.lo_open)
                || v.hi <= -1.0
                || (v.lo >= 0.0 && zsn(&t[1..], ops, d + 1, e2))
        }),
        // `max(a, b)` and `min(a, b)` are one of their members or NaN at every point (IEEE
        // 754-2019 `maximum`/`minimum`; at the zero tie a zero), so Z(max(a, b)) ⊆ Z(a) ∪ Z(b).
        ("max" | "min", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            zsn(&t[1..ja], ops, d + 1, e2) && zsn(&t[ja..], ops, d + 1, e2)
        }
        // `mod(a, b)` is 0 exactly where `a` is an integer multiple of a finite nonzero `b`
        // (`fmod` is exact), or at `a = 0` under an infinite `b`: countably many level sets
        // {a = k b}. Two shapes are read, each by Fubini over one variable:
        //   * `a` a variable x not in `b`: for fixed other variables `b` is one value, and
        //     {x = k b} ∪ {x = 0} is countable in x. `b` must hold a variable and have null
        //     level sets: a `b` constant on a region (`max(x2, 1)`, or a ground power of two)
        //     makes float64's `mod(x, b)` exactly 0 on the whole ray |x| >= 2^52 |b| there, a
        //     float64 tail the real reading does not see.
        //   * `b` a variable y not in `a`: for fixed other variables `a` is one value c; c = 0
        //     is `a`'s zero set (null by `zsn(a)`), and for c != 0, {y = c/k} is countable (an
        //     infinite `a` is NaN, no zero).
        // Anything else fails closed (`mod(x1, 2)`, `mod(x1 + x2, x1)`).
        ("mod", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            let (a, b) = (&t[1..ja], &t[ja..]);
            let lone_var = |s: &[String], other: &[String]| matches!(s, [x] if var_index(x).is_some() && !other.contains(x));
            (lone_var(a, b) && levels_null(b, ops)) || (lone_var(b, a) && zsn(a, ops, d + 1, e2))
        }
        // `harmonic(g)` = digamma(g + 1) + gamma vanishes at g = 0, increasing on (-1, inf),
        // and once more between each two of its poles -n-1 and -n (it runs from -inf to +inf
        // there); its poles are NaN, no zero. A `g` that stays >= -1 has Z(harmonic(g)) = Z(g)
        // (float64's kernel keeps every nonzero argument nonzero, `harmonic(5e-324) =
        // 1e-323`); otherwise the zero set is the union of countably many level sets of `g`,
        // null where every level set of `g` is (`levels_null`).
        ("harmonic", Some(1)) => {
            let g = &t[1..];
            if whole_range(g, ops).is_some_and(|v| v.has_fin && v.lo >= -1.0) {
                zsn(g, ops, d + 1, e2)
            } else {
                levels_null(g, ops)
            }
        }
        // `besselj(nu, x)` at a literal order: for each order J_nu is analytic in x on x > 0
        // (entire at an integer order) and not identically zero, so its zeros in x are
        // isolated -- countably many points z_k -- and Z ⊆ ∪_k {x = z_k}, null where every level
        // set of `x` is. Off x > 0 a non-integer order is NaN (x < 0) or a pole (x = 0, a
        // negative order), no zero. A non-literal order fails closed: `besselj(nu, 0)` vanishes
        // for every nu > 0, and a large order underflows to 0 on whole intervals of x.
        ("besselj", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            besselj_order(&t[1..ja], ops).is_some() && levels_null(&t[ja..], ops)
        }
        // `atan2(y, x)` is 0 where y = 0 (and x >= 0) and at x = +inf for a finite y; elsewhere
        // it is nonzero (underflow aside, the overflow reading's). Z ⊆ Z(y) ∪ {x = +inf}.
        ("atan2", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            zsn(&t[1..ja], ops, d + 1, e2) && isn_e(&t[ja..], ops, d + 1, e2)
        }
        // sums, sin, cos, and any other analytic composition: identity theorem + witness; on
        // refusal, a factor every term of a sum shares (`common_factor_witness`), a slope in one
        // variable (`slice_witness`), the identity theorem on the expression's domain
        // (`domain_witness`) and the E2 clearing fallback, each scoped as E2 is (see the fn doc).
        _ => {
            analytic_nonzero_witness_in(t, ops, e2)
                || (e2
                    && (common_factor_witness(t, ops, d)
                        || slice_witness(t, ops, d)
                        || domain_witness(t, ops)
                        || cleared_witness(t, ops)))
        }
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
            if let Some(ja) = rec(t, i + 1, ops) {
                if let Some(jb) = subtree_end(t, ja, ops) {
                    if matches!(int_literal(&t[ja..jb]), Some(q) if q >= 0) {
                        return Some(jb);
                    }
                }
            }
            return ground_leaf(t, i, ops);
        }
        if !op_entire_analytic(tok) {
            return ground_leaf(t, i, ops);
        }
        let mut j = i + 1;
        for _ in 0..ops.arity_of(tok).unwrap_or(0) as usize {
            j = rec(t, j, ops)?;
        }
        Some(j)
    }
    rec(t, 0, ops).is_some_and(|j| j == t.len())
}

/// Tier-0 (E2 rider, owner-ratified 2026-08-11, audit F81): a GROUND subtree --
/// no variables, no `<constant>` -- denotes a constant, and a constant function
/// is entire regardless of its SPELLING: the exact-fraction literal `/ 5 4` is
/// the number 5/4, not a quotient of functions, yet the op scan used to refuse
/// it (`x3 - 5/4` never reached the witness; the pow-completion note above is
/// the precedent for exactly this class of token-level refusal). Accept the
/// ground subtree as a leaf. Soundness rests on the same argument as opaque
/// leaf tokens: a non-finite constant (ground `1/0`, an inf literal) can never
/// yield the positive-measure FINITE value cell the witness demands, so the
/// witness stays fail-closed without inspecting the value here.
fn ground_leaf(t: &[String], i: usize, ops: &Operators) -> Option<usize> {
    let j = subtree_end(t, i, ops)?;
    if t[i..j]
        .iter()
        .all(|x| var_index(x).is_none() && x != "<constant>")
    {
        Some(j)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// E2 -- the denominator-clearing certificate (owner-ratified 2026-08-11, audit
// F81; full design + soundness chain in the research harness).
//
// A rational-level factor is rewritten as a PAIR (N, D) of ENTIRE spellings
// with factor = N/D wherever the factor is defined, and the EXISTING identity-
// theorem witness runs on N alone. The inclusion Z(factor) subset-of Z(N):
// at a zero of the factor the value is defined, so D != 0 there and
// N = factor * D = 0; the zeros of `inv u` / `pow(u, -q)` live exactly where
// u = +-inf, i.e. where u's own denominator vanishes -- and the pair SWAP puts
// that denominator into N, capturing them; D is entire, hence finite
// everywhere (variables are finite reals by contract), so no third place
// exists. `<constant>` never reaches this code un-renamed: the scoped entry
// renames occurrences to fresh variables BEFORE any recursion (E1), so a
// subterm the clearing DUPLICATES (`a + tan u` puts u into N via sin AND cos)
// keeps one shared variable per occurrence -- the rename-before-clear ordering
// requirement of the design is satisfied by construction here.
//
// Fail-closed everywhere: out-of-scope ops (abs, log, asin, acos, acosh,
// atanh, rootn, odd-root sugar), non-integer exponents, non-literal
// denominators inside transcendental function arguments, non-rational literal
// scales, and cleared spellings past the token budget all refuse.
// ---------------------------------------------------------------------------

/// Exact rational reading of a literal token (bare/parenthesized, decimal or
/// fraction). `np.pi`/`np.e`/inf/nan refuse: their exact reciprocals have no
/// token spelling, and this reader only serves the function-argument scale.
fn rat_of_token(tok: &str) -> Option<crate::ac::rat::Rat> {
    use crate::ac::rat::Rat;
    let bare = |s: &str| {
        if let Some(r) = Rat::parse_decimal(s) {
            return Some(r);
        }
        if let Some((p, q)) = crate::utils::split_fraction(s) {
            if let (Ok(p), Ok(q)) = (p.parse::<i128>(), q.parse::<i128>()) {
                return Rat::new(p, q);
            }
        }
        // beyond i128: the exact reader, within the number type's cap
        let (p, q) = crate::ac::rat::token_rational(s)?;
        Rat::from_big(p, q)
    };
    if let Some(inner) = tok.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
        return bare(inner);
    }
    bare(tok)
}

/// One-token spelling of an exact rational for WITNESS-INTERNAL use only --
/// this token feeds `analytic_nonzero_witness`, never the dialect, so no
/// argmin aesthetics apply. Negatives spell parenthesized (the readers'
/// composed grammar, H-012/H-048).
fn rat_to_token(r: &crate::ac::rat::Rat) -> String {
    let s = if r.is_integer() {
        r.numer_string()
    } else {
        format!("{}/{}", r.numer_string(), r.denom_string())
    };
    if r.is_negative() {
        format!("({s})")
    } else {
        s
    }
}

const CLEAR_BUDGET: usize = 3000; // tokens per pair side; past this: refuse

fn cw_mul(a: Vec<String>, b: Vec<String>) -> Option<Vec<String>> {
    if a.len() == 1 && a[0] == "1" {
        return Some(b);
    }
    if b.len() == 1 && b[0] == "1" {
        return Some(a);
    }
    if a.len() + b.len() + 1 > CLEAR_BUDGET {
        return None;
    }
    let mut out = Vec::with_capacity(a.len() + b.len() + 1);
    out.push("*".to_string());
    out.extend(a);
    out.extend(b);
    Some(out)
}

fn cw_pow(base: Vec<String>, q: i64) -> Option<Vec<String>> {
    if q == 1 {
        return Some(base);
    }
    if base.len() + 2 > CLEAR_BUDGET {
        return None;
    }
    let mut out = Vec::with_capacity(base.len() + 2);
    out.push("pow".to_string());
    out.extend(base);
    out.push(q.to_string());
    Some(out)
}

/// A transcendental function's argument must be denominator-free up to a
/// nonzero RATIONAL literal scale, rewritten as multiplication by the exact
/// reciprocal ("pole inside a function argument stays refused until its own
/// lemma" -- the ratified boundary).
fn cw_arg(n: Vec<String>, d: Vec<String>) -> Option<Vec<String>> {
    if d.len() == 1 && d[0] == "1" {
        return Some(n);
    }
    if d.len() == 1 {
        let r = rat_of_token(&d[0])?;
        if r.is_zero() {
            return None;
        }
        let inv = r.checked_inv()?;
        return cw_mul(vec![rat_to_token(&inv)], n);
    }
    None
}

/// The pair recursion: `(end, N, D)` for the subtree at `i`; `None` refuses.
fn clear_pair(
    t: &[String],
    i: usize,
    ops: &Operators,
    d: u32,
) -> Option<(usize, Vec<String>, Vec<String>)> {
    if d > CERT_RECURSION_MAX {
        return None;
    }
    let one = || vec!["1".to_string()];
    let tok = t.get(i)?.as_str();
    if !ops.is_operator(tok) {
        return Some((i + 1, vec![tok.to_string()], one()));
    }
    match (tok, ops.arity_of(tok)) {
        ("neg", Some(1)) => {
            let (j, n, dd) = clear_pair(t, i + 1, ops, d + 1)?;
            let mut nn = Vec::with_capacity(n.len() + 1);
            nn.push("neg".to_string());
            nn.extend(n);
            Some((j, nn, dd))
        }
        ("inv", Some(1)) => {
            let (j, n, dd) = clear_pair(t, i + 1, ops, d + 1)?;
            Some((j, dd, n)) // the swap: zeros of 1/u are u's infinities
        }
        // Linear literal scalings distribute over the quotient: k*(N/D) = (k*N)/D.
        ("mult2" | "mult3" | "mult4" | "mult5" | "div2" | "div3" | "div4" | "div5", Some(1)) => {
            let (j, n, dd) = clear_pair(t, i + 1, ops, d + 1)?;
            let mut nn = Vec::with_capacity(n.len() + 1);
            nn.push(tok.to_string());
            nn.extend(n);
            Some((j, nn, dd))
        }
        ("pow2" | "pow3" | "pow4" | "pow5", Some(1)) => {
            let q = i64::from(tok.as_bytes()[3] - b'0');
            let (j, n, dd) = clear_pair(t, i + 1, ops, d + 1)?;
            Some((j, cw_pow(n, q)?, cw_pow(dd, q)?))
        }
        ("tan", Some(1)) => {
            let (j, n, dd) = clear_pair(t, i + 1, ops, d + 1)?;
            let u = cw_arg(n, dd)?;
            let mut sn = Vec::with_capacity(u.len() + 1);
            sn.push("sin".to_string());
            sn.extend(u.iter().cloned());
            let mut cs = Vec::with_capacity(u.len() + 1);
            cs.push("cos".to_string());
            cs.extend(u);
            Some((j, sn, cs))
        }
        ("exp" | "sin" | "cos" | "sinh" | "cosh" | "tanh" | "atan" | "asinh", Some(1)) => {
            let (j, n, dd) = clear_pair(t, i + 1, ops, d + 1)?;
            let u = cw_arg(n, dd)?;
            let mut f = Vec::with_capacity(u.len() + 1);
            f.push(tok.to_string());
            f.extend(u);
            Some((j, f, one()))
        }
        ("pow", Some(2)) => {
            let (ja, n, dd) = clear_pair(t, i + 1, ops, d + 1)?;
            let jb = subtree_end(t, ja, ops)?;
            let q = int_literal(t.get(ja..jb)?)?;
            if q >= 0 {
                Some((jb, cw_pow(n, q)?, cw_pow(dd, q)?))
            } else {
                Some((jb, cw_pow(dd, -q)?, cw_pow(n, -q)?)) // swap, powered
            }
        }
        ("+" | "-", Some(2)) => {
            let (ja, na, da) = clear_pair(t, i + 1, ops, d + 1)?;
            let (jb, nb, db) = clear_pair(t, ja, ops, d + 1)?;
            let l = cw_mul(na, db.clone())?;
            let r = cw_mul(nb, da.clone())?;
            if l.len() + r.len() + 1 > CLEAR_BUDGET {
                return None;
            }
            let mut n = Vec::with_capacity(l.len() + r.len() + 1);
            n.push(tok.to_string());
            n.extend(l);
            n.extend(r);
            Some((jb, n, cw_mul(da, db)?))
        }
        ("*", Some(2)) => {
            let (ja, na, da) = clear_pair(t, i + 1, ops, d + 1)?;
            let (jb, nb, db) = clear_pair(t, ja, ops, d + 1)?;
            Some((jb, cw_mul(na, nb)?, cw_mul(da, db)?))
        }
        ("/", Some(2)) => {
            let (ja, na, da) = clear_pair(t, i + 1, ops, d + 1)?;
            let (jb, nb, db) = clear_pair(t, ja, ops, d + 1)?;
            Some((jb, cw_mul(na, db)?, cw_mul(da, nb)?))
        }
        // abs, log, asin, acos, acosh, atanh, rootn, odd-root sugar, anything
        // else: out of the ratified scope -- refuse.
        _ => None,
    }
}

/// The E2 certificate body: clear the whole slice, witness the numerator.
fn cleared_witness(t: &[String], ops: &Operators) -> bool {
    let Some((end, n, d)) = clear_pair(t, 0, ops, 0) else {
        return false;
    };
    if end != t.len() || n.len() > CLEAR_BUDGET {
        return false;
    }
    // The clearing identity Z(t) = Z(n) PRESUMES the cleared denominator is finite
    // a.e.: `1/g = 0` exactly on {g = +-inf}, so an a.e.-infinite denominator mints
    // a FAT zero set the numerator never sees. Without this side condition the
    // certificate declared `inv(inf + h)` nonzero-a.e. while it is identically 0,
    // licensing a per-factor move that flips the sign of infinity on positive
    // measure (F1, 2026-08-24, benchmark row 17425). `isn` is the existing
    // authority for exactly this question; an empty denominator cleared nothing.
    if !d.is_empty() && !isn(&d, ops, 0) {
        return false;
    }
    analytic_nonzero_witness(&n, ops)
}

fn analytic_nonzero_witness(t: &[String], ops: &Operators) -> bool {
    analytic_nonzero_witness_in(t, ops, false)
}

/// `analytic_nonzero_witness`, where `far` searches the largest box (`horizon`'s `R_MAX`) when the
/// expression's literals put its horizon past it, instead of refusing: one cell anywhere on which
/// an entire function is certainly nonzero proves its zero set null (the identity theorem on the
/// connected R^n), so the horizon only says where to look. `zsn` takes it in `e2`'s scope:
/// `0.41*x1^5 + 182*x1^4 + ... + 1.9e9` is read, its literals past `R_MAX / 2`.
fn analytic_nonzero_witness_in(t: &[String], ops: &Operators, far: bool) -> bool {
    const WITNESS_BUDGET: u32 = 2_000;
    if !entire_analytic_composition(t, ops) {
        return false;
    }
    let (r, depth_per_dim, decidable) = horizon(&[t], &[&[]], ops);
    if !decidable && !far {
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

// ---------------------------------------------------------------------------
// THE IDENTITY THEOREM ON THE EXPRESSION'S DOMAIN. `analytic_nonzero_witness` needs f analytic on
// all of R^n, so a sum over `log`, an even root, a fractional power, `asin`, a quotient or `abs`
// never got past it: `log(x2) - x2` is never zero, yet its zero set was "not proven null". A zero
// is a value, so only where f is DEFINED can it be one; where f is NaN it is no zero (and `1/NaN`
// carries no sign).
//
// The argument. Every operator of f is real-analytic on an open domain: the entire ones on R,
// `log`, an even root and a fractional or symbolic power where their argument is > 0, `asin`,
// `acos`, `atanh` on (-1, 1), `acosh` on (1, inf), and `inv`, a quotient's denominator, a negative
// integer power, `abs` and an odd root where their argument is != 0. So f is analytic on the open
// set D where every such argument lies in its domain. Off D f is undefined (NaN, no zero), or the
// argument sits on the domain's edge, a level set {g = c}. Where each restricted argument g is
// AFFINE in the variables (or its domain holds over the whole space, `vacuous`), every edge is a
// hyperplane, a null set, and D is a union of convex pieces, one per sign of the arguments that
// must be != 0; a convex set is connected. By the identity theorem, f vanishing on a set of
// positive measure in a piece vanishes on the whole piece. So the zero set is null as soon as
// each piece holds one cell of positive measure, certified inside the piece by interval
// evaluation of every restricted argument, on which f is certified nonzero. A piece in which f is
// identically zero never yields such a cell (`x1 - |x1|` on x1 > 0), and neither does an empty or
// unreachable piece: the test fails closed, it never guesses.
// ---------------------------------------------------------------------------

/// Where an operator is real-analytic: its argument `g` must be positive, in (-1, 1), above 1, or
/// nonzero (`domain_witness`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Need {
    Pos,
    Unit,
    AboveOne,
    NonZero,
}

/// The domain restrictions of `t`: each restricted argument as its token range, with what it needs.
/// `None` where an operator is outside the analytic vocabulary (`tan`, whose poles cut the line into
/// infinitely many pieces, a non-literal root index, ...). A ground subtree is a constant, decided
/// by its value.
fn analytic_domain(t: &[String], ops: &Operators) -> Option<Vec<(usize, usize, Need)>> {
    fn rec(
        t: &[String],
        i: usize,
        ops: &Operators,
        out: &mut Vec<(usize, usize, Need)>,
    ) -> Option<usize> {
        let tok = t.get(i)?.as_str();
        if !ops.is_operator(tok) {
            return Some(i + 1);
        }
        if let Some(j) = ground_leaf(t, i, ops) {
            return Some(j);
        }
        let one = |need: Need, out: &mut Vec<(usize, usize, Need)>| -> Option<usize> {
            let j = rec(t, i + 1, ops, out)?;
            out.push((i + 1, j, need));
            Some(j)
        };
        match (tok, ops.arity_of(tok)?) {
            ("log" | "pow1_2" | "pow1_4", 1) => one(Need::Pos, out),
            ("asin" | "acos" | "atanh", 1) => one(Need::Unit, out),
            ("acosh", 1) => one(Need::AboveOne, out),
            ("inv" | "abs" | "pow1_3" | "pow1_5", 1) => one(Need::NonZero, out),
            ("/", 2) => {
                let ja = rec(t, i + 1, ops, out)?;
                let jb = rec(t, ja, ops, out)?;
                out.push((ja, jb, Need::NonZero));
                Some(jb)
            }
            ("pow", 2) => {
                let ja = rec(t, i + 1, ops, out)?;
                let jb = subtree_end(t, ja, ops)?;
                match int_literal(&t[ja..jb]) {
                    Some(q) if q >= 0 => {}
                    Some(_) => out.push((i + 1, ja, Need::NonZero)),
                    // a fractional or symbolic exponent: `exp(e * log(b))`, analytic for b > 0
                    None => {
                        rec(t, ja, ops, out)?;
                        out.push((i + 1, ja, Need::Pos));
                    }
                }
                Some(jb)
            }
            ("rootn", 2) => {
                let ja = rec(t, i + 1, ops, out)?;
                let jb = subtree_end(t, ja, ops)?;
                match int_literal(&t[ja..jb]) {
                    Some(n) if n >= 2 && n % 2 == 0 => out.push((i + 1, ja, Need::Pos)),
                    Some(n) if n >= 3 => out.push((i + 1, ja, Need::NonZero)),
                    _ => return None,
                }
                Some(jb)
            }
            (op, n) if op_entire_analytic(op) => {
                let mut j = i + 1;
                for _ in 0..n {
                    j = rec(t, j, ops, out)?;
                }
                Some(j)
            }
            _ => None,
        }
    }
    let mut out = Vec::new();
    (rec(t, 0, ops, &mut out)? == t.len()).then_some(out)
}

/// Is `g` affine in the variables: a variable, a constant, their sums and differences, and their
/// multiples and quotients by constants? Its level sets are then hyperplanes and its sub- and
/// superlevel sets half-spaces.
fn affine(g: &[String], ops: &Operators) -> bool {
    fn rec(t: &[String], i: usize, ops: &Operators) -> Option<usize> {
        let tok = t.get(i)?.as_str();
        if !ops.is_operator(tok) {
            return Some(i + 1);
        }
        if let Some(j) = ground_leaf(t, i, ops) {
            return Some(j);
        }
        match (tok, ops.arity_of(tok)?) {
            ("+" | "-", 2) => {
                let ja = rec(t, i + 1, ops)?;
                rec(t, ja, ops)
            }
            (
                "neg" | "mult2" | "mult3" | "mult4" | "mult5" | "div2" | "div3" | "div4" | "div5",
                1,
            ) => rec(t, i + 1, ops),
            ("*", 2) => {
                let ja = subtree_end(t, i + 1, ops)?;
                let jb = subtree_end(t, ja, ops)?;
                if ground_leaf(t, i + 1, ops).is_some() {
                    rec(t, ja, ops).filter(|&j| j == jb)
                } else if ground_leaf(t, ja, ops).is_some() {
                    rec(t, i + 1, ops).filter(|&j| j == ja).map(|_| jb)
                } else {
                    None
                }
            }
            ("/", 2) => {
                let ja = rec(t, i + 1, ops)?;
                ground_leaf(t, ja, ops)
            }
            _ => None,
        }
    }
    rec(g, 0, ops) == Some(g.len())
}

/// Does `v`, a value set of a restricted argument, satisfy `need` throughout (`strict` on its finite
/// part; an argument that is NaN somewhere is restricted by its own operators)?
fn meets(v: &Vs, need: Need, sign: bool) -> bool {
    if !v.has_fin {
        return false;
    }
    let above = |c: f64| v.lo > c || (v.lo == c && v.lo_open);
    let below = |c: f64| v.hi < c || (v.hi == c && v.hi_open);
    match need {
        Need::Pos => above(0.0),
        Need::Unit => above(-1.0) && below(1.0),
        Need::AboveOne => above(1.0),
        Need::NonZero if sign => above(0.0),
        Need::NonZero => below(0.0),
    }
}

/// Does `v` certainly miss `need` everywhere (a cell wholly outside the piece)?
fn misses(v: &Vs, need: Need, sign: bool) -> bool {
    if !v.has_fin {
        return true;
    }
    let at_most = |c: f64| v.hi <= c;
    let at_least = |c: f64| v.lo >= c;
    match need {
        Need::Pos => at_most(0.0),
        Need::Unit => at_most(-1.0) || at_least(1.0),
        Need::AboveOne => at_most(1.0),
        Need::NonZero if sign => at_most(0.0),
        Need::NonZero => at_least(0.0),
    }
}

/// THE ZERO SET IS NULL ON THE DOMAIN (see the section note): every operator of `t` real-analytic
/// on a domain bounded by hyperplanes, and in each convex piece of it a cell where every restricted
/// argument certainly lies in its domain and `t` is certainly nonzero. Fail-closed on budget,
/// horizon, a non-affine restricted argument whose domain is not the whole space, and on more than
/// `MAX_SIGNS` arguments that must be nonzero (`2^MAX_SIGNS` pieces).
fn domain_witness(t: &[String], ops: &Operators) -> bool {
    const MAX_SIGNS: usize = 3;
    const WITNESS_BUDGET: u32 = 2_000;
    let Some(needs) = analytic_domain(t, ops) else {
        return false;
    };
    // No restriction: f is entire, `analytic_nonzero_witness`'s case.
    if needs.is_empty() {
        return false;
    }
    // Each restriction: dropped where it holds over the whole space, else its argument must be
    // affine. An argument that must be nonzero is read up to its sign (`neg g`), once, and not at
    // all where another restriction on it already keeps it away from zero.
    let mut kept: Vec<(&[String], Need)> = Vec::new();
    for &(a, b, need) in &needs {
        let mut g = &t[a..b];
        if need == Need::NonZero && g.first().is_some_and(|s| s == "neg") {
            g = &g[1..];
        }
        if kept.iter().any(|&(h, n)| h == g && n == need) {
            continue;
        }
        if whole_range(g, ops).is_some_and(|v| {
            meets(&v, need, true) || (need == Need::NonZero && meets(&v, need, false))
        }) {
            continue;
        }
        if !affine(g, ops) {
            return false;
        }
        kept.push((g, need));
    }
    kept.retain(|&(g, need)| need != Need::NonZero || !needs_positive_elsewhere(g, &needs, t));
    let signs: Vec<&[String]> = kept
        .iter()
        .filter(|(_, n)| *n == Need::NonZero)
        .map(|(g, _)| *g)
        .collect();
    if signs.len() > MAX_SIGNS {
        return false;
    }
    let pieces = 1usize << signs.len();
    let (r, depth_per_dim, decidable) = horizon(&[t], &[&[]], ops);
    if !decidable {
        return false;
    }
    let used = distinct_vars(&[t]);
    let width = n_var_slots(&[t]);
    let n = used.len().max(1);
    let mut bx = vec![(-r, r); n];
    let mut budget = WITNESS_BUDGET.saturating_mul(pieces as u32);
    let mut found = vec![false; pieces];
    struct Search<'a> {
        t: &'a [String],
        ops: &'a Operators,
        kept: &'a [(&'a [String], Need)],
        signs: &'a [&'a [String]],
        r: f64,
        used: &'a [usize],
        width: usize,
        depth_max: u32,
    }
    // The pieces this cell may still meet (bit k: the sign of `signs[k]`), and for each the
    // verdict: a witness, outside, or undecided.
    fn rec(s: &Search, bx: &mut [(f64, f64)], d: u32, budget: &mut u32, found: &mut [bool]) {
        if *budget == 0 || found.iter().all(|&f| f) {
            return;
        }
        *budget -= 1;
        let doms = doms_from_box(bx, s.used, s.width, (-s.r, s.r));
        let mut open: Vec<usize> = Vec::new();
        let vals: Vec<Option<Vs>> = s
            .kept
            .iter()
            .map(|(g, _)| value_set_p(g, s.ops, &doms, &[]))
            .collect();
        for (piece, done) in found.iter_mut().enumerate() {
            if *done {
                continue;
            }
            let sign_of = |g: &[String]| {
                let k = s
                    .signs
                    .iter()
                    .position(|h| *h == g)
                    .expect("a sign argument");
                piece >> k & 1 == 1
            };
            let mut inside = true;
            let mut outside = false;
            for ((g, need), v) in s.kept.iter().zip(&vals) {
                let sign = *need != Need::NonZero || sign_of(g);
                match v {
                    Some(v) if meets(v, *need, sign) => {}
                    Some(v) if misses(v, *need, sign) => {
                        outside = true;
                        inside = false;
                    }
                    _ => inside = false,
                }
            }
            if outside {
                continue;
            }
            if inside {
                if let Some(v) = value_set_p(s.t, s.ops, &doms, &[]) {
                    let zero_in = (v.lo < 0.0 || (v.lo == 0.0 && !v.lo_open))
                        && (v.hi > 0.0 || (v.hi == 0.0 && !v.hi_open));
                    if v.fin_pm() && !zero_in && !v.nan {
                        *done = true;
                        continue;
                    }
                }
            }
            open.push(piece);
        }
        if open.is_empty() || d >= s.depth_max {
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
        rec(s, bx, d + 1, budget, found);
        bx[i] = (mid, hi);
        rec(s, bx, d + 1, budget, found);
        bx[i] = (lo, hi);
    }
    let s = Search {
        t,
        ops,
        kept: &kept,
        signs: &signs,
        r,
        used: &used,
        width,
        depth_max: depth_per_dim * n as u32,
    };
    rec(&s, &mut bx, 0, &mut budget, &mut found);
    found.iter().all(|&f| f)
}

// ---------------------------------------------------------------------------
// THE ZERO SET IS NULL ON A SLICE (`slice_witness`). A sum over restricted arguments that are not
// affine (`log(x1*x2) - x1`, `sqrt(x1 - x2^2) - x3`, `log(log(x1)) - x2`) has a domain whose
// pieces `domain_witness` cannot list. The test reads it one variable at a time instead.
//
// Fix a variable x_k and write x' for the others. For a subexpression e and a value of x', the
// SLICE of e is the function t -> e(x', t) of one real variable. A SLOPE of e in x_k is a claim
// about the slice for almost every x' (all but a null set of x'): there is an open set U of the
// line on which the slice is a finite real-analytic function, with only countably many points
// outside U where the slice is a number or an infinity (not NaN), and
//   * Free: e does not hold x_k (its slice is one value, whatever t);
//   * Inc / Dec: on each interval of U the slice is strictly increasing / decreasing;
//   * Mono: on each interval of U it is strictly monotone (the direction may depend on x');
//   * Moving: on each interval of U it is not constant.
// Every slope but Free implies Moving. If the whole expression has a slope other than Free, its
// slice vanishes only at countably many points for almost every x': on an interval of U a
// nonconstant analytic function has isolated zeros, U has countably many intervals, and the
// points outside U are countable. By Fubini (the zero set is Borel) the zero set is then null.
//
// The slopes are built bottom-up; anything outside the rules below has no slope and the test
// fails closed.
//   * x_k is Inc. A subtree without x_k is Free.
//   * A Free summand c needs {c = +-inf} null (`isn`), so for almost every x' it is a finite
//     number or NaN, which shifts the slice or makes it NaN. A Free factor or divisor c needs
//     {c = 0} and {c = +-inf} null (`zsn`, `isn`), so it scales the slice by a finite nonzero
//     number (a direction flip where its sign is not certain: Mono) or makes it NaN.
//   * Two summands with x_k: both Inc or both Dec (on an interval of the intersection of their U
//     both are monotone the same way).
//   * A product or quotient of two strictly monotone factors with x_k whose signs are certain
//     (interval evaluation over the whole space): where their magnitudes rise together (or fall
//     together) on each interval, so does the product's, so it is Inc or Dec by its sign.
//   * Any other mix of two summands, and a product or quotient of two factors with x_k, each with
//     a slope: Moving where the derivative in x_k (`slice_derivative`, the textbook rules, exact
//     on the analytic domain) has a null zero set (`zsn`, read again). On an interval of the
//     intersection of their U both are analytic, so is the result, and its derivative there is
//     the derivative's slice; for almost every x' that slice vanishes on a null set, so the
//     result is constant on no interval. `x1 - log(x1)` (derivative `1 - 1/x1`) moves, and
//     `x1*(1/x1)` (derivative `1/x1 - x1/x1^2`, zero wherever defined) does not.
//   * h(g) for a unary h with g Moving: h is real-analytic on an open set D_h, its domain outside
//     D_h is countably many points (`log` and `sqrt` at 0, `abs` and `inv` at 0, `asin` at
//     +-1, `tan` at its poles), and h is not constant on any interval of D_h. A level set of
//     g on an interval of U is countable (g is analytic and not constant there), so h(g) has a
//     U of its own: U_g minus the countably many points where g leaves D_h; on each of its
//     intervals g covers an interval of D_h, where h is not constant (identity theorem), so h(g)
//     is Moving, and Inc / Dec / Mono where h is monotone on each interval of D_h. An infinity
//     of g that h maps to a number (`exp(-inf) = 0`, `1/inf = 0`) is one of g's countably many
//     points outside U_g.
//   * An integer power: odd ones are increasing, even ones Moving, negative ones the inverse of
//     the positive; a non-integer literal power is analytic and monotone where its base is
//     positive and NaN where it is negative (`numeric::aligned_pow`); a root of literal index is
//     increasing where it is defined.
// The reading is the real one, as `zsn`'s (the float64 tails are `on_float64_tails`'s).
// ---------------------------------------------------------------------------

/// A slope of a subexpression in one variable (see the section note).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slope {
    Free,
    Inc,
    Dec,
    Mono,
    Moving,
}

impl Slope {
    /// The slope of `-e` (and of `h(e)` for a decreasing `h`).
    fn flip(self) -> Slope {
        match self {
            Slope::Inc => Slope::Dec,
            Slope::Dec => Slope::Inc,
            s => s,
        }
    }

    /// The slope of `h(e)` for an `h` that is not constant on any interval of its analytic
    /// domain, nor monotone on each.
    fn moving(self) -> Slope {
        match self {
            Slope::Free => Slope::Free,
            _ => Slope::Moving,
        }
    }
}

/// THE ZERO SET IS NULL ON A SLICE (see the section note): some variable of `t` in which `t` has a
/// slope other than `Free`. Fail-closed: a subtree outside the rules has no slope.
fn slice_witness(t: &[String], ops: &Operators, d: u32) -> bool {
    let mut vars: Vec<usize> = t.iter().filter_map(|x| var_index(x)).collect();
    vars.sort_unstable();
    vars.dedup();
    vars.into_iter()
        .any(|k| matches!(slope(t, 0, ops, k, d), Some((j, s)) if j == t.len() && s != Slope::Free))
}

/// The slope of the subtree at `i` in the variable `x_k`, with its end; `None` where it has none.
fn slope(t: &[String], i: usize, ops: &Operators, k: usize, d: u32) -> Option<(usize, Slope)> {
    if d > CERT_RECURSION_MAX {
        return None;
    }
    let end = subtree_end(t, i, ops)?;
    if !t[i..end].iter().any(|x| var_index(x) == Some(k)) {
        return Some((end, Slope::Free));
    }
    let tok = t[i].as_str();
    if !ops.is_operator(tok) {
        return Some((end, Slope::Inc)); // x_k itself
    }
    // A Free summand: finite or NaN almost everywhere. A Free factor: finite, nonzero or NaN.
    let summand = |c: &[String]| isn_e(c, ops, d + 1, true);
    let factor = |c: &[String]| zsn(c, ops, d + 1, true) && isn_e(c, ops, d + 1, true);
    // The one sign `c` takes wherever it is a finite number other than zero (`Some(true)`: negative):
    // the zeros of a factor here are a null set of x' (a Free factor) or countably many points of a
    // slice (one with x_k), which U leaves out.
    let sign = |c: &[String]| match whole_range(c, ops) {
        Some(v) if v.has_fin && v.lo >= 0.0 => Some(false),
        Some(v) if v.has_fin && v.hi <= 0.0 => Some(true),
        _ => None,
    };
    // `c * s` for a Free factor `c`: the direction where the sign of `c` is certain.
    let scaled = |c: &[String], s: Slope| -> Slope {
        match sign(c) {
            Some(false) => s,
            Some(true) => s.flip(),
            None => match s {
                Slope::Inc | Slope::Dec => Slope::Mono,
                s => s,
            },
        }
    };
    // `a * b` for two strictly monotone factors of certain signs: where their magnitudes move the
    // same way, so does the magnitude of the product, whose sign is certain too.
    let monotone_product = |a: (Slope, Option<bool>), b: (Slope, Option<bool>)| -> Option<Slope> {
        let rises = |(s, neg): (Slope, Option<bool>)| match (s, neg?) {
            (Slope::Inc, n) | (Slope::Dec, n) => Some((s == Slope::Inc) != n),
            _ => None,
        };
        let (ra, rb) = (rises(a)?, rises(b)?);
        let negative = a.1? != b.1?;
        (ra == rb).then_some(if ra != negative {
            Slope::Inc
        } else {
            Slope::Dec
        })
    };
    let sum = |a: Slope, b: Slope| -> Option<Slope> {
        match (a, b) {
            (Slope::Free, s) | (s, Slope::Free) => Some(s),
            (Slope::Inc, Slope::Inc) => Some(Slope::Inc),
            (Slope::Dec, Slope::Dec) => Some(Slope::Dec),
            _ => None,
        }
    };
    let unary = |f: fn(Slope) -> Slope| -> Option<(usize, Slope)> {
        let (j, s) = slope(t, i + 1, ops, k, d + 1)?;
        Some((j, f(s)))
    };
    // Does the subtree ending at `j` move by its derivative (see the section note)? A derivative
    // is read with `DERIVATIVE_DEPTH` more of the recursion budget spent, and no derivative is read
    // within the reading of another (a cost bound: each would differentiate a larger tree).
    let moves = |j: usize| {
        d < DERIVATIVE_DEPTH
            && slice_derivative(&t[i..j], ops, k, 0)
                .is_some_and(|dt| zsn(&dt, ops, d + DERIVATIVE_DEPTH, true))
    };
    match (tok, ops.arity_of(tok)?) {
        // increasing on each interval of an analytic domain whose edge is countably many points
        (
            "mult2" | "mult3" | "mult4" | "mult5" | "div2" | "div3" | "div4" | "div5" | "exp"
            | "log" | "sinh" | "tanh" | "atan" | "asinh" | "asin" | "atanh" | "acosh" | "erf"
            | "tan" | "pow3" | "pow5" | "pow1_2" | "pow1_3" | "pow1_4" | "pow1_5",
            1,
        ) => unary(|s| s),
        // decreasing so
        ("neg" | "acos" | "inv", 1) => unary(Slope::flip),
        // not constant on any interval of such a domain
        ("sin" | "cos" | "cosh" | "abs" | "pow2" | "pow4", 1) => unary(Slope::moving),
        ("+" | "-", 2) => {
            let (ja, a) = slope(t, i + 1, ops, k, d + 1)?;
            let (jb, b) = slope(t, ja, ops, k, d + 1)?;
            if (a == Slope::Free && !summand(&t[i + 1..ja]))
                || (b == Slope::Free && !summand(&t[ja..jb]))
            {
                return None;
            }
            let b = if tok == "-" { b.flip() } else { b };
            match sum(a, b) {
                Some(s) => Some((jb, s)),
                None => moves(jb).then_some((jb, Slope::Moving)),
            }
        }
        ("*" | "/", 2) => {
            let (ja, a) = slope(t, i + 1, ops, k, d + 1)?;
            let (jb, b) = slope(t, ja, ops, k, d + 1)?;
            // a divisor is the inverse of its slope; a Free one scales as a factor (its zero and
            // infinite sets swap, its sign stays)
            let b = if tok == "/" { b.flip() } else { b };
            match (a, b) {
                (Slope::Free, s) if factor(&t[i + 1..ja]) => Some((jb, scaled(&t[i + 1..ja], s))),
                (s, Slope::Free) if factor(&t[ja..jb]) => Some((jb, scaled(&t[ja..jb], s))),
                (Slope::Free, _) | (_, Slope::Free) => None,
                _ => match monotone_product((a, sign(&t[i + 1..ja])), (b, sign(&t[ja..jb]))) {
                    Some(s) => Some((jb, s)),
                    None => moves(jb).then_some((jb, Slope::Moving)),
                },
            }
        }
        ("pow", 2) => {
            let (ja, b) = slope(t, i + 1, ops, k, d + 1)?;
            let jb = subtree_end(t, ja, ops)?;
            // the exponent: a ground subtree (a literal or a literal's spelling, `/ 1 2`)
            ground_leaf(t, ja, ops)?;
            let p = crate::numeric::evaluate_constant_value(&t[ja..jb], ops)?;
            if !p.is_finite() || p == 0.0 {
                return None;
            }
            // an integer power: odd ones increase, even ones move; a non-integer one is
            // x^p = exp(p log x) where x > 0, NaN where x < 0 (`numeric::aligned_pow`)
            let s = if p.fract() == 0.0 && (p / 2.0).fract() == 0.0 {
                b.moving()
            } else {
                b
            };
            Some((jb, if p < 0.0 { s.flip() } else { s }))
        }
        ("rootn", 2) => {
            let (ja, b) = slope(t, i + 1, ops, k, d + 1)?;
            let jb = subtree_end(t, ja, ops)?;
            match int_literal(&t[ja..jb]) {
                Some(n) if n >= 2 => Some((jb, b)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A COMMON FACTOR OF A SUM: `t` a sum (of `+`, `-` and `neg`) whose every term is a product (of
/// `*`, `neg` and literal scalings) holding one factor `g` in common. Where `g` is finite the sum
/// is `g` times the sum of the cofactors, and where `g` is infinite each term is infinite or NaN,
/// so the sum is never zero there: its zero set lies in `Z(g)` and the cofactors' sum's, and is
/// null where both are (`zsn`, read again). So `-10.7*cos(u) - 8.9*x2*cos(u)` is null, its
/// cofactors' sum `-10.7 - 8.9*x2` being so, wherever `cos(u)` is; interval evaluation alone loses
/// it where `u` moves fast. False where `t` is no such sum.
fn common_factor_witness(t: &[String], ops: &Operators, d: u32) -> bool {
    // the terms of the sum `t`, each with its sign
    fn terms<'a>(t: &'a [String], ops: &Operators, neg: bool, out: &mut Vec<(bool, &'a [String])>) {
        match t[0].as_str() {
            "+" | "-" if ops.arity_of(&t[0]) == Some(2) => {
                let Some(ja) = subtree_end(t, 1, ops) else {
                    out.clear();
                    return;
                };
                terms(&t[1..ja], ops, neg, out);
                terms(&t[ja..], ops, neg != (t[0] == "-"), out);
            }
            "neg" if ops.arity_of("neg") == Some(1) => terms(&t[1..], ops, !neg, out),
            _ => out.push((neg, t)),
        }
    }
    // the factors of a term (its signs and literal scalings dropped: they scale by a nonzero)
    fn factors<'a>(t: &'a [String], ops: &Operators, out: &mut Vec<&'a [String]>) -> Option<()> {
        match (t[0].as_str(), ops.arity_of(&t[0])) {
            ("*", Some(2)) => {
                let ja = subtree_end(t, 1, ops)?;
                factors(&t[1..ja], ops, out)?;
                factors(&t[ja..], ops, out)
            }
            (
                "neg" | "mult2" | "mult3" | "mult4" | "mult5" | "div2" | "div3" | "div4" | "div5",
                Some(1),
            ) => factors(&t[1..], ops, out),
            _ => {
                out.push(t);
                Some(())
            }
        }
    }
    if !matches!(t[0].as_str(), "+" | "-") || ops.arity_of(&t[0]) != Some(2) {
        return false;
    }
    let mut ts = Vec::new();
    terms(t, ops, false, &mut ts);
    if ts.len() < 2 {
        return false;
    }
    let mut fs: Vec<Vec<&[String]>> = Vec::with_capacity(ts.len());
    for (_, term) in &ts {
        let mut f = Vec::new();
        if factors(term, ops, &mut f).is_none() {
            return false;
        }
        fs.push(f);
    }
    // the first factor of the first term, holding a variable, that every other term holds
    let shared = fs[0].iter().copied().find(|g| {
        g.iter().any(|x| var_index(x).is_some()) && fs[1..].iter().all(|f| f.contains(g))
    });
    let Some(g) = shared else {
        return false;
    };
    // the cofactors' sum, each term's sign and literal scalings kept on its cofactor
    let mut rest: Vec<String> = Vec::new();
    for (n, ((neg, term), _)) in ts.iter().zip(&fs).enumerate() {
        let cof = drop_factor(term, g, ops).unwrap_or_else(|| vec!["1".to_string()]);
        let signed = if *neg {
            [vec!["neg".to_string()], cof].concat()
        } else {
            cof
        };
        rest = if n == 0 {
            signed
        } else {
            [vec!["+".to_string()], rest, signed].concat()
        };
        if rest.len() > DERIVATIVE_BUDGET {
            return false;
        }
    }
    zsn(g, ops, d + 1, true) && zsn(&rest, ops, d + 1, true)
}

/// `t` (a product of `*`, `neg` and literal scalings) with one occurrence of the factor `g`
/// removed; `None` where nothing is left (the term is `g` up to its scalings, a cofactor of 1).
fn drop_factor(t: &[String], g: &[String], ops: &Operators) -> Option<Vec<String>> {
    if t == g {
        return None;
    }
    match (t[0].as_str(), ops.arity_of(&t[0])) {
        ("*", Some(2)) => {
            let ja = subtree_end(t, 1, ops)?;
            let (a, b) = (&t[1..ja], &t[ja..]);
            if contains_factor(a, g, ops) {
                Some(match drop_factor(a, g, ops) {
                    Some(ra) => [vec!["*".to_string()], ra, b.to_vec()].concat(),
                    None => b.to_vec(),
                })
            } else {
                Some(match drop_factor(b, g, ops) {
                    Some(rb) => [vec!["*".to_string()], a.to_vec(), rb].concat(),
                    None => a.to_vec(),
                })
            }
        }
        (
            "neg" | "mult2" | "mult3" | "mult4" | "mult5" | "div2" | "div3" | "div4" | "div5",
            Some(1),
        ) => {
            let inner = drop_factor(&t[1..], g, ops).unwrap_or_else(|| vec!["1".to_string()]);
            Some([vec![t[0].clone()], inner].concat())
        }
        _ => Some(t.to_vec()),
    }
}

/// Does the product `t` hold the factor `g`?
fn contains_factor(t: &[String], g: &[String], ops: &Operators) -> bool {
    if t == g {
        return true;
    }
    match (t[0].as_str(), ops.arity_of(&t[0])) {
        ("*", Some(2)) => subtree_end(t, 1, ops).is_some_and(|ja| {
            contains_factor(&t[1..ja], g, ops) || contains_factor(&t[ja..], g, ops)
        }),
        (
            "neg" | "mult2" | "mult3" | "mult4" | "mult5" | "div2" | "div3" | "div4" | "div5",
            Some(1),
        ) => contains_factor(&t[1..], g, ops),
        _ => false,
    }
}

/// The recursion budget reading a derivative costs (`slope`), and the most tokens it may hold.
const DERIVATIVE_DEPTH: u32 = 16;
const DERIVATIVE_BUDGET: usize = 1_000;

/// The derivative of `t` in the variable `x_k`, as tokens, by the textbook rules: the derivative
/// wherever every operator of `t` is analytic (the domain `slope` reads). `None` past the token
/// budget, where an operator is outside the analytic vocabulary, or where a power's exponent is
/// not a ground subtree. The rules' constants are exact: integers, and a power's own exponent.
fn slice_derivative(t: &[String], ops: &Operators, k: usize, d: u32) -> Option<Vec<String>> {
    type T = Vec<String>;
    fn w(s: &str) -> T {
        vec![s.to_string()]
    }
    fn is(a: &[String], s: &str) -> bool {
        a.len() == 1 && a[0] == s
    }
    fn node(op: &str, args: &[&[String]]) -> Option<T> {
        let n = 1 + args.iter().map(|a| a.len()).sum::<usize>();
        (n <= DERIVATIVE_BUDGET).then(|| {
            let mut out = Vec::with_capacity(n);
            out.push(op.to_string());
            for a in args {
                out.extend_from_slice(a);
            }
            out
        })
    }
    // The builders drop the zeros and ones the rules make (`x_k` reads 1, a subtree without it 0).
    fn add(a: T, b: T) -> Option<T> {
        match (is(&a, "0"), is(&b, "0")) {
            (true, _) => Some(b),
            (_, true) => Some(a),
            _ => node("+", &[&a, &b]),
        }
    }
    fn sub(a: T, b: T) -> Option<T> {
        match (is(&a, "0"), is(&b, "0")) {
            (_, true) => Some(a),
            (true, _) => node("neg", &[&b]),
            _ => node("-", &[&a, &b]),
        }
    }
    fn mul(a: &[String], b: &[String]) -> Option<T> {
        if is(a, "0") || is(b, "0") {
            return Some(w("0"));
        }
        match (is(a, "1"), is(b, "1")) {
            (true, _) => Some(b.to_vec()),
            (_, true) => Some(a.to_vec()),
            _ => node("*", &[a, b]),
        }
    }
    fn div(a: &[String], b: &[String]) -> Option<T> {
        if is(a, "0") {
            return Some(w("0"));
        }
        if is(b, "1") {
            return Some(a.to_vec());
        }
        node("/", &[a, b])
    }
    // `u^(q - 1)` for an exponent subtree `q`: exact, an integer where `q` is one.
    fn power_less_one(u: &[String], q: &[String]) -> Option<T> {
        match int_literal(q) {
            Some(2) => Some(u.to_vec()),
            Some(1) => Some(w("1")),
            Some(n) => node("pow", &[u, &w(&(n - 1).to_string())]),
            None => node("pow", &[u, &node("-", &[q, &w("1")])?]),
        }
    }
    if d > CERT_RECURSION_MAX {
        return None;
    }
    if !t.iter().any(|x| var_index(x) == Some(k)) {
        return Some(w("0"));
    }
    let tok = t[0].as_str();
    if !ops.is_operator(tok) {
        return Some(w("1")); // x_k itself
    }
    let ja = subtree_end(t, 1, ops)?;
    let u = &t[1..ja];
    let v = &t[ja..];
    let du = slice_derivative(u, ops, k, d + 1)?;
    let dv = || slice_derivative(v, ops, k, d + 1);
    let one = w("1");
    let two = w("2");
    match (tok, ops.arity_of(tok)?) {
        ("+", 2) => add(du, dv()?),
        ("-", 2) => sub(du, dv()?),
        // u'v + uv'
        ("*", 2) => add(mul(&du, v)?, mul(u, &dv()?)?),
        // (u'v - uv') / v^2
        ("/", 2) => div(
            &sub(mul(&du, v)?, mul(u, &dv()?)?)?,
            &node("pow", &[v, &two])?,
        ),
        // q u^(q - 1) u', the exponent a ground subtree
        ("pow", 2) => {
            ground_leaf(t, ja, ops)?;
            mul(&mul(v, &power_less_one(u, v)?)?, &du)
        }
        // u' / (n rootn(u, n)^(n - 1))
        ("rootn", 2) => {
            int_literal(v).filter(|&n| n >= 2)?;
            let root = node("rootn", &[u, v])?;
            div(&du, &mul(v, &power_less_one(&root, v)?)?)
        }
        ("neg" | "mult2" | "mult3" | "mult4" | "mult5" | "div2" | "div3" | "div4" | "div5", 1) => {
            if is(&du, "0") {
                Some(du)
            } else {
                node(tok, &[&du])
            }
        }
        (op, 1) => {
            // h'(u), then times u'
            let sq = node("pow", &[u, &two])?;
            let inv = |a: T| node("inv", &[&a]);
            let sqrt = |a: T| node("rootn", &[&a, &two]);
            let dh: T = match op {
                "exp" => node("exp", &[u])?,
                "log" => inv(u.to_vec())?,
                "inv" => node("neg", &[&inv(sq)?])?,
                "sin" => node("cos", &[u])?,
                "cos" => node("neg", &[&node("sin", &[u])?])?,
                "tan" => inv(node("pow", &[&node("cos", &[u])?, &two])?)?,
                "sinh" => node("cosh", &[u])?,
                "cosh" => node("sinh", &[u])?,
                "tanh" => inv(node("pow", &[&node("cosh", &[u])?, &two])?)?,
                "asin" => inv(sqrt(node("-", &[&one, &sq])?)?)?,
                "acos" => node("neg", &[&inv(sqrt(node("-", &[&one, &sq])?)?)?])?,
                "atan" => inv(node("+", &[&one, &sq])?)?,
                "asinh" => inv(sqrt(node("+", &[&sq, &one])?)?)?,
                "acosh" => inv(sqrt(node("-", &[&sq, &one])?)?)?,
                "atanh" => inv(node("-", &[&one, &sq])?)?,
                // the sign of u, off its zero
                "abs" => node("/", &[u, &node("abs", &[u])?])?,
                "pow2" | "pow3" | "pow4" | "pow5" => {
                    let q = w(&op[3..]);
                    mul(&q, &power_less_one(u, &q)?)?
                }
                "pow1_2" | "pow1_3" | "pow1_4" | "pow1_5" => {
                    let n = w(&op[5..]);
                    let root = node("rootn", &[u, &n])?;
                    inv(mul(&n, &power_less_one(&root, &n)?)?)?
                }
                _ => return None,
            };
            mul(&dh, &du)
        }
        _ => None,
    }
}

/// Is `g` kept positive or above one by another restriction of `t`, so that it needs no sign?
fn needs_positive_elsewhere(g: &[String], needs: &[(usize, usize, Need)], t: &[String]) -> bool {
    needs
        .iter()
        .any(|&(a, b, n)| matches!(n, Need::Pos | Need::AboveOne) && &t[a..b] == g)
}

/// {x : f(x) not a finite real} is Lebesgue-null, for every `<constant>` value.
/// Fail-closed.
pub fn nonfinite_null(tokens: &[String], ops: &Operators) -> bool {
    if let Some(d) = dummy_spelling(&[tokens], ops) {
        return nonfinite_null(&d[0], ops);
    }
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
                && (!t[ja..].iter().any(|s| s == "<constant>") && zsn(&t[ja..], ops, d + 1, false))
        }
        ("inv", Some(1)) => {
            nfn(&t[1..], ops, d + 1)
                && (!t[1..].iter().any(|s| s == "<constant>") && zsn(&t[1..], ops, d + 1, false))
        }
        // tan poles sit on {cos(g) = 0}
        ("tan", Some(1)) => {
            let mut c: Vec<String> = Vec::with_capacity(t.len());
            c.push("cos".into());
            c.extend_from_slice(&t[1..]);
            nfn(&t[1..], ops, d + 1)
                && (!t[1..].iter().any(|s| s == "<constant>") && zsn(&c, ops, d + 1, false))
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
                        && zsn(&t[1..ja], ops, d + 1, false)
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
        // The 0.15 heads (`floor`, `ceil`, `erf` are total-finite, above).
        // `max`, `min` and `atan2` map two finite reals to a finite real (`atan2(0, 0) = 0`), so
        // they are not finite only where an operand is not.
        ("max" | "min" | "atan2", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            nfn(&t[1..ja], ops, d + 1) && nfn(&t[ja..], ops, d + 1)
        }
        // `mod(a, b)` of finite reals is finite (|mod| <= |b|) except at b = 0, where it is NaN:
        // nonfinite(mod) ⊆ nonfinite(a) ∪ nonfinite(b) ∪ Z(b), as for a quotient (`zsn` refuses
        // a `<constant>`-bearing divisor, which keeps the claim ∀C-sound).
        ("mod", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            nfn(&t[1..ja], ops, d + 1)
                && nfn(&t[ja..], ops, d + 1)
                && (!t[ja..].iter().any(|s| s == "<constant>") && zsn(&t[ja..], ops, d + 1, false))
        }
        // `harmonic(g)` is finite at every finite g but the negative integers (its poles, NaN):
        // nonfinite ⊆ nonfinite(g) ∪ ∪_{n >= 1} {g = -n}, null where `g` stays above -1 or
        // between two poles, or every level set of `g` is null. (float64's NaN tail at or below
        // -2^52 is the `f64` reading's, `harmonic_nan_tail`.)
        ("harmonic", Some(1)) => {
            let g = &t[1..];
            nfn(g, ops, d + 1)
                && (whole_range(g, ops).is_some_and(|v| {
                    v.has_fin && {
                        let k = v.lo.floor();
                        v.lo > -1.0
                            || (v.lo == -1.0 && v.lo_open)
                            || (v.lo > k && v.hi < k + 1.0 && v.hi < -1.0)
                    }
                }) || levels_null(g, ops))
        }
        // `besselj(nu, x)` at a literal order (`besselj_order`): an integer order is entire in x,
        // finite at every finite x; a positive non-integer order is NaN at every x < 0, so `x`
        // must stay >= 0; a negative non-integer order is NaN below 0 and a pole at x = 0, so `x`
        // must stay >= 0 with a null zero set. (Overflow near the pole is the overflow reading's.)
        // A non-literal order fails closed.
        ("besselj", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            let x = &t[ja..];
            let Some(nu) = besselj_order(&t[1..ja], ops) else {
                return false;
            };
            let nonneg = || whole_range(x, ops).is_some_and(|v| v.has_fin && v.lo >= 0.0);
            nfn(x, ops, d + 1)
                && match nu {
                    BesseljOrder::Integer => true,
                    BesseljOrder::PositiveFraction => nonneg(),
                    BesseljOrder::NegativeFraction => {
                        nonneg()
                            && !x.iter().any(|s| s == "<constant>")
                            && zsn(x, ops, d + 1, false)
                    }
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
// Test-only tokens interface (certified + poison batteries); in-crate consumers reach `isn`
// through the zsn/nfn arms.
#[cfg(test)]
pub fn infinite_set_null(tokens: &[String], ops: &Operators) -> bool {
    if let Some(d) = dummy_spelling(&[tokens], ops) {
        return infinite_set_null(&d[0], ops);
    }
    isn(tokens, ops, 0)
}

fn isn(t: &[String], ops: &Operators, d: u32) -> bool {
    isn_e(t, ops, d, false)
}

/// `isn` in the scope of `zsn`'s `e2`: its zero-set sub-claims read as `zsn` does in that scope,
/// and a power of a ground exponent that is not an integer is read (`x^p` is infinite only where
/// `x` is, for `p > 0`, and where `x` is zero, for `p < 0`; NaN where `x < 0`).
fn isn_e(t: &[String], ops: &Operators, d: u32, e2: bool) -> bool {
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
        ("sin" | "cos" | "tanh" | "atan" | "asin" | "acos" | "erf", Some(1)) => true,
        // `atan2` lies in [-pi, pi] at every pair of operands it is defined at.
        ("atan2", Some(2)) => true,
        // {op(u) = ±inf} ⊆ {u = ±inf}: every op here maps finite reals to
        // finite-or-NaN and propagates NaN -- the total-finite class plus the
        // fat-NaN domain-restricted ops whose only infinities are inherited
        // (`pow1_2(+inf) = +inf` but `pow1_2(finite)` is finite or NaN, etc.).
        (op, Some(1)) if op_total_finite(op) || matches!(op, "pow1_2" | "pow1_4" | "acosh") => {
            isn_e(&t[1..], ops, d + 1, e2)
        }
        // {log u = ±inf} = {u = 0} ∪ {u = +inf} (one-zero: log 0 = -inf).
        ("log", Some(1)) => {
            isn_e(&t[1..], ops, d + 1, e2)
                && !t[1..].iter().any(|s| s == "<constant>")
                && zsn(&t[1..], ops, d + 1, e2)
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
            zsn(&m, ops, d + 1, e2) && zsn(&p, ops, d + 1, e2)
        }
        // tan's poles sit on {cos(g) = 0}.
        ("tan", Some(1)) => {
            if t[1..].iter().any(|s| s == "<constant>") {
                return false;
            }
            let mut c: Vec<String> = Vec::with_capacity(t.len());
            c.push("cos".into());
            c.extend_from_slice(&t[1..]);
            zsn(&c, ops, d + 1, e2)
        }
        // {1/g = ±inf} = {g = 0} (1/±inf = 0, 1/nan = nan).
        ("inv", Some(1)) => {
            !t[1..].iter().any(|s| s == "<constant>") && zsn(&t[1..], ops, d + 1, e2)
        }
        // An infinite sum/product needs an infinite operand (finite ∘ finite is
        // finite; overflow is representation; NaN propagates).
        ("+" | "-" | "*", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            isn_e(&t[1..ja], ops, d + 1, e2) && isn_e(&t[ja..], ops, d + 1, e2)
        }
        // {a/b = ±inf} ⊆ {a = ±inf} ∪ {b = 0} (a/±inf = 0; NaN propagates).
        ("/", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            isn_e(&t[1..ja], ops, d + 1, e2)
                && !t[ja..].iter().any(|s| s == "<constant>")
                && zsn(&t[ja..], ops, d + 1, e2)
        }
        // pow with a literal integer exponent: k > 0 → {b = ±inf}; k < 0 → {b = 0}
        // (0^-k = +inf one-zero; ±inf^-k = 0); k = 0 → b^0 = 1, never infinite.
        ("pow", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            match int_literal(&t[ja..]) {
                Some(k) if k > 0 => isn_e(&t[1..ja], ops, d + 1, e2),
                Some(k) if k < 0 => {
                    !t[1..ja].iter().any(|s| s == "<constant>") && zsn(&t[1..ja], ops, d + 1, e2)
                }
                Some(_) => true,
                None if e2 && ground_leaf(t, ja, ops).is_some() => {
                    match crate::numeric::evaluate_constant_value(&t[ja..], ops) {
                        Some(p) if p.is_finite() && p > 0.0 => isn_e(&t[1..ja], ops, d + 1, e2),
                        Some(p) if p.is_finite() && p < 0.0 => {
                            !t[1..ja].iter().any(|s| s == "<constant>")
                                && zsn(&t[1..ja], ops, d + 1, e2)
                        }
                        Some(p) => p == 0.0,
                        None => false,
                    }
                }
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
                Some(n) if n >= 2 => isn_e(&t[1..ja], ops, d + 1, e2),
                _ => false,
            }
        }
        // `max(a, b)` and `min(a, b)` are a member or NaN: infinite only where a member is.
        ("max" | "min", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            isn_e(&t[1..ja], ops, d + 1, e2) && isn_e(&t[ja..], ops, d + 1, e2)
        }
        // `mod(a, b)` lies between 0 and a finite `b`; it is infinite only under an infinite `b`
        // (`mod(-3, inf) = inf`); an infinite `a` is NaN.
        ("mod", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            isn_e(&t[ja..], ops, d + 1, e2)
        }
        // `harmonic(g)` is +inf at g = +inf and finite or NaN elsewhere (its poles are NaN); the
        // poles' level sets are asked to be null as well, as the real function's infinities.
        ("harmonic", Some(1)) => {
            let g = &t[1..];
            isn_e(g, ops, d + 1, e2)
                && (whole_range(g, ops).is_some_and(|v| v.has_fin && v.lo >= -1.0)
                    || levels_null(g, ops))
        }
        // `besselj(nu, x)` at a literal order: never infinite at an integer or a positive order
        // (finite at every finite x, NaN at +-inf); at a negative non-integer order infinite only
        // at its pole x = 0.
        ("besselj", Some(2)) => {
            let Some(ja) = subtree_end(t, 1, ops) else {
                return false;
            };
            let x = &t[ja..];
            match besselj_order(&t[1..ja], ops) {
                Some(BesseljOrder::Integer | BesseljOrder::PositiveFraction) => true,
                Some(BesseljOrder::NegativeFraction) => {
                    !x.iter().any(|s| s == "<constant>") && zsn(x, ops, d + 1, e2)
                }
                None => false,
            }
        }
        _ => false, // unknown ops: fail-closed
    }
}

/// The multiplicative group's regular domain: defined, finite AND nonzero a.e. -- the
/// exact soundness domain of `A/A -> 1` (SELFCANCEL T1). Refuses `<constant>`-bearing
/// trees via `zero_set_null`.
pub fn finite_nonzero_ae(tokens: &[String], ops: &Operators, f64: bool) -> bool {
    zero_set_null(tokens, ops, f64) && finite_ae(tokens, ops, f64)
}

/// Is `f` an ENTIRE-analytic composition (no branch cuts, poles, or abs) that is
/// certainly NOT identically constant? Certified by a PAIR of boxes whose rigorous
/// value intervals are DISJOINT. Consumed by the symbolic-exponent occurrence merge
/// (`ac::expr::mul`): for a nonconstant entire exponent y every level set {y = q} is
/// null (identity theorem), so {y in (1/n)Z \ Z} is null and merging n occurrences of
/// base^y into base^(n*y) changes nothing off a null set. The traps are refused by
/// construction: an identically-constant-but-unfolded y (sin^2 + cos^2) admits no
/// disjoint pair; an abs-bearing y is not entire; a ground y is constant.
///
/// `f64` (the `f64` mode) reads y in float64 (`on_float64_tails`), where `tanh` and `atan` are
/// constant on whole tails: there y is constant too, and its level set at that value is the whole
/// tail. The merge reads only the levels off the integers, so a tail on which y is an integer
/// (`tanh(x1)`, at +-1) is harmless, and one on which it is any other constant (`3*atan(x1)`, at
/// +-3*fl(pi/2)) is refused. The constant is the one float64 computes there (the engine's
/// evaluator, whose `tanh` saturates after the deployed one's: where it reads an integer, so does
/// the deployed one). `real` and `permissive` read y over the reals.
pub fn nonconstant_entire(t: &[String], ops: &Operators, f64: bool) -> bool {
    if let Some(d) = dummy_spelling(&[t], ops) {
        return nonconstant_entire(&d[0], ops, f64);
    }
    let real = |t: &[String]| nonconstant_entire_real(t, ops);
    let integer = |t: &[String]| {
        crate::numeric::evaluate_constant_subtree(t, ops)
            .and_then(|v| v.parse::<f64>().ok())
            .is_some_and(|v| v.is_finite() && v.fract() == 0.0)
    };
    if f64 {
        on_float64_tails(t, ops, 0, &real, &integer)
    } else {
        real(t)
    }
}

/// `nonconstant_entire` over the reals.
fn nonconstant_entire_real(t: &[String], ops: &Operators) -> bool {
    if t.iter()
        .any(|s| ops.is_operator(s) && !op_entire_analytic(s))
    {
        return false;
    }
    disjoint_value_pair(t, ops, false)
}

/// A pair of boxes whose rigorous value intervals of `t` are disjoint: `t` is not constant.
/// `fine` also tries the pairs at the scales r/4, ..., r/4^6 of the horizon r, where a periodic
/// or saturating `t` (`sin(x)` over boxes wider than its period) can part.
fn disjoint_value_pair(t: &[String], ops: &Operators, fine: bool) -> bool {
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
    let disjoint = |a: &Vs, b: &Vs| {
        a.has_fin
            && b.has_fin
            && a.lo.is_finite()
            && a.hi.is_finite()
            && b.lo.is_finite()
            && b.hi.is_finite()
            && (a.hi < b.lo || b.hi < a.lo)
    };
    for k in 0..=if fine { 6 } else { 0 } {
        let s = r / 4f64.powi(k);
        let center = (0.05 * s, 0.15 * s);
        let lo = (-0.95 * s, -0.8 * s);
        let hi = (0.8 * s, 0.95 * s);
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
    }
    false
}

/// f > 0 a.e.: the pow-base predicate completing the algebra. Conservative v1, and test-only:
/// it has no production consumer (the flatten-and-collect pass, proposal P4, would be one);
/// the certified/poison batteries below pin its behavior.
#[cfg(test)]
pub fn positive_ae(tokens: &[String], ops: &Operators) -> bool {
    if let Some(d) = dummy_spelling(&[tokens], ops) {
        return positive_ae(&d[0], ops);
    }
    if tokens.is_empty() {
        return false;
    }
    match tokens[0].as_str() {
        "exp" | "cosh" => nonfinite_null(&tokens[1..], ops),
        // squares / abs of a null-zero-set argument: >= 0 with a null zero set
        "pow2" | "pow4" | "abs" => {
            zero_set_null(&tokens[1..], ops, false) && nonfinite_null(&tokens[1..], ops)
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
                && zero_set_null(&tokens[1..ja], ops, false)
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
#[cfg(test)] // the in-crate gate deliberately uses `_at` (see below); the tests use this one
pub fn defined_measure_p(expr: &[String], params: &[f64], ops: &Operators) -> Option<f64> {
    if let Some(d) = dummy_spelling(&[expr], ops) {
        return defined_measure_p(&d[0], params, ops);
    }
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
    if let Some(d) = dummy_spelling(&[expr], ops) {
        return defined_measure_p_at(&d[0], params, ops, r, depth_per_dim);
    }
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

/// FLOAT64'S TAILS on one cell, for the evidence certificate (`finite_cells`): `tanh`, `atan` and
/// `erf` ARE their limit wherever float64 rounds them to it (`tail_of`), but the enclosure of such
/// a node straddles the limit by its outward step, so a source finite only because a tail is exact
/// -- `acosh(erf(x))`, finite where float64's erf is 1, x >= 5.92 -- never reads finite. Where a
/// node's argument lies past the threshold over the whole cell (`doms`), the node is its limit
/// at every point of it: this returns the expression with each such node replaced by the literal
/// limit, its `<constant>` bindings dropped with it, or `None` where no node is. A float64
/// reading, sound for the evidence a mine counts (its rows are evaluated in float64 in every
/// mode) and for nothing that reads the reals.
fn f64_tail_reading(
    e: &[String],
    ops: &Operators,
    doms: &[Vs],
    params: &[f64],
) -> Option<(Vec<String>, Vec<f64>)> {
    let mut t = e.to_vec();
    let mut p = params.to_vec();
    let mut changed = false;
    let mut i = 0usize;
    while i < t.len() {
        let tail = tail_of(&t[i]).filter(|_| ops.arity_of(&t[i]) == Some(1));
        let Some((threshold, limit)) = tail else {
            i += 1;
            continue;
        };
        let j = subtree_end(&t, i, ops)?;
        let k0 = t[..i].iter().filter(|x| *x == "<constant>").count();
        let kn = t[i + 1..j].iter().filter(|x| *x == "<constant>").count();
        let side = (p.len() >= k0 + kn)
            .then(|| value_set_p(&t[i + 1..j], ops, doms, &p[k0..k0 + kn]))
            .flatten()
            .filter(|a| a.has_fin && !a.nan && !a.pinf && !a.ninf)
            .and_then(|a| {
                if a.lo >= threshold {
                    Some(limit)
                } else if a.hi <= -threshold {
                    Some(-limit)
                } else {
                    None
                }
            });
        if let Some(c) = side {
            t.splice(i..j, [format!("{c}")]);
            p.drain(k0..k0 + kn);
            changed = true;
        }
        i += 1;
    }
    changed.then_some((t, p))
}

/// Cells of the horizon box on which an expression is FINITE everywhere (`finite_cells`): one
/// interval per variable the expression references, `vars[d]` naming the variable of dimension
/// `d` in the expression's own spelling, inside the box `[-r, r]^n`.
pub struct FiniteCells {
    pub vars: Vec<String>,
    pub cells: Vec<Vec<(f64, f64)>>,
    pub r: f64,
}

/// THE EVIDENCE CERTIFICATE: is `expr` (its `<constant>`s bound to `params`) FINITE on a
/// positive-measure set of its horizon box, and where? With no `params`, its `<constant>`s are
/// free and range over the box too (dimension 0, beside a first variable), so the question is
/// asked of the constants themselves.
///
/// A cell is a witness when the outward-rounded value set over it holds a finite part and
/// nothing else (no NaN, no infinity, and the finite part not confined to a null set): every
/// point of the cell evaluates finite in the deployed float64 evaluator, which is the
/// evidence a mine counts (`worker::candidate_matches` binds source-finite rows only, every
/// mode evaluating its rows in float64). A cell is read through float64's tails as well
/// (`f64_tail_reading`), where the evaluator is exact. The cells come from the same box,
/// horizon and dyadic subdivision `defined_measure_p_at` reads, so "positive measure" means the
/// same thing to the evidence decision and to the domain gate.
///
/// The search is BREADTH-FIRST, so the witnesses are the coarsest cells of the finite set and
/// spread over all of it: it stops at the end of the first level holding at least `want`
/// cells. A cell with no positive-measure finite part is dropped (it holds no witness); a mixed
/// cell is split along its widest dimension down to the resolution floor.
///
/// FAIL CLOSED: `None` when the horizon is undecidable, the node budget runs out, or an
/// unanalyzable cell was met before any witness; `Some` with no cells when the search completed
/// without one. Only a witness certifies; the caller reads both other outcomes as "no evidence".
pub fn finite_cells(
    expr: &[String],
    params: &[f64],
    ops: &Operators,
    want: usize,
) -> Option<FiniteCells> {
    let respelled = dummy_spelling(&[expr], ops);
    let e: &[String] = respelled.as_ref().map_or(expr, |d| &d[0]);
    let (r, depth_per_dim, decidable) = horizon(&[e], &[params], ops);
    if !decidable {
        HORIZON_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    let used = distinct_vars(&[e]);
    let width = n_var_slots(&[e]);
    // `dummy_spelling` renames token for token, so a dimension's variable is the original
    // token at the position of its dummy.
    let vars: Vec<String> = used
        .iter()
        .map(|&i| {
            e.iter()
                .position(|t| var_index(t) == Some(i))
                .map(|j| expr[j].clone())
                .unwrap_or_default()
        })
        .collect();
    let n = used.len().max(1);
    // A variable-free expression at bound constants is one value: its box has nothing to split.
    // Free constants (fewer `params` than `<constant>`s) live on dimension 0 (`leaf_vs`), and
    // the box over them is split like a variable's.
    let free = e.iter().filter(|t| *t == "<constant>").count() > params.len();
    let depth_max = if used.is_empty() && !free {
        0
    } else {
        depth_per_dim * n as u32
    };
    let mut budget = max_nodes_budget();
    let mut undecided = false;
    let mut cells: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut level: Vec<Vec<(f64, f64)>> = vec![vec![(-r, r); n]];
    let mut d = 0u32;
    while !level.is_empty() && cells.len() < want && budget > 0 {
        let mut next: Vec<Vec<(f64, f64)>> = Vec::new();
        for bx in level {
            if budget == 0 {
                break;
            }
            budget -= 1;
            let doms = doms_from_box(&bx, &used, width, (-r, r));
            let Some(v) = value_set_p(e, ops, &doms, params) else {
                undecided = true;
                continue;
            };
            if !v.fin_pm() {
                continue; // no positive-measure finite part anywhere in this cell
            }
            let finite = |w: &Vs| w.fin_pm() && !w.nan && !w.pinf && !w.ninf;
            if finite(&v)
                || f64_tail_reading(e, ops, &doms, params).is_some_and(|(t, p)| {
                    value_set_p(&t, ops, &doms, &p).is_some_and(|w| finite(&w))
                })
            {
                cells.push(bx);
                continue;
            }
            if d < depth_max {
                let i = (0..bx.len())
                    .max_by(|&a, &b| {
                        (bx[a].1 - bx[a].0)
                            .partial_cmp(&(bx[b].1 - bx[b].0))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                let (lo, hi) = bx[i];
                let mid = 0.5 * (lo + hi);
                let mut left = bx.clone();
                left[i] = (lo, mid);
                let mut right = bx;
                right[i] = (mid, hi);
                next.push(left);
                next.push(right);
            }
        }
        level = next;
        d += 1;
    }
    if cells.is_empty() {
        if budget == 0 {
            NODE_BUDGET_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
        if undecided {
            UNANALYZABLE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    }
    Some(FiniteCells { vars, cells, r })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::RuleMode;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// A root's index decides it: an index that reaches no nonzero integer is NaN at every
    /// radicand, and an index range that holds one stays unanalyzable (fail closed).
    #[test]
    fn a_root_index_that_reaches_no_integer_is_nan() {
        let pt = |x: f64| Vs::interval(x, x, false, false);
        assert!(!index_reaches_a_nonzero_integer(&Vs::interval(
            0.88, 0.89, false, false
        )));
        assert!(!index_reaches_a_nonzero_integer(&pt(0.0)));
        assert!(!index_reaches_a_nonzero_integer(&Vs::interval(
            -0.5, 0.5, false, false
        )));
        assert!(!index_reaches_a_nonzero_integer(&Vs::interval(
            1.0, 2.0, true, true
        )));
        assert!(index_reaches_a_nonzero_integer(&Vs::interval(
            1.0, 2.0, true, false
        )));
        assert!(index_reaches_a_nonzero_integer(&Vs::interval(
            -1.2, -0.8, false, false
        )));
        assert!(index_reaches_a_nonzero_integer(&pt(3.0)));
        assert!(index_reaches_a_nonzero_integer(&pt(1e300)));
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        for src in [
            &["rootn", "(-1)", "acos", "0"][..],
            &["rootn", "x0", "asinh", "tanh", "4"],
            &["rootn", "2", "log", "1"],
        ] {
            assert_eq!(value_class(&s(src), ops), Some(Class::Nan), "{src:?}");
        }
        // an index whose enclosure holds an integer: no verdict
        assert_eq!(
            value_class(&s(&["rootn", "2", "+", "1", "sin", "np.pi"]), ops),
            None
        );
    }

    /// A positive base under a non-integer exponent is bounded by its corner powers: a ground
    /// `1/pi^(1/6)` reads as the number below 1 it is, so `acosh` of it classifies as NaN.
    #[test]
    fn a_positive_base_power_is_bounded_by_its_corners() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let v = value_set(&s(&["pow", "np.pi", "/", "1", "6"]), ops, &Vs::reals()).unwrap();
        let want = std::f64::consts::PI.powf(1.0 / 6.0);
        assert!(v.lo <= want && want <= v.hi && v.hi - v.lo < 1e-12, "{v:?}");
        assert_eq!(
            value_class(&s(&["acosh", "inv", "pow", "np.pi", "/", "1", "6"]), ops),
            Some(Class::Nan)
        );
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
    /// A pole reached at a SATURATION-ATTAINED argument is an INF VALUE, not an unbounded
    /// finite edge. `exp` is exactly +0 on the whole ray below `EXP_UNDERFLOW` and `tanh`
    /// exactly +-1 past +-`TANH_SAT`, so `cosh(exp C) = 1` and `exp(exp C) = 1` on a ray of
    /// POSITIVE measure -- putting `atanh`'s pole and `log`'s pole at an attained argument.
    ///
    /// This is NOT the overflow case the module header rules on ("a float OVERFLOW means the
    /// value is finite but unrepresentable -> an unbounded EDGE, never an inf value"): there,
    /// `f` of a representable argument leaves f64's range and the true value is finite. Here
    /// the argument lands exactly ON a pole and the true value IS infinite -- which is what the
    /// deployed evaluator returns, on 2005 of 4177 sampled C for `atanh(cosh(exp C))`, with a
    /// finite value on NONE of them. Reporting `has_fin` there certifies a finite value the
    /// evaluator never produces.
    #[test]
    fn an_attained_pole_is_an_infinity_not_an_unbounded_edge() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        for src in [
            &["atanh", "cosh", "exp", "<constant>"][..],
            &["atanh", "exp", "exp", "<constant>"][..],
        ] {
            let v = value_set(&s(src), ops, &Vs::reals()).unwrap();
            assert!(
                v.pinf && !v.has_fin && v.nan,
                "{src:?}: atanh at an attained +1 is +inf a.e. on the ray, never finite: {v:?}"
            );
            assert_eq!(value_class(&s(src), ops), Some(Class::Mixed));
        }
        // log at an attained -0: `neg(exp C)` is exactly -0 on the same ray.
        let v = value_set(&s(&["log", "neg", "exp", "<constant>"]), ops, &Vs::reals()).unwrap();
        assert!(
            v.ninf && !v.has_fin && v.nan,
            "log at an attained zero is -inf, never an unbounded finite edge: {v:?}"
        );
        assert_eq!(
            value_class(&s(&["log", "neg", "exp", "<constant>"]), ops),
            Some(Class::Mixed)
        );
    }

    /// The twin of the above on the FINITE side: where the grazed domain edge carries no pole,
    /// the attained graze is a real finite value on positive measure, so it must survive the
    /// R3 null-kill and classify Mixed -- not NAN. `acos(cosh(exp C)) = 0` on 2005 of 4177
    /// sampled C. The pre-image branch kept the graze but marked it null, which deleted a
    /// finite part the evaluator really produces.
    #[test]
    fn an_attained_graze_is_supported_on_positive_measure() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        for src in [
            &["acos", "cosh", "exp", "<constant>"][..],
            &["asin", "exp", "exp", "<constant>"][..],
            &["acosh", "tanh", "neg", "<constant>"][..],
            &["acosh", "neg", "tanh", "<constant>"][..],
        ] {
            let v = value_set(&s(src), ops, &Vs::reals()).unwrap();
            assert!(
                v.has_fin && !v.fin_null && v.nan,
                "{src:?}: the graze is attained on a saturation ray, not on a null set: {v:?}"
            );
            assert_eq!(
                value_class(&s(src), ops),
                Some(Class::Mixed),
                "{src:?} is finite on one ray and nan on another: neither collapse is licensed"
            );
        }
    }

    /// ...and the conservatism it must NOT overturn. `cosh(inv C)`'s `lo = 1` is the
    /// MATHEMATICAL minimum: reached only where `inv C = 0`, which the interval layer merely
    /// over-approximates as reachable and exact arithmetic never reaches at all. No `attain_pt`
    /// provenance stands behind it, so the graze stays NULL-supported and these keep the
    /// verdicts they had before the f64-image work -- the flag is PROVEN-only, and absence of
    /// proof stays fail-closed.
    #[test]
    fn an_unproven_graze_keeps_its_conservative_null_marking() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        for src in [
            &["acos", "cosh", "inv", "<constant>"][..],
            &["asin", "cosh", "inv", "<constant>"][..],
            &["atanh", "cosh", "inv", "<constant>"][..],
        ] {
            let v = value_set(&s(src), ops, &Vs::reals()).unwrap();
            assert!(
                v.has_fin && v.fin_null && !v.pinf && !v.ninf,
                "{src:?}: an unproven graze must stay the null-marked edge: {v:?}"
            );
            assert_eq!(value_class(&s(src), ops), Some(Class::Nan));
        }
        // The outward-rounding artifact graze (`acosh(cos x0)`, cos's enclosure [1-8u, 1]
        // meeting acosh's domain in {1}) is the shape the null marking was introduced for.
        let v = value_set(&s(&["acosh", "cos", "x0"]), ops, &Vs::reals()).unwrap();
        assert!(
            v.has_fin && v.fin_null,
            "the artifact graze lost its null marking -- witnesses would unblock: {v:?}"
        );
    }

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
    /// E1 (F80): `<constant>` is a generic parameter for the DISTRIBUTION licence's
    /// zero-set question -- and ONLY there. The plain `zero_set_null` (and through it
    /// the `$`-certificate `finite_nonzero_ae`) keeps the blanket refusal: the
    /// SELFCANCEL asymmetry protects the c = 0 atom.
    #[test]
    fn e1_const_as_generic_parameter() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // The acos(C) witness class: zero only at the single setting C = 1.
        assert!(
            zero_set_null_generic_const(&s(&["acos", "<constant>"]), ops, false),
            "acos(<constant>) vanishes only at C = 1: null in parameter space"
        );
        // The bare atom, consistent with the ruled licence doctrine.
        assert!(
            zero_set_null_generic_const(&s(&["<constant>"]), ops, false),
            "{{c = 0}} is a null set of the mask's own parameter space"
        );
        // Independent occurrences: C1 - C2 vanishes only on the diagonal.
        assert!(
            zero_set_null_generic_const(&s(&["-", "<constant>", "<constant>"]), ops, false),
            "two <constant>s are INDEPENDENT: the diagonal is null"
        );
        // A zero literal still kills the product unconditionally.
        assert!(
            !zero_set_null_generic_const(&s(&["*", "0", "<constant>"]), ops, false),
            "an exact zero factor is identically zero"
        );
        // The genuinely-forbidden fat-zero shape stays refused (abs is not analytic).
        assert!(
            !zero_set_null_generic_const(&s(&["-", "x0", "abs", "x0"]), ops, false),
            "x - |x| vanishes on a half-line: must refuse"
        );
        // THE ASYMMETRY PINS: the plain certificate and the $-certificate are
        // untouched by E1 -- Const-bearing streams still refuse there.
        assert!(
            !zero_set_null(&s(&["acos", "<constant>"]), ops, false),
            "plain zero_set_null keeps the Const blanket"
        );
        assert!(
            !finite_nonzero_ae(&s(&["acos", "<constant>"]), ops, false),
            "the $-cancel licence keeps its SELFCANCEL protection of the c = 0 atom"
        );
    }

    /// E2 (F81, owner-ratified 2026-08-11; design + soundness chain in
    /// the research harness): the denominator-clearing certificate -- rational-
    /// level factors rewrite to an entire numerator N with Z(factor) subset
    /// Z(N), and the EXISTING identity-theorem witness decides N. Unlock pins,
    /// sound-refusal pins, the ratified boundary, the tier-0 ground-literal
    /// gate rider, and the scoping asymmetry (plain zsn / $-cancel untouched).
    #[test]
    fn e2_cleared_numerator_witness() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // --- UNLOCK PINS (refused before E2, certified now) ---
        // The design's worked example: an affine sum whose literal spells as a
        // fraction; tier-0 ground-leaf acceptance lets the witness see it.
        assert!(
            zero_set_null_generic_const(&s(&["-", "x3", "/", "5", "4"]), ops, false),
            "x3 - 5/4 vanishes on one hyperplane: the literal SPELLING must not refuse"
        );
        assert!(
            entire_analytic_composition(&s(&["-", "x3", "/", "5", "4"]), ops),
            "tier-0: a ground subtree is a constant, hence entire (gate rider)"
        );
        // 1/x - x clears to N = 1 - x^2: zeros {-1, +1}, null.
        assert!(
            zero_set_null_generic_const(&s(&["-", "inv", "x7", "x7"]), ops, false),
            "1/x - x = (1 - x^2)/x: witness the entire numerator"
        );
        // Zeros AT the inner pole: 1/(1/x + 1) is genuinely 0 at x = 0
        // (1/0 = +inf, 1/inf = 0); the swap makes N = x, catching it.
        assert!(
            zero_set_null_generic_const(&s(&["inv", "+", "inv", "x1", "1"]), ops, false),
            "the pair swap captures zeros-at-poles: Z = {{0}}, null"
        );
        // tan clears at head level: x2 + tan(x3) -> N = x2*cos(x3) + sin(x3).
        assert!(
            zero_set_null_generic_const(&s(&["+", "x2", "tan", "x3"]), ops, false),
            "tan u = sin u / cos u clears; N is entire and not identically zero"
        );
        // E1 x E2 composition: 1/c - c with the generic-parameter rename.
        assert!(
            zero_set_null_generic_const(&s(&["-", "inv", "<constant>", "<constant>"]), ops, false),
            "Const renames to a generic variable BEFORE clearing (ordering law)"
        );
        // Negative literal exponents are cleared by the powered swap.
        assert!(
            zero_set_null_generic_const(&s(&["-", "pow", "x2", "-2", "x2"]), ops, false),
            "x^-2 - x = (1 - x^3)/x^2: the powered swap clears it"
        );
        // --- SOUND-REFUSAL PINS (must stay refused forever) ---
        // The fat class: x - |x| vanishes on a half-line; abs refuses clearing.
        assert!(
            !zero_set_null_generic_const(&s(&["-", "x0", "abs", "x0"]), ops, false),
            "x - |x| is the genuinely forbidden class: no spelling may certify it"
        );
        // Pole inside a transcendental argument: the ratified boundary of the clearing.
        assert!(
            !cleared_witness(&s(&["+", "x1", "sin", "inv", "x2"]), ops),
            "sin(1/x) class: clearing cannot pull a denominator through sin"
        );
        // The identity theorem on the domain reads it: analytic on each half-space x2 > 0 and
        // x2 < 0, nonzero on a cell of each, and NaN on x2 = 0 (sin(inf)), which holds no zero.
        assert!(
            zero_set_null_generic_const(&s(&["+", "x1", "sin", "inv", "x2"]), ops, false),
            "sin(1/x) class: null on each piece of the domain"
        );
        // Identically-zero disguise: sin^2 + cos^2 - 1 clears to itself (D = 1)
        // and the witness can never find a nonzero cell. Fail-closed.
        assert!(
            !zero_set_null_generic_const(
                &s(&["-", "+", "*", "sin", "x1", "sin", "x1", "*", "cos", "x1", "cos", "x1", "1"]),
                ops,
                false
            ),
            "an identically-zero factor has no nonzero witness cell"
        );
        // Branch-cut exponent: the pair algebra refuses non-integer powers, but `1/x^(1/2)` is zero
        // only where `x^(1/2)` is infinite, which a finite `x` never makes it (`isn_e`).
        assert!(
            zero_set_null_generic_const(&s(&["inv", "pow", "x1", "/", "1", "2"]), ops, false),
            "x^(-1/2) is never zero"
        );
        assert!(
            !zero_set_null(&s(&["inv", "pow", "x1", "/", "1", "2"]), ops, false),
            "the plain certificate keeps its strength"
        );
        // Ground infinity: x1 + 1/0 = +inf everywhere -- EMPTY zero set, so a
        // TRUE verdict is sound. The pair is (1 + x1*0, 0): the two-sided
        // invariant (factor-zeros in Z(N), factor-INFINITIES in Z(D)) parks
        // the everywhere-infinity in D = 0 and hands the witness N == 1.
        assert!(
            zero_set_null_generic_const(&s(&["+", "x1", "/", "1", "0"]), ops, false),
            "x1 + 1/0 is +inf everywhere: empty zero set certifies"
        );
        // The mirror image: inv(inv(0)) = inv(+inf) IS identically zero -- the
        // swap carries the zero into N = 0 and the witness can never certify.
        assert!(
            !zero_set_null_generic_const(&s(&["inv", "inv", "0"]), ops, false),
            "1/(1/0) = 1/inf = 0 identically: the swap parks it in N = 0"
        );
        // --- SCOPING ASYMMETRY (ask b, ratified: ac_zsn-scoped only) ---
        assert!(
            !zero_set_null(&s(&["-", "inv", "x7", "x7"]), ops, false),
            "the PLAIN certificate keeps its pre-E2 strength (scoped wiring)"
        );
        assert!(
            !finite_nonzero_ae(&s(&["-", "inv", "x7", "x7"]), ops, false),
            "the $-cancel licence is untouched by E2 (widening is its own ruling)"
        );
    }

    /// The identity theorem on the expression's domain: a sum over `log`, roots, `asin`, `acosh`, a
    /// quotient or `abs` has a null zero set where every piece of its domain holds a cell on which
    /// it is certainly nonzero; a sum that vanishes on a region of its domain never does.
    #[test]
    fn a_zero_set_is_read_on_the_domain() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        for null in [
            &["-", "log", "x2", "x2"][..],
            &["-", "rootn", "x2", "2", "x2"],
            &["-", "log", "x3", "log", "x2"],
            &["-", "asin", "x1", "x1"],
            &["-", "acosh", "x1", "x1"],
            &["-", "/", "sin", "x3", "x1", "rootn", "x2", "2"],
            &["-", "pow", "x2", "/", "1", "3", "x2"],
            &["+", "abs", "x1", "1"],
            &["-", "*", "x2", "log", "+", "x1", "1", "x1"],
        ] {
            assert!(
                zero_set_null_generic_const(&s(null), ops, false),
                "{null:?}: never zero on a region of its domain"
            );
            // the plain certificate keeps its strength (the `$`-cancel's ruling)
            assert!(!zero_set_null(&s(null), ops, false), "{null:?}");
        }
        for region in [
            &["-", "x1", "abs", "x1"][..],
            &["-", "abs", "x1", "x1"],
            &["-", "x1", "pow", "rootn", "x1", "2", "2"],
            &["-", "asin", "sin", "x1", "x1"],
            &["-", "log", "exp", "x1", "x1"],
            &["-", "x1", "atan", "tan", "x1"],
            &["-", "acos", "cos", "x1", "x1"],
        ] {
            assert!(
                !zero_set_null_generic_const(&s(region), ops, false),
                "{region:?} vanishes on a region"
            );
        }
    }

    /// A slope in one variable: a sum over restricted arguments that are not affine has a null zero
    /// set where it moves strictly with one variable on each piece of every slice; a sum that may
    /// vanish on a region never has such a slope.
    #[test]
    fn a_zero_set_is_read_on_a_slice() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        for null in [
            &["-", "log", "*", "x1", "x2", "x1"][..],
            &["-", "rootn", "-", "x1", "pow", "x2", "2", "2", "x3"],
            &["-", "log", "log", "x1", "x2"],
            &["-", "rootn", "sin", "x1", "2", "x2"],
            &["+", "log", "cos", "x2", "x1"],
            &["-", "rootn", "*", "x2", "+", "x4", "x5", "2", "x1"],
            &["-", "tan", "x1", "x2"],
            &["+", "x2", "-", "x1", "abs", "x1"],
            &["-", "inv", "log", "*", "x1", "x2", "exp", "x3"],
            &["-", "*", "x3", "log", "*", "x1", "x2", "sin", "x1"],
            &["-", "pow", "-", "x1", "pow", "x2", "2", "/", "3", "2", "x3"],
            // by the derivative: `1 - tan(x1)`,
            // `log(x1) + 1`, `1 - 0.5 x2/|x2|`
            &["+", "x1", "log", "cos", "x1"],
            &["-", "*", "x1", "log", "x1", "1"],
            &["-", "x2", "*", "0.5", "abs", "x2"],
            &["-", "x1", "log", "rootn", "x1", "2"],
        ] {
            assert!(
                zero_set_null_generic_const(&s(null), ops, false),
                "{null:?}: moves with a variable on every slice"
            );
            assert!(!zero_set_null(&s(null), ops, false), "{null:?}");
        }
        // a factor every term shares: `cos(500 x1 x2)*(2 - x3)`, which interval evaluation alone
        // does not read; a polynomial whose literals put the horizon past the largest box
        for null in [
            &[
                "-", "*", "2", "cos", "*", "500", "*", "x1", "x2", "*", "x3", "cos", "*", "500",
                "*", "x1", "x2",
            ][..],
            &[
                "+",
                "*",
                "x1",
                "+",
                "*",
                "x1",
                "+",
                "*",
                "0.41",
                "x1",
                "182.26",
                "2763462.464",
                "1916567309.85",
            ],
        ] {
            assert!(
                zero_set_null_generic_const(&s(null), ops, false),
                "{null:?}"
            );
        }
        for region in [
            &["-", "x1", "abs", "x1"][..],
            &["*", "x2", "-", "abs", "x1", "x1"],
            &["-", "log", "exp", "x1", "x1"],
            &["-", "x2", "pow", "rootn", "x2", "2", "2"],
            &["-", "*", "x1", "inv", "x1", "1"],
            &["-", "pow", "x1", "+", "1", "1", "pow", "abs", "x1", "2"],
            &["-", "exp", "neg", "+", "x1", "inv", "*", "0", "x2", "0"],
            &["+", "x1", "abs", "x1"],
            &["-", "x2", "*", "1", "abs", "x2"],
            &["-", "*", "x2", "*", "x1", "inv", "x1", "x2"],
            &["-", "log", "exp", "*", "x1", "x2", "*", "x2", "x1"],
            &["-", "*", "x2", "abs", "x1", "*", "x2", "x1"],
            &["+", "*", "x1", "abs", "inv", "x1", "1"],
        ] {
            assert!(
                !zero_set_null_generic_const(&s(region), ops, false),
                "{region:?} may vanish on a region"
            );
        }
    }

    /// The f64 saturation thresholds are MEASURED, not declared: each is where the deployed
    /// kernel (`numeric::kernels`, which the evaluator and `simplipy.operators` both call)
    /// attains its limit, to the ulp.
    #[test]
    fn saturation_thresholds_are_the_deployed_kernels() {
        use crate::numeric::kernels::{atan, exp, tanh};
        let toward_zero = |x: f64| f64::from_bits(x.to_bits() - 1);
        assert_eq!(tanh(TANH_SAT), 1.0);
        assert!(tanh(toward_zero(TANH_SAT)) < 1.0);
        assert_eq!(tanh(-TANH_SAT), -1.0);
        assert_eq!(atan(ATAN_SAT), std::f64::consts::FRAC_PI_2);
        assert!(atan(toward_zero(ATAN_SAT)) < std::f64::consts::FRAC_PI_2);
        assert_eq!(exp(EXP_UNDERFLOW), 0.0);
        assert!(exp(toward_zero(EXP_UNDERFLOW)) > 0.0);
    }

    /// float64's `tanh` is exactly +-1 for every |g| >= 19.06 and its `atan` exactly +-fl(pi/2)
    /// for every |g| >= 5.8e15, so in f64 a sum that cancels the limit vanishes on a region,
    /// where over the reals it never vanishes.
    #[test]
    fn float64_tails_hold_zero_sets() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let read = |t: &[&str], f64: bool| zero_set_null_generic_const(&s(t), ops, f64);
        for region in [
            &["-", "*", "x2", "tanh", "x1", "x2"][..],
            &["-", "tanh", "x1", "1"],
            &["+", "tanh", "-", "x1", "x3", "1"],
            &["-", "atan", "x1", "1.5707963267948966"],
            &["-", "atan", "x1", "/", "np.pi", "2"],
            &["*", "<constant>", "-", "tanh", "x1", "1"],
            // tanh(tanh(x1)) is tanh(1) wherever x1 >= 19.06
            &["-", "tanh", "tanh", "x1", "tanh", "1"],
            // 1/(tanh(x1) - 1) is +inf on the tail, so its inverse is 0 there
            &["inv", "inv", "-", "tanh", "x1", "1"],
            // a training-data denominator: x2 - x2 wherever its tanh saturates
            &[
                "-", "*", "x2", "tanh", "-", "x8", "/", "*", "x8", "asin", "x2", "x9", "x2",
            ],
        ] {
            assert!(read(region, false), "{region:?}: never zero over the reals");
            assert!(!read(region, true), "{region:?}: zero on a float64 tail");
        }
        // the tails do not cancel, or the argument never reaches them
        for null in [
            &["-", "atan", "x1", "x1"][..],
            &["-", "*", "x2", "tanh", "x1", "x1"],
            &["-", "*", "x2", "tanh", "*", "x1", "x3", "x1"],
            &["-", "tanh", "x1", "/", "1", "2"],
            &["sin", "tanh", "pow", "x1", "8"],
            &["-", "tanh", "sin", "x1", "1"],
            &["-", "log", "x2", "tanh", "x2"],
            &["-", "x1", "tanh", "1"],
        ] {
            assert!(read(null, false), "{null:?}");
            assert!(read(null, true), "{null:?}");
        }
    }

    /// The `!` and `$` certificates read float64's tails in f64: `1/(tanh(x1) - 1)` is `+inf`
    /// and `tanh(x1) - 1` is `+0` wherever x1 >= 19.06, which the real functions never are.
    #[test]
    fn float64_tails_hold_infinities() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let fin = |t: &[&str], f64: bool| finite_ae(&s(t), ops, f64);
        let finnz = |t: &[&str], f64: bool| finite_nonzero_ae(&s(t), ops, f64);
        let zero = |t: &[&str], f64: bool| zero_set_null(&s(t), ops, f64);
        for t in [
            &["inv", "-", "tanh", "x1", "1"][..],
            &["*", "x2", "inv", "-", "tanh", "x1", "1"],
            &["inv", "-", "atan", "x1", "1.5707963267948966"],
            &["/", "x2", "+", "tanh", "-", "x1", "x3", "1"],
        ] {
            assert!(fin(t, false), "{t:?}: finite over the reals");
            assert!(!fin(t, true), "{t:?}: infinite on a float64 tail");
        }
        for t in [
            &["-", "tanh", "x1", "1"][..],
            &["*", "x2", "-", "tanh", "x1", "1"],
            &["-", "atan", "x1", "1.5707963267948966"],
        ] {
            assert!(
                finnz(t, false) && zero(t, false),
                "{t:?}: nonzero over the reals"
            );
            assert!(
                !finnz(t, true) && !zero(t, true),
                "{t:?}: zero on a float64 tail"
            );
        }
        // the tails stay finite and nonzero
        for t in [
            &["inv", "-", "tanh", "x1", "/", "1", "2"][..],
            &["tanh", "x1"],
            &["-", "*", "x2", "tanh", "x1", "x1"],
            &["inv", "-", "atan", "x1", "x1"],
        ] {
            assert!(finnz(t, false) && finnz(t, true), "{t:?}");
        }
    }

    /// The symbolic-exponent merge certificate reads float64's tails in f64: on a tail the
    /// exponent is constant, so its level set there is the whole tail. A level at an integer is
    /// harmless to the merge; one at any other value is refused.
    #[test]
    fn float64_tails_hold_exponent_levels() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let nce = |t: &[&str], f64: bool| nonconstant_entire(&s(t), ops, f64);
        for t in [
            &["atan", "x1"][..],
            &["*", "3", "atan", "x1"],
            &["sin", "tanh", "x1"],
            // fl(pi) where x1 and -x2 are both on their upper tails
            &["-", "atan", "x1", "atan", "x2"],
        ] {
            assert!(nce(t, false), "{t:?}: every level null over the reals");
            assert!(
                !nce(t, true),
                "{t:?}: a non-integer level holds a float64 tail"
            );
        }
        for t in [
            &["tanh", "x1"][..],
            &["+", "tanh", "x1", "1"],
            &["*", "3", "tanh", "x1"],
            &["*", "x2", "tanh", "x1"],
            &["-", "atan", "x1", "x2"],
            &["exp", "x1"],
        ] {
            assert!(nce(t, false) && nce(t, true), "{t:?}");
        }
    }

    /// enclosure makes every exact-zero identity ground contain 0 (certificates fail closed)
    /// while grounds rigorously bounded away from 0 keep certifying.
    /// F82 (2026-08-11, extreme row 433087): b_mul's infinite blocks collapsed a
    /// MIXED operand to a single sign, and `cls()` read the under-set flags as a
    /// DEFINITE PosInf -- the classification fold then shipped `+inf` for a value
    /// whose exact truth is -inf. The flags are reachability claims and Vs is an
    /// over-approximation for every fold/witness consumer: dropping a true flag
    /// fabricates a definite class.
    #[test]
    fn f82_mixed_infinity_survives_multiplication() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let cls = |t: &[&str]| value_set(&s(t), ops, &Vs::reals()).unwrap().cls();
        // -inf * sin(x0): honest Mixed at two factors (sin's range crosses zero).
        assert_eq!(
            cls(&["*", "float(\"-inf\")", "sin", "x0"]),
            Class::Mixed,
            "mixed set forms honestly at two factors"
        );
        // (-inf * sin(x0)) * 2: the pre-fix collapse read this as DEFINITE PosInf.
        assert_eq!(
            cls(&["*", "*", "float(\"-inf\")", "sin", "x0", "2"]),
            Class::Mixed,
            "MIXED x positive must stay mixed -- the ninf flag may not drop"
        );
        assert_eq!(
            cls(&["*", "*", "float(\"-inf\")", "sin", "x0", "-2"]),
            Class::Mixed,
            "MIXED x negative likewise"
        );
        // The inf x inf block has the same collapse shape.
        assert_eq!(
            cls(&["*", "*", "float(\"-inf\")", "sin", "x0", "float(\"inf\")"]),
            Class::Mixed,
            "MIXED x (+inf) must stay mixed"
        );
        // Single-signed behaviour is UNCHANGED: the fix adds flags only for mixed.
        assert_eq!(cls(&["*", "float(\"inf\")", "2"]), Class::PosInf);
        assert_eq!(cls(&["*", "float(\"inf\")", "-2"]), Class::NegInf);
        assert_eq!(
            cls(&["*", "float(\"-inf\")", "float(\"inf\")"]),
            Class::NegInf
        );
        // End-to-end (the 433087 shape, reduced): the engine must never ship the
        // fabricated +inf literal for A * -inf / pi with A sign-unknown to the
        // walker but certainly positive in truth.
        let row = [
            "/",
            "*",
            "-",
            "9007199254740991",
            "log",
            "pow",
            "np.pi",
            "/",
            "1",
            "170141183460469231731687303715884105727",
            "float(\"-inf\")",
            "np.pi",
        ];
        let out = e.ac_simplify(&s(&row), 48, RuleMode::Default).unwrap();
        assert_ne!(
            out,
            s(&["float(\"inf\")"]),
            "the fabricated +inf literal must be gone (true value is -inf)"
        );
    }

    #[test]
    fn transcendental_atoms_are_enclosures_not_points() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // Exact-zero identity grounds: the nonzero certificate must REFUSE.
        assert!(
            !zero_set_null(&s(&["sin", "np.pi"]), ops, false),
            "sin(pi) is exactly 0: a nonzero-a.e. certificate must not exist"
        );
        assert!(
            !zero_set_null(&s(&["abs", "sin", "np.pi"]), ops, false),
            "abs(sin(pi)) is exactly 0: certificate must refuse"
        );
        // Grounds bounded away from 0 still certify (no blanket refusal).
        assert!(
            zero_set_null(&s(&["np.pi"]), ops, false),
            "pi itself is rigorously nonzero"
        );
        assert!(
            zero_set_null(&s(&["+", "sin", "np.pi", "2"]), ops, false),
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

    /// A verdict must not depend on what a variable is CALLED: the vocabulary says what a variable
    /// is, and every name is read as a dummy of its own (`dummy_spelling`).
    #[test]
    fn verdicts_are_invariant_to_what_a_variable_is_called() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let spell = |src: &[&str], a: &str, b: &str| -> Vec<String> {
            src.iter()
                .map(|t| match *t {
                    "A" => a.to_string(),
                    "B" => b.to_string(),
                    t => t.to_string(),
                })
                .collect()
        };
        let exprs: [&[&str]; 7] = [
            &["sin", "inv", "A"],
            &["log", "A"],
            &["-", "A", "sin", "B"],
            &["-", "A", "abs", "A"],
            &["+", "exp", "A", "*", "B", "<constant>"],
            &["/", "A", "-", "B", "B"],
            &["exp", "log", "A"],
        ];
        for src in exprs {
            let at = |a: &str, b: &str| {
                let t = spell(src, a, b);
                (
                    finite_ae(&t, ops, false),
                    zero_set_null(&t, ops, false),
                    zero_set_null_generic_const(&t, ops, false),
                    nonfinite_null(&t, ops),
                    infinite_set_null(&t, ops),
                    nonconstant_entire(&t, ops, false),
                    value_class(&t, ops),
                    domain_extension(&t, &spell(&["A"], a, b), ops),
                )
            };
            let want = at("x1", "x2");
            for (a, b) in [
                ("x_1", "x_2"),
                ("v1", "v2"),
                ("alpha", "beta"),
                ("x2", "x1"),
            ] {
                assert_eq!(at(a, b), want, "{src:?} over {a}, {b}");
            }
        }
        // a dummy keeps its dimension; another name takes one past it, never the constants' axis
        let t = s(&["+", "x0", "*", "y", "x_1"]);
        assert_eq!(
            dummy_spelling(&[&t], ops),
            Some(vec![s(&["+", "x0", "*", "x1", "x2"])])
        );
        assert_eq!(
            dummy_spelling(&[&s(&["sin", "theta"]), &s(&["theta"])], ops),
            Some(vec![s(&["sin", "x1"]), s(&["x1"])])
        );
        assert_eq!(
            dummy_spelling(&[&s(&["sin", "x3", "np.pi", "1/3"])], ops),
            None
        );
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
        let Some(e) = crate::test_engine() else {
            return;
        };
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
        // An integer literal certifies at any size when its nearest f64 IS that integer:
        // 2^200 written out is a point, and `(-inf)^(2^200)` is +inf.
        let two200 = num_traits::pow(num_bigint::BigInt::from(2), 200).to_string();
        assert_eq!(
            value_class(&s(&["pow", "float(\"-inf\")", &two200]), ops),
            Some(Class::PosInf)
        );
        // INTERVAL-LAYER CONVENTION, documented (H-045-R CLOSED 2026-08-05, owner
        // Option B): an integer literal whose nearest f64 is NOT that integer (`1e40`:
        // 10^40 is no double) cannot certify here, keeps the
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
            assert!(finite_ae(&s(&expr), ops, false), "{expr:?} must certify");
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
            assert!(
                !finite_ae(&s(&expr), ops, false),
                "{expr:?} must NOT certify"
            );
        }
    }

    /// B2: the subdivision arm proves finiteness on `[-R, R]^n` ONLY, and nothing in it
    /// looked outside -- so a nan onset past the horizon (R >= 64) was invisible and
    /// `- !0 !0 -> 0` cancelled expressions that are nan on a positive-measure tail.
    /// The mechanism was bisected to exactly the box boundary (X3: k = 0.015625, onset
    /// 64.000, refuses; k = 0.015624, onset 64.004, collapses). The tail arm conjoins
    /// slab evaluations over `{|x_i| > R}` onto the SUBDIVISION disjunct only --
    /// `nonfinite_null` is already global and sound and must not pay for it.
    #[test]
    fn finite_ae_tail_arm() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        // The two plan-listed witnesses: nan onset at 1/k = 1000, far outside the box.
        for expr in [
            vec!["log", "-", "1", "*", "0.001", "x1"],
            vec!["asin", "*", "0.001", "x1"],
        ] {
            assert!(
                !finite_ae(&s(&expr), ops, false),
                "{expr:?} is nan on a positive-measure tail and must NOT certify"
            );
        }
        // The T3 predicate's whole point: pole-free growth keeps its certificate even
        // though its tail enclosure is unbounded (exp/sinh are the flagship binders).
        for expr in [
            vec!["exp", "x0"],
            vec!["sinh", "x0"],
            vec!["tanh", "*", "x0", "x1"],
        ] {
            assert!(
                finite_ae(&s(&expr), ops, false),
                "{expr:?} must keep certifying"
            );
        }
        // In-box onsets were always refused and must stay refused (the control that
        // pins the boundary from the other side: 1/0.0157 = 63.69 < 64).
        assert!(!finite_ae(
            &s(&["log", "-", "1", "*", "0.0157", "x1"]),
            ops,
            false
        ));
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
                zero_set_null(&v(&expr), ops, false),
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
            assert!(
                !zero_set_null(&v(&expr), ops, false),
                "{expr:?} must NOT certify"
            );
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
            assert!(
                finite_nonzero_ae(&v(&expr), ops, false),
                "{expr:?} must certify"
            );
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
                !finite_nonzero_ae(&v(&expr), ops, false),
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
            assert!(
                zero_set_null(&v(&expr), ops, false),
                "{expr:?} must certify"
            );
        }
        // zero_set_null: sharpened arms must keep refusing the fat sets
        for expr in [
            vec!["inv", "inv", "-", "x0", "abs", "x0"], // zero on the fat half-line
            vec!["pow", "-", "x0", "x0", "2"],          // identically zero, no witness
            vec!["+", "abs", "x0", "x0"],               // plateau (unchanged refusal)
        ] {
            assert!(
                !zero_set_null(&v(&expr), ops, false),
                "{expr:?} must NOT certify"
            );
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

#[cfg(test)]
mod enclosure_fuzz {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
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

    /// THE enclosure property, and the reason it is written before the fixes it fails on
    /// (audit B4/B6, plan sequencing: "write the enclosure fuzz before B4/B6, not after --
    /// writing it afterwards loses the proof that it works").
    ///
    /// An interval box is only sound if it CONTAINS the pointwise value. The
    /// `interval_value_set_box` contract (rust/lib.rs) calls the opposite -- an
    /// under-approximation -- fatal, and it is: the box is what licenses `finite_ae`, which
    /// licenses cancelling `A - A` to `0`. A box that misses the pole loses the nan and the cancellation ships a wrong answer at runtime.
    ///
    /// This sweeps a dense grid through the DEGENERATE box `[c, c]`: whatever the fold
    /// computes at `c`, the box at `c` must contain it.
    #[test]
    fn box_contains_the_pointwise_value() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let mut violations: Vec<String> = Vec::new();
        let mut checked = 0usize; // anti-vacuity: a sweep that skipped every point is not a pass

        // integer exponents, both signs, straddling the parity split that B4 gets wrong
        for k in [-5i32, -4, -3, -2, -1, 1, 2, 3, 4, 5] {
            for step in 0..=80 {
                let c = -4.0 + (step as f64) * 0.1; // [-4, 4], through 0
                if c == 0.0 {
                    continue;
                }
                let expected = c.powi(k);
                if !expected.is_finite() {
                    continue;
                }
                let dom = pt(c);
                let v = value_set_p(&s(&["pow", "x0", &k.to_string()]), ops, &[dom], &[])
                    .expect("pow over a degenerate box must evaluate");
                checked += 1;
                let inside = (v.has_fin && v.lo <= expected && expected <= v.hi)
                    || (expected > 0.0 && v.pinf)
                    || (expected < 0.0 && v.ninf);
                if !inside {
                    violations.push(format!(
                        "pow({c}, {k}) = {expected} outside box [{}, {}] (pinf={} ninf={})",
                        v.lo, v.hi, v.pinf, v.ninf
                    ));
                }
            }
        }
        assert!(
            checked > 500,
            "vacuous sweep: only {checked} points evaluated"
        );
        assert!(
            violations.is_empty(),
            "{} enclosure violation(s) over {checked} points; first 5:\n  {}",
            violations.len(),
            violations
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }

    /// B6, stated as the plan's general property: for EVERY unary operator over a dense
    /// grid (plus domain-edge-hugging points), the value the FOLD KERNEL computes at `c`
    /// (`numeric::apply_op` -- the system libm, exactly what `evaluate_constant_subtree`
    /// ships) lies inside the degenerate box `[c, c]`. The arms evaluate endpoint images
    /// with whatever function they call and step outward `ULP_LIBM` = 8 ulps; an arm whose
    /// function DIFFERS from the fold's by more than that budget breaks enclosure. Not
    /// hypothetical: Rust std's `f64::atanh` is Rust's OWN composition (not a libm
    /// forward), and against glibc it is off by 161 ulps at -0.999 and by 1.5e13 ulps at
    /// -(1 - 1e-15) -- the 4th significant digit (measured 2026-08-15; `asinh`/`acosh`
    /// measured 0-ulp identical to libm over 2.2M points on the same build).
    #[test]
    fn every_unary_fold_lands_inside_its_own_box() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();

        // dense coarse grid, then edge-hugging points for the region-restricted domains
        // (atanh/asin/acos at +-1, acosh at 1, log/inv at 0) where composition error peaks
        let mut grid: Vec<f64> = (0..=160).map(|i| -4.0 + (i as f64) * 0.05).collect();
        for j in 1..=15 {
            let eps = 10f64.powi(-j);
            grid.extend([1.0 - eps, -(1.0 - eps), 1.0 + eps, eps, -eps]);
        }

        let mut unary: Vec<String> = ops
            .arity
            .iter()
            .filter(|(_, &a)| a == 1)
            .map(|(n, _)| n.clone())
            .collect();
        unary.sort(); // FxHashMap order is arbitrary; deterministic sweep + report order
        let mut checked = 0usize;
        let mut swept_atanh = false;
        let mut violations: Vec<String> = Vec::new();
        for name in &unary {
            if crate::numeric::unary_fn(name).is_none() {
                continue; // no fold kernel -> nothing whose enclosure could be claimed
            }
            for &c in &grid {
                let Some(expected) = crate::numeric::apply_op(name, &[c]) else {
                    continue;
                };
                if !expected.is_finite() {
                    continue; // poles/domain misses: the finite-value containment is the B6 claim
                }
                let Some(v) = value_set_p(&s(&[name, "x0"]), ops, &[pt(c)], &[]) else {
                    continue;
                };
                checked += 1;
                if name == "atanh" {
                    swept_atanh = true;
                }
                let inside = (v.has_fin && v.lo <= expected && expected <= v.hi)
                    || (expected > 0.0 && v.pinf)
                    || (expected < 0.0 && v.ninf);
                if !inside {
                    violations.push(format!(
                        "{name}({c:?}) = {expected:?} outside box [{:?}, {:?}] (pinf={} ninf={})",
                        v.lo, v.hi, v.pinf, v.ninf
                    ));
                }
            }
        }
        assert!(
            checked > 2000,
            "vacuous sweep: only {checked} points evaluated"
        );
        // the operator this test was written to convict must actually be in the sweep
        assert!(swept_atanh, "atanh missing from the swept unary vocabulary");
        assert!(
            violations.is_empty(),
            "{} enclosure violation(s) over {checked} points; first 8:\n  {}",
            violations.len(),
            violations
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }

    /// B1/D8(iii): the magnitude-step arms must reach the value 1 whenever the base's
    /// magnitude range CONTAINS 1 -- `pow(t, +-inf)` is 1 at |t| = 1 (ratified
    /// exception), and the old exact-point row missed both the bracketed grounds
    /// (`cos(np.pi)`: the 46-row fabricated-Inf family) and every straddling range.
    /// The box surface hulls {0} with the inf flag into [0, inf], hiding the interior
    /// gap, so this pin reads the `Vs` directly: the FINITE part must contain 1.
    #[test]
    fn magnitude_step_reaches_one() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        for exp_tok in ["float(\"inf\")", "float(\"-inf\")"] {
            let v = value_set_p(&s(&["pow", "x0", exp_tok]), ops, &[range(0.5, 2.0)], &[])
                .expect("evaluable");
            assert!(
                v.has_fin && v.lo <= 1.0 && 1.0 <= v.hi,
                "pow(x0, {exp_tok}) over [1/2, 2] must reach 1 at |x0| = 1; \
                 fin part is [{}, {}]",
                v.lo,
                v.hi
            );
        }
    }

    /// B4, stated as the property rather than as the endpoint: for an ODD NEGATIVE exponent
    /// the pole is two-sided, so a box straddling zero from below MUST reach -inf. Rust's `%`
    /// truncates, so `k % 2 == 1` is FALSE for every negative odd k -- the `-INF` candidate
    /// was never pushed and the box came back reflected. The correct idiom (`k % 2 != 0`)
    /// already sits 18 lines above the defect.
    #[test]
    fn odd_negative_powers_reach_the_lower_pole() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        for k in [-1i32, -3, -5] {
            let dom = range(-4.0, 0.0);
            let v = value_set_p(&s(&["pow", "x0", &k.to_string()]), ops, &[dom], &[])
                .expect("evaluable");
            // The pin is the PROPERTY -- unbounded below -- not an endpoint pair. The plan's
            // own acceptance value `(-inf, -0.25)` is NOT what a correct box returns here and
            // was rewritten on that measurement (audit D10 / verify_X2): with the pole reached
            // from both sides the enclosure is `(-inf, +inf)`, wider and sound, and a box may
            // always be wider. What must never happen again is the REFLECTED box that misses
            // the lower pole entirely (measured pre-fix: lo=-0.25000000000000017, hi=inf).
            let unbounded_below = v.ninf || (v.has_fin && v.lo == -INF);
            assert!(
                unbounded_below,
                "pow(x, {k}) over [-4, 0] must reach the lower pole (two-sided at an odd \
                 negative exponent); got lo={} hi={} pinf={} ninf={}",
                v.lo, v.hi, v.pinf, v.ninf
            );
        }
    }
}

/// The 0.15 heads: every enclosure contains what the deployed kernel computes, and the arms
/// that have no simple sound answer refuse.
#[cfg(test)]
mod new_head_tests {
    use super::*;
    use crate::numeric::kernels;

    /// A named unary and binary kernel.
    type Unary = (&'static str, fn(f64) -> f64);
    type Binary = (&'static str, fn(f64, f64) -> f64);

    fn engine() -> crate::engine::Engine {
        crate::engine::Engine::from_strs(include_str!("../assets/engines/acj2-4/config.yaml"), "[]")
            .expect("acj2-4 builds")
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    fn inside(v: &Vs, y: f64) -> bool {
        if y.is_nan() {
            return v.nan;
        }
        if y == INF {
            return v.pinf || (v.has_fin && v.hi == INF);
        }
        if y == -INF {
            return v.ninf || (v.has_fin && v.lo == -INF);
        }
        v.has_fin && v.lo <= y && y <= v.hi
    }

    /// A deterministic spread of boxes: points, narrow and wide intervals, both signs,
    /// integer and half-integer edges, huge magnitudes.
    fn boxes() -> Vec<(f64, f64)> {
        let edges = [
            -1e300, -1e17, -7.5, -3.0, -2.5, -1.0, -0.75, -0.5, -1e-300, 0.0, 1e-300, 0.25, 0.5,
            1.0, 1.5, 2.0, 3.0, 5.92, 6.0, 40.0, 1e17, 1e300,
        ];
        let mut out = Vec::new();
        for (i, &a) in edges.iter().enumerate() {
            for &b in &edges[i..] {
                out.push((a, b));
            }
        }
        out
    }

    fn samples(lo: f64, hi: f64) -> Vec<f64> {
        let mut v = vec![lo, hi];
        for k in 1..16 {
            let t = k as f64 / 16.0;
            v.push(lo + t * (hi - lo));
        }
        v.retain(|x| x.is_finite() && *x >= lo && *x <= hi);
        v
    }

    #[test]
    fn new_head_enclosures_contain_the_kernel() {
        let e = engine();
        let ops = e.operators_ref();
        let unary: [Unary; 4] = [
            ("floor", kernels::floor),
            ("ceil", kernels::ceil),
            ("erf", kernels::erf),
            ("harmonic", kernels::harmonic),
        ];
        let mut checked = 0usize;
        for (name, f) in unary {
            for (lo, hi) in boxes() {
                let dom = Vs::interval(lo, hi, false, false);
                let Some(v) = value_set_p(&s(&[name, "x0"]), ops, &[dom], &[]) else {
                    assert!(
                        name == "harmonic" && lo <= -1.0,
                        "{name} on [{lo}, {hi}] refused"
                    );
                    continue;
                };
                for x in samples(lo, hi) {
                    checked += 1;
                    assert!(inside(&v, f(x)), "{name}({x}) = {} outside {v:?}", f(x));
                }
            }
        }
        let binary: [Binary; 5] = [
            ("max", kernels::max),
            ("min", kernels::min),
            ("mod", kernels::modulo),
            ("atan2", kernels::atan2),
            ("besselj", kernels::besselj),
        ];
        let bx = boxes();
        for (name, f) in binary {
            for (i, &(alo, ahi)) in bx.iter().enumerate().step_by(7) {
                for &(blo, bhi) in bx.iter().skip(i % 5).step_by(5) {
                    let doms = [
                        Vs::interval(alo, ahi, false, false),
                        Vs::interval(blo, bhi, false, false),
                    ];
                    let Some(v) = value_set_p(&s(&[name, "x0", "x1"]), ops, &doms, &[]) else {
                        continue;
                    };
                    for a in samples(alo, ahi) {
                        for b in samples(blo, bhi) {
                            checked += 1;
                            assert!(
                                inside(&v, f(a, b)),
                                "{name}({a}, {b}) = {} outside {v:?}",
                                f(a, b)
                            );
                        }
                    }
                }
            }
        }
        assert!(checked > 20_000, "vacuous sweep: {checked}");
    }

    #[test]
    fn new_heads_refuse_where_no_simple_enclosure_is_sound() {
        let e = engine();
        let ops = e.operators_ref();
        let iv = |lo: f64, hi: f64| Vs::interval(lo, hi, false, false);
        let vs = |t: &[&str], doms: &[Vs]| value_set_p(&s(t), ops, doms, &[]);
        // harmonic meets its poles at or below -1
        assert!(vs(&["harmonic", "x0"], &[iv(-1.0, 2.0)]).is_none());
        assert!(vs(&["harmonic", "x0"], &[iv(-0.5, 2.0)]).is_some());
        // besselj: an order that may be a negative non-integer, at an argument that may reach
        // its pole at 0; away from 0 the recurrence bounds it, and a negative integer order is
        // bounded by 1
        assert!(vs(&["besselj", "x0", "x1"], &[iv(-0.5, 0.5), iv(0.0, 1.0)]).is_none());
        let b = vs(&["besselj", "x0", "x1"], &[iv(-0.5, 0.5), iv(0.5, 1.0)]).unwrap();
        assert!(b.has_fin && !b.nan && b.hi >= 14.0 && b.hi < 15.0, "{b:?}");
        let b = vs(&["besselj", "-1.5", "x0"], &[iv(2.0, 3.0)]).unwrap();
        assert!(
            b.has_fin && b.lo < -1.0 && b.hi > 1.0 && b.hi < 10.0,
            "{b:?}"
        );
        let n = vs(&["besselj", "-3", "x0"], &[iv(-1.0, 1.0)]).unwrap();
        assert!(n.has_fin && !n.nan && n.hi < 1.01 && n.lo > -1.01, "{n:?}");
        assert!(vs(&["besselj", "x0", "x1"], &[iv(0.0, 3.0), iv(-1.0, 1.0)]).is_some());
        // mod: a divisor that may be zero among other values
        assert!(vs(&["mod", "x0", "x1"], &[iv(0.0, 1.0), iv(-1.0, 1.0)]).is_none());
        // ... and a zero divisor is NaN, everywhere
        let z = vs(&["mod", "x0", "0"], &[iv(0.0, 1.0)]).unwrap();
        assert!(z.nan && !z.has_fin);
        // a non-integer order at a negative argument is NaN
        let n = vs(&["besselj", "0.5", "x0"], &[iv(-2.0, -1.0)]).unwrap();
        assert!(n.nan);
        // every unknown head is unanalyzable
        let cfg = "operators:\n  \"+\": {realization: '+', arity: 2, precedence: 1, commutative: true}\n  gamma: {realization: math.gamma, arity: 1}\n";
        let e2 = crate::engine::Engine::from_strs(cfg, "[]").unwrap();
        assert!(value_set_p(
            &s(&["gamma", "x0"]),
            e2.operators_ref(),
            &[iv(0.0, 1.0)],
            &[]
        )
        .is_none());
    }

    /// A ground class through a 0.15 head is decided by the interval layer: `mod(1, 0)` is NaN,
    /// `floor(inf)` is +inf, and a finite one stays finite.
    #[test]
    fn new_head_ground_classes() {
        let e = engine();
        let ops = e.operators_ref();
        assert_eq!(value_class(&s(&["mod", "1", "0"]), ops), Some(Class::Nan));
        assert_eq!(
            value_class(&s(&["floor", "float(\"inf\")"]), ops),
            Some(Class::PosInf)
        );
        assert_eq!(
            value_class(&s(&["erf", "float(\"-inf\")"]), ops),
            Some(Class::Finite)
        );
        assert_eq!(
            value_class(&s(&["besselj", "-0.5", "0"]), ops),
            Some(Class::PosInf)
        );
        assert_eq!(
            value_class(&s(&["besselj", "-1.5", "0"]), ops),
            Some(Class::NegInf)
        );
        assert_eq!(
            value_class(&s(&["besselj", "sin", "-1", "0"]), ops),
            Some(Class::PosInf)
        );
        assert_eq!(
            value_class(&s(&["besselj", "sin", "-1", "-1"]), ops),
            Some(Class::Nan)
        );
        assert_eq!(
            value_class(&s(&["besselj", "0.5", "-2"]), ops),
            Some(Class::Nan)
        );
        // a NaN order, and an infinite one on the negative axis (scipy's cells)
        assert_eq!(
            value_class(&s(&["besselj", "log", "-1", "1"]), ops),
            Some(Class::Nan)
        );
        assert_eq!(
            value_class(&s(&["besselj", "atanh", "1", "-1"]), ops),
            Some(Class::Nan)
        );
        assert_eq!(
            value_class(&s(&["besselj", "atanh", "1", "2"]), ops),
            Some(Class::Finite)
        );
        // harmonic is NaN at a negative integer (D6) and at -inf
        assert_eq!(value_class(&s(&["harmonic", "-2"]), ops), Some(Class::Nan));
        assert_eq!(
            value_class(&s(&["harmonic", "atanh", "ceil", "-1"]), ops),
            Some(Class::Nan)
        );
    }

    /// The infinite cells of `mod` are numpy's (D7): an infinite dividend is NaN; under an
    /// infinite divisor the dividend is its own remainder where it is zero or has the divisor's
    /// sign, and the divisor's infinity otherwise.
    #[test]
    fn mod_reads_numpys_infinite_cells() {
        let e = engine();
        let ops = e.operators_ref();
        let class = |t: &[&str]| value_class(&s(t), ops);
        let set = |t: &[&str]| value_set(&s(t), ops, &Vs::reals()).unwrap();
        let (pinf, ninf) = ("float(\"inf\")", "float(\"-inf\")");
        assert_eq!(class(&["mod", "1", ninf]), Some(Class::NegInf));
        assert_eq!(class(&["mod", "-1", pinf]), Some(Class::PosInf));
        assert_eq!(class(&["mod", pinf, "1"]), Some(Class::Nan));
        assert_eq!(class(&["mod", "log", "0", "0"]), Some(Class::Nan));
        for (a, b, v) in [("-1", ninf, -1.0), ("1", pinf, 1.0), ("0", ninf, 0.0)] {
            let r = set(&["mod", a, b]);
            assert!(r.is_const() && r.lo == v, "mod({a}, {b}): {r:?}");
        }
        for (a, b) in [
            (-1.0, INF),
            (1.0, -INF),
            (0.0, INF),
            (-0.0, -INF),
            (2.5, INF),
            (-2.5, -INF),
        ] {
            let r = set(&["mod", &format!("{a:?}"), if b > 0.0 { pinf } else { ninf }]);
            assert!(
                inside(&r, kernels::modulo(a, b)),
                "mod({a}, {b}) outside {r:?}"
            );
        }
        // a dividend of either sign under +inf: itself where non-negative, +inf below
        let r = value_set_p(
            &s(&["mod", "x0", pinf]),
            ops,
            &[Vs::interval(-1.0, 2.0, false, false)],
            &[],
        )
        .unwrap();
        assert!(
            r.has_fin && r.lo == 0.0 && r.hi == 2.0 && r.pinf && !r.ninf,
            "{r:?}"
        );
    }

    /// harmonic is enclosed strictly between two poles, where it increases: the enclosure holds
    /// the kernel at every sample and the true value `H(-3/2) = 2 - 2 ln 2`.
    #[test]
    fn harmonic_between_its_poles() {
        let e = engine();
        let ops = e.operators_ref();
        let vs = |lo: f64, hi: f64| {
            value_set_p(
                &s(&["harmonic", "x0"]),
                ops,
                &[Vs::interval(lo, hi, false, false)],
                &[],
            )
        };
        let h = vs(-1.5, -1.5).unwrap();
        let exact = 2.0 - 2.0 * std::f64::consts::LN_2;
        assert!(h.lo < exact && exact < h.hi && h.hi - h.lo < 1e-13, "{h:?}");
        for n in 1..40 {
            let k = -(n as f64) - 1.0;
            for (lo, hi) in [
                (k + 1e-9, k + 0.5),
                (k + 0.25, k + 0.75),
                (k + 0.5, k + 1.0 - 1e-9),
            ] {
                let v = vs(lo, hi).unwrap();
                assert!(!v.nan && v.has_fin, "{v:?}");
                for x in samples(lo, hi) {
                    assert!(
                        inside(&v, kernels::harmonic(x)),
                        "harmonic({x}) outside {v:?}"
                    );
                }
            }
            // a box that reaches a pole is refused
            assert!(vs(k - 0.5, k + 0.5).is_none());
            assert!(vs(k, k + 0.5).is_none());
        }
    }

    /// besselj's exact and bounded cells: `J_nu(0)` at a positive order, an integer order and
    /// order 0; `|J_n| <= 1/sqrt(2)` at an integer n != 0, and the alternating series' bracket on
    /// 0 < |x| <= 2, with its sign, each holding the kernel.
    #[test]
    fn besselj_exact_and_bounded_cells() {
        let e = engine();
        let ops = e.operators_ref();
        let set = |t: &[&str]| value_set(&s(t), ops, &Vs::reals()).unwrap();
        for (nu, v) in [
            (&["sin", "1"][..], 0.0),
            (&["-2"][..], 0.0),
            (&["0"][..], 1.0),
        ] {
            let mut t = vec!["besselj"];
            t.extend_from_slice(nu);
            t.push("0");
            let r = set(&t);
            assert!(r.is_const() && r.lo == v, "{t:?}: {r:?}");
        }
        let r = set(&["besselj", "1", "1"]);
        assert!(r.lo > 0.43 && r.hi < 0.51, "{r:?}");
        let r = set(&["besselj", "-1", "1"]);
        assert!(r.hi < -0.43 && r.lo > -0.51, "{r:?}");
        let r = set(&["besselj", "1", "-1"]);
        assert!(r.hi < -0.43, "{r:?}");
        let r = set(&["besselj", "-2", "-1"]);
        assert!(r.lo > 0.1, "{r:?}");
        let r = set(&["besselj", "0", "1"]);
        assert!(r.lo > 0.74 && r.hi < 0.77, "{r:?}");
        assert_eq!(
            value_class(&s(&["acosh", "besselj", "-1", "1"]), ops),
            Some(Class::Nan)
        );
        assert_eq!(
            value_class(&s(&["acosh", "besselj", "3", "x0"]), ops),
            Some(Class::Nan)
        );
        assert_eq!(
            value_class(&s(&["log", "besselj", "-1", "1"]), ops),
            Some(Class::Nan)
        );
        // the brackets hold the kernel, every integer order to 50 and both signs of x
        let mut checked = 0;
        for n in -50..=50 {
            for (lo, hi) in [
                (1e-3, 0.5),
                (0.5, 1.0),
                (1.0, 2.0),
                (1.9, 2.0),
                (0.01, 0.01),
            ] {
                for (a, b) in [(lo, hi), (-hi, -lo)] {
                    let order = format!("{n}");
                    let v = value_set_p(
                        &s(&["besselj", &order, "x0"]),
                        ops,
                        &[Vs::interval(a, b, false, false)],
                        &[],
                    )
                    .unwrap();
                    for x in samples(a, b) {
                        checked += 1;
                        let y = kernels::besselj(n as f64, x);
                        assert!(inside(&v, y), "J_{n}({x}) = {y} outside {v:?}");
                    }
                }
            }
        }
        assert!(checked > 10_000, "{checked}");
    }

    /// atan2 on the positive x axis and at x = +inf is a zero.
    #[test]
    fn atan2_zero_on_the_positive_axis() {
        let e = engine();
        let ops = e.operators_ref();
        let set = |t: &[&str]| value_set(&s(t), ops, &Vs::reals()).unwrap();
        for t in [
            &["atan2", "0", "cosh", "1"][..],
            &["atan2", "sin", "1", "float(\"inf\")"][..],
        ] {
            let r = set(t);
            assert!(r.is_const() && r.lo == 0.0, "{t:?}: {r:?}");
        }
        let r = set(&["atan2", "0", "neg", "cosh", "1"]);
        assert!(!r.is_const(), "{r:?}");
        // on the y axis: +-pi/2 by y's sign, holding the kernel at both zeros
        let half = std::f64::consts::FRAC_PI_2;
        let r = set(&["atan2", "1", "0"]);
        assert!(r.lo < half && half < r.hi && r.hi - r.lo < 1e-14, "{r:?}");
        let r = set(&["atan2", "neg", "sin", "1", "-0.0"]);
        assert!(
            inside(&r, kernels::atan2(-1.0, -0.0)) && r.hi < -1.5,
            "{r:?}"
        );
    }

    /// The pole-free table is keyed the safe way round: a head it does not name makes the tree
    /// pole-bearing (bounded slabs only).
    #[test]
    fn tail_arm_reads_unknown_heads_as_pole_bearing() {
        let e = engine();
        let ops = e.operators_ref();
        // erf is pole-free and bounded; harmonic of exp is unbounded and harmonic is not named
        assert!(tail_finite_ae(&s(&["erf", "x0"]), ops));
        assert!(!tail_finite_ae(&s(&["harmonic", "exp", "x0"]), ops));
        // the measured float64 tail of erf
        assert_eq!(kernels::erf(ERF_SAT), 1.0);
        assert!(kernels::erf(f64::from_bits(ERF_SAT.to_bits() - 1)) < 1.0);
        assert_eq!(tail_of("erf"), Some((ERF_SAT, 1.0)));
    }

    /// The `!` (finite a.e.), `$` (finite and nonzero a.e.) and zero-set certificates read each
    /// 0.15 head by its own argument, and refuse where the head is undefined, zero or infinite
    /// on a region: a step head on [0, 1) or (-1, 0], `mod` by a ground divisor (float64's zero
    /// tail), `besselj` at a non-literal order or a non-integer order below 0, `harmonic`'s
    /// float64 NaN tail at or below -2^52.
    #[test]
    fn certificates_read_the_new_heads() {
        let e = engine();
        let ops = e.operators_ref();
        let t = |x: &str| x.split(' ').map(String::from).collect::<Vec<String>>();
        let fin = |x: &str, f64: bool| finite_ae(&t(x), ops, f64);
        let fnz = |x: &str, f64: bool| finite_nonzero_ae(&t(x), ops, f64);
        let zs = |x: &str| zero_set_null(&t(x), ops, true) && zero_set_null(&t(x), ops, false);
        let isn = |x: &str| infinite_set_null(&t(x), ops);
        for x in [
            "erf x0",
            "erf - x0 x1",
            "besselj 0 x0",
            "besselj 1 x0",
            "besselj 3 * 2 x0",
            "besselj 1.5 exp x0",
            "besselj -1.5 pow x0 2",
            "harmonic exp x0",
            "harmonic sin x0",
            "max x0 x1",
            "min x0 sin x1",
            "mod x0 x1",
            "mod x0 sinh x1",
            "mod exp x0 x1",
            "atan2 x0 x1",
            "atan2 sin x0 exp x1",
            "floor + exp x0 2",
            "ceil - -2 exp x0",
        ] {
            for f64 in [true, false] {
                assert!(fnz(x, f64), "$ {x} f64={f64}");
                assert!(fin(x, f64), "! {x} f64={f64}");
            }
        }
        // real: harmonic's poles are a null set; float64: its NaN tail at or below -2^52 is not
        for x in [
            "harmonic x0",
            "harmonic neg exp x0",
            "harmonic + -1.5 sin x0",
        ] {
            assert!(fnz(x, false) && fin(x, false), "{x}");
        }
        assert!(!fin("harmonic x0", true) && !fnz("harmonic x0", true));
        assert!(!fin("harmonic neg exp x0", true));
        assert!(fin("harmonic + -1.5 sin x0", true));
        // finite, but zero on a region: `!` holds, `$` and the zero set do not
        for x in [
            "floor x0",
            "ceil x0",
            "floor sin x0",
            "ceil + exp x0 -1",
            "max x0 0",
            "min x0 0",
            "max x0 floor x1",
            "mod x0 2",
            "mod x0 1",
            "mod x0 max x1 1",
            "atan2 0 x0",
            "atan2 floor x0 x1",
        ] {
            assert!(!zs(x), "zero set of {x}");
            assert!(!fnz(x, true) && !fnz(x, false), "$ {x}");
        }
        assert!(fin("mod x0 2", true) && fin("floor x0", true) && fin("max x0 0", true));
        // the pole of a negative non-integer order at |x0| = 0 is a point; `abs` is no entire
        // head, so the zero set of the argument's levels fails closed
        assert!(fin("besselj -1.5 abs x0", true) && !fnz("besselj -1.5 abs x0", false));
        // float64's `mod(x, 1)` is 0 on |x| >= 2^52, the tail `tanh(x1) = 1` reaches
        assert!(fnz("mod x0 tanh x1", false) && !fnz("mod x0 tanh x1", true));
        // NaN or a pole on a region: neither `!` nor `$`
        for x in [
            "besselj 1.5 x0",
            "besselj -1.5 x0",
            "besselj x1 x0",
            "besselj 2000 x0",
            "mod x0 0",
            "mod x0 floor x1",
            "harmonic floor x0",
            "harmonic neg abs floor x0",
        ] {
            for f64 in [true, false] {
                assert!(!fin(x, f64), "! {x} f64={f64}");
                assert!(!fnz(x, f64), "$ {x} f64={f64}");
            }
        }
        // a `<constant>` never certifies a zero set or a divisor
        for x in [
            "mod x0 <constant>",
            "besselj -1.5 + x0 <constant>",
            "erf * <constant> x0",
        ] {
            assert!(!zs(x) && !fnz(x, false), "{x}");
        }
        assert!(fin("mod <constant> x0", false) && fin("erf * <constant> x0", false));
        // infinite sets: the bounded heads never, `mod` only under an infinite divisor,
        // `besselj` only at a negative non-integer order's pole
        for x in [
            "erf x0",
            "atan2 x0 x1",
            "max x0 x1",
            "mod x0 x1",
            "mod inv x0 x1",
            "harmonic x0",
            "floor x0",
            "besselj 2 x0",
            "besselj 0.5 x0",
            "besselj -1.5 x0",
        ] {
            assert!(isn(x), "infinite set of {x}");
        }
        for x in [
            "mod x0 float(\"inf\")",
            "max x0 float(\"inf\")",
            "besselj -1.5 0",
            "besselj x1 x0",
        ] {
            assert!(!isn(x), "infinite set of {x}");
        }
        // the float64 kernels behind the arms: no nonzero argument of erf or harmonic rounds to
        // zero, harmonic is NaN at every float at or below -2^52, besselj at a non-integer order
        // is NaN below 0, and float64's mod by 2 is 0 on the whole ray past 2^53
        for x in [5e-324, -5e-324, 1e-300, -1e-300] {
            assert!(kernels::erf(x) != 0.0 && kernels::harmonic(x) != 0.0, "{x}");
        }
        for x in [-4503599627370496.0, -4503599627370498.0, -1e300] {
            assert!(kernels::harmonic(x).is_nan(), "{x}");
        }
        assert!(!kernels::harmonic(-4503599627370495.5).is_nan());
        assert!(kernels::besselj(1.5, -1e-300).is_nan() && kernels::besselj(-1.5, -2.0).is_nan());
        assert_eq!(kernels::modulo(9007199254740993.0 * 2.0 + 2.0, 2.0), 0.0);
    }
}
