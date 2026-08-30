//! The licence registry -- C1.20's dual, owner-approved 2026-08-08.
//!
//! The miner's only probe used to be "does the engine already derive this?" (the
//! Kruskal prune), which reads a doctrinal REFUSAL as ignorance and mints the
//! forbidden rewrite as a discovery. Three shipped/blocked incidents (the 2026-08-01
//! `+ <constant> np.pi -> <constant>` mint, F49's seven `e`-absorptions, the
//! F55-closed `pow acos (-1) <constant>`) were each patched with a bespoke mint-side
//! twin keyed on a proxy (tokens -> spellings -> endpoints), and each proxy rotted as
//! the canon evolved. This module is the single home the register asked for: every
//! reified refusal lives here, the mint path consults it at candidate judgment, and
//! the AC load path consults it as defense in depth. The completeness argument -- one
//! disposition per doctrinal refusal, with the proof or falsifier that earned it --
//! is the research harness; the standing obligation (every future
//! engine-side refusal ships with a registry disposition IN THE SAME COMMIT) is the
//! contract clause recorded with this build.
//!
//! Design rules, all load-bearing:
//! - Entries key on VALUES or STATES, never token spellings (the proxy-rot lesson).
//! - The registry never over-refuses what the contract licenses: `Ext` at a point is
//!   R3's null-set licence and passes; only defined-value changes (R2, "sign
//!   included") refuse.
//! - Fail-open on unevaluable adjudications (`contract_point_verdict`'s own
//!   convention): the registry is a doctrine gate, not a substitute for the numeric
//!   challenge bars, which keep running unchanged.

use crate::hiprec::{contract_point_verdict, PointCmp, ProbeAtom};
use crate::operators::Operators;

/// One doctrinal refusal. The variants mirror SWEEP.md's REGISTRY rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Refusal {
    /// SWEEP row 2 (relocated): a DETERMINED source (var-free, Const-free) never
    /// accepts a Const-INTRODUCING target -- `<constant>` for a single value is
    /// vacuous abstraction the engine's own semantics refuse.
    ConstIntroduction,
    /// SWEEP row 1 (relocated, content = contract 10.11): a `<constant>`-bearing
    /// target may not lower the special count of a special-bearing source. Since
    /// 10.11 the constructor absorbs specials at the bijective bag sites, so such
    /// sources reach atomic endpoints and never arrive here; what this refuses is
    /// the non-absorbable remainder (introduced masks; non-bag positions like
    /// `pow np.pi <constant>`).
    SpecialAbsorption,
    /// SWEEP row 3 (new): R2 at a constructed exceptional point -- the candidate
    /// changes a DEFINED extended-real value (sign of infinity included) at a
    /// definable point of the pattern space. `1/(1-w)` vs `-1/(w-1)` at w=1.
    PoleClassChange,
}

/// Count of special-constant tokens (`np.pi` / `np.e`) in a token slice. The raw
/// token count is only ever used together with the ENDPOINT count (spelling
/// independence); see `mint_refusals`.
fn n_special(ts: &[String]) -> usize {
    ts.iter().filter(|t| *t == "np.pi" || *t == "np.e").count()
}

impl Refusal {
    /// The wire name a refusal reports to the Python layer. Exhaustive on purpose: a new
    /// variant must be named here or the crate stops compiling. The callers used to spell
    /// these literals themselves, which put an UNREACHABLE arm in `lib.rs` (mint_refusals
    /// has three exits and none is `PoleClassChange`) and duplicated "pole_class_change"
    /// at the two sites that can emit it (audit A9 / X9). A `_ =>` fallback would silence
    /// exactly the rename this arm exists to catch, so there is none.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Refusal::ConstIntroduction => "const_introduction",
            Refusal::SpecialAbsorption => "special_absorption",
            Refusal::PoleClassChange => "pole_class_change",
        }
    }
}

/// The cheap, syntactic registry entries at MINT-candidate judgment. EXACT
/// relocation of the two licences that lived inline in `mine_one_length`'s accept
/// closure (behavior-identical by construction; the triple re-mine byte-identity is
/// the falsifier for this relocation).
///
/// `src` is the enumerated source, `ac_out` the engine's own endpoint for it (the
/// state a bare `<constant>` target would replace -- counting there is
/// spelling-independent, F49's lesson), `target` the candidate.
pub fn mint_refusals(
    src: &[String],
    ac_out: &[String],
    target: &[String],
    var_names: &[String],
) -> Option<Refusal> {
    let src_determined =
        !src.iter().any(|x| x == "<constant>") && !src.iter().any(|x| var_names.contains(x));
    if src_determined && target.iter().any(|x| x == "<constant>") {
        return Some(Refusal::ConstIntroduction);
    }
    let src_specials = n_special(src).max(n_special(ac_out));
    if src_specials > 0
        && target.iter().any(|x| x == "<constant>")
        && n_special(target) < src_specials
    {
        return Some(Refusal::SpecialAbsorption);
    }
    None
}

/// An affine form `a + b*w` over exact i64 rationals (small, overflow-checked): the
/// symbolic value of a one-variable subtree, when the subtree is affine in it.
#[derive(Clone, Copy, Debug)]
struct Affine {
    // a = an/ad, b = bn/bd, all i64 with positive denominators
    an: i64,
    ad: i64,
    bn: i64,
    bd: i64,
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    if a == 0 {
        1
    } else {
        a
    }
}

fn norm(n: i64, d: i64) -> Option<(i64, i64)> {
    if d == 0 {
        return None;
    }
    let s = if d < 0 { -1 } else { 1 };
    let g = gcd(n, d);
    Some((s * (n / g), (d / g).abs()))
}

fn add_q(an: i64, ad: i64, bn: i64, bd: i64) -> Option<(i64, i64)> {
    let n = an.checked_mul(bd)?.checked_add(bn.checked_mul(ad)?)?;
    let d = ad.checked_mul(bd)?;
    norm(n, d)
}

fn mul_q(an: i64, ad: i64, bn: i64, bd: i64) -> Option<(i64, i64)> {
    norm(an.checked_mul(bn)?, ad.checked_mul(bd)?)
}

impl Affine {
    fn constant(n: i64, d: i64) -> Option<Affine> {
        let (an, ad) = norm(n, d)?;
        Some(Affine {
            an,
            ad,
            bn: 0,
            bd: 1,
        })
    }
    fn var() -> Affine {
        Affine {
            an: 0,
            ad: 1,
            bn: 1,
            bd: 1,
        }
    }
    fn add(self, o: Affine) -> Option<Affine> {
        let (an, ad) = add_q(self.an, self.ad, o.an, o.ad)?;
        let (bn, bd) = add_q(self.bn, self.bd, o.bn, o.bd)?;
        Some(Affine { an, ad, bn, bd })
    }
    fn neg(self) -> Affine {
        Affine {
            an: -self.an,
            ad: self.ad,
            bn: -self.bn,
            bd: self.bd,
        }
    }
    fn scale(self, n: i64, d: i64) -> Option<Affine> {
        let (an, ad) = mul_q(self.an, self.ad, n, d)?;
        let (bn, bd) = mul_q(self.bn, self.bd, n, d)?;
        Some(Affine { an, ad, bn, bd })
    }
    /// The exact root of `a + b*w = 0`, when b != 0.
    fn root(self) -> Option<(i64, u64)> {
        if self.bn == 0 {
            return None;
        }
        // w = -a/b = -(an * bd) / (ad * bn)
        let n = self.an.checked_mul(self.bd)?.checked_neg()?;
        let d = self.ad.checked_mul(self.bn)?;
        let (n, d) = norm(n, d)?;
        Some((n, d as u64))
    }
}

/// Parse an integer or simple rational literal token (the mine alphabet's literal
/// spellings: `3`, `(-3)`, `0.5` is NOT in the alphabet -- integers and the
/// parenthesized negatives are). Returns (n, d).
fn lit_q(tok: &str) -> Option<(i64, i64)> {
    let s = tok
        .strip_prefix('(')
        .and_then(|t| t.strip_suffix(')'))
        .unwrap_or(tok);
    if let Ok(n) = s.parse::<i64>() {
        return Some((n, 1));
    }
    // decimal literals occur in ac_out spellings (`0.5`): parse small terminating
    // decimals exactly; anything unparseable is simply "not affine" (fail-soft).
    if let Some(dot) = s.find('.') {
        let (int_part, frac_part) = (&s[..dot], &s[dot + 1..]);
        if frac_part.len() <= 9 && !frac_part.is_empty() {
            let neg = int_part.starts_with('-');
            let i: i64 = int_part.parse().ok()?;
            let f: i64 = frac_part.parse().ok()?;
            let scale = 10_i64.checked_pow(frac_part.len() as u32)?;
            let n = i.checked_mul(scale)?;
            let n = if neg {
                n.checked_sub(f)?
            } else {
                n.checked_add(f)?
            };
            return Some((n, scale));
        }
    }
    None
}

/// One parsed prefix node: (token, child subslices). Returns (consumed, ...).
fn subtree_len(tokens: &[String], ops: &Operators) -> Option<usize> {
    let t = tokens.first()?;
    match ops.arity_of(t) {
        Some(2) => {
            let a = subtree_len(&tokens[1..], ops)?;
            let b = subtree_len(&tokens[1 + a..], ops)?;
            Some(1 + a + b)
        }
        Some(1) => Some(1 + subtree_len(&tokens[1..], ops)?),
        _ => Some(1), // leaf (variable, literal, special, <constant>)
    }
}

/// The affine value of `tokens` in the single variable `w` (all other variables make
/// the subtree non-affine -> None). Ops covered: `+ - * / neg` with the division
/// restricted to a constant divisor and multiplication to one constant side.
fn affine_of(tokens: &[String], w: &str, var_names: &[String], ops: &Operators) -> Option<Affine> {
    let t = &tokens[0];
    match ops.arity_of(t.as_str()) {
        Some(2) => {
            let la = subtree_len(&tokens[1..], ops)?;
            let lhs = &tokens[1..1 + la];
            let rhs = &tokens[1 + la..];
            let a = affine_of(lhs, w, var_names, ops);
            let b = affine_of(rhs, w, var_names, ops);
            match (t.as_str(), a, b) {
                ("+", Some(a), Some(b)) => a.add(b),
                ("-", Some(a), Some(b)) => a.add(b.neg()),
                ("*", Some(a), Some(b)) => {
                    if a.bn == 0 {
                        b.scale(a.an, a.ad)
                    } else if b.bn == 0 {
                        a.scale(b.an, b.ad)
                    } else {
                        None
                    }
                }
                ("/", Some(a), Some(b)) if b.bn == 0 && b.an != 0 => a.scale(b.ad, b.an),
                _ => None,
            }
        }
        Some(1) => match t.as_str() {
            "neg" => Some(affine_of(&tokens[1..], w, var_names, ops)?.neg()),
            _ => None,
        },
        _ => {
            if t == w {
                Some(Affine::var())
            } else if var_names.contains(t) || t == "<constant>" {
                None
            } else if let Some((n, d)) = lit_q(t) {
                Affine::constant(n, d)
            } else {
                None // specials/inf/nan: exact but not rational -- out of the affine scope
            }
        }
    }
}

/// Collect the EXCEPTIONAL POINTS of a pattern: exact rational bindings of one
/// variable at which a denominator (or `log` argument) vanishes. Scope (documented in
/// SWEEP.md row 3): `inv X` / `/ A B` / `pow B <negative integer>` denominators and
/// `log X` at X=0, with the vanishing subtree AFFINE in exactly one variable. This is
/// the entire F63 orbit surface; transcendental locations (tan) are H-017's lane.
fn exceptional_points(
    tokens: &[String],
    var_names: &[String],
    ops: &Operators,
    out: &mut Vec<(String, (i64, u64))>,
) {
    if tokens.is_empty() {
        return;
    }
    let t = &tokens[0];
    let mut denoms: Vec<&[String]> = Vec::new();
    match (t.as_str(), ops.arity_of(t.as_str())) {
        // The token names are only trusted as structure when the config's own arity
        // table agrees -- a leaf that happens to be SPELLED "inv" must not have the
        // following tokens read as its child.
        ("inv" | "log", Some(1)) => denoms.push(&tokens[1..]),
        ("/", Some(2)) => {
            if let Some(la) = subtree_len(&tokens[1..], ops) {
                denoms.push(&tokens[1 + la..]);
            }
        }
        ("pow", Some(2)) => {
            if let Some(lb) = subtree_len(&tokens[1..], ops) {
                let exp = &tokens[1 + lb..];
                if exp.len() == 1 {
                    if let Some((n, 1)) = lit_q(&exp[0]) {
                        if n < 0 {
                            denoms.push(&tokens[1..1 + lb]);
                        }
                    }
                }
            }
        }
        _ => {}
    }
    for d in denoms {
        // exactly one variable, occurring exactly once, affine
        let vars: Vec<&String> = d.iter().filter(|x| var_names.contains(x)).collect();
        if vars.len() == 1 {
            let w = vars[0].clone();
            if let Some(aff) = affine_of(d, &w, var_names, ops) {
                if let Some(root) = aff.root() {
                    if !out.iter().any(|(v, r)| *v == w && *r == root) {
                        out.push((w, root));
                    }
                }
            }
        }
    }
    // recurse into children
    if let Some(ar) = ops.arity_of(t.as_str()) {
        let mut rest = &tokens[1..];
        for _ in 0..ar {
            let Some(l) = subtree_len(rest, ops) else {
                return;
            };
            exceptional_points(&rest[..l], var_names, ops, out);
            rest = &rest[l..];
        }
    }
}

/// Generic (off-pole) values for the OTHER variables: odd rationals chosen to avoid
/// coincidental zeros of small-integer affine forms.
const GENERIC: [(i64, u64); 3] = [(3, 7), (-5, 11), (9, 13)];

/// SWEEP row 3 -- the pole-class entry. Constructs the exceptional points of BOTH
/// sides and adjudicates each with the contract evaluator (one-zero conventions,
/// precision rungs): a candidate that changes a DEFINED extended-real value at a
/// definable point (InfChange / RealChange / a lost value) is refused; `Ext` (a nan
/// hole filled at the point) is R3's licence and passes. Fail-open on unevaluable
/// points, per `contract_point_verdict`'s own convention.
///
/// Cost discipline: callers pre-screen (the function returns immediately when
/// neither side has a solvable exceptional point), and the mint path consults this
/// LAST, after the cheap licences and the ordering acceptance, so it runs only on
/// candidates that would otherwise be accepted.
pub fn pole_refusal(
    src: &[String],
    target: &[String],
    ops: &Operators,
    var_names: &[String],
) -> Option<Refusal> {
    let mut pts: Vec<(String, (i64, u64))> = Vec::new();
    exceptional_points(src, var_names, ops, &mut pts);
    exceptional_points(target, var_names, ops, &mut pts);
    if pts.is_empty() {
        return None;
    }
    // Variables appearing anywhere (order = var_names order, the atoms vector's frame)
    let used: Vec<String> = var_names
        .iter()
        .filter(|v| src.contains(v) || target.contains(v))
        .cloned()
        .collect();
    for (w, (n, d)) in &pts {
        for (gi, g) in GENERIC.iter().enumerate() {
            let atoms: Vec<ProbeAtom> = var_names
                .iter()
                .map(|v| {
                    if v == w {
                        ProbeAtom::Rat(*n, *d)
                    } else if used.contains(v) {
                        // rotate the generic values so multi-var patterns see varied
                        // off-pole coordinates across the three rounds
                        let k =
                            (gi + used.iter().position(|u| u == v).unwrap_or(0)) % GENERIC.len();
                        ProbeAtom::Rat(GENERIC[k].0, GENERIC[k].1)
                    } else {
                        ProbeAtom::Rat(g.0, g.1)
                    }
                })
                .collect();
            let (verdict, _snapped) =
                contract_point_verdict(src, &[], target, &[], ops, var_names, &atoms);
            match verdict {
                Some(PointCmp::InfChange) | Some(PointCmp::RealChange) | Some(PointCmp::Shrink) => {
                    return Some(Refusal::PoleClassChange);
                }
                _ => {}
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    fn test_engine() -> Engine {
        let cfg = r#"operators:
  "+": {realization: '+', arity: 2, precedence: 1, commutative: true}
  "-": {realization: '-', arity: 2, precedence: 1}
  "*": {realization: '*', arity: 2, precedence: 2, commutative: true}
  "/": {realization: '/', arity: 2, precedence: 2}
  neg: {realization: simplipy.operators.neg, arity: 1, precedence: 2.5}
  inv: {realization: simplipy.operators.inv, arity: 1, precedence: 4}
  pow: {realization: simplipy.operators.pow, arity: 2, precedence: 3}
  log: {realization: simplipy.operators.log, arity: 1, precedence: 4}
"#;
        Engine::from_strs(cfg, "[]").expect("engine builds")
    }

    /// THE falsifier this entry exists for -- the verified hostile pair of the C1.20
    /// design session: `-(1/(1-w))` and `1/(w-1)` are stable states, a.e.-equal,
    /// mu-ordered mirror-below-source (42,000 -> 26,000), invisible to every other
    /// gate (challenges sample no measure-zero set; POLE_GRID is constants-only; all
    /// seven load drops are syntactic; promotion is flatteners-only), and differ at
    /// exactly w=1: -inf vs +inf under one-zero. The registry must refuse it.
    #[test]
    fn pole_entry_refuses_the_mirror_family() {
        let e = test_engine();
        let vars = s(&["x0"]);
        let src = s(&["neg", "inv", "-", "1", "x0"]);
        let tgt = s(&["inv", "-", "x0", "1"]);
        assert_eq!(
            pole_refusal(&src, &tgt, &e.operators, &vars),
            Some(Refusal::PoleClassChange),
            "the a.e. pole mirror must be refused at its constructed exceptional point"
        );
        // The same trade through a NON-DYADIC pole location (1 - 3w = 0 at w = 1/3):
        // exercises ProbeAtom::Rat exactness -- the f64 rounding of 1/3 sits BESIDE
        // the pole and would judge Eq.
        let src = s(&["neg", "inv", "-", "1", "*", "3", "x0"]);
        let tgt = s(&["inv", "-", "*", "3", "x0", "1"]);
        assert_eq!(
            pole_refusal(&src, &tgt, &e.operators, &vars),
            Some(Refusal::PoleClassChange),
            "non-dyadic pole locations must be hit exactly (ProbeAtom::Rat)"
        );
    }

    /// Sound denominator-bearing rewrites KEEP: agreement at the exceptional point is
    /// what sound rules do there (`inv(inv(w))` at w=0: inv(+inf) = 0 = w), and R3's
    /// point-level licence (`Ext`: a nan hole filled) must not be over-refused --
    /// `w/w -> 1` at w=0 is 0/0 = nan -> 1, the contract's own forcing example.
    #[test]
    fn pole_entry_keeps_sound_and_ext_licensed() {
        let e = test_engine();
        let vars = s(&["x0"]);
        assert_eq!(
            pole_refusal(&s(&["inv", "inv", "x0"]), &s(&["x0"]), &e.operators, &vars),
            None,
            "inv(inv(w)) -> w agrees at the pole (0 -> +inf -> 0)"
        );
        assert_eq!(
            pole_refusal(&s(&["/", "x0", "x0"]), &s(&["1"]), &e.operators, &vars),
            None,
            "w/w -> 1 is Ext at w=0 (nan hole filled): R3-licensed, never refused"
        );
        // No solvable exceptional point at all -> immediate None (the pre-screen arm).
        assert_eq!(
            pole_refusal(&s(&["+", "x0", "1"]), &s(&["x0"]), &e.operators, &vars),
            None
        );
    }

    /// The relocated syntactic licences behave exactly as they did inline (the
    /// re-mine byte-identity is the integration falsifier; these pin the unit level).
    #[test]
    fn relocated_licences_hold() {
        let vars = s(&["x0", "x1"]);
        // Determined source (var-free, Const-free) refuses a Const-introducing target.
        assert_eq!(
            mint_refusals(
                &s(&["+", "np.pi", "1"]),
                &s(&["+", "np.pi", "1"]),
                &s(&["<constant>"]),
                &vars
            ),
            Some(Refusal::ConstIntroduction)
        );
        // Special-bearing source: a Const-bearing target may not lower the count --
        // counted on the MAX of raw source and endpoint (spelling independence, F49).
        assert_eq!(
            mint_refusals(
                &s(&["pow", "np.pi", "<constant>"]),
                &s(&["pow", "np.pi", "<constant>"]),
                &s(&["<constant>"]),
                &vars
            ),
            Some(Refusal::SpecialAbsorption)
        );
        // The endpoint side of the max: raw source spells no special, the endpoint does.
        assert_eq!(
            mint_refusals(
                &s(&["*", "<constant>", "exp", "1"]),
                &s(&["*", "np.e", "<constant>"]),
                &s(&["<constant>"]),
                &vars
            ),
            Some(Refusal::SpecialAbsorption)
        );
        // Benign: variable-bearing source, special-preserving target.
        assert_eq!(
            mint_refusals(
                &s(&["*", "np.pi", "x0"]),
                &s(&["*", "np.pi", "x0"]),
                &s(&["*", "np.pi", "x0"]),
                &vars
            ),
            None
        );
    }

    /// The affine solver: exact rational roots, including the overflow-checked path.
    #[test]
    fn affine_solver_exact_roots() {
        let e = test_engine();
        let vars = s(&["x0"]);
        let mut pts = Vec::new();
        // 1/(2 - 4w): root at w = 1/2
        exceptional_points(
            &s(&["inv", "-", "2", "*", "4", "x0"]),
            &vars,
            &e.operators,
            &mut pts,
        );
        assert_eq!(pts, vec![("x0".to_string(), (1, 2))]);
        pts.clear();
        // log argument: log(w + 3) at w = -3
        exceptional_points(&s(&["log", "+", "x0", "3"]), &vars, &e.operators, &mut pts);
        assert_eq!(pts, vec![("x0".to_string(), (-3, 1))]);
        pts.clear();
        // pow negative-integer exponent: (5 - w)^-2 at w = 5
        exceptional_points(
            &s(&["pow", "-", "5", "x0", "(-2)"]),
            &vars,
            &e.operators,
            &mut pts,
        );
        assert_eq!(pts, vec![("x0".to_string(), (5, 1))]);
        pts.clear();
        // two variables in the denominator -> not one-variable-affine, no point
        exceptional_points(
            &s(&["inv", "-", "x1", "x0"]),
            &s(&["x0", "x1"]),
            &e.operators,
            &mut pts,
        );
        assert!(pts.is_empty());
    }
}
