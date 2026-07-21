//! Constant-fitting for the OFFLINE rule miner -- the native replacement for
//! `exist_constants_that_fit` (engine.py@0.2.15:2272), which called `scipy.optimize.curve_fit` (whose
//! cost is dominated by scipy/MINPACK/Python overhead, not math). The key lever: many candidates
//! are AFFINE in their `<constant>`s, for which the best fit is a single CLOSED-FORM
//! least-squares solve -- no iteration, no random `p0`, no retries, and a DETERMINISTIC
//! accept/reject decision.
//!
//! Pieces: the linearity classifier + the affine closed-form path (`exist_constants_fit_linear`),
//! and a native Levenberg-Marquardt (`lm_fit`) with random restarts for nonlinear-in-params
//! candidates. The complete native `exist_constants_that_fit` is `exist_constants_fit` (affine ->
//! closed-form, nonlinear -> LM). The accept/reject gate stays `allclose` (crate::eval), so soundness
//! is identical to scipy's: the optimizer only PROPOSES constants, `allclose` DISPOSES.

use crate::eval::{allclose_extends, columns_from_row_major, Tape};
use crate::operators::Operators;

/// Degree of an expression in its `<constant>` placeholders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linearity {
    /// no `<constant>` appears (a pure function of the variables).
    ConstFree,
    /// the output is affine in the constants: `y = b(X) + sum_j C_j a_j(X)` (a_j, b free of C).
    Affine,
    /// at least one constant enters nonlinearly (inside a transcendental/power, a denominator, a
    /// product of two constant-bearing factors, ...).
    Nonlinear,
}

/// Operators that are LINEAR maps in their (single) operand: `op(a*x+b) = a*op(x)+op(b)`-compatible
/// for the purpose of constant-degree, i.e. they preserve the affine/constfree/nonlinear class of the
/// operand. `neg` and the fixed scalar multiples/divisors. (NOT `abs` -- abs is nonlinear.)
fn is_linear_unary(op: &str) -> bool {
    matches!(
        op,
        "neg" | "mult2" | "mult3" | "mult4" | "mult5" | "div2" | "div3" | "div4" | "div5"
    )
}

fn combine_add(l: Linearity, r: Linearity) -> Linearity {
    use Linearity::*;
    if l == Nonlinear || r == Nonlinear {
        Nonlinear
    } else if l == Affine || r == Affine {
        Affine
    } else {
        ConstFree
    }
}

fn combine_mul(l: Linearity, r: Linearity) -> Linearity {
    use Linearity::*;
    if l == Nonlinear || r == Nonlinear {
        return Nonlinear;
    }
    // count constant-bearing factors: 0 -> constfree, 1 -> affine, 2 -> C*C nonlinear.
    let c = (l == Affine) as u8 + (r == Affine) as u8;
    match c {
        0 => ConstFree,
        1 => Affine,
        _ => Nonlinear,
    }
}

fn combine_div(num: Linearity, den: Linearity) -> Linearity {
    use Linearity::*;
    // a constant in the DENOMINATOR is nonlinear (1/C); a constant-free denominator is a linear
    // scaling of the numerator, so the result inherits the numerator's class.
    if den != ConstFree {
        Nonlinear
    } else {
        num
    }
}

/// Classify a prefix expression's degree in its `<constant>` placeholders.
pub fn classify(tokens: &[String], ops: &Operators) -> Result<Linearity, String> {
    let mut idx = 0usize;
    let lin = classify_node(tokens, &mut idx, ops)?;
    if idx != tokens.len() {
        return Err(format!("trailing tokens after position {idx}"));
    }
    Ok(lin)
}

fn classify_node(tokens: &[String], idx: &mut usize, ops: &Operators) -> Result<Linearity, String> {
    let tok = tokens
        .get(*idx)
        .ok_or_else(|| format!("truncated prefix expression at position {idx}"))?
        .clone();
    *idx += 1;
    if let Some(arity) = ops.arity_of(&tok) {
        let arity = arity as usize;
        let mut children = Vec::with_capacity(arity);
        for _ in 0..arity {
            children.push(classify_node(tokens, idx, ops)?);
        }
        Ok(match (tok.as_str(), arity) {
            ("+", 2) | ("-", 2) => combine_add(children[0], children[1]),
            ("*", 2) => combine_mul(children[0], children[1]),
            ("/", 2) => combine_div(children[0], children[1]),
            (op, 1) if is_linear_unary(op) => children[0],
            // any other operator (sin/exp/log/pow*/abs/inv/binary pow): constant-free if all operands
            // are constant-free, else nonlinear.
            _ => {
                if children.iter().all(|c| *c == Linearity::ConstFree) {
                    Linearity::ConstFree
                } else {
                    Linearity::Nonlinear
                }
            }
        })
    } else if tok == "<constant>" {
        Ok(Linearity::Affine)
    } else {
        // variable or numeric/special literal
        Ok(Linearity::ConstFree)
    }
}

/// Solve a small dense linear system `A x = b` (k x k, k tiny) by Gaussian elimination with partial
/// pivoting. `None` if singular (should not happen after ridge regularization).
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in (col + 1)..n {
            let f = a[r][col] / d;
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in (col + 1)..n {
            s -= a[col][c] * x[c];
        }
        x[col] = s / a[col][col];
    }
    Some(x)
}

/// Verdict of a constant-fit check. `NearMiss` carries the best fitted constants when the
/// residual check failed: the caller MAY re-judge exactly the failing rows at P bits
/// (`crate::hiprec::rescue`) at those constants -- an accept there is still a legitimate
/// "constants EXIST" witness (the fitted constants are the witness), so the rescue can only
/// recover true rules that f64 evaluation of the SOURCE corrupted (e.g. `atanh(tanh(C*x))`
/// on saturation rows), never admit a new accept class. Paths that produce no usable
/// constants (underdetermined, rank-deficient, the bare-`<constant>` exact-interval
/// decision) return plain `Fail`.
///
/// `Pass` carries the WITNESS the fit chose: `Some(c)` = these constants reproduce the source
/// (`c` empty = the candidate is const-free, which is still a determined witness); `None` = the
/// pass was VACUOUS -- every row was domain-extendable, so NO constants were determined and any
/// value would have passed. The domain gate needs the witness to bind the candidate's
/// `<constant>` leaves and must SKIP vacuous instances, which bind nothing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum FitVerdict {
    Pass(Option<Vec<f64>>),
    Fail,
    NearMiss(Vec<f64>),
}

impl FitVerdict {
    pub(crate) fn passed(&self) -> bool {
        matches!(self, FitVerdict::Pass(_))
    }
}

/// AFFINE closed-form fit + accept/reject. The candidate is affine in its k constants,
/// so `y(C) = b(X) + sum_j C_j a_j(X)` with `b = eval(C=0)` and `a_j = eval(C=e_j) - b`. We solve the
/// least-squares system by HOUSEHOLDER QR on the FINITE-row mask (mirroring scipy's `is_valid` mask and
/// its `n_const > n_valid` bail; QR avoids the normal-equations conditioning squaring, see
/// `householder_lstsq`), evaluate the fitted candidate on ALL rows, and gate on
/// `allclose_extends(y_target, fitted)` -- source-finite rows bind, source-nonfinite rows are
/// domain-extendable (generic equivalence). A gate failure returns
/// `NearMiss(c)` with the solved constants (see `FitVerdict`).
fn fit_affine_check(
    tape: &Tape,
    x_cols: &[Vec<f64>],
    y_target: &[f64],
    n_rows: usize,
    rtol: f64,
    atol: f64,
) -> FitVerdict {
    let k = tape.n_params;
    let zeros = vec![0.0f64; k];
    let b = tape.eval_columns(x_cols, &zeros, n_rows);
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(k);
    for j in 0..k {
        let mut e = vec![0.0f64; k];
        e[j] = 1.0;
        let yj = tape.eval_columns(x_cols, &e, n_rows);
        a.push(yj.iter().zip(&b).map(|(v, bb)| v - bb).collect());
    }
    // valid rows: every variable finite AND y_target finite (scipy issue 13969 mask).
    let valid: Vec<usize> = (0..n_rows)
        .filter(|&r| y_target[r].is_finite() && x_cols.iter().all(|c| c[r].is_finite()))
        .collect();
    if valid.is_empty() {
        // No fittable binding row. NaN source rows are domain-extendable (genuinely vacuous),
        // but an INF source row is a pole the candidate must reproduce (escalation): with no
        // finite row to fit a constant, a finite-valued candidate cannot certify it, so an
        // instance carrying inf source rows is rejected -- NOT vacuously passed.
        // (`pow(acos(C), inf) -> <constant>` reached this path via all-inf vacuity and is
        // FALSE on the positive-measure pole region C < cos(1).)
        return if y_target.iter().any(|v| v.is_infinite()) {
            FitVerdict::Fail
        } else {
            FitVerdict::Pass(None) // VACUOUS: every row extendable, no constant determined
        };
    }
    if k > valid.len() {
        return FitVerdict::Fail; // underdetermined -> scipy bails -> False
    }
    // BARE `<constant>` candidate (fitted == v on every row; detected structurally as k == 1
    // with a zero base and an all-ones basis): decide by EXACT interval-intersection
    // feasibility instead of least squares. The accept gate is per-row RELATIVE, so its
    // feasible set is an intersection of intervals in v -- and a (weighted or unweighted)
    // least-squares mean is NOT a Chebyshev center: for skewed rows the mean can sit outside
    // a NONEMPTY feasible set (adversarially demonstrated: 63 rows at e-2.4e-9, one at
    // e+2.4e-9, all within the band of v = e, mean rejected), which broke the fold-filter
    // dominance argument (`<constant>` must accept whenever ANY var-free candidate's value
    // does). The exact decision restores dominance as a theorem, up to f64 rounding of the
    // interval endpoints themselves (~1 ulp, 7 orders below the rtol band).
    if k == 1 && (0..n_rows).all(|r| b[r] == 0.0 && a[0][r] == 1.0) {
        // exact interval decision on the FINITE rows: no witness constant fits => the feasible
        // set is empty (Fail, not a near miss). A witness that fits every finite row is a Pass
        // ONLY when the source has no inf rows -- an inf source row is a pole or a saturation and
        // a bare finite constant cannot be assumed to complete it (`pow(0, cos x) = +inf` where
        // cos < 0 is a pole, NOT extendable to a constant). When inf rows are present, hand the
        // witness to the near-miss rescue, which escalates each inf row (pole -> the constant
        // must reproduce the inf, so reject; saturation -> resolves finite, compare to c0).
        return match fits_constant_witness(y_target, rtol, atol) {
            None => FitVerdict::Fail,
            Some(c0) => {
                if y_target.iter().any(|v| v.is_infinite()) {
                    FitVerdict::NearMiss(vec![c0])
                } else {
                    FitVerdict::Pass(Some(vec![c0]))
                }
            }
        };
    }
    match affine_solve(&b, &a, y_target, &valid, rtol, atol) {
        Some(c) => {
            // fitted on ALL rows = eval at the solved constants (exact for an affine model).
            // GENERIC EQUIVALENCE: source-nonfinite rows are extendable, source-finite bind.
            let fitted = tape.eval_columns(x_cols, &c, n_rows);
            if allclose_extends(y_target, &fitted, rtol, atol) {
                return FitVerdict::Pass(Some(c));
            }
            // KNIFE-EDGE rescue: exact-cancellation rows (cosh(x) - sinh(x)
            // = 0.0 bitwise at large x) tolerate ~atol while d(fitted)/dc ~ 1e303, so they demand
            // constants BITWISE equal to the true ones -- any solver's +-1-ulp output is a coin flip
            // (the old solver "won" some seeds by luck, lost others). When the solved constants sit
            // within ~rtol of integers, re-gate at the snapped integers: deterministic accept for the
            // integer-constant rule class, and STILL gate-certified (no new accept class -- an accepted
            // snap is just another "constants exist" witness). The nearness guard keeps the extra eval
            // off the ordinary reject path.
            let near_int = c
                .iter()
                .all(|v| (v - v.round()).abs() <= 1e-9 * v.abs().max(1.0));
            if near_int && c.iter().any(|v| *v != v.round()) {
                let snapped: Vec<f64> = c.iter().map(|v| v.round()).collect();
                let fitted2 = tape.eval_columns(x_cols, &snapped, n_rows);
                if allclose_extends(y_target, &fitted2, rtol, atol) {
                    return FitVerdict::Pass(Some(snapped));
                }
            }
            FitVerdict::NearMiss(c)
        }
        None => FitVerdict::Fail, // rank-deficient with an inconsistent RHS -> no exact fit
    }
}

/// Weighted Householder least-squares solve of the affine system over the given `rows`, with the
/// three stabilizers documented below. `b`/`a` are the affine basis evaluated on ALL rows
/// (`b = eval(C=0)`, `a_j = eval(C=e_j) - b`); `rows` selects which rows constrain the solve --
/// `fit_affine_check` passes every valid row, the trimmed refit (`refit_trimmed`) passes the
/// f64-passing subset. Returns the solved constants (unscaled), or `None` when the factorization
/// or solve fails (rank deficiency).
fn affine_solve(
    b: &[f64],
    a: &[Vec<f64>],
    y_target: &[f64],
    rows: &[usize],
    rtol: f64,
    atol: f64,
) -> Option<Vec<f64>> {
    let k = a.len();
    let valid = rows;
    // Solve the least-squares system A c ~= t over the valid rows by HOUSEHOLDER QR
    // (the former normal-equations path SQUARED the condition number, see `QrFactor`), with
    // three stabilizers for the GROWING-BASIS recall gap:
    //
    // 1. ROW WEIGHTS mirroring the accept gate. `allclose_extends` is RELATIVE per row
    //    (|y - fitted| <= atol + rtol*|fitted|), but an unweighted solve minimizes ABSOLUTE
    //    residuals, so rows with huge |y| dominate. With a growing basis (exp/cosh/pow3+ on the
    //    heavy-tailed mine X), y's own f64 rounding on those rows (eps*|y| ~ 1e5 at |y| ~ 1e21)
    //    exceeds an O(1) intercept, whose information lives ONLY in the small-|y| rows: the
    //    unweighted solve buried it and the whole C0*f(x)+C1 / f(x)+C1 family rejected on
    //    exactly-true constants (0/4) while interceptless C0*f(x) passed. Weighting each row by
    //    ~1/tolerance_r makes the solve optimize the criterion the gate actually checks. The gate
    //    itself is UNCHANGED: a solved c still has to pass `allclose_extends` on all rows, so
    //    this is recall-only -- no new accept class exists that the gate would not certify.
    // 2. COLUMN max-abs equilibration: keeps the QR's working magnitudes O(1) (a raw exp-basis
    //    column reaches ~1e308, where the reflector norm itself can overflow) and removes the
    //    column-imbalance part of the conditioning; solutions are rescaled back.
    // 3. FIXED-ROUND iterative refinement on the retained factor: for a CONSISTENT system (a
    //    true rule) it polishes the solution toward machine precision even at large kappa; for
    //    an inconsistent one the residual is ~orthogonal to range(A) and the update is ~0.
    //    Fixed rounds keep the path deterministic.
    let m = valid.len();
    // (a) column max-abs equilibration on the RAW columns, BEFORE weighting: a huge-but-FINITE
    // basis entry (sinh(705) ~ 7.5e305) times a large row weight (up to 1/atol = 1e12 where
    // y ~ 0, e.g. the exact-cancellation rows of cosh(x) - sinh(x)) overflows to inf and NaNs
    // the whole solve -- a false REJECT of true fits, adversarially demonstrated on the real
    // mining X. Scaled first, basis entries are <= 1 and every later product stays finite.
    let mut acols: Vec<Vec<f64>> = (0..k)
        .map(|j| valid.iter().map(|&r| a[j][r]).collect::<Vec<f64>>())
        .collect(); // column-major over valid rows
    let traw: Vec<f64> = valid.iter().map(|&r| y_target[r] - b[r]).collect();
    let mut scale = vec![1.0f64; k];
    for (j, col) in acols.iter_mut().enumerate() {
        let s = col.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        if s.is_finite() && s > 0.0 {
            scale[j] = s;
            for v in col.iter_mut() {
                *v /= s;
            }
        }
    }
    // (b) row weights, capped so the weighted RHS also stays finite (a row needing the cap has
    // its residual target astronomically outside the band -- the fit is infeasible there and
    // the gate rejects it; the cap only keeps the algebra finite instead of NaN).
    let w: Vec<f64> = valid
        .iter()
        .zip(&traw)
        .map(|(&r, tr)| {
            let wr = 1.0 / (atol + rtol * y_target[r].abs()).max(1e-300);
            if tr.abs() > 1.0 {
                wr.min(1e290 / tr.abs())
            } else {
                wr
            }
        })
        .collect();
    let aw: Vec<Vec<f64>> = acols
        .iter()
        .map(|col| {
            col.iter()
                .zip(&w)
                .map(|(v, wi)| v * wi)
                .collect::<Vec<f64>>()
        })
        .collect(); // column-major, scaled then row-weighted; entries bounded by max(w) <= 1e300
    let tw: Vec<f64> = traw.iter().zip(&w).map(|(tr, wi)| tr * wi).collect();
    // NO second (post-weighting) equilibration: the combined per-column scale s1*s2 (basis
    // max-abs ~ 1e303 times weight scale ~ 1e12) overflows f64, turning `scale[j]` into inf and
    // the unscaled solution into 0 -- a false reject on exactly the huge-basis rows the
    // pre-weighting scale exists for (cosh/sinh at |x| ~ 705). The single raw-column scale keeps
    // every intermediate (matrix entries, scaled solution c*s1, and the unscale) representable.
    let factor = QrFactor::factor(aw.clone(), m, k)?;
    let mut cs = factor.solve(tw.clone())?;
    for _ in 0..2 {
        let mut resid = tw.clone();
        for (col, c_j) in aw.iter().zip(&cs) {
            for (res, v) in resid.iter_mut().zip(col) {
                *res -= v * c_j;
            }
        }
        match factor.solve(resid) {
            Some(d) => {
                for (c_j, d_j) in cs.iter_mut().zip(&d) {
                    *c_j += d_j;
                }
            }
            None => break,
        }
    }
    Some(cs.iter().zip(&scale).map(|(v, s)| v / s).collect())
}

/// ROBUST REFIT for the hiprec escalation. A near-miss fit's constants are
/// CONTAMINATED by the very rows that need rescuing: least squares over all valid rows pulls
/// the solution toward the f64-corrupted evaluations (for `atanh(tanh(C*x)) -> C*x`, ~5
/// dead-zone rows of 60 shift C by ~4% -- 7 orders of magnitude beyond the rtol band), and at
/// the contaminated constants EVERY row fails the strict gate, so no pass/fail split at `c0`
/// can identify the outliers. LTS-style iterative median trimming can: rows whose absolute
/// residual exceeds `TRIM_KAPPA` times the median residual are outliers RELATIVE TO THE
/// CONSENSUS regardless of where the contaminated fit sits (for the target class the
/// corrupted rows carry ~2e-1 errors vs the ~majority's uniform ~|dC|*|f(x)| drift), so
/// trim -> refit -> repeat converges to the majority fit in <= `TRIM_ROUNDS` rounds. For a
/// structurally wrong candidate the residuals have no small-majority consensus: nothing (or
/// noise) is trimmed, the refit barely moves, and the caller's P-bit gate rejects as before.
/// Refusing to certify when the trimmed fraction exceeds `TRIM_MAX_FRAC` of the binding rows
/// mirrors `hiprec`'s near-miss cost gate. Soundness: the refit constants are still just a
/// "constants exist" witness; every row is re-checked by the caller (f64 on kept rows, P bits
/// on escalated rows).
const TRIM_ROUNDS: usize = 3;
const TRIM_KAPPA: f64 = 5.0;
const TRIM_MAX_FRAC: f64 = 0.5;

pub(crate) fn refit_robust(
    tape: &Tape,
    lin: Linearity,
    cols: &[Vec<f64>],
    n_rows: usize,
    y_target: &[f64],
    c0: &[f64],
    rtol: f64,
    atol: f64,
) -> Option<Vec<f64>> {
    let k = tape.n_params;
    let valid: Vec<usize> = (0..n_rows)
        .filter(|&r| y_target[r].is_finite() && cols.iter().all(|c| c[r].is_finite()))
        .collect();
    if valid.is_empty() {
        return None;
    }
    // Affine basis reused across rounds (the model is fixed; only the row subset moves).
    let affine_basis = if lin == Linearity::Affine {
        let zeros = vec![0.0f64; k];
        let b = tape.eval_columns(cols, &zeros, n_rows);
        let mut a: Vec<Vec<f64>> = Vec::with_capacity(k);
        for j in 0..k {
            let mut e = vec![0.0f64; k];
            e[j] = 1.0;
            let yj = tape.eval_columns(cols, &e, n_rows);
            a.push(yj.iter().zip(&b).map(|(v, bb)| v - bb).collect());
        }
        Some((b, a))
    } else {
        None
    };
    let mut c: Vec<f64> = c0.to_vec();
    for _ in 0..TRIM_ROUNDS {
        let fitted = tape.eval_columns(cols, &c, n_rows);
        let mut resid: Vec<f64> = valid
            .iter()
            .map(|&r| {
                if fitted[r].is_finite() {
                    (y_target[r] - fitted[r]).abs()
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        let mut sorted = resid.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let med = sorted[sorted.len() / 2];
        // trim threshold: consensus-relative, but never below the accept band itself
        let kept: Vec<usize> = valid
            .iter()
            .zip(resid.drain(..))
            .filter(|(&r, res)| *res <= (TRIM_KAPPA * med).max(atol + rtol * y_target[r].abs()))
            .map(|(&r, _)| r)
            .collect();
        if kept.len() == valid.len() {
            return Some(c); // consensus already covers every row: c is the majority fit
        }
        if (kept.len() as f64) < (1.0 - TRIM_MAX_FRAC) * (valid.len() as f64) || k > kept.len() {
            return None; // no small-outlier consensus: nothing to certify robustly
        }
        let c_new = match &affine_basis {
            Some((b, a)) => affine_solve(b, a, y_target, &kept, rtol, atol)?,
            None => {
                // Nonlinear: ONE LM run on the kept rows, seeded at the current constants --
                // the trimmed problem shares the basin, only the corrupting rows are gone.
                let m = kept.len();
                let xv: Vec<Vec<f64>> = cols
                    .iter()
                    .map(|col| kept.iter().map(|&r| col[r]).collect())
                    .collect();
                let yv: Vec<f64> = kept.iter().map(|&r| y_target[r]).collect();
                lm_fit(tape, &xv, &yv, m, k, &c, rtol, atol)
            }
        };
        let moved = c
            .iter()
            .zip(&c_new)
            .any(|(a_, b_)| (a_ - b_).abs() > 1e-12 * a_.abs().max(1.0));
        c = c_new;
        if !moved {
            break;
        }
    }
    Some(c)
}

/// EXACT feasibility decision for fitting a bare constant: is there a `v` satisfying the accept
/// gate |y_r - v| <= atol + rtol*|v| on every FINITE row of `y` (nonfinite rows handled by the
/// caller: nan is extendable, inf is escalated)? Returns a WITNESS `v` in the feasible set (its
/// nonemptiness is what the fold-filter dominance theorem needs: any var-free candidate's fitted
/// value is one such v, so its acceptance implies a nonempty set here). The feasible set is an
/// intersection of per-row intervals, tracked separately for the v >= 0 and v <= 0 branches (the
/// gate's |v| makes the bounds branch-dependent); the witness is the low endpoint of whichever
/// branch is nonempty.
fn fits_constant_witness(y: &[f64], rtol: f64, atol: f64) -> Option<f64> {
    let mut any = false;
    let (mut lo_p, mut hi_p) = (0.0f64, f64::INFINITY); // v >= 0 branch
    let (mut lo_n, mut hi_n) = (f64::NEG_INFINITY, 0.0f64); // v <= 0 branch
    for &yr in y {
        if !yr.is_finite() {
            continue;
        }
        any = true;
        // v >= 0: |y - v| <= atol + rtol*v  <=>  (y-atol)/(1+rtol) <= v <= (y+atol)/(1-rtol)
        lo_p = lo_p.max((yr - atol) / (1.0 + rtol));
        hi_p = hi_p.min((yr + atol) / (1.0 - rtol));
        // v <= 0: |y - v| <= atol - rtol*v  <=>  (y-atol)/(1-rtol) <= v <= (y+atol)/(1+rtol)
        lo_n = lo_n.max((yr - atol) / (1.0 - rtol));
        hi_n = hi_n.min((yr + atol) / (1.0 + rtol));
    }
    if !any {
        return None; // no binding row -> no evidence for a constant (mirrors the k > valid bail)
    }
    if lo_p <= hi_p {
        Some(lo_p)
    } else if lo_n <= hi_n {
        Some(hi_n)
    } else {
        None
    }
}

/// Householder QR factorization of an m x k column-major design (`qr[j][row]`), reusable across
/// multiple right-hand sides -- the iterative-refinement loop in `fit_affine_check` re-solves the
/// SAME factor for each residual RHS. Never forms A^T A, so the working conditioning is cond(A),
/// not its square -- the whole reason for QR over the normal equations here. After `factor`, the
/// reflectors live below/on the diagonal, R above it, and R's diagonal in `r_diag` (the reflector
/// maps column j to -sigma*e_j, so R[j][j] = -norm; the sign convention is pinned by
/// `householder_lstsq_recovers_exact_solution`). A column with an all-zero subdiagonal gets the
/// IDENTITY reflector (LAPACK tau = 0): the general formula would place v_j = 0 and divide by it,
/// NaN-ing the solve -- the m == k last column ALWAYS lands there, so exact-interpolation fits
/// (k constants on exactly k valid rows) used to reject spuriously.
struct QrFactor {
    qr: Vec<Vec<f64>>,
    r_diag: Vec<f64>,
    identity: Vec<bool>,
    m: usize,
    k: usize,
}

impl QrFactor {
    fn factor(mut qr: Vec<Vec<f64>>, m: usize, k: usize) -> Option<Self> {
        let mut r_diag = vec![0.0f64; k];
        let mut identity = vec![false; k];
        for j in 0..k {
            // Householder reflector zeroing qr[j][j+1..m].
            let mut norm_below = 0.0f64;
            for row in (j + 1)..m {
                norm_below = norm_below.hypot(qr[j][row]);
            }
            if norm_below == 0.0 {
                if qr[j][j] == 0.0 {
                    return None; // zero column -> rank-deficient
                }
                // already upper-triangular at j: H_j = I, nothing to apply
                r_diag[j] = qr[j][j];
                identity[j] = true;
                continue;
            }
            let mut norm = norm_below.hypot(qr[j][j]);
            // pick the sign that avoids cancellation
            if qr[j][j] > 0.0 {
                norm = -norm;
            }
            for row in j..m {
                qr[j][row] /= norm;
            }
            qr[j][j] += 1.0;
            // apply the reflector to the remaining columns
            for c in (j + 1)..k {
                let mut s = 0.0f64;
                for row in j..m {
                    s += qr[j][row] * qr[c][row];
                }
                let s = -s / qr[j][j];
                for row in j..m {
                    qr[c][row] += s * qr[j][row];
                }
            }
            r_diag[j] = -norm;
        }
        Some(QrFactor {
            qr,
            r_diag,
            identity,
            m,
            k,
        })
    }

    /// Solve min ||A x - t|| for one RHS using the stored reflectors; consumes `t`.
    fn solve(&self, mut t: Vec<f64>) -> Option<Vec<f64>> {
        for j in 0..self.k {
            if self.identity[j] {
                continue;
            }
            let mut s = 0.0f64;
            for row in j..self.m {
                s += self.qr[j][row] * t[row];
            }
            let s = -s / self.qr[j][j];
            for row in j..self.m {
                t[row] += s * self.qr[j][row];
            }
        }
        // back-substitute R x = (Q^T t)[..k]: diag in r_diag, above-diag in qr.
        let mut x = vec![0.0f64; self.k];
        for j in (0..self.k).rev() {
            if self.r_diag[j] == 0.0 {
                return None;
            }
            let mut s = t[j];
            for c in (j + 1)..self.k {
                s -= self.qr[c][j] * x[c];
            }
            x[j] = s / self.r_diag[j];
        }
        Some(x)
    }
}

/// One-shot factor + solve (kept for the unit test pinning the R[j][j] = -norm convention).
#[cfg(test)]
fn householder_lstsq(qr: Vec<Vec<f64>>, t: Vec<f64>, m: usize, k: usize) -> Option<Vec<f64>> {
    QrFactor::factor(qr, m, k)?.solve(t)
}

/// Native `exist_constants_that_fit` for the AFFINE case. Returns `Some(decision)` if the
/// candidate is affine in its constants (handled here), or `None` if it is nonlinear-in-params
/// (the native-LM path). `Err` only on a malformed candidate / shape mismatch.
pub fn exist_constants_fit_linear(
    ops: &Operators,
    candidate: &[String],
    var_names: &[String],
    x_flat: &[f64],
    n_rows: usize,
    y_target: &[f64],
    rtol: f64,
    atol: f64,
) -> Result<Option<bool>, String> {
    if classify(candidate, ops)? != Linearity::Affine {
        return Ok(None);
    }
    let tape = Tape::compile(candidate, ops, var_names)?;
    let n_vars = var_names.len();
    if x_flat.len() != n_rows * n_vars {
        return Err("x_flat shape mismatch".to_string());
    }
    if y_target.len() != n_rows {
        return Err("y_target length mismatch".to_string());
    }
    let cols = columns_from_row_major(x_flat, n_rows, n_vars);
    // bool FFI surface: a NearMiss stays a reject here -- this entry point receives raw
    // `y_target` DATA (no source expression), so there is nothing to re-evaluate at P bits.
    Ok(Some(
        fit_affine_check(&tape, &cols, y_target, n_rows, rtol, atol).passed(),
    ))
}

/// A tiny deterministic PRNG (splitmix64) + Box-Muller normal, for the LM's random restarts. We do
/// NOT reproduce numpy's Mersenne-Twister stream (RNG divergence is expected and fine -- the miner is
/// stochastic and re-mining yields a new engine-id); we only need reproducible N(0,5) restart points.
pub(crate) struct Rng(u64);
impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E3779B97F4A7C15)
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa in [0, 1)
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
    pub(crate) fn normal(&mut self, mean: f64, sd: f64) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        mean + sd * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    /// Magnitude for a mined source constant. HALF the draws are moderate-scale |N(0, 5)|
    /// (typical constants); HALF are log-uniform over [1e-3, 1e3] so pole thresholds -- where a
    /// subexpression crosses +-1, e.g. |3*C| = 1 in `pow(mult3(C), -inf)` -- are reliably sampled
    /// at every scale. A plain half-normal draws |C| < 1/3 only ~5% of the time and misses such
    /// POSITIVE-measure poles, admitting false rules (`pow(mult3(C), -inf) -> 0`, which is +inf on
    /// |C| < 1/3). The two tiers together put ~21% of draws below 1/3, so 16 challenges sample the
    /// region with >97% reliability.
    pub(crate) fn const_magnitude(&mut self) -> f64 {
        if self.next_f64() < 0.5 {
            self.normal(0.0, 5.0).abs()
        } else {
            const L: f64 = 3.0 * std::f64::consts::LN_10; // ln(1e3)
            (self.next_f64() * (2.0 * L) - L).exp() // log-uniform over [1e-3, 1e3]
        }
    }
}

/// One Levenberg-Marquardt least-squares solve (Marquardt diagonal scaling, forward-difference
/// Jacobian) from start point `p0`, minimizing `||fitted(C) - y_valid||^2` over the valid rows
/// `xv_cols` (m rows). The native analog of MINPACK `lmdif` (what scipy `curve_fit` calls). Returns
/// the best parameters found. EARLY-EXITS as soon as the valid-row residual passes `allclose` (the
/// goal is the accept/reject gate, not full convergence) -- this keeps positives cheap.
#[allow(clippy::too_many_arguments)]
fn lm_fit(
    tape: &Tape,
    xv_cols: &[Vec<f64>],
    y_valid: &[f64],
    m: usize,
    k: usize,
    p0: &[f64],
    rtol: f64,
    atol: f64,
) -> Vec<f64> {
    let resid = |c: &[f64]| -> (Vec<f64>, f64) {
        let f = tape.eval_columns(xv_cols, c, m);
        let r: Vec<f64> = f.iter().zip(y_valid).map(|(a, b)| a - b).collect();
        let cost = r.iter().map(|x| x * x).sum::<f64>();
        (r, cost)
    };
    let mut c = p0.to_vec();
    let (mut r, mut cost) = resid(&c);
    if !cost.is_finite() {
        return c;
    }
    let close = |r: &[f64]| -> bool {
        // valid-row allclose proxy: fitted = r + y_valid, allclose(y_valid, fitted).
        r.iter()
            .zip(y_valid)
            .all(|(ri, &y)| ri.abs() <= atol + rtol * (ri + y).abs())
    };
    if close(&r) {
        return c;
    }
    let mut lambda = 1e-3f64;
    let eps = f64::EPSILON.sqrt();
    for _iter in 0..50 {
        // Jacobian transpose (k x m) by forward differences.
        let fitted: Vec<f64> = r.iter().zip(y_valid).map(|(ri, y)| ri + y).collect();
        let mut jt = vec![vec![0.0f64; m]; k];
        for j in 0..k {
            let h = eps * c[j].abs().max(1.0);
            let mut cj = c.clone();
            cj[j] += h;
            let fj = tape.eval_columns(xv_cols, &cj, m);
            for i in 0..m {
                jt[j][i] = (fj[i] - fitted[i]) / h;
            }
        }
        // JtJ (k x k) and Jtr (k).
        let mut jtj = vec![vec![0.0f64; k]; k];
        let mut jtr = vec![0.0f64; k];
        for a in 0..k {
            for i in 0..m {
                jtr[a] += jt[a][i] * r[i];
            }
            for b in 0..k {
                let mut s = 0.0;
                for i in 0..m {
                    s += jt[a][i] * jt[b][i];
                }
                jtj[a][b] = s;
            }
        }
        // Inner loop: grow lambda until a step reduces the cost.
        let mut stepped = false;
        for _ in 0..30 {
            let mut aug = jtj.clone();
            for d in 0..k {
                aug[d][d] += lambda * jtj[d][d].max(1e-12); // Marquardt diagonal scaling
            }
            let neg = jtr.iter().map(|v| -v).collect::<Vec<_>>();
            let Some(delta) = solve(aug, neg) else {
                lambda *= 3.0;
                continue;
            };
            let c_new: Vec<f64> = c.iter().zip(&delta).map(|(a, b)| a + b).collect();
            let (r_new, cost_new) = resid(&c_new);
            if cost_new.is_finite() && cost_new < cost {
                let dn: f64 = delta.iter().map(|x| x * x).sum::<f64>().sqrt();
                let cn: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
                let red = cost - cost_new;
                c = c_new;
                r = r_new;
                lambda = (lambda * 0.3).max(1e-12);
                stepped = true;
                if close(&r) {
                    return c; // accept/reject gate satisfied -> done
                }
                if red < 1e-12 * cost.max(1e-300) || dn < 1e-10 * (cn + 1e-10) {
                    return c; // converged
                }
                cost = cost_new;
                break;
            } else {
                lambda *= 3.0;
                if lambda > 1e14 {
                    return c; // stuck
                }
            }
        }
        if !stepped {
            return c;
        }
    }
    c
}

/// The two log-linearizable single-constant power forms (Improvement 2): `pow(<constant>, g)` = C^g
/// (constant base) and `pow(g, <constant>)` = g^C (constant exponent), with `g` const-free. ~65% of
/// nonlinear-in-params candidates -- a closed-form least-squares solve in log-space instead of the LM.
#[derive(Clone, Copy)]
pub(crate) enum LogLinForm {
    CPowG,
    GPowC,
}

/// End index (exclusive) of the prefix subtree starting at `start` (arity walk).
fn subtree_end(tokens: &[String], start: usize, ops: &Operators) -> Option<usize> {
    let mut i = start;
    let mut need = 1usize;
    while need > 0 {
        let tok = tokens.get(i)?;
        i += 1;
        need -= 1;
        if let Some(a) = ops.arity_of(tok) {
            need += a as usize;
        }
    }
    Some(i)
}

/// Recognize `pow(<constant>, g)` / `pow(g, <constant>)` with a single `<constant>` and a const-free
/// `g`. Returns (form, g_tokens) or None (-> the LM path).
pub(crate) fn detect_log_linear(
    tokens: &[String],
    ops: &Operators,
) -> Option<(LogLinForm, Vec<String>)> {
    if tokens.first().map(|s| s.as_str()) != Some("pow") {
        return None;
    }
    if tokens.iter().filter(|t| t.as_str() == "<constant>").count() != 1 {
        return None;
    }
    let base_end = subtree_end(tokens, 1, ops)?;
    let base = &tokens[1..base_end];
    let exp = &tokens[base_end..];
    let const_free = |toks: &[String]| toks.iter().all(|t| t != "<constant>");
    if base.len() == 1 && base[0] == "<constant>" && const_free(exp) {
        return Some((LogLinForm::CPowG, exp.to_vec()));
    }
    if exp.len() == 1 && exp[0] == "<constant>" && const_free(base) {
        return Some((LogLinForm::GPowC, base.to_vec()));
    }
    None
}

/// Closed-form log-space least-squares for the recognized power forms. `cand_tape` evaluates the final
/// fit at the solved constant (so the accept/reject `allclose` uses the EXACT operator semantics, same
/// gate as the LM). Returns `Some((decision, c))` with the solved constant `c` when the solve is
/// computable (enough positive rows), else `None` -> fall back to the LM (e.g. negative-base
/// integer-power cases). The log-space fit is UNWEIGHTED, so on a heavy-tailed X its ~1e-10 base
/// error is amplified by large exponents past rtol; the caller therefore treats `Some((false, c))`
/// as an LM SEED, not a verdict, and only `Some((true, _))` short-circuits.
fn try_log_linear_fit(
    form: LogLinForm,
    g_tape: &Tape,
    cand_tape: &Tape,
    cols: &[Vec<f64>],
    n_rows: usize,
    y: &[f64],
    rtol: f64,
    atol: f64,
) -> Option<(bool, f64)> {
    let g = g_tape.eval_columns(cols, &[], n_rows);
    let (mut num, mut den, mut nvalid) = (0.0f64, 0.0f64, 0usize);
    let c = match form {
        LogLinForm::CPowG => {
            // y = C^g -> ln y = g * ln C ; solve u = ln C, C = exp(u).
            for r in 0..n_rows {
                let (gi, yi) = (g[r], y[r]);
                if gi.is_finite() && yi.is_finite() && yi > 0.0 {
                    num += gi * yi.ln();
                    den += gi * gi;
                    nvalid += 1;
                }
            }
            if nvalid < 1 || den <= 0.0 {
                return None;
            }
            (num / den).exp()
        }
        LogLinForm::GPowC => {
            // y = g^C -> ln y = C * ln g ; solve C.
            for r in 0..n_rows {
                let (gi, yi) = (g[r], y[r]);
                if gi.is_finite() && gi > 0.0 && yi.is_finite() && yi > 0.0 {
                    let lg = gi.ln();
                    num += lg * yi.ln();
                    den += lg * lg;
                    nvalid += 1;
                }
            }
            if nvalid < 1 || den <= 0.0 {
                return None;
            }
            num / den
        }
    };
    let fitted = cand_tape.eval_columns(cols, &[c], n_rows);
    Some((allclose_extends(y, &fitted, rtol, atol), c))
}

/// The COMPLETE native `exist_constants_that_fit`: affine candidates -> the closed-form path
/// (deterministic, no restarts); nonlinear-in-params candidates -> `n_restarts` LM solves from random
/// N(0,5) starts (mirroring the worker's retry loop), accept iff any makes `allclose_extends(
/// y_target, fitted)` pass (source-finite rows bind; generic equivalence). Only the
/// optimizer that proposes constants differs. `seed` makes the restarts reproducible.
#[allow(clippy::too_many_arguments)]
pub fn exist_constants_fit(
    ops: &Operators,
    candidate: &[String],
    var_names: &[String],
    x_flat: &[f64],
    n_rows: usize,
    y_target: &[f64],
    rtol: f64,
    atol: f64,
    n_restarts: usize,
    seed: u64,
) -> Result<bool, String> {
    let lin = classify(candidate, ops)?;
    let tape = Tape::compile(candidate, ops, var_names)?;
    let n_vars = var_names.len();
    if x_flat.len() != n_rows * n_vars {
        return Err("x_flat shape mismatch".to_string());
    }
    if y_target.len() != n_rows {
        return Err("y_target length mismatch".to_string());
    }
    let cols = columns_from_row_major(x_flat, n_rows, n_vars);
    // detect + compile the log-linear g-subtree once (nonlinear pow forms only).
    let g_tape = if lin == Linearity::Nonlinear {
        match detect_log_linear(candidate, ops) {
            Some((form, g)) => Some((form, Tape::compile(&g, ops, var_names)?)),
            None => None,
        }
    } else {
        None
    };
    let loglin = g_tape.as_ref().map(|(f, t)| (*f, t));
    // bool FFI surface: a NearMiss stays a reject here -- this entry point receives raw
    // `y_target` DATA (no source expression), so there is nothing to re-evaluate at P bits.
    Ok(exist_constants_fit_prepared(
        &tape, lin, &cols, n_rows, y_target, rtol, atol, n_restarts, seed, loglin,
    )
    .passed())
}

/// The post-compile core of `exist_constants_fit`, on a PRE-COMPILED tape + precomputed columns +
/// known `Linearity`. The OFFLINE worker (`crate::worker`) calls this so each candidate's tape and
/// classification are computed ONCE (at library build), not per challenge/source.
/// Returns a `FitVerdict`: a `NearMiss` carries the best fitted constants so the caller can
/// escalate the failing rows to P-bit re-judgment (see `crate::hiprec::rescue`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn exist_constants_fit_prepared(
    tape: &Tape,
    lin: Linearity,
    cols: &[Vec<f64>],
    n_rows: usize,
    y_target: &[f64],
    rtol: f64,
    atol: f64,
    n_restarts: usize,
    seed: u64,
    loglin: Option<(LogLinForm, &Tape)>,
) -> FitVerdict {
    if lin == Linearity::ConstFree {
        let fitted = tape.eval_columns(cols, &[], n_rows);
        return if allclose_extends(y_target, &fitted, rtol, atol) {
            FitVerdict::Pass(Some(Vec::new())) // const-free: determined, nothing to bind
        } else {
            FitVerdict::NearMiss(Vec::new())
        };
    }
    // GENERIC-EQUIVALENCE VACUITY: an instance whose target is nowhere finite and
    // has only NaN nonfinite rows binds NOTHING -- every row is domain-extendable, exactly as the
    // const-free arm's `allclose_extends` treats NaN. Falling through instead reached the
    // underdetermined bail (`k > valid.len()`) which returned FALSE and VETOED the candidate.
    // The original casualty (`/ (-1) <constant>`, all rows -inf) came from the C=0 constant
    // atom, which the signed sign-grid (no zero atom) removed at the root -- so a remaining
    // all-INF instance is genuine poles the candidate must reproduce (escalation); with no
    // finite row to fit a constant, a
    // finite-valued candidate cannot certify it and the instance is REJECTED, not passed.
    // A vacuous (all-NaN) instance contributes zero evidence; the caller's `min_informative`
    // gate still decides overall.
    if !y_target.iter().any(|v| v.is_finite()) {
        return if y_target.iter().any(|v| v.is_infinite()) {
            FitVerdict::Fail
        } else {
            FitVerdict::Pass(None) // VACUOUS: no constant determined
        };
    }
    if lin == Linearity::Affine {
        return fit_affine_check(tape, cols, y_target, n_rows, rtol, atol);
    }
    // Log-linearizable power forms -> closed-form (deterministic). Only a closed-form
    // ACCEPT short-circuits; a `Some((false, c))` is imprecise (unweighted log-space fit, ~1e-10 base
    // error amplified past rtol by large exponents on the heavy-tailed X), so `c` becomes the LM's
    // first seed rather than a verdict, and `None` (uncomputable) falls through unseeded. This keeps
    // the declared const-bearing policy ("accept iff constants EXIST that fit").
    let mut loglin_seed: Option<f64> = None;
    if let Some((form, g_tape)) = loglin {
        if let Some((dec, c)) =
            try_log_linear_fit(form, g_tape, tape, cols, n_rows, y_target, rtol, atol)
        {
            if dec {
                return FitVerdict::Pass(Some(vec![c]));
            }
            loglin_seed = Some(c);
        }
    }
    // Nonlinear-in-params: LM with random restarts (first seeded by the closed-form constant if any).
    let k = tape.n_params;
    let valid: Vec<usize> = (0..n_rows)
        .filter(|&r| y_target[r].is_finite() && cols.iter().all(|c| c[r].is_finite()))
        .collect();
    if k > valid.len() {
        return FitVerdict::Fail; // scipy issue 13969 bail
    }
    let m = valid.len();
    let xv: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| valid.iter().map(|&r| c[r]).collect())
        .collect();
    let yv: Vec<f64> = valid.iter().map(|&r| y_target[r]).collect();

    let mut rng = Rng::new(seed);
    // Track the best failing restart (fewest failing BINDING rows) so a final reject can hand
    // the hiprec escalation the most promising constants: for the target rescue class (few
    // f64-corrupted source rows contaminating an otherwise-true fit) that restart's params sit
    // closest to the true constants.
    let mut best: Option<(usize, Vec<f64>)> = None;
    for restart in 0..n_restarts.max(1) {
        // Seed restart 0 with the closed-form constant (k==1 for the recognized loglin forms) so the
        // LM refines it in the ORIGINAL space against the rtol gate; other restarts are random.
        let p0: Vec<f64> = match loglin_seed {
            Some(c0) if restart == 0 && k == 1 => vec![c0],
            _ => (0..k).map(|_| rng.normal(0.0, 5.0)).collect(),
        };
        let c = lm_fit(tape, &xv, &yv, m, k, &p0, rtol, atol);
        let fitted = tape.eval_columns(cols, &c, n_rows);
        if allclose_extends(y_target, &fitted, rtol, atol) {
            return FitVerdict::Pass(Some(c));
        }
        let n_fail = y_target
            .iter()
            .zip(&fitted)
            .filter(|(y, f)| {
                y.is_finite() && (!f.is_finite() || (*y - **f).abs() > atol + rtol * y.abs())
            })
            .count();
        if best.as_ref().is_none_or(|(bf, _)| n_fail < *bf) {
            best = Some((n_fail, c));
        }
    }
    match best {
        Some((_, c)) => FitVerdict::NearMiss(c),
        None => FitVerdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn classify_cases() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let c = |t: &[&str]| classify(&s(t), ops).unwrap();
        assert_eq!(c(&["_0"]), Linearity::ConstFree);
        assert_eq!(c(&["<constant>"]), Linearity::Affine);
        assert_eq!(
            c(&["+", "*", "<constant>", "_0", "<constant>"]),
            Linearity::Affine
        ); // C0*_0+C1
        assert_eq!(c(&["*", "<constant>", "sin", "_0"]), Linearity::Affine); // C0*sin(_0) affine
        assert_eq!(c(&["neg", "*", "<constant>", "_0"]), Linearity::Affine); // linear unary preserves
        assert_eq!(c(&["sin", "*", "<constant>", "_0"]), Linearity::Nonlinear); // C inside sin
        assert_eq!(c(&["*", "<constant>", "<constant>"]), Linearity::Nonlinear); // C*C
        assert_eq!(c(&["/", "_0", "<constant>"]), Linearity::Nonlinear); // C in denominator
        assert_eq!(c(&["/", "<constant>", "_0"]), Linearity::Affine); // C/_0 = C*(1/_0) affine
        assert_eq!(c(&["pow", "<constant>", "2"]), Linearity::Nonlinear); // C in pow base
        assert_eq!(c(&["sin", "_0"]), Linearity::ConstFree);
    }

    #[test]
    fn affine_fit_accepts_and_rejects() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["_0", "_1"]);
        let n = 64usize;
        // X rows
        let mut xf = Vec::with_capacity(n * 2);
        for r in 0..n {
            xf.push((r as f64) * 0.1 - 3.0);
            xf.push((r as f64) * -0.07 + 1.0);
        }
        // y_target = 2.5*_0 - 1.3  (affine, exactly representable by C0*_0+C1)
        let y: Vec<f64> = (0..n).map(|r| 2.5 * xf[2 * r] - 1.3).collect();
        let cand = s(&["+", "*", "<constant>", "_0", "<constant>"]);
        assert_eq!(
            e.exist_constants_fit_linear(&cand, &vars, &xf, n, &y, 1e-5, 1e-8)
                .unwrap(),
            Some(true)
        );
        // y2 = sin(_0): NOT representable by C0*_0+C1 -> reject
        let y2: Vec<f64> = (0..n).map(|r| xf[2 * r].sin()).collect();
        assert_eq!(
            e.exist_constants_fit_linear(&cand, &vars, &xf, n, &y2, 1e-5, 1e-8)
                .unwrap(),
            Some(false)
        );
        // nonlinear-in-params candidate -> None (the LM path handles it)
        let nl = s(&["sin", "*", "<constant>", "_0"]);
        assert_eq!(
            e.exist_constants_fit_linear(&nl, &vars, &xf, n, &y, 1e-5, 1e-8)
                .unwrap(),
            None
        );
    }

    #[test]
    fn householder_lstsq_recovers_exact_solution() {
        // overdetermined well-conditioned system with an EXACT solution; pins the R[j][j] = -norm
        // sign convention (a +norm bug returns an inconsistent, wrong solution).
        let m = 50usize;
        let col0: Vec<f64> = (0..m).map(|r| (r as f64) * 0.1 - 2.5).collect();
        let col1: Vec<f64> = vec![1.0; m];
        // t = 3.0*col0 - 1.7*col1 exactly
        let t: Vec<f64> = (0..m).map(|r| 3.0 * col0[r] - 1.7).collect();
        let c = super::householder_lstsq(vec![col0, col1], t, m, 2).unwrap();
        assert!((c[0] - 3.0).abs() < 1e-12, "slope {} != 3.0", c[0]);
        assert!((c[1] + 1.7).abs() < 1e-12, "intercept {} != -1.7", c[1]);
        // wide-magnitude column (the conditioning case): still exact via QR
        let big0: Vec<f64> = (0..m).map(|r| ((r as f64) - 25.0) * 40.0).collect();
        let big1: Vec<f64> = vec![1.0; m];
        let bt: Vec<f64> = (0..m).map(|r| 7.0 * big0[r]).collect(); // intercept exactly 0
        let bc = super::householder_lstsq(vec![big0, big1], bt, m, 2).unwrap();
        assert!((bc[0] - 7.0).abs() < 1e-12);
        assert!(bc[1].abs() < 1e-9, "intercept {} not ~0", bc[1]);
    }

    #[test]
    fn affine_growing_basis_intercept_recovers() {
        // Regression: with a growing basis (exp/cosh/pow3+)
        // an additive intercept was unrecoverable -- the unweighted solve let rows with |y|~1e21
        // dominate, whose f64 rounding exceeds an O(1) intercept. C0*f(x)+C1 and f(x)+C1 rejected
        // on exactly-true constants while interceptless C0*f(x) passed. The row-weighted solve
        // must certify the family at mine tolerances; a non-member must still reject.
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["_0"]);
        let n = 256usize;
        // dense sweep incl. rows where exp(x) spans 1e-300 .. 1e304 (no overflow rows: those are
        // masked by the y-finite gate anyway; the precision cliff is what this test pins)
        let xf: Vec<f64> = (0..n)
            .map(|r| ((r as f64) / (n as f64) - 0.5) * 1400.0)
            .collect();
        let y: Vec<f64> = xf.iter().map(|&x| 2.5 * x.exp() - 1.3).collect();
        let cand = s(&["+", "*", "<constant>", "exp", "_0", "<constant>"]);
        assert_eq!(
            e.exist_constants_fit_linear(&cand, &vars, &xf, n, &y, 1e-9, 1e-12)
                .unwrap(),
            Some(true),
            "C0*exp(_0)+C1 must certify on exactly-true constants"
        );
        // single-constant intercept form f(x)+C (also 0/4 before the fix)
        let y2: Vec<f64> = xf.iter().map(|&x| x.exp() + 4.2).collect();
        let cand2 = s(&["+", "exp", "_0", "<constant>"]);
        assert_eq!(
            e.exist_constants_fit_linear(&cand2, &vars, &xf, n, &y2, 1e-9, 1e-12)
                .unwrap(),
            Some(true),
            "exp(_0)+C must certify on an exactly-true constant"
        );
        // soundness: a non-member (additive non-constant part) must still reject
        let y3: Vec<f64> = xf.iter().map(|&x| x.exp() + x.sin()).collect();
        assert_eq!(
            e.exist_constants_fit_linear(&cand, &vars, &xf, n, &y3, 1e-9, 1e-12)
                .unwrap(),
            Some(false),
            "exp(_0)+sin(_0) is not in the C0*exp(_0)+C1 family"
        );
    }

    #[test]
    fn bare_constant_exact_feasibility() {
        // Regression: the accept gate's feasible set for a
        // bare <constant> is an interval intersection; a least-squares mean is NOT its Chebyshev
        // center and rejected a feasible skewed case (63 rows at e-d, 1 at e+d, d inside the
        // band) -- breaking the fold-filter dominance. The exact decision must accept it.
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["_0"]);
        let n = 64usize;
        let xf: Vec<f64> = (0..n).map(|r| r as f64).collect();
        let ec = std::f64::consts::E;
        let d = 2.4e-9; // inside the band atol + rtol*e = 1e-12 + 1e-9*e ~ 2.72e-9
        let mut y = vec![ec - d; n];
        y[n - 1] = ec + d;
        let cand = s(&["<constant>"]);
        assert_eq!(
            e.exist_constants_fit_linear(&cand, &vars, &xf, n, &y, 1e-9, 1e-12)
                .unwrap(),
            Some(true),
            "v = e is feasible on every row; the exact decision must find it"
        );
        // infeasible: spread wider than any band -> reject
        let mut y2 = vec![ec - 1.0e-8; n];
        y2[n - 1] = ec + 1.0e-8;
        assert_eq!(
            e.exist_constants_fit_linear(&cand, &vars, &xf, n, &y2, 1e-9, 1e-12)
                .unwrap(),
            Some(false)
        );
        // negative-branch sanity
        let y3 = vec![-3.5f64; n];
        assert_eq!(
            e.exist_constants_fit_linear(&cand, &vars, &xf, n, &y3, 1e-9, 1e-12)
                .unwrap(),
            Some(true)
        );
    }

    #[test]
    fn weighted_design_overflow_is_safe() {
        // Regression: rows pairing a huge-but-FINITE basis
        // (sinh(705) ~ 7.5e305) with y ~ 0 (cosh(x) - sinh(x) cancels bitwise to 0.0 at large x)
        // get weight ~ 1/atol = 1e12; weighting the RAW basis overflowed to inf and NaN'd the
        // solve -> false reject of the exactly-true C = -1. Column-scale-then-weight must accept.
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["_0"]);
        let xs: Vec<f64> = vec![
            -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 300.0, 705.0, 705.5,
            706.0,
        ];
        let n = xs.len();
        // y = cosh(x) + C*sinh(x) at C = -1 (= exp(-x); 0.0 bitwise at x >= ~705)
        let y: Vec<f64> = xs.iter().map(|&x| x.cosh() - x.sinh()).collect();
        let cand = s(&["+", "cosh", "_0", "*", "<constant>", "sinh", "_0"]);
        assert_eq!(
            e.exist_constants_fit_linear(&cand, &vars, &xs, n, &y, 1e-9, 1e-12)
                .unwrap(),
            Some(true),
            "exactly-true C = -1 must certify despite 1e305-scale basis rows at y ~ 0"
        );
    }

    #[test]
    fn exact_interpolation_k_equals_m() {
        // Degenerate-reflector regression (pre-existing): with exactly
        // k valid rows the QR's last column has an EMPTY subdiagonal; the general reflector
        // formula divides by zero and NaNs the solve -> exact interpolation fits rejected.
        // The identity-reflector branch must handle it.
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["_0"]);
        let xf: Vec<f64> = vec![1.0, 3.0];
        let y: Vec<f64> = vec![2.0 * 1.0 - 5.0, 2.0 * 3.0 - 5.0]; // 2x - 5 on 2 rows, k = 2
        let cand = s(&["+", "*", "<constant>", "_0", "<constant>"]);
        assert_eq!(
            e.exist_constants_fit_linear(&cand, &vars, &xf, 2, &y, 1e-9, 1e-12)
                .unwrap(),
            Some(true),
            "k == m exact interpolation must certify"
        );
    }

    #[test]
    fn lm_fit_accepts_and_rejects() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["_0", "_1"]);
        let n = 200usize;
        let mut xf = Vec::with_capacity(n * 2);
        for r in 0..n {
            xf.push((r as f64) * 0.05 - 5.0);
            xf.push((r as f64) * -0.03 + 2.0);
        }
        // nonlinear-in-params candidate C0*sin(C1*_0)+C2
        let cand = s(&[
            "+",
            "*",
            "<constant>",
            "sin",
            "*",
            "<constant>",
            "_0",
            "<constant>",
        ]);
        // y from KNOWN constants -> a fit exists -> accept (within a few restarts)
        let y: Vec<f64> = (0..n)
            .map(|r| 2.0 * (1.5 * xf[2 * r]).sin() + 0.5)
            .collect();
        assert!(e
            .exist_constants_fit(&cand, &vars, &xf, n, &y, 1e-5, 1e-8, 16, 0)
            .unwrap());
        // y2 = exp(_0): not representable by C0*sin(C1*_0)+C2 -> reject
        let y2: Vec<f64> = (0..n).map(|r| xf[2 * r].exp()).collect();
        assert!(!e
            .exist_constants_fit(&cand, &vars, &xf, n, &y2, 1e-5, 1e-8, 16, 0)
            .unwrap());
    }

    /// The fit-arm rescue flagship, piecewise: y = atanh(tanh(1.3*x)) evaluated in f64 has
    /// dead-zone rows (|1.3x| in [10, 19.6]) corrupted by up to 2e-1, which CONTAMINATE the
    /// plain least-squares fit of candidate `* <constant> _0` far beyond the rtol band. The
    /// robust refit must (a) recover C = 1.3 to machine precision from the majority rows,
    /// after which (b) `hiprec::rescue` certifies the corrupted rows at P bits.
    #[test]
    fn robust_refit_recovers_contaminated_constants() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let ops = e.operators_ref();
        let vars = s(&["_0"]);
        // 56 well-conditioned rows + 6 dead-zone rows (|1.3x| in [10, 19.6])
        let mut xs: Vec<f64> = (0..56).map(|i| -2.0 + 4.0 * (i as f64) / 55.0).collect();
        xs.extend([8.0, 9.5, 11.0, 12.5, 14.0, -10.0]);
        let n = xs.len();
        let cols = vec![xs.clone()];
        let y: Vec<f64> = xs.iter().map(|&x| (1.3 * x).tanh().atanh()).collect();
        // sanity: the dead-zone rows really are corrupted beyond the mine tolerance
        let n_bad = xs
            .iter()
            .zip(&y)
            .filter(|(&x, &yr)| yr.is_finite() && (yr - 1.3 * x).abs() > 1e-12 + 1e-9 * yr.abs())
            .count();
        assert!(n_bad >= 4, "expected corrupted dead-zone rows, got {n_bad}");
        let cand = s(&["*", "<constant>", "_0"]);
        let tape = Tape::compile(&cand, ops, &vars).unwrap();
        let lin = classify(&cand, ops).unwrap();
        // (a) the plain fit must NOT pass (that would make the rescue untestable here)...
        let verdict =
            exist_constants_fit_prepared(&tape, lin, &cols, n, &y, 1e-9, 1e-12, 16, 7, None);
        let FitVerdict::NearMiss(c0) = verdict else {
            panic!("expected NearMiss, got {verdict:?}");
        };
        // ...and its constants are contaminated (visibly off 1.3).
        // (b) robust refit recovers the true constant from the majority rows.
        let c1 = refit_robust(&tape, lin, &cols, n, &y, &c0, 1e-9, 1e-12)
            .expect("consensus exists: only 6 of 62 rows are corrupted");
        assert!(
            (c1[0] - 1.3).abs() <= 1e-9 * 1.3,
            "refit constant {} not within rtol of 1.3 (contaminated fit was {})",
            c1[0],
            c0[0]
        );
        // (c) the P-bit rescue certifies the corrupted rows at the refit constants.
        let y_fit = tape.eval_columns(&cols, &c1, n);
        assert!(crate::hiprec::rescue(
            &y,
            &y_fit,
            &s(&["atanh", "tanh", "*", "<constant>", "_0"]),
            &[1.3],
            &cand,
            &c1,
            ops,
            &vars,
            &cols,
            1e-9,
            1e-12,
        ));
    }
}
