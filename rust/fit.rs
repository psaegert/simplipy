//! Constant-fitting for the OFFLINE rule miner (Phase B, Milestone 3) -- the native replacement for
//! `exist_constants_that_fit` (engine.py:2272), which currently calls `scipy.optimize.curve_fit`.
//!
//! Measured (M1): one `curve_fit` is ~782 us (nonlinear) / ~103 us (linear) vs ~11 us for a residual
//! eval -- i.e. ~90-98% is scipy/MINPACK/Python overhead, NOT math. The biggest lever (the user's
//! "maybe we don't even need an optimizer") is to recognize that many candidates are AFFINE in their
//! `<constant>`s, for which the best fit is a single CLOSED-FORM least-squares solve -- no iteration,
//! no random `p0`, no retries, and a DETERMINISTIC accept/reject decision.
//!
//! M3a: the linearity classifier + the affine closed-form path (`exist_constants_fit_linear`).
//! M3b: a native Levenberg-Marquardt (`lm_fit`) with random restarts for nonlinear-in-params
//! candidates. The complete native `exist_constants_that_fit` is `exist_constants_fit` (affine ->
//! closed-form, nonlinear -> LM). The accept/reject gate stays `allclose` (crate::eval), so soundness
//! is identical to scipy's: the optimizer only PROPOSES constants, `allclose` DISPOSES.

use crate::eval::{allclose, columns_from_row_major, Tape};
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

impl Linearity {
    pub fn as_str(self) -> &'static str {
        match self {
            Linearity::ConstFree => "constfree",
            Linearity::Affine => "affine",
            Linearity::Nonlinear => "nonlinear",
        }
    }
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

/// AFFINE closed-form fit + accept/reject (the M3a path). The candidate is affine in its k constants,
/// so `y(C) = b(X) + sum_j C_j a_j(X)` with `b = eval(C=0)` and `a_j = eval(C=e_j) - b`. We solve the
/// (ridge-regularized) normal equations on the FINITE-row mask (mirroring scipy's `is_valid` mask and
/// its `n_const > n_valid` bail), evaluate the fitted candidate on ALL rows, and return
/// `allclose(y_target, fitted)` -- the exact same decision gate scipy's path uses.
fn fit_affine_check(
    tape: &Tape,
    x_cols: &[Vec<f64>],
    y_target: &[f64],
    n_rows: usize,
    rtol: f64,
    atol: f64,
) -> bool {
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
    if k > valid.len() {
        return false; // underdetermined -> scipy bails -> False
    }
    // normal equations A^T A c = A^T t over valid rows, t = y_target - b.
    let mut ata = vec![vec![0.0f64; k]; k];
    let mut att = vec![0.0f64; k];
    for &r in &valid {
        let t = y_target[r] - b[r];
        for i in 0..k {
            let ai = a[i][r];
            att[i] += ai * t;
            for j in 0..k {
                ata[i][j] += ai * a[j][r];
            }
        }
    }
    // tiny Tikhonov ridge for a stable solve on rank-deficient/collinear bases (the fitted PROJECTION
    // -- hence the allclose decision -- is unchanged for well-conditioned accept cases).
    let tr: f64 = (0..k).map(|i| ata[i][i]).sum();
    let lambda = if tr > 0.0 { tr * 1e-12 } else { 1e-12 };
    for (i, row) in ata.iter_mut().enumerate() {
        row[i] += lambda;
    }
    let Some(c) = solve(ata, att) else {
        return false;
    };
    // fitted on ALL rows = eval at the solved constants (exact for an affine model).
    let fitted = tape.eval_columns(x_cols, &c, n_rows);
    allclose(y_target, &fitted, rtol, atol)
}

/// Native `exist_constants_that_fit` for the AFFINE case (M3a). Returns `Some(decision)` if the
/// candidate is affine in its constants (handled here), or `None` if it is nonlinear-in-params
/// (deferred to the M3b native LM). `Err` only on a malformed candidate / shape mismatch.
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
    Ok(Some(fit_affine_check(
        &tape, &cols, y_target, n_rows, rtol, atol,
    )))
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
/// gate as the LM). Returns Some(decision) when the solve is computable (enough positive rows), else
/// None -> fall back to the LM (preserves recall; e.g. negative-base integer-power cases).
fn try_log_linear_fit(
    form: LogLinForm,
    g_tape: &Tape,
    cand_tape: &Tape,
    cols: &[Vec<f64>],
    n_rows: usize,
    y: &[f64],
    rtol: f64,
    atol: f64,
) -> Option<bool> {
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
    Some(allclose(y, &fitted, rtol, atol))
}

/// The COMPLETE native `exist_constants_that_fit` (M3): affine candidates -> the closed-form path
/// (deterministic, no restarts); nonlinear-in-params candidates -> `n_restarts` LM solves from random
/// N(0,5) starts (mirroring the worker's retry loop), accept iff any makes `allclose(y_target,
/// fitted)` pass on ALL rows. The accept/reject GATE is identical to scipy's (allclose); only the
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
    Ok(exist_constants_fit_prepared(
        &tape, lin, &cols, n_rows, y_target, rtol, atol, n_restarts, seed, loglin,
    ))
}

/// The post-compile core of `exist_constants_fit`, on a PRE-COMPILED tape + precomputed columns +
/// known `Linearity`. The OFFLINE worker (`crate::worker`) calls this so each candidate's tape and
/// classification are computed ONCE (at library build), not per challenge/source.
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
) -> bool {
    if lin == Linearity::ConstFree {
        let fitted = tape.eval_columns(cols, &[], n_rows);
        return allclose(y_target, &fitted, rtol, atol);
    }
    if lin == Linearity::Affine {
        return fit_affine_check(tape, cols, y_target, n_rows, rtol, atol);
    }
    // Improvement 2: log-linearizable power forms -> closed-form (deterministic, no LM). The accept
    // gate stays `allclose`; an uncomputable solve (None) falls through to the LM (preserves recall).
    if let Some((form, g_tape)) = loglin {
        if let Some(dec) =
            try_log_linear_fit(form, g_tape, tape, cols, n_rows, y_target, rtol, atol)
        {
            return dec;
        }
    }
    // Nonlinear-in-params: LM with random restarts.
    let k = tape.n_params;
    let valid: Vec<usize> = (0..n_rows)
        .filter(|&r| y_target[r].is_finite() && cols.iter().all(|c| c[r].is_finite()))
        .collect();
    if k > valid.len() {
        return false; // scipy issue 13969 bail
    }
    let m = valid.len();
    let xv: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| valid.iter().map(|&r| c[r]).collect())
        .collect();
    let yv: Vec<f64> = valid.iter().map(|&r| y_target[r]).collect();

    let mut rng = Rng::new(seed);
    for _ in 0..n_restarts.max(1) {
        let p0: Vec<f64> = (0..k).map(|_| rng.normal(0.0, 5.0)).collect();
        let c = lm_fit(tape, &xv, &yv, m, k, &p0, rtol, atol);
        let fitted = tape.eval_columns(cols, &c, n_rows);
        if allclose(y_target, &fitted, rtol, atol) {
            return true;
        }
    }
    false
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
        // nonlinear-in-params candidate -> None (deferred to M3b)
        let nl = s(&["sin", "*", "<constant>", "_0"]);
        assert_eq!(
            e.exist_constants_fit_linear(&nl, &vars, &xf, n, &y, 1e-5, 1e-8)
                .unwrap(),
            None
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
}
