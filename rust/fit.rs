//! Constant-fitting for the OFFLINE rule miner (Phase B, Milestone 3) -- the native replacement for
//! `exist_constants_that_fit` (engine.py:2272), which currently calls `scipy.optimize.curve_fit`.
//!
//! Measured (M1): one `curve_fit` is ~782 us (nonlinear) / ~103 us (linear) vs ~11 us for a residual
//! eval -- i.e. ~90-98% is scipy/MINPACK/Python overhead, NOT math. The biggest lever (the user's
//! "maybe we don't even need an optimizer") is to recognize that many candidates are AFFINE in their
//! `<constant>`s, for which the best fit is a single CLOSED-FORM least-squares solve -- no iteration,
//! no random `p0`, no retries, and a DETERMINISTIC accept/reject decision.
//!
//! M3a (this file): the linearity classifier + the affine closed-form path. Nonlinear-in-params
//! candidates return `None` (deferred to M3b, a native LM). The accept/reject gate stays `allclose`
//! (crate::eval), so soundness is identical to scipy's: the optimizer only PROPOSES, allclose DISPOSES.

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
}
