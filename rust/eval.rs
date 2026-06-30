//! Vectorized "tape" evaluator for the OFFLINE rule miner (Phase B, Milestone 1).
//!
//! The Python miner re-runs `operators_to_realizations -> explicit_constant_placeholders ->
//! prefix_to_infix -> codify -> code_to_lambda` per candidate and evaluates with numpy over a
//! (n_samples x n_vars) array. This module replaces that with a compile-once / evaluate-many tape:
//!
//!   * compile a prefix expression ONCE to a flat postfix program. Each operator is resolved at
//!     COMPILE to its scalar kernel as a function pointer (`numeric::unary_fn`/`binary_fn`), so the
//!     hot loop has NO per-element string dispatch -- the key to matching numpy's vectorized speed;
//!   * evaluate COLUMN-WISE (`eval_columns`): each operator runs a tight per-column loop, exactly
//!     numpy's strategy (arithmetic auto-vectorizes; transcendentals call the same libm numpy does).
//!     A row-wise `eval_row` is kept for one-off use / small batches.
//!
//! Parity policy is `numeric.rs`'s: per-element f64 + the SYSTEM libm. Differences vs numpy are at
//! most a last ULP (absorbed by the `allclose` decision gate, rtol=1e-5); the soundness-critical
//! special values (div0 -> signed inf, sqrt(neg) -> nan) are bit-exact via the shared kernel.

use crate::numeric::{binary_fn, unary_fn};
use crate::operators::Operators;

/// One postfix instruction. Operators carry their scalar kernel as a function pointer (resolved at
/// compile), so neither row nor column evaluation does any string matching.
enum Instr {
    Const(f64),
    Var(usize),
    Param(usize),
    Unary(fn(f64) -> f64),
    Binary(fn(f64, f64) -> f64),
}

/// A prefix expression compiled to a postfix program. Compile ONCE, evaluate over many rows / params.
pub struct Tape {
    instrs: Vec<Instr>,
    /// number of distinct `<constant>` slots (left-to-right appearance order = param index).
    pub n_params: usize,
    /// peak operand-stack depth.
    max_stack: usize,
}

impl Tape {
    /// Compile a flat prefix token slice to a postfix tape. Leaves: a variable (name in `var_names`
    /// -> column index), a `<constant>` (-> next param slot, numbered left-to-right exactly like
    /// `explicit_constant_placeholders`), or a numeric/special literal (`numeric::leaf_value`).
    pub fn compile(
        tokens: &[String],
        ops: &Operators,
        var_names: &[String],
    ) -> Result<Tape, String> {
        let mut instrs = Vec::with_capacity(tokens.len());
        let mut n_params = 0usize;
        let mut idx = 0usize;
        compile_node(tokens, &mut idx, ops, var_names, &mut instrs, &mut n_params)?;
        if idx != tokens.len() {
            return Err(format!(
                "trailing tokens after position {idx} (expression length {})",
                tokens.len()
            ));
        }
        // Peak depth: a leaf pushes 1; a unary op is net 0; a binary op is net -1.
        let mut depth = 0i64;
        let mut max = 0i64;
        for ins in &instrs {
            match ins {
                Instr::Binary(_) => depth -= 1,
                Instr::Unary(_) => {}
                _ => depth += 1,
            }
            if depth > max {
                max = depth;
            }
        }
        Ok(Tape {
            instrs,
            n_params,
            max_stack: max.max(1) as usize,
        })
    }

    /// Row-wise evaluation (one row at a time) with a caller-owned scratch stack. Kept for small
    /// batches / one-off use; the miner hot path uses `eval_columns`.
    #[inline]
    #[allow(dead_code)]
    pub fn eval_row(&self, x_row: &[f64], params: &[f64], stack: &mut Vec<f64>) -> f64 {
        stack.clear();
        for ins in &self.instrs {
            match ins {
                Instr::Const(v) => stack.push(*v),
                Instr::Var(c) => stack.push(x_row[*c]),
                Instr::Param(p) => stack.push(params[*p]),
                Instr::Unary(f) => {
                    let t = stack.last_mut().unwrap();
                    *t = f(*t);
                }
                Instr::Binary(g) => {
                    let b = stack.pop().unwrap();
                    let a = stack.last_mut().unwrap();
                    *a = g(*a, b);
                }
            }
        }
        stack.pop().unwrap_or(f64::NAN)
    }

    /// Column-wise (vectorized) evaluation -- numpy's strategy. `x_cols[c]` is variable `c`'s column
    /// (length `n_rows`); `<constant>` slots bind to `params`. Each operator runs a tight per-column
    /// loop. Returns the length-`n_rows` result column.
    pub fn eval_columns(&self, x_cols: &[Vec<f64>], params: &[f64], n_rows: usize) -> Vec<f64> {
        let mut stack: Vec<Vec<f64>> = Vec::with_capacity(self.max_stack);
        for ins in &self.instrs {
            match ins {
                Instr::Const(v) => stack.push(vec![*v; n_rows]),
                Instr::Var(c) => stack.push(x_cols[*c].clone()),
                Instr::Param(p) => stack.push(vec![params[*p]; n_rows]),
                Instr::Unary(f) => {
                    for x in stack.last_mut().unwrap().iter_mut() {
                        *x = f(*x);
                    }
                }
                Instr::Binary(g) => {
                    let b = stack.pop().unwrap();
                    let a = stack.last_mut().unwrap();
                    for (x, y) in a.iter_mut().zip(b.iter()) {
                        *x = g(*x, *y);
                    }
                }
            }
        }
        stack.pop().unwrap_or_else(|| vec![f64::NAN; n_rows])
    }
}

fn compile_node(
    tokens: &[String],
    idx: &mut usize,
    ops: &Operators,
    var_names: &[String],
    instrs: &mut Vec<Instr>,
    n_params: &mut usize,
) -> Result<(), String> {
    let tok = tokens
        .get(*idx)
        .ok_or_else(|| format!("truncated prefix expression at position {idx}"))?
        .clone();
    *idx += 1;
    if let Some(arity) = ops.arity_of(&tok) {
        let arity = arity as usize;
        for _ in 0..arity {
            compile_node(tokens, idx, ops, var_names, instrs, n_params)?;
        }
        match arity {
            1 => {
                let f = unary_fn(&tok).ok_or_else(|| format!("no unary kernel for {tok:?}"))?;
                instrs.push(Instr::Unary(f));
            }
            2 => {
                let g = binary_fn(&tok).ok_or_else(|| format!("no binary kernel for {tok:?}"))?;
                instrs.push(Instr::Binary(g));
            }
            other => return Err(format!("unsupported arity {other} for {tok:?}")),
        }
    } else if tok == "<constant>" {
        instrs.push(Instr::Param(*n_params));
        *n_params += 1;
    } else if let Some(col) = var_names.iter().position(|v| *v == tok) {
        instrs.push(Instr::Var(col));
    } else if let Some(v) = crate::numeric::leaf_value(&tok) {
        instrs.push(Instr::Const(v));
    } else {
        return Err(format!("unknown leaf token {tok:?}"));
    }
    Ok(())
}

/// Column-major variable columns from a row-major `x_flat` (shape `n_rows x n_vars`).
pub fn columns_from_row_major(x_flat: &[f64], n_rows: usize, n_vars: usize) -> Vec<Vec<f64>> {
    let mut cols = vec![Vec::with_capacity(n_rows); n_vars];
    for r in 0..n_rows {
        let base = r * n_vars;
        for (c, col) in cols.iter_mut().enumerate() {
            col.push(x_flat[base + c]);
        }
    }
    cols
}

/// Evaluate one prefix expression over `n_rows` rows of row-major `x_flat` (shape
/// `n_rows x var_names.len()`), `<constant>` placeholders bound to `params` left-to-right. Column-wise.
pub fn evaluate_batch(
    ops: &Operators,
    tokens: &[String],
    var_names: &[String],
    x_flat: &[f64],
    n_rows: usize,
    params: &[f64],
) -> Result<Vec<f64>, String> {
    let tape = Tape::compile(tokens, ops, var_names)?;
    let n_vars = var_names.len();
    if x_flat.len() != n_rows * n_vars {
        return Err(format!(
            "x_flat has {} elements, expected n_rows*n_vars = {}*{} = {}",
            x_flat.len(),
            n_rows,
            n_vars,
            n_rows * n_vars
        ));
    }
    if params.len() < tape.n_params {
        return Err(format!(
            "expression has {} <constant> slot(s) but only {} param(s) given",
            tape.n_params,
            params.len()
        ));
    }
    let cols = columns_from_row_major(x_flat, n_rows, n_vars);
    Ok(tape.eval_columns(&cols, params, n_rows))
}

/// `numpy.allclose(a, b, rtol, atol, equal_nan=True)`: `|a-b| <= atol + rtol*|b|` elementwise, with
/// numpy's special handling -- two NaNs are equal; infinities are close iff equal (same sign), never
/// close to a finite or opposite-sign inf. `b` is the asymmetric reference (second arg), matching the
/// miner's call order. Empty inputs -> True.
pub fn allclose(a: &[f64], b: &[f64], rtol: f64, atol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.is_nan() || y.is_nan() {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            return false;
        }
        if x.is_infinite() || y.is_infinite() {
            if x == y {
                continue;
            }
            return false;
        }
        if (x - y).abs() <= atol + rtol * y.abs() {
            continue;
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn allclose_semantics() {
        assert!(allclose(&[1.0, 2.0], &[1.0, 2.0 + 1e-9], 1e-5, 1e-8));
        assert!(!allclose(&[1.0], &[1.1], 1e-5, 1e-8));
        assert!(!allclose(&[1.0, 2.0], &[1.0], 1e-5, 1e-8));
        assert!(allclose(&[f64::NAN], &[f64::NAN], 1e-5, 1e-8));
        assert!(!allclose(&[f64::NAN], &[1.0], 1e-5, 1e-8));
        assert!(allclose(&[f64::INFINITY], &[f64::INFINITY], 1e-5, 1e-8));
        assert!(!allclose(
            &[f64::INFINITY],
            &[f64::NEG_INFINITY],
            1e-5,
            1e-8
        ));
        assert!(!allclose(&[f64::INFINITY], &[1e300], 1e-5, 1e-8));
        assert!(allclose(&[], &[], 1e-5, 1e-8));
    }

    #[test]
    fn eval_with_engine() {
        let Some(e) = crate::test_engine() else {
            return;
        };
        let vars = s(&["x0", "x1"]);
        let x = vec![2.0, 3.0, -1.0, 4.0]; // row-major 2x2

        assert_eq!(
            e.evaluate_batch(&s(&["+", "x0", "x1"]), &vars, &x, 2, &[])
                .unwrap(),
            vec![5.0, 3.0]
        );
        assert_eq!(
            e.evaluate_batch(&s(&["*", "<constant>", "x0"]), &vars, &x, 2, &[10.0])
                .unwrap(),
            vec![20.0, -10.0]
        );
        assert_eq!(
            e.evaluate_batch(&s(&["-", "x0", "x1"]), &vars, &x, 2, &[])
                .unwrap(),
            vec![-1.0, -5.0]
        );
        let d = e
            .evaluate_batch(&s(&["/", "x0", "0"]), &vars, &x, 2, &[])
            .unwrap();
        assert!(d[0].is_infinite() && d[0] > 0.0);
        assert!(d[1].is_infinite() && d[1] < 0.0);
        assert_eq!(
            e.evaluate_batch(&s(&["pow2", "x0"]), &vars, &x, 2, &[])
                .unwrap(),
            vec![4.0, 1.0]
        );
        let pi = e.evaluate_batch(&s(&["np.pi"]), &vars, &x, 2, &[]).unwrap();
        assert!(pi.iter().all(|v| (v - std::f64::consts::PI).abs() < 1e-12));
        assert_eq!(
            e.evaluate_batch(&s(&["(-1)"]), &vars, &x, 2, &[]).unwrap(),
            vec![-1.0, -1.0]
        );
        assert_eq!(
            e.evaluate_batch(
                &s(&["+", "<constant>", "<constant>"]),
                &vars,
                &x,
                2,
                &[3.0, 4.0]
            )
            .unwrap(),
            vec![7.0, 7.0]
        );
        assert!(e
            .evaluate_batch(&s(&["+", "<constant>", "x0"]), &vars, &x, 2, &[])
            .is_err());

        // row-wise and column-wise must agree on a nested expression
        let expr = s(&["+", "*", "<constant>", "sin", "x0", "x1"]);
        let tape = Tape::compile(&expr, e.operators_ref(), &vars).unwrap();
        let cols = columns_from_row_major(&x, 2, 2);
        let col = tape.eval_columns(&cols, &[7.0], 2);
        let mut st = Vec::new();
        let row0 = tape.eval_row(&x[0..2], &[7.0], &mut st);
        let row1 = tape.eval_row(&x[2..4], &[7.0], &mut st);
        assert!((col[0] - row0).abs() < 1e-12 && (col[1] - row1).abs() < 1e-12);
    }
}
