//! The tests of the constructors, by concern, and the helpers they share.

use super::*;
use crate::operators::{OperatorSpec, Operators};
use crate::tokens::{TokenOverlay, TokenTable, TokenView};
use std::cell::RefCell;

mod add;
mod canon;
mod literals;
mod mul;
mod pow;
mod price;
mod sign;
mod zero_sign;

fn test_ops() -> (Vec<String>, Operators) {
    let unary = ["sin", "cos", "abs", "exp", "log", "cosh"];
    let mut order = Vec::new();
    let mut specs: rustc_hash::FxHashMap<String, OperatorSpec> = Default::default();
    for n in ["+", "-", "*", "/", "pow"] {
        order.push(n.to_string());
        specs.insert(
            n.to_string(),
            OperatorSpec {
                realization: String::new(),
                alias: vec![],
                inverse: None,
                arity: 2,
                precedence: None,
                commutative: n == "+" || n == "*",
            },
        );
    }
    for n in unary {
        order.push(n.to_string());
        specs.insert(
            n.to_string(),
            OperatorSpec {
                realization: String::new(),
                alias: vec![],
                inverse: None,
                arity: 1,
                precedence: None,
                commutative: false,
            },
        );
    }
    let ops = Operators::from_specs(order.clone(), specs);
    (order, ops)
}

/// Run `f` with a fresh view over a minimal operator universe.
fn with_view<R>(f: impl FnOnce(&TokenView) -> R) -> R {
    let (order, ops) = test_ops();
    let table = TokenTable::build(&order, &ops);
    let overlay = RefCell::new(TokenOverlay::new(table.len()));
    let view = TokenView::new(&table, &overlay);
    f(&view)
}

fn x(view: &TokenView) -> Ex {
    Ex::Leaf(view.intern("x0"))
}

fn y(view: &TokenView) -> Ex {
    Ex::Leaf(view.intern("x1"))
}

/// Resolved-string rendering for CROSS-VIEW comparisons: two views may intern the
/// same names to different ids, so `Ex` equality cannot compare their outputs --
/// the resolved shape can.
fn ex_str(e: &Ex, view: &TokenView) -> String {
    match e {
        Ex::Num(r) => format!("{r:?}"),
        Ex::Pi => "pi".into(),
        Ex::E => "e".into(),
        Ex::PosInf => "inf".into(),
        Ex::NegInf => "-inf".into(),
        Ex::NaN => "nan".into(),
        Ex::Const(_) => "C".into(),
        Ex::Leaf(t) => view.resolve_owned(*t),
        Ex::Fun(f, args) => {
            let a: Vec<String> = args.iter().map(|x| ex_str(x, view)).collect();
            format!("{}({})", view.resolve_owned(*f), a.join(","))
        }
        Ex::Pow(b, x) => format!("pow({},{})", ex_str(b, view), ex_str(x, view)),
        Ex::Add(v) => {
            let a: Vec<String> = v.iter().map(|x| ex_str(x, view)).collect();
            format!("add[{}]", a.join(","))
        }
        Ex::Mul(v) => {
            let a: Vec<String> = v.iter().map(|x| ex_str(x, view)).collect();
            format!("mul[{}]", a.join(","))
        }
    }
}
