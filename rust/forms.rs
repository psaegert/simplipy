//! SYNTACTIC conversions between the explicit binary PREFIX dialect and the TAGGED
//! dialect -- token-level regrouping and nothing else.
//!
//! This module is deliberately not the AC core. It never builds an [`crate::ac::expr::Ex`],
//! never runs a canonical constructor, never consults the ruleset, never reorders a bag and
//! never re-spells a literal. Its answers therefore do not depend on the engine ARTIFACT,
//! which is the whole point of the split: `simplify` canonicalises, a conversion re-notates
//! (design: the research harness).
//!
//! The governing rule: **`to_X` leaves structure that is already in form X untouched and
//! rewrites only the foreign structure.** [`tagged_to_prefix`] expands bags and touches
//! nothing else, so it is the identity on tag-free input by construction; [`prefix_to_tagged`]
//! regroups `+ - * / neg inv` into bags and merges only a SAME-POLARITY same-kind nesting, so
//! it is the identity on input already in tagged normal form.
//!
//! FLATTENING IS CONSERVATIVE -- same polarity only. The right operand of `-`/`/` and the
//! operand of `neg`/`inv` become ONE section member, never a flattened run, because
//! `1/(a*b)` and `(1/a)*(1/b)` are different expressions (they part company at `0` and
//! `inf`, which is exactly why the AC core gates that distribution behind a nonzero
//! certificate). The inverse-of-product law is not a syntactic regrouping.

use crate::operators::Operators;

/// The tagged dialect's delimiters and inverse-section markers. They belong to the AC
/// language, not to any operator config, so no vocabulary can shadow them.
pub const TAG_TOKENS: [&str; 6] = ["<add>", "</add>", "<mul>", "</mul>", "<sub>", "<div>"];

/// One node of the shared syntax tree. `Add`/`Mul` are the tagged bags, carrying their
/// members in SOURCE ORDER and their inverse section separately.
enum Node {
    Leaf(String),
    App(String, Vec<Node>),
    Add(Vec<Node>, Vec<Node>),
    Mul(Vec<Node>, Vec<Node>),
}

/// Regroup a PREFIX token sequence into the TAGGED dialect. `Err` names the malformation.
pub fn prefix_to_tagged(tokens: &[String], ops: &Operators) -> Result<Vec<String>, String> {
    let node = parse(tokens, ops)?;
    let mut out = Vec::with_capacity(tokens.len() + 4);
    emit_tagged(&node, &mut out);
    Ok(out)
}

/// Expand a TAGGED token sequence into the explicit binary PREFIX dialect.
pub fn tagged_to_prefix(tokens: &[String], ops: &Operators) -> Result<Vec<String>, String> {
    let node = parse(tokens, ops)?;
    let mut out = Vec::with_capacity(tokens.len());
    emit_prefix(&node, &mut out);
    Ok(out)
}

/// Well-formedness of either dialect, without producing anything: the conversion parse IS
/// the validation, so `to_X` and this share one arbiter.
pub fn check(tokens: &[String], ops: &Operators) -> Result<(), String> {
    parse(tokens, ops).map(|_| ())
}

// --------------------------------------------------------------------------- parsing

/// One LIBERAL parser for both dialects (and mixes of them), exactly as the AC parser is
/// liberal on input. Tags are recognised first; everything else is an operator application
/// at its declared arity, or a leaf.
fn parse(tokens: &[String], ops: &Operators) -> Result<Node, String> {
    let (node, at) = parse_one(tokens, 0, ops)?;
    if at != tokens.len() {
        return Err(format!(
            "malformed expression: {} trailing token(s) after a complete expression",
            tokens.len() - at
        ));
    }
    Ok(node)
}

fn parse_one(tokens: &[String], idx: usize, ops: &Operators) -> Result<(Node, usize), String> {
    let t = tokens
        .get(idx)
        .ok_or_else(|| "malformed expression: operand expected, input ended".to_string())?;
    if t == "<add>" || t == "<mul>" {
        let is_add = t == "<add>";
        let (closer, section) = if is_add {
            ("</add>", "<sub>")
        } else {
            ("</mul>", "<div>")
        };
        let (mut pos, mut neg) = (Vec::new(), Vec::new());
        let mut inverted = false;
        let mut at = idx + 1;
        loop {
            let s = tokens.get(at).ok_or_else(|| {
                format!("malformed expression: bag opened with {t} is never closed")
            })?;
            if s == closer {
                at += 1;
                break;
            }
            if s == section {
                if inverted {
                    return Err(format!(
                        "malformed expression: two {section} sections in one bag"
                    ));
                }
                inverted = true;
                at += 1;
                continue;
            }
            let (child, next) = parse_one(tokens, at, ops)?;
            if inverted {
                neg.push(child)
            } else {
                pos.push(child)
            }
            at = next;
        }
        if pos.len() + neg.len() < 2 {
            return Err(format!(
                "malformed expression: bag opened with {t} holds fewer than two members"
            ));
        }
        let node = if is_add {
            Node::Add(pos, neg)
        } else {
            Node::Mul(pos, neg)
        };
        return Ok((node, at));
    }
    if TAG_TOKENS.contains(&t.as_str()) {
        return Err(format!("malformed expression: {t} with no enclosing bag"));
    }
    match ops.arity_of(t) {
        None => Ok((Node::Leaf(t.clone()), idx + 1)),
        Some(n) => {
            let mut args = Vec::with_capacity(n as usize);
            let mut at = idx + 1;
            for _ in 0..n {
                let (a, next) = parse_one(tokens, at, ops)?;
                args.push(a);
                at = next;
            }
            Ok((Node::App(t.clone(), args), at))
        }
    }
}

// ------------------------------------------------------------------ prefix -> tagged

/// Collect one SAME-POLARITY additive run. The right operand of `-` and the operand of
/// `neg` are single subtracted members, never flattened runs.
fn flatten_add<'a>(node: &'a Node, pos: &mut Vec<&'a Node>, neg: &mut Vec<&'a Node>) {
    match node {
        Node::App(op, args) if op == "+" && args.len() == 2 => {
            flatten_add(&args[0], pos, neg);
            flatten_add(&args[1], pos, neg);
        }
        Node::App(op, args) if op == "-" && args.len() == 2 => {
            flatten_add(&args[0], pos, neg);
            neg.push(&args[1]);
        }
        Node::App(op, args) if op == "neg" && args.len() == 1 => neg.push(&args[0]),
        Node::Add(p, n) => {
            pos.extend(p.iter());
            neg.extend(n.iter());
        }
        other => pos.push(other),
    }
}

/// The multiplicative twin.
fn flatten_mul<'a>(node: &'a Node, num: &mut Vec<&'a Node>, den: &mut Vec<&'a Node>) {
    match node {
        Node::App(op, args) if op == "*" && args.len() == 2 => {
            flatten_mul(&args[0], num, den);
            flatten_mul(&args[1], num, den);
        }
        Node::App(op, args) if op == "/" && args.len() == 2 => {
            flatten_mul(&args[0], num, den);
            den.push(&args[1]);
        }
        Node::App(op, args) if op == "inv" && args.len() == 1 => den.push(&args[0]),
        Node::Mul(n, d) => {
            num.extend(n.iter());
            den.extend(d.iter());
        }
        other => num.push(other),
    }
}

fn is_add_shaped(node: &Node) -> bool {
    matches!(node, Node::Add(..))
        || matches!(node, Node::App(op, args)
            if (op == "+" || op == "-") && args.len() == 2 || op == "neg" && args.len() == 1)
}

fn is_mul_shaped(node: &Node) -> bool {
    matches!(node, Node::Mul(..))
        || matches!(node, Node::App(op, args)
            if (op == "*" || op == "/") && args.len() == 2 || op == "inv" && args.len() == 1)
}

fn emit_tagged(node: &Node, out: &mut Vec<String>) {
    if is_add_shaped(node) {
        let (mut pos, mut neg) = (Vec::new(), Vec::new());
        flatten_add(node, &mut pos, &mut neg);
        emit_bag(&pos, &neg, "<add>", "<sub>", "</add>", "neg", out);
        return;
    }
    if is_mul_shaped(node) {
        let (mut num, mut den) = (Vec::new(), Vec::new());
        flatten_mul(node, &mut num, &mut den);
        emit_bag(&num, &den, "<mul>", "<div>", "</mul>", "inv", out);
        return;
    }
    match node {
        Node::Leaf(t) => out.push(t.clone()),
        Node::App(op, args) => {
            out.push(op.clone());
            for a in args {
                emit_tagged(a, out);
            }
        }
        _ => unreachable!("bags are handled above"),
    }
}

/// One bag, or -- when the regrouping yields a LONE inverted member, which the AC grammar
/// cannot spell as a bag -- the unary group-inverse spelling it keeps instead.
fn emit_bag(
    pos: &[&Node],
    neg: &[&Node],
    open: &str,
    section: &str,
    close: &str,
    unary: &str,
    out: &mut Vec<String>,
) {
    if pos.len() + neg.len() < 2 {
        // `neg x0` / `inv x0`: one member, inverted -- there is no one-member bag.
        out.push(unary.to_string());
        emit_tagged(neg[0], out);
        return;
    }
    out.push(open.to_string());
    for m in pos {
        emit_tagged(m, out);
    }
    if !neg.is_empty() {
        out.push(section.to_string());
        for m in neg {
            emit_tagged(m, out);
        }
    }
    out.push(close.to_string());
}

// ------------------------------------------------------------------ tagged -> prefix

fn emit_prefix(node: &Node, out: &mut Vec<String>) {
    match node {
        Node::Leaf(t) => out.push(t.clone()),
        Node::App(op, args) => {
            out.push(op.clone());
            for a in args {
                emit_prefix(a, out);
            }
        }
        Node::Add(pos, neg) => emit_expansion(pos, neg, "+", "-", "neg", out),
        Node::Mul(num, den) => emit_expansion(num, den, "*", "/", "inv", out),
    }
}

/// Expand one bag. The positive part is a RIGHT-nested chain (`+ a + b c`) -- the
/// association the shipped explicit emitter already produces. The section is a LEFT-nested
/// chain (`- - P c d`), which is forced rather than chosen: a section inverts each member
/// INDIVIDUALLY, and only a left chain spells that without inventing the product `c*d`.
/// An empty positive part opens with the unary group inverse, so no `0`/`1` unit is ever
/// materialised and the cycle stays byte-exact.
fn emit_expansion(
    pos: &[Node],
    neg: &[Node],
    chain: &str,
    inverse: &str,
    unary: &str,
    out: &mut Vec<String>,
) {
    let (head, rest): (Vec<String>, &[Node]) = if pos.is_empty() {
        let mut head = vec![unary.to_string()];
        emit_prefix(&neg[0], &mut head);
        (head, &neg[1..])
    } else {
        let mut head = Vec::new();
        for p in &pos[..pos.len() - 1] {
            head.push(chain.to_string());
            emit_prefix(p, &mut head);
        }
        emit_prefix(&pos[pos.len() - 1], &mut head);
        (head, neg)
    };
    // Left nesting: each subtracted/divided member wraps everything emitted so far.
    for _ in rest {
        out.push(inverse.to_string());
    }
    out.extend(head);
    for m in rest {
        emit_prefix(m, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{OperatorSpec, Operators};
    use rustc_hash::FxHashMap;

    fn ops() -> Operators {
        let mut specs = FxHashMap::default();
        let mut order = Vec::new();
        for (name, arity) in [
            ("+", 2u8),
            ("-", 2),
            ("*", 2),
            ("/", 2),
            ("neg", 1),
            ("inv", 1),
            ("pow", 2),
            ("rootn", 2),
            ("abs", 1),
            ("sin", 1),
        ] {
            order.push(name.to_string());
            specs.insert(
                name.to_string(),
                OperatorSpec {
                    realization: String::new(),
                    alias: Vec::new(),
                    inverse: None,
                    arity,
                    precedence: None,
                    commutative: false,
                },
            );
        }
        Operators::from_specs(order, specs)
    }

    fn t(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn prefix_input_is_returned_unchanged_by_the_prefix_expansion() {
        let o = ops();
        for row in [
            t(&["+", "x0", "+", "x1", "x2"]),
            t(&["-", "-", "x0", "x1", "x2"]),
            t(&["/", "x0", "*", "x1", "x2"]),
            t(&["pow", "abs", "x0", "2"]),
            t(&["neg", "x0"]),
        ] {
            assert_eq!(tagged_to_prefix(&row, &o).unwrap(), row);
        }
    }

    #[test]
    fn tagged_prefix_tagged_is_byte_exact() {
        let o = ops();
        for row in [
            t(&["<add>", "x0", "x1", "x2", "</add>"]),
            t(&["<add>", "x0", "<sub>", "x1", "x2", "</add>"]),
            t(&["<mul>", "x0", "<div>", "x1", "x2", "</mul>"]),
            t(&["<mul>", "<div>", "x1", "x2", "</mul>"]),
            t(&["<add>", "<sub>", "x1", "x2", "</add>"]),
            t(&[
                "<mul>", "x0", "<div>", "<mul>", "x1", "x2", "</mul>", "</mul>",
            ]),
            t(&["neg", "x0"]),
            t(&["inv", "x0"]),
        ] {
            let p = tagged_to_prefix(&row, &o).unwrap();
            assert_eq!(prefix_to_tagged(&p, &o).unwrap(), row, "via {p:?}");
        }
    }

    #[test]
    fn both_associations_reach_one_bag() {
        let o = ops();
        let a = prefix_to_tagged(&t(&["*", "x0", "*", "x1", "x2"]), &o).unwrap();
        let b = prefix_to_tagged(&t(&["*", "*", "x0", "x1", "x2"]), &o).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, t(&["<mul>", "x0", "x1", "x2", "</mul>"]));
    }

    #[test]
    fn the_inverse_of_a_product_is_never_distributed() {
        let o = ops();
        assert_eq!(
            prefix_to_tagged(&t(&["/", "x0", "*", "x1", "x2"]), &o).unwrap(),
            t(&["<mul>", "x0", "<div>", "<mul>", "x1", "x2", "</mul>", "</mul>"])
        );
    }

    #[test]
    fn malformed_input_is_named() {
        let o = ops();
        assert!(prefix_to_tagged(&t(&["*", "x0"]), &o).is_err());
        assert!(tagged_to_prefix(&t(&["<mul>", "x0", "x1"]), &o).is_err());
        assert!(tagged_to_prefix(&t(&["</mul>"]), &o).is_err());
        assert!(tagged_to_prefix(&t(&["<mul>", "x0", "</mul>"]), &o).is_err());
        assert!(prefix_to_tagged(&t(&["bogusfn", "x0"]), &o).is_err());
    }
}
