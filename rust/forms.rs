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
//! it is the identity on input already in tagged normal form. The lattice heads `max` and `min`
//! (where the config declares them binary) are bags too: `max a max b c` regroups into
//! `<max> a b c </max>`, which expands back to the right-nested chain.
//!
//! FLATTENING IS CONSERVATIVE -- same polarity only. The right operand of `-`/`/` and the
//! operand of `neg`/`inv` become ONE section member, never a flattened run, because
//! `1/(a*b)` and `(1/a)*(1/b)` are different expressions (they part company at `0` and
//! `inf`, which is exactly why the AC core gates that distribution behind a nonzero
//! certificate). The inverse-of-product law is not a syntactic regrouping.

use crate::operators::Operators;

/// The tagged dialect's delimiters and inverse-section markers. They belong to the AC
/// language, not to any operator config, so no vocabulary can shadow them. The lattice bags'
/// delimiters read only where the config declares `max` (`min`) as a binary operator.
pub const TAG_TOKENS: [&str; 10] = [
    "<add>", "</add>", "<mul>", "</mul>", "<sub>", "<div>", "<max>", "</max>", "<min>", "</min>",
];

/// The lattice bags' (opener, closer, head).
const LATTICE_TAGS: [(&str, &str, &str); 2] =
    [("<max>", "</max>", "max"), ("<min>", "</min>", "min")];

/// Is `op`, applied to `n` operands, a lattice head the config declares binary?
fn lattice_app(op: &str, n: usize, ops: &Operators) -> bool {
    n == 2 && (op == "max" || op == "min") && ops.arity_of(op) == Some(2)
}

/// One node of the shared syntax tree. `Add`/`Mul` are the tagged bags, carrying their
/// members in SOURCE ORDER and their inverse section separately.
enum Node {
    Leaf(String),
    App(String, Vec<Node>),
    Add(Vec<Node>, Vec<Node>),
    Mul(Vec<Node>, Vec<Node>),
    /// A lattice bag, `<max> ... </max>` or `<min> ... </min>`: its head and members.
    Lat(String, Vec<Node>),
}

/// Regroup a PREFIX token sequence into the TAGGED dialect. `Err` names the malformation.
pub fn prefix_to_tagged(tokens: &[String], ops: &Operators) -> Result<Vec<String>, String> {
    let node = parse(tokens, ops)?;
    let mut out = Vec::with_capacity(tokens.len() + 4);
    emit_tagged(&node, ops, &mut out);
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
    if let Some(&(open, closer, head)) = LATTICE_TAGS.iter().find(|(o, ..)| o == t) {
        if ops.arity_of(head) != Some(2) {
            return Err(format!(
                "malformed expression: {open} needs {head} declared as a binary operator"
            ));
        }
        let mut members = Vec::new();
        let mut at = idx + 1;
        loop {
            let s = tokens.get(at).ok_or_else(|| {
                format!("malformed expression: bag opened with {open} is never closed")
            })?;
            if s == closer {
                at += 1;
                break;
            }
            let (child, next) = parse_one(tokens, at, ops)?;
            members.push(child);
            at = next;
        }
        if members.len() < 2 {
            return Err(format!(
                "malformed expression: bag opened with {open} holds fewer than two members"
            ));
        }
        return Ok((Node::Lat(head.to_string(), members), at));
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

/// Collect one lattice run of the head `head`: a same-head application or bag gives its
/// operands, anything else is one member.
fn flatten_lat<'a>(node: &'a Node, head: &str, ops: &Operators, out: &mut Vec<&'a Node>) {
    match node {
        Node::App(op, args) if op == head && lattice_app(op, args.len(), ops) => {
            for a in args {
                flatten_lat(a, head, ops, out);
            }
        }
        Node::Lat(op, members) if op == head => {
            for m in members {
                flatten_lat(m, head, ops, out);
            }
        }
        other => out.push(other),
    }
}

fn emit_tagged(node: &Node, ops: &Operators, out: &mut Vec<String>) {
    if is_add_shaped(node) {
        let (mut pos, mut neg) = (Vec::new(), Vec::new());
        flatten_add(node, &mut pos, &mut neg);
        emit_bag(&pos, &neg, "<add>", "<sub>", "</add>", "neg", ops, out);
        return;
    }
    if is_mul_shaped(node) {
        let (mut num, mut den) = (Vec::new(), Vec::new());
        flatten_mul(node, &mut num, &mut den);
        emit_bag(&num, &den, "<mul>", "<div>", "</mul>", "inv", ops, out);
        return;
    }
    let lattice = match node {
        Node::App(op, args) if lattice_app(op, args.len(), ops) => Some(op.as_str()),
        Node::Lat(op, _) => Some(op.as_str()),
        _ => None,
    };
    if let Some(head) = lattice {
        let (open, close, _) = LATTICE_TAGS
            .iter()
            .find(|(.., h)| *h == head)
            .expect("a lattice head");
        let mut members = Vec::new();
        flatten_lat(node, head, ops, &mut members);
        out.push(open.to_string());
        for m in members {
            emit_tagged(m, ops, out);
        }
        out.push(close.to_string());
        return;
    }
    match node {
        Node::Leaf(t) => match fraction_parts(t) {
            // A fraction is a quotient, so the tagged dialect spells it as one: a tagged sequence
            // declares its dialect by its bags, and `exp 1/2` declared none (review T7).
            Some((p, q)) => out.extend(["<mul>", p, "<div>", q, "</mul>"].map(String::from)),
            None => out.push(t.clone()),
        },
        Node::App(op, args) => {
            out.push(op.clone());
            for a in args {
                emit_tagged(a, ops, out);
            }
        }
        _ => unreachable!("bags are handled above"),
    }
}

/// The numerator and denominator of a one-token exact fraction (`1/2`, `-7/4`, `(-1/3)`).
fn fraction_parts(t: &str) -> Option<(&str, &str)> {
    let bare = t
        .strip_prefix('(')
        .and_then(|x| x.strip_suffix(')'))
        .unwrap_or(t);
    crate::utils::split_fraction(bare)
}

/// One bag. A LONE inverted member has no bag of its own (a bag holds two members or more): a
/// reciprocal is bagged with the unit, `<mul> 1 <div> x </mul>`, so every quotient of a tagged
/// sequence carries a bag and the sequence declares its dialect (review T7: `neg inv x1` declared
/// none, so `simplify` read it as explicit and answered explicit); a negation keeps its unary
/// spelling `neg x`, which both dialects read and answer alike.
#[allow(clippy::too_many_arguments)]
fn emit_bag(
    pos: &[&Node],
    neg: &[&Node],
    open: &str,
    section: &str,
    close: &str,
    unary: &str,
    ops: &Operators,
    out: &mut Vec<String>,
) {
    if pos.len() + neg.len() < 2 {
        // One member, inverted: there is no one-member bag.
        if unary == "inv" {
            out.extend(["<mul>", "1", "<div>"].map(String::from));
            emit_tagged(neg[0], ops, out);
            out.push(close.to_string());
        } else {
            out.push(unary.to_string());
            emit_tagged(neg[0], ops, out);
        }
        return;
    }
    out.push(open.to_string());
    for m in pos {
        emit_tagged(m, ops, out);
    }
    if !neg.is_empty() {
        out.push(section.to_string());
        for m in neg {
            emit_tagged(m, ops, out);
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
        // A lattice bag expands to the right-nested chain of its members, in their order.
        Node::Lat(op, members) => {
            for m in &members[..members.len() - 1] {
                out.push(op.clone());
                emit_prefix(m, out);
            }
            emit_prefix(&members[members.len() - 1], out);
        }
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
            t(&["<mul>", "1", "<div>", "x0", "</mul>"]),
        ] {
            let p = tagged_to_prefix(&row, &o).unwrap();
            assert_eq!(prefix_to_tagged(&p, &o).unwrap(), row, "via {p:?}");
        }
    }

    /// Every quotient of a tagged sequence carries a bag (review T7): a lone reciprocal is
    /// bagged with the unit and a fraction token as its quotient, a negation stays unary.
    #[test]
    fn every_reciprocal_and_fraction_is_bagged() {
        let o = ops();
        for (prefix, tagged) in [
            (
                t(&["inv", "x0"]),
                t(&["<mul>", "1", "<div>", "x0", "</mul>"]),
            ),
            (
                t(&["neg", "inv", "x1"]),
                t(&["neg", "<mul>", "1", "<div>", "x1", "</mul>"]),
            ),
            (
                t(&["sin", "1/2"]),
                t(&["sin", "<mul>", "1", "<div>", "2", "</mul>"]),
            ),
            (
                t(&["sin", "(-1/3)"]),
                t(&["sin", "<mul>", "-1", "<div>", "3", "</mul>"]),
            ),
            (t(&["neg", "x0"]), t(&["neg", "x0"])),
            (t(&["sin", "0.5"]), t(&["sin", "0.5"])),
        ] {
            assert_eq!(prefix_to_tagged(&prefix, &o).unwrap(), tagged, "{prefix:?}");
            let back = tagged_to_prefix(&tagged, &o).unwrap();
            assert_eq!(prefix_to_tagged(&back, &o).unwrap(), tagged, "via {back:?}");
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
