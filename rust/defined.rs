//! DEFINED FUNCTIONS (ruling 40, 2026-11-26): "We should always support at least that what is
//! written in the config. This can also be more."
//!
//! A config's `defined_functions:` section gives a function by a name, an arity and a
//! DEFINITION over the engine's own operators, written in infix with the arguments `_0`, `_1`,
//! ...:
//!
//! ```yaml
//! defined_functions:
//!   sqrt: {arity: 1, definition: "rootn(_0, 2)"}
//!   cot:  {arity: 1, definition: "1/tan(_0)"}
//!   acot: {arity: 1, definition: "atan(1/_0)"}
//! ```
//!
//! A defined function is SUPPORTED, not mined: the readers keep its head as written (it is a
//! function of the language, so `read_infix`, the conversions and `is_valid` read it like a
//! declared one), every entry into the canonical state and every evaluation EXPANDS it into its
//! definition first ([`expand_defined`]), so the canonical state, the answer, the price and the
//! value are the definition's, and nothing that enumerates or samples the config's operators
//! (the miner, the monitor, the rule files' fingerprints) sees it: the operator table the
//! kernel, the miner and the measure read ([`Operators::arity`]) does not hold it.

use rustc_hash::FxHashMap;
use serde::Deserialize;

use crate::operators::Operators;

/// One token of a definition: an operator, a number or a constant written in it, or an
/// argument (`_k`, by position).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BodyToken {
    Token(String),
    Argument(usize),
}

/// A defined function: its arity and its definition, in explicit prefix with its arguments.
#[derive(Debug, Clone)]
pub struct DefinedFunction {
    pub arity: u8,
    /// The definition as written in the config (infix), kept for messages.
    pub definition: String,
    pub body: Vec<BodyToken>,
}

/// One entry of the config section.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DefinedSpec {
    arity: u8,
    definition: String,
}

/// Read a config's `defined_functions:` section against the operator table `ops` (which holds
/// no defined function yet): every definition is read by the strict infix reader, so it may
/// call only the engine's operators -- declared, aliased or core -- and never another defined
/// function, and it may hold numbers, `np.pi`/`np.e` and its arguments `_0` to `_{arity-1}`,
/// each at least once. A name must be a fresh identifier: no operator, alias, reserved constant
/// name or argument spelling.
pub fn read_section(
    section: serde_yaml_ng::Mapping,
    ops: &Operators,
) -> Result<Vec<(String, DefinedFunction)>, String> {
    let mut out = Vec::with_capacity(section.len());
    for (key, value) in section {
        let name = key
            .as_str()
            .ok_or("defined_functions: a function's name must be a string")?
            .to_string();
        let spec: DefinedSpec = serde_yaml_ng::from_value(value)
            .map_err(|e| format!("defined_functions: {name:?}: {e}"))?;
        check_name(&name, ops)?;
        if spec.arity == 0 {
            return Err(format!(
                "defined_functions: {name:?} declares arity 0; a defined function takes at \
                 least one argument"
            ));
        }
        let body = read_definition(&name, spec.arity, &spec.definition, ops)?;
        if out
            .iter()
            .any(|(n, _): &(String, DefinedFunction)| *n == name)
        {
            return Err(format!("defined_functions: {name:?} is defined twice"));
        }
        out.push((
            name,
            DefinedFunction {
                arity: spec.arity,
                definition: spec.definition,
                body,
            },
        ));
    }
    Ok(out)
}

/// A defined function's name: an identifier starting with a letter (so never an argument or a
/// sigil), not one of the engine's operators or aliases, and not a name the infix language
/// reserves for a constant.
fn check_name(name: &str, ops: &Operators) -> Result<(), String> {
    let identifier = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !identifier {
        return Err(format!(
            "defined_functions: {name:?} is not a function name (letters, digits and '_', \
             starting with a letter)"
        ));
    }
    if ops.arity_of(name).is_some() || ops.operator_aliases.contains_key(name) {
        return Err(format!(
            "defined_functions: {name:?} is already an operator or an alias of this config"
        ));
    }
    if matches!(name, "pi" | "e" | "inf" | "nan") {
        return Err(format!(
            "defined_functions: {name:?} is the infix spelling of a constant"
        ));
    }
    Ok(())
}

/// The argument a token spells (`_k`), if any.
fn argument_index(token: &str) -> Option<usize> {
    let digits = token.strip_prefix('_')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Read and check one definition (see [`read_section`]).
fn read_definition(
    name: &str,
    arity: u8,
    definition: &str,
    ops: &Operators,
) -> Result<Vec<BodyToken>, String> {
    let tokens = crate::convert::parse(definition, ops, true, false, true)
        .map_err(|e| format!("defined_functions: {name:?}: the definition {definition:?}: {e}"))?;
    if tokens.is_empty() {
        return Err(format!(
            "defined_functions: {name:?}: the definition is empty"
        ));
    }
    let mut used = vec![false; usize::from(arity)];
    let mut body = Vec::with_capacity(tokens.len());
    // Well-formedness by the arity scan, from the right: every operator finds its operands and
    // one expression remains.
    let mut depth = 0usize;
    for token in tokens.iter().rev() {
        if let Some(a) = ops.arity_of(token) {
            let a = usize::from(a);
            if depth < a {
                return Err(format!(
                    "defined_functions: {name:?}: the definition {definition:?} is malformed"
                ));
            }
            depth -= a;
        } else if let Some(k) = argument_index(token) {
            if k >= usize::from(arity) {
                return Err(format!(
                    "defined_functions: {name:?} has arity {arity}, but its definition uses \
                     the argument _{k}"
                ));
            }
            used[k] = true;
        } else if crate::numeric::leaf_value(token).is_none()
            && !crate::utils::is_numeric_string(token)
        {
            return Err(format!(
                "defined_functions: {name:?}: the definition {definition:?} holds {token:?}, \
                 which is neither an operator of this config, a number, a constant nor one of \
                 its arguments _0 to _{}",
                arity - 1
            ));
        }
        depth += 1;
    }
    if depth != 1 {
        return Err(format!(
            "defined_functions: {name:?}: the definition {definition:?} is malformed"
        ));
    }
    if let Some(k) = used.iter().position(|u| !u) {
        return Err(format!(
            "defined_functions: {name:?} has arity {arity}, but its definition never uses the \
             argument _{k}"
        ));
    }
    for token in tokens {
        body.push(match argument_index(&token) {
            Some(k) => BodyToken::Argument(k),
            None => BodyToken::Token(token),
        });
    }
    Ok(body)
}

/// Whether a token opens or closes a bag, or opens an inverted section, of the tagged dialect.
fn tagged_marker(token: &str) -> Option<TagRole> {
    match token {
        "<add>" | "<mul>" | "<max>" | "<min>" => Some(TagRole::Open),
        "</add>" | "</mul>" | "</max>" | "</min>" => Some(TagRole::Close),
        "<sub>" | "<div>" => Some(TagRole::Section),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TagRole {
    Open,
    Close,
    Section,
}

/// The index after the subtree that starts at each position of a well-formed sequence, in
/// either dialect; `None` when the sequence is not one well-formed expression.
fn subtree_ends(tokens: &[String], ops: &Operators) -> Option<Vec<usize>> {
    #[derive(Clone, Copy)]
    enum Item {
        Tree(usize),
        Close(usize),
        Section,
    }
    let n = tokens.len();
    let mut end = vec![0usize; n];
    let mut stack: Vec<Item> = Vec::with_capacity(n);
    for i in (0..n).rev() {
        let t = tokens[i].as_str();
        match tagged_marker(t) {
            Some(TagRole::Close) => stack.push(Item::Close(i)),
            Some(TagRole::Section) => stack.push(Item::Section),
            Some(TagRole::Open) => loop {
                match stack.pop()? {
                    Item::Close(c) => {
                        end[i] = c + 1;
                        stack.push(Item::Tree(i));
                        break;
                    }
                    Item::Tree(_) | Item::Section => {}
                }
            },
            None => {
                let arity = usize::from(ops.syntax_arity(t).unwrap_or(0));
                let mut last = i + 1;
                for _ in 0..arity {
                    match stack.pop()? {
                        Item::Tree(j) => last = end[j],
                        Item::Close(_) | Item::Section => return None,
                    }
                }
                end[i] = last;
                stack.push(Item::Tree(i));
            }
        }
    }
    match stack.as_slice() {
        [Item::Tree(0)] if end[0] == n => Some(end),
        _ => None,
    }
}

/// EXPAND every defined function of `tokens` (explicit prefix or tagged) into its definition,
/// the arguments' own definitions expanded too: `sqrt cot x0` is `rootn / 1 tan x0 2`. The
/// identity, as the same vector, when the engine defines nothing or the sequence holds no
/// defined head; a sequence that is not one well-formed expression is returned unchanged, for
/// its consumer's own validation to refuse.
pub fn expand_defined(tokens: &[String], ops: &Operators) -> Vec<String> {
    if ops.defined.is_empty() || !tokens.iter().any(|t| ops.defined.contains_key(t.as_str())) {
        return tokens.to_vec();
    }
    let Some(end) = subtree_ends(tokens, ops) else {
        return tokens.to_vec();
    };
    enum Work<'a> {
        /// Copy `tokens[pos..stop]`, expanding the defined heads met.
        Span { pos: usize, stop: usize },
        /// Write a definition from body position `at`, its arguments the given spans.
        Body {
            body: &'a [BodyToken],
            at: usize,
            args: Vec<(usize, usize)>,
        },
    }
    let mut out = Vec::with_capacity(tokens.len() + 8);
    let mut work = vec![Work::Span {
        pos: 0,
        stop: tokens.len(),
    }];
    while let Some(item) = work.pop() {
        match item {
            Work::Span { mut pos, stop } => {
                while pos < stop {
                    let t = tokens[pos].as_str();
                    if let Some(def) = ops.defined.get(t) {
                        let mut args = Vec::with_capacity(usize::from(def.arity));
                        let mut child = pos + 1;
                        for _ in 0..def.arity {
                            args.push((child, end[child]));
                            child = end[child];
                        }
                        work.push(Work::Span {
                            pos: end[pos],
                            stop,
                        });
                        work.push(Work::Body {
                            body: &def.body,
                            at: 0,
                            args,
                        });
                        break;
                    }
                    out.push(tokens[pos].clone());
                    pos += 1;
                }
            }
            Work::Body { body, mut at, args } => {
                while at < body.len() {
                    match &body[at] {
                        BodyToken::Token(t) => out.push(t.clone()),
                        BodyToken::Argument(k) => {
                            let (pos, stop) = args[*k];
                            work.push(Work::Body {
                                body,
                                at: at + 1,
                                args,
                            });
                            work.push(Work::Span { pos, stop });
                            break;
                        }
                    }
                    at += 1;
                }
            }
        }
    }
    out
}

/// The section as the engine holds it: name -> function.
pub type DefinedTable = FxHashMap<String, DefinedFunction>;

#[cfg(test)]
mod tests {
    use super::*;

    /// acj2-4's operators with `section` in place of the config's own defined functions (the
    /// last section of the file).
    fn engine(section: &str) -> Result<crate::Engine, String> {
        let shipped = include_str!("../assets/engines/acj2-4/config.yaml");
        let own = shipped
            .find("\ndefined_functions:")
            .expect("acj2-4 defines its functions last");
        let cfg = format!("{}\n{section}", &shipped[..own]);
        crate::Engine::from_strs(&cfg, "[]").map_err(|e| e.to_string())
    }

    #[test]
    fn the_shipped_config_defines_the_three() {
        let shipped = include_str!("../assets/engines/acj2-4/config.yaml");
        let e = crate::Engine::from_strs(shipped, "[]").unwrap();
        let ops = e.operators_ref();
        let mut names: Vec<&str> = ops.defined.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(names, ["acot", "cot", "sqrt"]);
        assert_eq!(expand_defined(&t("acot x0"), ops), t("atan / 1 x0"));
    }

    fn t(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    const SECTION: &str =
        "defined_functions:\n  sqrt: {arity: 1, definition: \"rootn(_0, 2)\"}\n  \
                           cot: {arity: 1, definition: \"1/tan(_0)\"}\n  \
                           acot: {arity: 1, definition: \"atan(1/_0)\"}\n";

    #[test]
    fn expansion_substitutes_arguments_in_both_dialects() {
        let e = engine(SECTION).unwrap();
        let ops = e.operators_ref();
        assert_eq!(expand_defined(&t("sqrt x0"), ops), t("rootn x0 2"));
        assert_eq!(expand_defined(&t("cot + x0 x1"), ops), t("/ 1 tan + x0 x1"));
        assert_eq!(expand_defined(&t("acot x0"), ops), t("atan / 1 x0"));
        assert_eq!(
            expand_defined(&t("sqrt cot acot x0"), ops),
            t("rootn / 1 tan atan / 1 x0 2")
        );
        assert_eq!(
            expand_defined(&t("* sqrt x0 sqrt x1"), ops),
            t("* rootn x0 2 rootn x1 2")
        );
        assert_eq!(
            expand_defined(
                &t("<add> x0 sqrt <mul> x1 x2 </mul> <sub> cot x3 </add>"),
                ops
            ),
            t("<add> x0 rootn <mul> x1 x2 </mul> 2 <sub> / 1 tan x3 </add>")
        );
        // nothing defined in it, or malformed: unchanged
        assert_eq!(expand_defined(&t("+ x0 x1"), ops), t("+ x0 x1"));
        assert_eq!(expand_defined(&t("sqrt"), ops), t("sqrt"));
        assert_eq!(expand_defined(&t("sqrt x0 x1"), ops), t("sqrt x0 x1"));
    }

    #[test]
    fn a_section_is_checked_when_the_engine_is_built() {
        for (section, why) in [
            (
                "defined_functions:\n  sin: {arity: 1, definition: \"cos(_0)\"}",
                "already an operator",
            ),
            (
                "defined_functions:\n  Abs2: {arity: 1, definition: \"foo(_0)\"}",
                "foo",
            ),
            (
                "defined_functions:\n  f: {arity: 1, definition: \"_0 + x1\"}",
                "\"x1\"",
            ),
            (
                "defined_functions:\n  f: {arity: 2, definition: \"tan(_0)\"}",
                "never uses the argument _1",
            ),
            (
                "defined_functions:\n  f: {arity: 1, definition: \"_1\"}",
                "uses the argument _1",
            ),
            (
                "defined_functions:\n  f: {arity: 0, definition: \"2\"}",
                "arity 0",
            ),
            (
                "defined_functions:\n  _f: {arity: 1, definition: \"_0\"}",
                "not a function name",
            ),
            (
                "defined_functions:\n  pi: {arity: 1, definition: \"_0\"}",
                "constant",
            ),
            (
                "defined_functions:\n  f: {arity: 1, definition: \"sqrt(_0)\"}",
                "sqrt",
            ),
            (
                "defined_functions:\n  f: {arity: 1, defn: \"_0\"}",
                "unknown field",
            ),
        ] {
            let err = engine(section).err().unwrap_or_default();
            assert!(err.contains(why), "{section}: {err}");
        }
    }

    #[test]
    fn the_readers_keep_the_head() {
        let e = engine(SECTION).unwrap();
        assert_eq!(
            e.parse("sqrt(x0)^2", true, false, true).unwrap(),
            t("pow sqrt x0 2")
        );
        assert_eq!(
            e.parse("cot x0 + 1", true, false, true).unwrap(),
            t("+ cot x0 1")
        );
        assert!(e.parse("sqrt(x0, x1)", true, false, true).is_err());
        assert!(e.parse("foo(x0)", true, false, true).is_err());
        assert_eq!(
            e.prefix_to_infix(&t("+ acot x0 sqrt x1"), crate::convert::Power::Func, false)
                .unwrap(),
            "acot(x0) + sqrt(x1)"
        );
        assert!(e.is_valid(&t("* sqrt x0 cot x1")));
        assert!(!e.is_valid(&t("sqrt x0 x1")));
        assert!(e.check_form(&t("<mul> sqrt x0 cot x1 </mul>")).is_ok());
    }
}
