//! Token interning for the simplify hot path.
//!
//! The hot path used to shuttle `String` tokens through every tree clone, flatten, memo key and
//! hash lookup. This module replaces the token representation with an interned `Copy` id
//! ([`Tok`]) plus per-id PRECOMPUTED properties (arity, sigil class), so the
//! kernel compares/copies `u32`s and consults property vectors instead of re-running string
//! predicates. Interning is injective: same string <-> same id within a call.
//!
//! ## Two-level store: per-Engine table + per-call overlay (NO locks)
//! * [`TokenTable`] -- built once in `Engine::from_strs` (operators, aliases, `**`, every rule
//!   LHS/RHS token, `<constant>`, the literal vocabulary, x0..x63) and IMMUTABLE afterwards
//!   (only `&mut Engine` entry points -- `set_rules` -- may extend it; ids are append-only and
//!   never re-assigned, so existing `Tok`s stay valid). Parallel Python callers share the Engine
//!   and read the table without synchronization.
//! * [`TokenOverlay`] -- a per-CALL append-only extension for tokens not in the table (arbitrary
//!   numeric literals, higher-index variables, folded results). Ids start at `table.len()`.
//!   Overlay ids are only meaningful within their call, so they must never leak into per-Engine
//!   state (the `!`-certificate cache guards on [`TokenTable::is_table_id`]).
//! * [`TokenView`] -- the `(table, &RefCell<overlay>)` pair threaded through the kernel:
//!   `intern`, `to_string`, and all property lookups.

use std::cell::RefCell;
use std::cmp::Ordering;

use rustc_hash::FxHashMap;

use crate::operators::Operators;

/// An interned token id. `Copy` + 4 bytes: the whole point. (`Default` only serves the
/// `TokenTable` builder's pre-init placeholder ids.)
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct Tok(pub u32);

/// Precomputed per-token properties -- each field is the result of the EXACT predicate the hot
/// path previously ran on the token string at use time (same functions, run once at intern).
#[derive(Debug, Clone, Copy)]
pub struct TokProps {
    /// `Operators::arity_of` (None = leaf). Every operator name is interned into the TABLE at
    /// build, so overlay tokens are never operators (arity None by construction).
    pub arity: Option<u8>,
    /// First-character sigil class: b'_' / b'?' / b'!' / b'$' or 0. Placeholder classification is
    /// BY FIRST CHARACTER (Python keys on the sigil prefix, not the full `^[_?!$]\d+$` regex).
    pub sigil: u8,
    /// Is the token a named constant ([`is_named_constant`])?
    pub named: bool,
}

impl TokProps {
    /// Properties derivable from the string alone (the overlay path: never an operator/alias).
    fn leaf_only(s: &str) -> Self {
        Self {
            arity: None,
            sigil: sigil_of(s),
            named: is_named_constant(s),
        }
    }
}

#[inline]
fn sigil_of(s: &str) -> u8 {
    match s.as_bytes().first() {
        Some(b @ (b'_' | b'?' | b'!' | b'$')) => *b,
        _ => 0,
    }
}

/// Is `s` a NAMED CONSTANT, `<constant_NAME>` with NAME alphanumeric (`[A-Za-z0-9]+`)? One
/// unknown finite real per name, distinct names independent; `<constant>` is the anonymous one.
/// A named constant is a parameter, never a data variable: the engine reads a name used more
/// than once as it reads a variable (every answer holds for every value), and a name used once
/// as exactly a `<constant>`, keeping its name ([`TokenView::is_free_name`]).
pub fn is_named_constant(s: &str) -> bool {
    s.strip_prefix("<constant_")
        .and_then(|r| r.strip_suffix('>'))
        .is_some_and(|name| !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric()))
}

/// Per-Engine interner: strings, ids, properties.
///
/// (The deleted binary kernel's distinguished-token FIELDS -- `constant`/`nan`/`plus`/
/// .../`cc_inverse_op`, its id-comparison anchors -- are gone:
/// zero consumers since the 0.12.0 kernel deletion. The AC core interns through
/// `TokenView::intern` and compares by string, never against stored ids. Their eager
/// `intern` CALLS survive at the FRONT of [`TokenTable::build`]'s literal list, in the
/// ORIGINAL order, so table ids are unchanged by the removal.)
#[derive(Debug, Default)]
pub struct TokenTable {
    strings: Vec<Box<str>>,
    ids: FxHashMap<Box<str>, Tok>,
    props: Vec<TokProps>,
}

impl TokenTable {
    /// Build the per-Engine table: the eagerly-interned common tokens, the literal vocabulary,
    /// x0..x63, every operator (config order, deterministic ids), every alias, and `**`.
    /// Rule LHS/RHS tokens are interned by `CompiledRules::compile` (which receives `&mut self`).
    pub fn build(operator_order: &[String], ops: &Operators) -> Self {
        let mut t = TokenTable::default();
        // Eager membership is PERFORMANCE-only (a missing literal just lands in the
        // per-call overlay) -- except `rootn`, see below. The first eleven entries are
        // the deleted kernel's distinguished tokens, kept FIRST in their historical
        // order so removing the fields left every table id unchanged.
        for s in [
            "<constant>",
            "float(\"nan\")",
            "+",
            "-",
            "*",
            "/",
            "0",
            "1",
            "neg",
            "inv",
            "pow",
            "float(\"inf\")",
            "float(\"-inf\")",
            "np.pi",
            "np.e",
            "(-1)",
            "(-0.5)",
            "(-2)",
            "0.5",
            "2.0",
            "3.0",
            "**",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "-1",
            "-2",
            // The AC core's general signed-root operator (IEEE rootn): rule translation
            // desugars pow1_3/pow1_5 into it and stores the resulting expressions
            // per-Engine, so the token must be a TABLE id, never an overlay id.
            "rootn",
        ] {
            t.intern(s, ops);
        }
        for i in 0..64 {
            t.intern(&format!("x{i}"), ops);
        }
        // Operators in config order (deterministic ids), then their aliases (sorted for
        // determinism -- the per-op alias lists are consumed by `Operators::from_specs`).
        for name in operator_order {
            t.intern(name, ops);
        }
        let mut aliases: Vec<&String> = ops.operator_aliases.keys().collect();
        aliases.sort();
        for a in aliases {
            t.intern(a, ops);
        }
        // `f64`'s written negative zero (`ac::expr::NEG_ZERO`): the constructors build it, and
        // a rule holding one is stored per-Engine, so it is a TABLE id (appended: every other
        // id is unchanged).
        t.intern(crate::ac::expr::NEG_ZERO, ops);
        // (The former cc_inverse_op tail interned the config's +/* inverse names here.
        // For every real config that is a no-op on table contents -- `-` and `/` are
        // eagerly interned above, and an inverse naming another OPERATOR is interned
        // via operator_order; only a malformed non-operator inverse name ever added a
        // token, and such a string now simply lands in per-call overlays, where the
        // string-based comparisons treat it identically. Removed with the dead
        // field it fed.)
        t
    }

    /// Intern into the TABLE (build / `&mut Engine` paths only). Append-only: ids are stable.
    pub fn intern(&mut self, s: &str, ops: &Operators) -> Tok {
        if let Some(&t) = self.ids.get(s) {
            return t;
        }
        let id = Tok(self.strings.len() as u32);
        self.strings.push(s.into());
        self.ids.insert(s.into(), id);
        self.props.push(TokProps {
            arity: ops.arity_of(s),
            ..TokProps::leaf_only(s)
        });
        id
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Is this a TABLE id (vs a per-call overlay id)? The guard that keeps overlay ids out of
    /// per-Engine state (the `!`-certificate cache).
    #[inline]
    pub fn is_table_id(&self, t: Tok) -> bool {
        (t.0 as usize) < self.strings.len()
    }

    /// Resolve a TABLE token to its string. Panics on an overlay id (per-Engine state never
    /// holds one -- see [`TokenTable::is_table_id`]).
    #[inline]
    pub fn resolve(&self, t: Tok) -> &str {
        &self.strings[t.0 as usize]
    }

    #[inline]
    fn props(&self, t: Tok) -> &TokProps {
        &self.props[t.0 as usize]
    }
}

/// Per-call overlay for tokens not in the table. Append-only; ids start at `table.len()`.
#[derive(Debug)]
pub struct TokenOverlay {
    base: u32,
    strings: Vec<Box<str>>,
    ids: FxHashMap<Box<str>, Tok>,
    props: Vec<TokProps>,
    /// The call's FREE NAMES: the named constants the expression of the call uses once
    /// ([`TokenView::is_free_name`]). Empty unless the call's entry classifies its input.
    free_names: Vec<Tok>,
}

impl TokenOverlay {
    pub fn new(table_len: usize) -> Self {
        Self {
            base: table_len as u32,
            strings: Vec::new(),
            ids: FxHashMap::default(),
            props: Vec::new(),
            free_names: Vec::new(),
        }
    }

    fn intern(&mut self, s: &str) -> Tok {
        if let Some(&t) = self.ids.get(s) {
            return t;
        }
        let id = Tok(self.base + self.strings.len() as u32);
        self.strings.push(s.into());
        self.ids.insert(s.into(), id);
        self.props.push(TokProps::leaf_only(s));
        id
    }

    #[inline]
    fn str_of(&self, t: Tok) -> &str {
        &self.strings[(t.0 - self.base) as usize]
    }

    #[inline]
    fn props(&self, t: Tok) -> &TokProps {
        &self.props[(t.0 - self.base) as usize]
    }
}

/// The table+overlay pair the kernel threads through one call. NO locks: the table is shared
/// read-only, the overlay is per-call behind a plain (non-atomic) `RefCell`.
#[derive(Clone, Copy)]
pub struct TokenView<'a> {
    pub table: &'a TokenTable,
    pub overlay: &'a RefCell<TokenOverlay>,
}

impl<'a> TokenView<'a> {
    pub fn new(table: &'a TokenTable, overlay: &'a RefCell<TokenOverlay>) -> Self {
        Self { table, overlay }
    }

    /// Intern a token: table hit (common) or per-call overlay append. Injective within the call.
    pub fn intern(&self, s: &str) -> Tok {
        if let Some(&t) = self.table.ids.get(s) {
            return t;
        }
        self.overlay.borrow_mut().intern(s)
    }

    /// Resolve to an owned `String` (boundary conversions only -- FFI exit, `finite_ae`,
    /// `evaluate_constant_subtree`).
    pub fn resolve_owned(&self, t: Tok) -> String {
        if self.table.is_table_id(t) {
            self.table.resolve(t).to_string()
        } else {
            self.overlay.borrow().str_of(t).to_string()
        }
    }

    /// Name equality WITHOUT resolving to an owned string (constructor hot paths --
    /// `fun`'s head dispatch must not allocate per construction).
    pub fn tok_is(&self, t: Tok, s: &str) -> bool {
        if self.table.is_table_id(t) {
            self.table.resolve(t) == s
        } else {
            self.overlay.borrow().str_of(t) == s
        }
    }

    /// Run a closure over a token's string WITHOUT resolving to an owned string
    /// (the mu Leaf arm sits on the ordering hot path -- pricing a numeric-string
    /// leaf must not allocate per comparison).
    pub fn with_str<R>(&self, t: Tok, f: impl FnOnce(&str) -> R) -> R {
        if self.table.is_table_id(t) {
            f(self.table.resolve(t))
        } else {
            f(self.overlay.borrow().str_of(t))
        }
    }

    /// String-order comparison of two tokens WITHOUT resolving to owned strings (the sort
    /// `operand_key` Var/op comparisons -- Python compares the token strings).
    pub fn str_cmp(&self, a: Tok, b: Tok) -> Ordering {
        if a == b {
            return Ordering::Equal;
        }
        match (self.table.is_table_id(a), self.table.is_table_id(b)) {
            (true, true) => self.table.resolve(a).cmp(self.table.resolve(b)),
            (true, false) => self.table.resolve(a).cmp(self.overlay.borrow().str_of(b)),
            (false, true) => self.overlay.borrow().str_of(a).cmp(self.table.resolve(b)),
            (false, false) => {
                let ov = self.overlay.borrow();
                ov.str_of(a).cmp(ov.str_of(b))
            }
        }
    }

    #[inline]
    fn props(&self, t: Tok) -> TokProps {
        if self.table.is_table_id(t) {
            *self.table.props(t)
        } else {
            *self.overlay.borrow().props(t)
        }
    }

    #[inline]
    pub fn arity(&self, t: Tok) -> Option<u8> {
        self.props(t).arity
    }

    #[inline]
    pub fn sigil(&self, t: Tok) -> u8 {
        self.props(t).sigil
    }

    /// Is `t` a FREE NAME of this call: a named constant ([`is_named_constant`]) the call's
    /// expression uses exactly once? Such a name is exactly a `<constant>` (one unknown
    /// finite real nothing else refers to), so the parse reads it as the fitted constant
    /// `Ex::Const` carrying its name, which the constructors may absorb as they absorb any
    /// fitted constant, keeping the name (`sin(<constant_a>)` is `<constant_a>`). A name used
    /// more than once is a leaf, read as a variable: absorbing it would cut the tie between
    /// its uses. Table tokens are never free.
    pub fn is_free_name(&self, t: Tok) -> bool {
        !self.table.is_table_id(t) && self.overlay.borrow().free_names.contains(&t)
    }

    /// Does this call read any free name ([`Self::is_free_name`])?
    pub fn has_free_names(&self) -> bool {
        !self.overlay.borrow().free_names.is_empty()
    }

    /// Is `t` a named constant ([`is_named_constant`])?
    #[inline]
    pub fn is_named(&self, t: Tok) -> bool {
        self.props(t).named
    }

    /// CLASSIFY the named constants of one expression for this call (`tokens`, interned through
    /// this view): each name used exactly once becomes a free name ([`Self::is_free_name`]).
    /// Called once, by an entry point, on the whole expression it reads: a part of an expression
    /// does not say how often the whole uses a name.
    pub fn classify_names(&self, tokens: &[Tok]) {
        let mut counts: Vec<(Tok, usize)> = Vec::new();
        for &t in tokens {
            if self.table.is_table_id(t) || !self.is_named(t) {
                continue;
            }
            match counts.iter_mut().find(|(u, _)| *u == t) {
                Some((_, n)) => *n += 1,
                None => counts.push((t, 1)),
            }
        }
        let free: Vec<Tok> = counts
            .into_iter()
            .filter(|&(_, n)| n == 1)
            .map(|(t, _)| t)
            .collect();
        self.overlay.borrow_mut().free_names = free;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_constants_are_alphanumeric_names() {
        for s in [
            "<constant_a>",
            "<constant_k12>",
            "<constant_1>",
            "<constant_X1y>",
        ] {
            assert!(is_named_constant(s), "{s}");
        }
        for s in [
            "<constant>",
            "<constant_>",
            "<constant_a_b>",
            "<constant_a-b>",
            "<constant_a",
            "constant_a",
            "<constant_\u{e9}>",
            "x1",
        ] {
            assert!(!is_named_constant(s), "{s}");
        }
    }

    #[test]
    fn a_name_used_once_is_free() {
        let ops = Operators::from_specs(Vec::new(), FxHashMap::default());
        let table = TokenTable::build(&[], &ops);
        let overlay = RefCell::new(TokenOverlay::new(table.len()));
        let view = TokenView::new(&table, &overlay);
        let toks: Vec<Tok> = [
            "<constant_a>",
            "<constant_b>",
            "<constant_a>",
            "x1",
            "<constant>",
        ]
        .iter()
        .map(|s| view.intern(s))
        .collect();
        view.classify_names(&toks);
        assert!(!view.is_free_name(toks[0]), "a is used twice");
        assert!(view.is_free_name(toks[1]), "b is used once");
        assert!(!view.is_free_name(toks[3]) && !view.is_free_name(toks[4]));
        assert!(view.is_named(toks[0]) && !view.is_named(toks[4]));
    }
}
