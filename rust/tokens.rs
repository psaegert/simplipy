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
}

impl TokProps {
    /// Properties derivable from the string alone (the overlay path: never an operator/alias).
    fn leaf_only(s: &str) -> Self {
        Self {
            arity: None,
            sigil: sigil_of(s),
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

/// Per-Engine interner: strings, ids, properties.
///
/// (The deleted binary kernel's distinguished-token FIELDS -- `constant`/`nan`/`plus`/
/// .../`cc_inverse_op`, its id-comparison anchors -- were removed in D3, 2026-08-05:
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
        // order so the D3 field removal left every table id unchanged.
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
        // (The former cc_inverse_op tail interned the config's +/* inverse names here.
        // For every real config that is a no-op on table contents -- `-` and `/` are
        // eagerly interned above, and an inverse naming another OPERATOR is interned
        // via operator_order; only a malformed non-operator inverse name ever added a
        // token, and such a string now simply lands in per-call overlays, where the
        // string-based comparisons treat it identically. Removed in D3 with the dead
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
}

impl TokenOverlay {
    pub fn new(table_len: usize) -> Self {
        Self {
            base: table_len as u32,
            strings: Vec::new(),
            ids: FxHashMap::default(),
            props: Vec::new(),
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
}
