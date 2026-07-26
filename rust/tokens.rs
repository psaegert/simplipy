//! Token interning for the simplify hot path.
//!
//! The hot path used to shuttle `String` tokens through every tree clone, flatten, memo key and
//! hash lookup. This module replaces the token representation with an interned `Copy` id
//! ([`Tok`]) plus per-id PRECOMPUTED properties (arity, leaf value, sigil class, ...), so the
//! kernel compares/copies `u32`s and consults property vectors instead of re-running string
//! predicates. Outputs are BYTE-IDENTICAL: interning is injective (same string <-> same id within
//! a call), and every property is computed by the exact same function that previously ran on the
//! string at use time.
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
    /// `numeric::leaf_value` -- the shared literal table (numerics, np.pi/np.e, float("..."), (-1)).
    pub leaf_value: Option<f64>,
    /// `token.parse::<f64>().ok()` -- the sort `operand_key` numeric test (Python `float(token)`;
    /// DISTINCT from `leaf_value`, faithfully so).
    pub sort_float: Option<f64>,
    /// `utils::is_numeric_string` -- the `mask_elementary_literals` predicate.
    pub is_num_str: bool,
    /// First-character sigil class: b'_' / b'?' / b'!' / b'$' or 0. Placeholder classification is
    /// BY FIRST CHARACTER (Python keys on the sigil prefix, not the full `^[_?!$]\d+$` regex).
    pub sigil: u8,
    /// `rules::is_wildcard` (the full sigil+digits check) -- rule-compile classification.
    pub is_wildcard: bool,
    /// `Operators::is_commutative`.
    pub commutative: bool,
    /// `Operators::sort_resolve` (alias-canonicalized operator + compat arity), with the canonical
    /// operator interned. Every alias / compat key is interned into the table at build, so overlay
    /// tokens always resolve to None here (they cannot be aliases).
    pub sort_resolve: Option<(Tok, u8)>,
}

impl TokProps {
    /// Properties derivable from the string alone (the overlay path: never an operator/alias).
    fn leaf_only(s: &str) -> Self {
        Self {
            arity: None,
            leaf_value: crate::numeric::leaf_value(s),
            sort_float: s.parse::<f64>().ok(),
            is_num_str: crate::utils::is_numeric_string(s),
            sigil: sigil_of(s),
            is_wildcard: crate::rules::is_wildcard(s),
            commutative: false,
            sort_resolve: None,
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

/// Per-Engine interner: strings, ids, properties, plus the handful of distinguished tokens the
/// kernel compares against by id (`<constant>`, the connection-class operators/neutrals, ...).
#[derive(Debug, Default)]
pub struct TokenTable {
    strings: Vec<Box<str>>,
    ids: FxHashMap<Box<str>, Tok>,
    props: Vec<TokProps>,
    /// `<constant>` -- the abstract-constant token, keyed everywhere by this id.
    pub constant: Tok,
    /// `float("nan")` -- the NaN literal (numeric-line propagation).
    pub nan: Tok,
    pub plus: Tok,
    pub minus: Tok,
    pub star: Tok,
    pub slash: Tok,
    /// The connection-class neutrals `0` / `1` (cancel emission).
    pub zero: Tok,
    pub one: Tok,
    /// The class inverse prefixes `neg` / `inv` and the `pow` fallback head (cancel emission).
    pub neg: Tok,
    pub inv: Tok,
    pub pow: Tok,
    /// `operator_inverses["+"]` / `["*"]` interned -- cancel's parity-flip comparison
    /// (`ops.operator_inverse(operator_set0) == Some(operator)`), per connection class.
    pub cc_inverse_op: [Option<Tok>; 2],
}

impl TokenTable {
    /// Build the per-Engine table: distinguished tokens, the literal vocabulary, x0..x63, every
    /// operator (config order, deterministic ids), every alias, `**`, and the two class inverses.
    /// Rule LHS/RHS tokens are interned by `CompiledRules::compile` (which receives `&mut self`).
    pub fn build(operator_order: &[String], ops: &Operators) -> Self {
        let mut t = TokenTable::default();
        t.constant = t.intern("<constant>", ops);
        t.nan = t.intern("float(\"nan\")", ops);
        t.plus = t.intern("+", ops);
        t.minus = t.intern("-", ops);
        t.star = t.intern("*", ops);
        t.slash = t.intern("/", ops);
        t.zero = t.intern("0", ops);
        t.one = t.intern("1", ops);
        t.neg = t.intern("neg", ops);
        t.inv = t.intern("inv", ops);
        t.pow = t.intern("pow", ops);
        // Literal vocabulary the hot path (masking / folding / cancel emission) commonly meets.
        // Membership here is PERFORMANCE-only (a missing literal just lands in the overlay).
        for s in [
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
        let inv_add = ops.operator_inverse("+").map(str::to_string);
        let inv_mult = ops.operator_inverse("*").map(str::to_string);
        t.cc_inverse_op = [
            inv_add.map(|s| t.intern(&s, ops)),
            inv_mult.map(|s| t.intern(&s, ops)),
        ];
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
        // Placeholder first: `sort_resolve` may recursively intern the canonical operator, and the
        // canonical operator of an alias may be... this very token (self-resolution).
        self.props.push(TokProps::leaf_only(s));
        let sort_resolve = ops.sort_resolve(s).map(|(op, ar)| {
            let target = if op == s { id } else { self.intern(&op, ops) };
            (target, ar as u8)
        });
        self.props[id.0 as usize] = TokProps {
            arity: ops.arity_of(s),
            commutative: ops.is_commutative(s),
            sort_resolve,
            ..TokProps::leaf_only(s)
        };
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
    pub fn lookup(&self, s: &str) -> Option<Tok> {
        self.ids.get(s).copied()
    }

    /// Look up a whole token sequence; `None` if ANY token is absent from the table (in which
    /// case the sequence cannot equal any table-token key, e.g. a compiled-rule LHS).
    pub fn lookup_seq(&self, tokens: &[String]) -> Option<Vec<Tok>> {
        tokens.iter().map(|s| self.lookup(s)).collect()
    }

    #[inline]
    fn props(&self, t: Tok) -> &TokProps {
        &self.props[t.0 as usize]
    }

    /// `arity` for a TABLE token (rule compilation, where every token is interned here).
    #[inline]
    pub fn arity(&self, t: Tok) -> Option<u8> {
        self.props(t).arity
    }

    /// Per-id `rules::is_wildcard` for a TABLE token.
    #[inline]
    pub fn is_wildcard_tok(&self, t: Tok) -> bool {
        self.props(t).is_wildcard
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
    pub fn to_string(&self, t: Tok) -> String {
        if self.table.is_table_id(t) {
            self.table.resolve(t).to_string()
        } else {
            self.overlay.borrow().str_of(t).to_string()
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
    pub fn is_operator(&self, t: Tok) -> bool {
        self.props(t).arity.is_some()
    }

    #[inline]
    pub fn leaf_value(&self, t: Tok) -> Option<f64> {
        self.props(t).leaf_value
    }

    #[inline]
    pub fn sort_float(&self, t: Tok) -> Option<f64> {
        self.props(t).sort_float
    }

    #[inline]
    pub fn is_num_str(&self, t: Tok) -> bool {
        self.props(t).is_num_str
    }

    #[inline]
    pub fn sigil(&self, t: Tok) -> u8 {
        self.props(t).sigil
    }

    #[inline]
    pub fn commutative(&self, t: Tok) -> bool {
        self.props(t).commutative
    }

    #[inline]
    pub fn sort_resolve(&self, t: Tok) -> Option<(Tok, u8)> {
        self.props(t).sort_resolve
    }
}
