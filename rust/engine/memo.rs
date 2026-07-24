//! The kernel's memo state: the per-Engine generational `!`-certificate cache ([`BangCache`])
//! and the per-simplify-call memo context ([`SimplifyCtx`]).

use std::cell::RefCell;

use rustc_hash::FxHashMap;

use crate::tokens::{Tok, TokenOverlay};

/// Generational memo for the `!`-certificate. Two generations of <= GEN_CAP entries each:
/// lookups check `cur` then `prev` (promoting prev-hits), inserts go to `cur`, and a full
/// `cur` becomes `prev` (the old `prev` is dropped). Hot entries survive generations; memory
/// stays bounded; memoization never stops.
///
/// Keys are `Vec<Tok>` of TABLE ids only -- per-call overlay ids must never enter this
/// per-Engine map (`bang_certified` guards with `TokenTable::is_table_id`). Table ids are stable
/// for the Engine's lifetime (append-only), so keys stay valid across `set_rules`.
pub(super) struct BangCache {
    cur: rustc_hash::FxHashMap<Vec<Tok>, bool>,
    prev: rustc_hash::FxHashMap<Vec<Tok>, bool>,
}

const BANG_GEN_CAP: usize = 100_000;

impl BangCache {
    pub(super) fn new() -> Self {
        Self {
            cur: FxHashMap::default(),
            prev: FxHashMap::default(),
        }
    }

    pub(super) fn get_promoting(&mut self, key: &[Tok]) -> Option<bool> {
        if let Some(&b) = self.cur.get(key) {
            return Some(b);
        }
        if let Some(&b) = self.prev.get(key) {
            self.insert(key.to_vec(), b);
            return Some(b);
        }
        None
    }

    pub(super) fn insert(&mut self, key: Vec<Tok>, b: bool) {
        self.cur.insert(key, b);
        if self.cur.len() >= BANG_GEN_CAP {
            self.prev = std::mem::take(&mut self.cur);
        }
    }
}

/// Per-simplify-call memo context. Pure-function memoization, valid because both passes are
/// deterministic for a fixed engine + max_pattern_length (fixed within one call):
/// - `rules_memo`: a whole-pass input->output token map. The tree search reaches the same node
///   by many paths, so the rules pass for a node is computed once no matter how often it recurs.
///   The cancel enumeration needs no memo: the search's `visited` set already dedupes its nodes.
/// - `normal_forms`: flattened subtrees the rule walk returned UNCHANGED (no fire anywhere
///   inside; guarded against the faithful-mode all-`<constant>`-operand entry collapse).
///   Reaching an identical subtree in a later iteration (or a repeated substructure in the
///   same pass) returns immediately -- only the changed spine is re-scanned.
/// - `cert_scratch`: the per-call certificate scratch (see `bang_certified`).
///
/// The ctx also owns the per-call [`TokenOverlay`] (tokens outside the per-Engine table);
/// all memo keys are `Vec<Tok>`, consistent within the call by interning injectivity.
pub(super) struct SimplifyCtx {
    pub(super) overlay: RefCell<TokenOverlay>,
    pub(super) cert_scratch: RefCell<FxHashMap<Vec<Tok>, bool>>,
    pub(super) rules_memo: RefCell<FxHashMap<Vec<Tok>, Vec<Tok>>>,
    pub(super) normal_forms: RefCell<rustc_hash::FxHashSet<Vec<Tok>>>,
    /// AGGRESSIVE apply-time mode: bind every placeholder as `_` and skip the `!` certificate
    /// (see `matcher::match_pattern_with_cert`). Set once per call at the simplify entry.
    pub(super) wildcard_all: bool,
}

impl SimplifyCtx {
    pub(super) fn new(table_len: usize, wildcard_all: bool) -> Self {
        Self {
            overlay: RefCell::new(TokenOverlay::new(table_len)),
            cert_scratch: RefCell::new(FxHashMap::default()),
            rules_memo: RefCell::new(FxHashMap::default()),
            normal_forms: RefCell::new(rustc_hash::FxHashSet::default()),
            wildcard_all,
        }
    }
}
