//! The kernel's memo state: the per-Engine generational `!`-certificate cache ([`BangCache`])
//! and the per-simplify-call memo context ([`SimplifyCtx`]).

use std::cell::RefCell;

use rustc_hash::FxHashMap;

use crate::tokens::{Tok, TokenOverlay, TokenTable};

/// Generational memo for the `!`-certificate. Two generations of <= GEN_CAP entries each:
/// lookups check `cur` then `prev` (promoting prev-hits), inserts go to `cur`, and a full
/// `cur` becomes `prev` (the old `prev` is dropped). Hot entries survive generations; memory
/// stays bounded; memoization never stops.
///
/// Keys are `Vec<Tok>` of TABLE ids only -- per-call overlay ids must never enter this
/// per-Engine map. C1.11: the TYPE owns that invariant at the one unsound door,
/// insertion ([`BangCache::insert`] requires the table and DROPS any key carrying a
/// non-table id): the table is append-only, so a poisoned overlay-keyed entry would not
/// stay unreachable -- once the table grows past the overlay id, a legitimate key can
/// collide with it and read a wrong cached certificate. Lookups need no guard: an
/// overlay id is >= the CURRENT table length, every stored key's ids are below the
/// table length at their insert time, and the table only grows -- so an overlay-bearing
/// probe can never equal a stored key (it just misses, which is harmless). Table ids
/// are stable for the Engine's lifetime (append-only), so keys stay valid across
/// `set_rules`.
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

    pub(super) fn get_promoting(&mut self, key: &[Tok], table: &TokenTable) -> Option<bool> {
        if let Some(&b) = self.cur.get(key) {
            return Some(b);
        }
        if let Some(&b) = self.prev.get(key) {
            self.insert(key.to_vec(), b, table);
            return Some(b);
        }
        None
    }

    /// Insert a certificate. Keys carrying a per-call overlay id are DROPPED (see the
    /// type doc: not caching is always sound; caching under a reusable id is not).
    pub(super) fn insert(&mut self, key: Vec<Tok>, b: bool, table: &TokenTable) {
        if !key.iter().all(|&t| table.is_table_id(t)) {
            debug_assert!(false, "overlay id in a per-Engine certificate cache key");
            return;
        }
        self.cur.insert(key, b);
        if self.cur.len() >= BANG_GEN_CAP {
            self.prev = std::mem::take(&mut self.cur);
        }
    }
}

/// Per-simplify-call memo context: the certificate scratches (`cert_scratch` for the
/// `!`-certificate, `cert_mult_scratch` for its `$`-sort twin -- both consumed by
/// `engine/ac.rs::ac_cert`, the successor of the deleted kernel's `bang_certified`) plus
/// the per-call [`TokenOverlay`] (tokens outside the per-Engine table). Pure-function
/// memoization, valid because certification is deterministic for a fixed engine; all memo
/// keys are `Vec<Tok>`, consistent within the call by interning injectivity.
pub(super) struct SimplifyCtx {
    pub(super) overlay: RefCell<TokenOverlay>,
    pub(super) cert_scratch: RefCell<FxHashMap<Vec<Tok>, bool>>,
    /// The `$`-sort twin of `cert_scratch` (see `ac_cert`'s `mult` arm).
    pub(super) cert_mult_scratch: RefCell<FxHashMap<Vec<Tok>, bool>>,
}

impl SimplifyCtx {
    pub(super) fn new(table_len: usize) -> Self {
        Self {
            overlay: RefCell::new(TokenOverlay::new(table_len)),
            cert_scratch: RefCell::new(FxHashMap::default()),
            cert_mult_scratch: RefCell::new(FxHashMap::default()),
        }
    }
}
