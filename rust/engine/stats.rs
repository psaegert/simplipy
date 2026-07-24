//! ONLINE simplify observability: process-global counters over the simplify hot path, plus
//! coarse nanosecond accounting on the phase boundaries (cancel / rules / cert / mask+sort).
//! Relaxed atomics, no locks, no control-flow effect. Aggregates across engines and threads;
//! read/reset between batch runs via `simplify_counters()` / `reset_simplify_counters()`
//! (lib.rs). Caveat: the certificate timer brackets the whole `bang_certified` call, so
//! memo-HIT time includes the timer overhead itself -- fine for share-of-runtime ranking,
//! not for ns-exact hit-path costing.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub static SIMPLIFY_CALLS: AtomicU64 = AtomicU64::new(0);
pub static SIMPLIFY_ITERS: AtomicU64 = AtomicU64::new(0);
pub static EXACT_HITS: AtomicU64 = AtomicU64::new(0);
pub static PATTERN_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
pub static PATTERN_FIRES: AtomicU64 = AtomicU64::new(0);
pub static CERT_CALLS: AtomicU64 = AtomicU64::new(0);
pub static CERT_HITS: AtomicU64 = AtomicU64::new(0);
pub static NANOS_CANCEL: AtomicU64 = AtomicU64::new(0);
pub static NANOS_RULES: AtomicU64 = AtomicU64::new(0);
pub static NANOS_CERT: AtomicU64 = AtomicU64::new(0);
pub static NANOS_MASK_SORT: AtomicU64 = AtomicU64::new(0);

pub static ALL: [(&str, &AtomicU64); 11] = [
    ("simplify_calls", &SIMPLIFY_CALLS),
    ("simplify_iters", &SIMPLIFY_ITERS),
    ("exact_hits", &EXACT_HITS),
    ("pattern_attempts", &PATTERN_ATTEMPTS),
    ("pattern_fires", &PATTERN_FIRES),
    ("cert_calls", &CERT_CALLS),
    ("cert_hits", &CERT_HITS),
    ("nanos_cancel", &NANOS_CANCEL),
    ("nanos_rules", &NANOS_RULES),
    ("nanos_cert", &NANOS_CERT),
    ("nanos_mask_sort", &NANOS_MASK_SORT),
];

// --- OFFLINE mining progress: a within-tier "sources done / total" signal a driver can poll
// while `mine_one_length` (one blocking, rayon-parallel call) is running, so a length tier is no
// longer an opaque black box. Process-global relaxed atomics, incremented once per source.
pub static MINE_SOURCES_DONE: AtomicU64 = AtomicU64::new(0);
pub static MINE_SOURCES_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Start a tier: publish the total source count and reset the done counter.
pub fn mine_begin(total: u64) {
    MINE_SOURCES_TOTAL.store(total, Relaxed);
    MINE_SOURCES_DONE.store(0, Relaxed);
}

/// One source finished (called from every rayon worker).
#[inline]
pub fn mine_tick() {
    MINE_SOURCES_DONE.fetch_add(1, Relaxed);
}

/// `(done, total)` for the tier currently mining.
pub fn mine_progress() -> (u64, u64) {
    (
        MINE_SOURCES_DONE.load(Relaxed),
        MINE_SOURCES_TOTAL.load(Relaxed),
    )
}

#[inline]
pub fn bump(c: &AtomicU64) {
    c.fetch_add(1, Relaxed);
}

#[inline]
pub fn add(c: &AtomicU64, v: u64) {
    c.fetch_add(v, Relaxed);
}

pub fn snapshot() -> Vec<(&'static str, u64)> {
    ALL.iter().map(|(k, c)| (*k, c.load(Relaxed))).collect()
}

pub fn reset() {
    for (_, c) in ALL.iter() {
        c.store(0, Relaxed);
    }
}
