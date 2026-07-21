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
