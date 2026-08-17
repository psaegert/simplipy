//! Mining observability. (The deleted binary kernel's twelve simplify-hot-path counters,
//! phase timers, `bump`/`Timer` plumbing and the `snapshot`/`reset` FFI schema were
//! removed in D3, 2026-08-05, together with the `stats-counters`/`stats-timers`/
//! `profiling` feature gates that existed only for them: their bump sites went with the
//! kernel, so every read was 0 under any build -- schema without signal. Everything below
//! is LIVE and unconditional: once-per-mined-source frequency, off any hot path.)

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

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

/// Mining ordering-acceptance decisions REFUSED because `ac_ordered_below` could not parse
/// a side. The core serialization language is config-independent, so the engine's own
/// output and every library candidate always parse -- any nonzero count is a BUG tripwire
/// (the F0 dormancy lesson: a silent conservative arm hides its own trigger), asserted
/// loudly in debug builds at the refusal site.
pub static ACCEPT_UNDECIDED_REFUSALS: AtomicU64 = AtomicU64::new(0);

/// Candidates refused by the registry's POLE entry (refusals.rs, SWEEP row 3): the
/// candidate changed a DEFINED extended-real value at a constructed exceptional point.
/// Expected 0 at the (4,3) tiers (the mirror family needs one extra token); a nonzero
/// count at raised tiers is the registry doing exactly its job -- report it, never
/// silence it.
pub static REGISTRY_POLE_REFUSALS: AtomicU64 = AtomicU64::new(0);
