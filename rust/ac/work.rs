//! DETERMINISTIC WORK ACCOUNTING -- the unit of the search's work budget.
//!
//! One unit is one step of the AC matcher (`matcher::try_assign`, a placement attempt of one
//! pattern element) or one canonical-constructor call (`add`, `mul`, `pow`, `fun`). These are
//! the two places the search's time goes: on srbf's 120 slowest-and-sampled permissive
//! predictions the matcher alone is 87% of the call's time (median share), and the per-unit cost
//! of matcher steps plus constructor calls varies by less than a factor of two across those rows
//! (q90/q10 = 1.8). The count is a pure function of the walk, so a budget in these units cuts the
//! same walk at the same place on every machine and under any load -- unlike a wall clock.
//!
//! The counter is thread-local and only ever read as a difference, so nothing needs resetting.
//! [`limit`] arms a ceiling for the dynamic extent of a guard; while the ceiling is exceeded,
//! [`over`] is true and the matcher refuses to enumerate further (`try_assign`), so a candidate
//! descent that crosses the ceiling finishes quickly -- and the search then discards it.

use std::cell::Cell;

thread_local! {
    static WORK: Cell<u64> = const { Cell::new(0) };
    static LIMIT: Cell<u64> = const { Cell::new(u64::MAX) };
    /// The per-search budget the current call runs under (`u64::MAX`: unbounded), set at the
    /// FFI boundary for one simplify call ([`search_budget_scope`]).
    static SEARCH_BUDGET: Cell<u64> = const { Cell::new(u64::MAX) };
}

/// Count `n` units of work.
#[inline]
pub fn tick(n: u64) {
    WORK.with(|w| w.set(w.get().wrapping_add(n)));
}

/// The units counted so far on this thread (read as a difference).
#[inline]
pub fn now() -> u64 {
    WORK.with(|w| w.get())
}

/// Is an armed ceiling exceeded?
#[inline]
pub fn over() -> bool {
    WORK.with(|w| w.get()) >= LIMIT.with(|l| l.get())
}

/// Restores the previous ceiling when dropped.
pub struct LimitGuard(u64);

impl Drop for LimitGuard {
    fn drop(&mut self) {
        LIMIT.with(|l| l.set(self.0));
    }
}

/// Arm a ceiling `budget` units from now (never above an enclosing one) until the guard drops.
pub fn limit(budget: u64) -> LimitGuard {
    let ceiling = now().saturating_add(budget);
    LimitGuard(LIMIT.with(|l| l.replace(l.get().min(ceiling))))
}

/// The per-search budget of the current call.
pub fn search_budget() -> u64 {
    SEARCH_BUDGET.with(|b| b.get())
}

/// Restores the previous per-search budget when dropped.
pub struct BudgetGuard(u64);

impl Drop for BudgetGuard {
    fn drop(&mut self) {
        SEARCH_BUDGET.with(|b| b.set(self.0));
    }
}

/// Run under a per-search budget (`None`: unbounded) until the guard drops.
pub fn search_budget_scope(budget: Option<u64>) -> BudgetGuard {
    BudgetGuard(SEARCH_BUDGET.with(|b| b.replace(budget.unwrap_or(u64::MAX))))
}
