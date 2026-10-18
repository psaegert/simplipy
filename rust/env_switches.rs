//! THE ARTIFACT-AFFECTING ENVIRONMENT SWITCHES: every environment variable that can change
//! a mined artifact, a certification verdict or a served answer. Each read site names its
//! variable through these constants, and the extension exports the list as
//! `simplipy._core.ARTIFACT_ENV_SWITCHES` -- the registry the mine's provenance records every
//! set entry of, so it cannot drift from what the core actually reads. Diagnostics (the
//! `*_TRACE` family), the test-asset switch and the build stamp change no artifact and are
//! not listed.

/// The interval layer's subdivision node budget (`interval::max_nodes_budget`).
pub const IVL_NODE_BUDGET: &str = "SIMPLIPY_IVL_NODE_BUDGET";
/// The high-precision near-miss escalation fraction (`hiprec::near_miss_frac`).
pub const HIPREC_FRAC: &str = "SIMPLIPY_HIPREC_FRAC";
/// The largest decimal the tagged form spells as a fraction bag (`ac::convert`).
pub const TAGGED_FRACTION_MAX: &str = "SIMPLIPY_TAGGED_FRACTION_MAX";

/// Every artifact-affecting switch, in one list.
pub const ARTIFACT_ENV_SWITCHES: [&str; 3] = [IVL_NODE_BUDGET, HIPREC_FRAC, TAGGED_FRACTION_MAX];
