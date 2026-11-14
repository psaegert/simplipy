//! THE ARTIFACT-AFFECTING ENVIRONMENT SWITCHES: every environment variable that can change
//! a mined artifact, a certification verdict or a served answer. Each read site names its
//! variable through a constant declared here, and the extension exports the list as
//! `simplipy._core.ARTIFACT_ENV_SWITCHES` -- the registry a mine's provenance and
//! `simplipy verify-mine` check, so it cannot drift from what the core actually reads.
//! Diagnostics (the `*_TRACE` family), the test-asset switch, the kernel thread count
//! (`SIMPLIPY_KERNEL_THREADS`, which splits a long element-wise map over threads without
//! changing a bit) and the build stamp change no artifact and are not listed.
//!
//! The registry is EMPTY: the interval node budget is a recorded `find_rules` argument and
//! the tagged form's fraction bound is a constant, so no environment variable changes what
//! the core mines, certifies or serves.

/// Every artifact-affecting switch, in one list.
pub const ARTIFACT_ENV_SWITCHES: [&str; 0] = [];
