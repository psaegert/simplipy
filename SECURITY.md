# Security Policy

## Supported versions

Only the latest released minor version receives security fixes.

| Version | Supported |
|---|---|
| 0.13.x (latest) | yes |
| < 0.13 | no |

## Reporting a vulnerability

Please report vulnerabilities **privately** via GitHub's private vulnerability
reporting: open the repository's **Security** tab and choose **Report a
vulnerability**. Do not open a public issue for anything you believe is
exploitable.

You can expect an acknowledgement within a week. Please include a minimal
reproducer where possible.

## Scope — what is and is not a security issue here

**In scope:**

- Code execution beyond the declared trust model. SimpliPy evaluates operator
  *realizations* (Python callables named in engine configs) and refuses
  modules outside the trusted set (`simplipy.trust`,
  `SIMPLIPY_TRUSTED_MODULES`). Any path that executes code from a config or
  artifact **without** tripping `UntrustedModuleError` is a vulnerability.
- Artifact integrity bypass. Installed artifacts are pinned by manifest
  `revision` and per-file `sha256` digests, enforced at install and at cache
  resolution. Any way to make the engine load rule content that does not
  match the manifest digests is a vulnerability, as is path escape during
  asset install/uninstall (asset paths are resolved and contained under the
  asset root by design).
- Denial of service through crafted expressions or configs that bypass the
  documented budgets (`max_passes`, mining caps).

**Out of scope (report as ordinary bugs instead):**

- Wrong simplification results. Soundness violations are treated as
  high-priority *correctness* bugs with their own falsifier-first process —
  please report them publicly in the issue tracker.
- Anything reachable only through `codify` / `code_to_lambda` on
  attacker-supplied input. These compile strings to Python source and are
  documented as unsafe by construction on untrusted input; do not feed them
  attacker-controlled strings.
- Behaviour of the deprecated/undocumented surface (names outside the
  declared `__all__`).
