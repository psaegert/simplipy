# Contributing

Thanks for your interest. This project moves carefully: it ships a soundness
contract, and most of the process below exists to protect it.

## Dev setup

- Python ≥ 3.12 and a Rust toolchain (stable) with `cargo`.
- `pip install -e .[dev]` builds the required compiled core
  (`simplipy._core`) via maturin; there is no pure-Python fallback.
- Run the suite with `pytest tests` (all tests must pass; there are no
  known-failing tests).

## The rules of the road

- **Falsifier first.** A bug fix lands with a test that is red before the fix
  and green after. A claim in the docs is backed by an executable example
  (the doc-runner test executes fenced blocks and checks their printed
  output).
- **Artifact-affecting changes are gated.** Anything that could change a
  mined artifact or a certification verdict (rule mining, certification,
  promotion, the interval kernel, environment switches in
  `simplipy.engine.ARTIFACT_ENV_SWITCHES`) must come with evidence that the
  reference mine is byte-identical — or a changelog entry explaining exactly
  which rows move and why.
- **One miner per process.** Mining/certification is single-flighted through
  `simplipy.mining._MINE_LOCK`; do not spawn concurrent miners in one
  process.
- **No drive-by refactors.** Mechanical moves and behaviour changes go in
  separate commits so equality gates keep their meaning.

## Pull requests

Keep PRs small and single-purpose. State what the change does, how it is
tested, and — for anything near the mine or the kernel — why the artifact
does not move. CI runs the test suite on Python 3.12/3.13 plus the Rust
tests; all lanes must be green.

## Security

See [SECURITY.md](SECURITY.md) — vulnerabilities go through private
reporting, not the public tracker.
