#!/usr/bin/env python3
"""Assert the sdist's disclosure boundary. Run in CI on every build; runnable by hand.

A PyPI upload is irrevocable and mirrored, so what the tarball contains is a boundary,
not a packaging preference -- and the project has now been bitten twice by trusting the
globs that implement it (audit R4/B15). On 2026-08-15 the exclude list, untouched, let
491 `remine/` entries into the sdist (19.2 MB, 439 of them the private pre-release audit
corpus) because it denied by extension at a single level while allowing recursion.

This script is the check that fails instead. It asserts BOTH halves, because either
alone is silently wrong:

* nothing under `remine/` or `design/` ships except the twelve allowlisted files -- a
  new private file is out by default, and a hole shows up here rather than on PyPI;
* all twelve DO ship -- the shipped test suite loads them, so a tarball without them is
  one whose tests cannot run at all (audit C54).

Usage:  python .github/check_sdist_boundary.py dist/simplipy-*.tar.gz
"""
from __future__ import annotations

import glob
import sys
import tarfile

ALLOWED_PRIVATE = {
    f'assets/engines/acj-{cell}/{name}'
    for cell in ('2-1', '3-2', '4-3')
    for name in ('config.yaml', 'mine.yaml', 'rules.json', 'rules.json.provenance.json')
}
PRIVATE_PREFIXES = ('remine/', 'design/')


def check(tarball: str) -> list[str]:
    """Return the list of failures (empty means the boundary holds)."""
    with tarfile.open(tarball) as archive:
        members = archive.getnames()
    # strip the `simplipy-<version>/` root every sdist member carries
    paths = {name.split('/', 1)[1] for name in members if '/' in name}

    failures = []
    leaked = sorted(
        path for path in paths
        if path.startswith(PRIVATE_PREFIXES) and path not in ALLOWED_PRIVATE)
    if leaked:
        failures.append(
            f'{len(leaked)} private path(s) in the sdist that the allowlist does not permit:\n  '
            + '\n  '.join(leaked[:40])
            + (f'\n  ... and {len(leaked) - 40} more' if len(leaked) > 40 else ''))

    missing = sorted(ALLOWED_PRIVATE - paths)
    if missing:
        failures.append(
            f'{len(missing)} allowlisted artifact file(s) MISSING -- the shipped tests cannot '
            f'load an engine without them:\n  ' + '\n  '.join(missing))
    return failures


def main() -> int:
    patterns = sys.argv[1:] or ['dist/*.tar.gz']
    tarballs = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not tarballs:
        print(f'no sdist matched {patterns} -- nothing to check, which is itself a failure')
        return 2
    exit_code = 0
    for tarball in tarballs:
        failures = check(tarball)
        if failures:
            exit_code = 1
            print(f'FAIL {tarball}')
            for failure in failures:
                print(f'  {failure}')
        else:
            print(f'ok   {tarball}: {len(ALLOWED_PRIVATE)} allowlisted artifact files, no other private path')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
