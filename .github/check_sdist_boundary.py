#!/usr/bin/env python3
"""Assert the sdist's disclosure boundary. Run in CI on every build; runnable by hand.

A PyPI upload is irrevocable and mirrored, so what the tarball contains is a boundary,
not a packaging preference -- and the project has now been bitten twice by trusting the
globs that implement it (audit R4/B15). On 2026-08-15 the exclude list, untouched, let
491 `remine/` entries into the sdist (19.2 MB, 439 of them the private pre-release audit
corpus) because it denied by extension at a single level while allowing recursion.

This script is the check that fails instead. It asserts BOTH halves, because either
alone is silently wrong:

* nothing under `remine/` or `design/` ships, no exceptions -- a private file is out
  by default, and a hole shows up here rather than on PyPI;
* every artifact file the CHECKOUT carries under `assets/engines/` ships -- the shipped
  test suite loads them, so a tarball missing one is a tarball whose tests cannot run
  (audit C54). The required set is DERIVED from the checkout, not hardcoded: a
  hardcoded list demanded the retired `acj-2-1`/`acj-3-2` cells forever (audit U1),
  and would go stale again the day the acj-4-3 cell grows its `real`/`corpus` thirds.
  Deriving it keeps the meaning fixed while the inventory moves: whatever the tree
  ships, the sdist ships whole.

Usage:  python .github/check_sdist_boundary.py dist/simplipy-*.tar.gz
"""
from __future__ import annotations

import glob
import os
import pathlib
import sys
import tarfile

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_ARTIFACT_TREE = _REPO_ROOT / 'assets' / 'engines'

PRIVATE_PREFIXES = ('remine/', 'design/')


def required_artifacts() -> frozenset[str]:
    """Every artifact file the checkout carries, as sdist-relative posix paths."""
    return frozenset(
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _ARTIFACT_TREE.rglob('*') if path.is_file())


def check(tarball: str, required: frozenset[str]) -> list[str]:
    """Return the list of failures (empty means the boundary holds)."""
    with tarfile.open(tarball) as archive:
        members = archive.getnames()
    # strip the `simplipy-<version>/` root every sdist member carries
    paths = {name.split('/', 1)[1] for name in members if '/' in name}

    failures = []
    leaked = sorted(path for path in paths if path.startswith(PRIVATE_PREFIXES))
    if leaked:
        failures.append(
            f'{len(leaked)} private path(s) in the sdist:\n  '
            + '\n  '.join(leaked[:40])
            + (f'\n  ... and {len(leaked) - 40} more' if len(leaked) > 40 else ''))

    missing = sorted(required - paths)
    if missing:
        failures.append(
            f'{len(missing)} artifact file(s) in the checkout but MISSING from the sdist -- '
            f'the shipped tests cannot load an engine without them:\n  ' + '\n  '.join(missing))
    return failures


def main() -> int:
    required = required_artifacts()
    if not required:
        print(f'no artifact files under {_ARTIFACT_TREE} -- the checkout this script derives '
              f'the required set from is not one that could have built a shippable sdist')
        return 2
    patterns = sys.argv[1:] or ['dist/*.tar.gz']
    tarballs = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not tarballs:
        print(f'no sdist matched {patterns} -- nothing to check, which is itself a failure')
        return 2
    exit_code = 0
    for tarball in tarballs:
        failures = check(tarball, required)
        if failures:
            exit_code = 1
            print(f'FAIL {tarball}')
            for failure in failures:
                print(f'  {failure}')
        else:
            print(f'ok   {os.path.basename(tarball)}: all {len(required)} checkout artifact '
                  f'files ship, no private path')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
