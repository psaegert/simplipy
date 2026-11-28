"""The Rust crates' notices in LICENSES_bundled.txt, generated from the Cargo tree.

usage: python rust/bundled_notices.py [--check]

The wheel links the compiled core with the crates of its normal-dependency closure: ``cargo metadata``
under the features the wheel builds with (pyproject's ``[tool.maturin] features``, which compile
mimalloc in), over every target platform, procedural macros included. Their licences ask a binary
distribution to carry their notices, so LICENSES_bundled.txt carries one entry per crate between the
markers below: its name and version, its licence expression as published, the licence the wheel uses
it under (the first of MIT, BSD-2-Clause, BSD-3-Clause, Zlib, BSL-1.0, Unicode-3.0, Apache-2.0 a
choice offers) and that licence's text from the crate's own files, with the notices of the native
sources a crate vendors and builds (mimalloc's, in libmimalloc-sys's ``c_src``). A crate that ships
no text for its licence gets the licence's standard text over its published authors (MIT only).

Run it after any change to Cargo.lock or the wheel's features; ``--check`` exits 1 when the file is
not what it would write (the CI runs it).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTICES = os.path.join(ROOT, 'LICENSES_bundled.txt')
BEGIN = '=' * 100 + '\nBEGIN GENERATED: the Rust crates the compiled core links (rust/bundled_notices.py)\n'
END = 'END GENERATED: the Rust crates the compiled core links\n' + '=' * 100 + '\n'

#: The licence used from a choice, most preferred first.
PREFERENCE = ['MIT', 'BSD-2-Clause', 'BSD-3-Clause', 'Zlib', 'BSL-1.0', 'Unicode-3.0', 'Apache-2.0',
              'Apache-2.0 WITH LLVM-exception', 'Unlicense']
#: The file names (lower case, a prefix) a crate carries each licence's text under.
FILES = {'MIT': ('license-mit',), 'Apache-2.0': ('license-apache',), 'BSL-1.0': ('license-boost', 'license-bsl'),
         'BSD-2-Clause': ('license-bsd',), 'BSD-3-Clause': ('license-bsd',), 'Unicode-3.0': ('license-unicode',),
         'Zlib': ('license-zlib',), 'Unlicense': ('unlicense', 'copying')}
#: Directories holding native sources a crate vendors and compiles.
VENDORED = ('c_src', 'vendor', 'third_party')
MIT_TEXT = """Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE."""


def _closure() -> list[dict]:
    """The packages of the root's normal-dependency closure, by name and version."""
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as file:
        features = tomllib.load(file)['tool']['maturin']['features']
    meta = json.loads(subprocess.run(
        ['cargo', 'metadata', '--format-version', '1', '--locked', '--features', ','.join(features),
         '--manifest-path', os.path.join(ROOT, 'Cargo.toml')],
        check=True, capture_output=True, text=True).stdout)
    packages = {p['id']: p for p in meta['packages']}
    nodes = {n['id']: n for n in meta['resolve']['nodes']}
    root = meta['resolve']['root']
    seen, stack = set(), [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(d['pkg'] for d in nodes[node]['deps']
                     if any(k['kind'] is None for k in d['dep_kinds']))
    seen.discard(root)
    return sorted((packages[i] for i in seen), key=lambda p: (p['name'], p['version']))


def _chosen(expression: str) -> list[str]:
    """The licences the wheel uses a crate under: every conjunct of the SPDX expression, the most
    preferred alternative of each disjunction (`/` is the old spelling of `OR`)."""
    tokens = re.findall(r'\(|\)|[^\s()]+', expression.replace('/', ' OR '))
    position = 0

    def term() -> list[list[str]]:
        nonlocal position
        if tokens[position] == '(':
            position += 1
            inner = disjunction()
            position += 1  # ')'
            return inner
        name = tokens[position]
        position += 1
        if position < len(tokens) and tokens[position] == 'WITH':
            name += f' WITH {tokens[position + 1]}'
            position += 2
        return [[name]]

    def conjunction() -> list[list[str]]:
        nonlocal position
        out = term()
        while position < len(tokens) and tokens[position] == 'AND':
            position += 1
            right = term()
            out = [a + b for a in out for b in right]
        return out

    def disjunction() -> list[list[str]]:
        nonlocal position
        out = conjunction()
        while position < len(tokens) and tokens[position] == 'OR':
            position += 1
            out += conjunction()
        return out

    def rank(ids: list[str]) -> tuple:
        return tuple(sorted(PREFERENCE.index(i) if i in PREFERENCE else len(PREFERENCE) for i in ids))

    return min(disjunction(), key=rank)


def _licence_files(package: dict) -> list[str]:
    directory = os.path.dirname(package['manifest_path'])
    out = []
    for path, dirs, files in os.walk(directory):
        rel = os.path.relpath(path, directory)
        dirs[:] = sorted(d for d in dirs if d not in ('tests', 'benches', 'examples', 'target', '.git'))
        if rel != '.' and not set(rel.split(os.sep)) & set(VENDORED):
            continue
        out += [os.path.join(rel, f) if rel != '.' else f for f in sorted(files)
                if f.upper().startswith(('LICEN', 'COPYING', 'UNLICENSE'))]
    return out


def _entry(package: dict, printed: dict[str, str]) -> str:
    directory = os.path.dirname(package['manifest_path'])
    files = _licence_files(package)
    top = [f for f in files if os.sep not in f]
    chosen = _chosen(package['license'] or 'NONE')
    lines = [f"Name: {package['name']} {package['version']}"]
    if package.get('repository'):
        lines.append(f"Origin: crates.io, {package['repository']}")
    lines.append(f"License: {package['license']}; used under {' AND '.join(chosen)}")
    texts: list[tuple[str, str]] = []
    for licence in chosen:
        named = [f for f in top if f.lower().startswith(FILES.get(licence, ('\0',)))]
        if not named and len(chosen) == 1:
            named = [f for f in top if f.lower().split('.')[0] in ('license', 'licence', 'copying')]
        if named:
            texts += [(f, open(os.path.join(directory, f), encoding='utf-8', errors='replace').read())
                      for f in named[:1]]
        elif licence == 'MIT':
            authors = ', '.join(re.sub(r'\s*<[^>]*>', '', a) for a in package.get('authors') or []) or \
                f"the authors of {package['name']}"
            texts.append(('(none shipped; the standard text over the published authors)',
                          f'Copyright (c) {authors}\n\n{MIT_TEXT}'))
        else:
            texts.append(('(none shipped)', f'The {licence} licence, by its SPDX identifier.'))
    texts += [(f, open(os.path.join(directory, f), encoding='utf-8', errors='replace').read())
              for f in files if os.sep in f]  # the vendored native sources' notices
    lines.append('Notice: ' + ', '.join(f for f, _ in texts))
    out = '\n'.join(lines) + '\n'
    for name, text in texts:
        key = re.sub(r'\s+', ' ', text).strip()
        if key in printed:
            out += f'\n  ({name}: the same text as {printed[key]} above)\n'
            continue
        printed[key] = f"{package['name']} {package['version']}'s {name}"
        out += '\n' + '\n'.join(('  ' + line).rstrip() for line in text.strip('\n').splitlines()) + '\n'
    return out


def generated() -> str:
    printed: dict[str, str] = {}
    body = '\n\n'.join(_entry(p, printed) for p in _closure())
    return (BEGIN + 'Do not edit by hand: run `python rust/bundled_notices.py` after a change to Cargo.lock or\n'
            'to the wheel\'s features.\n' + '=' * 100 + '\n\n' + body + '\n' + END)


def main() -> int:
    with open(NOTICES, encoding='utf-8') as file:
        current = file.read()
    head = current.split(BEGIN)[0].rstrip('\n') + '\n\n\n'
    wanted = head + generated()
    if '--check' in sys.argv[1:]:
        if wanted != current:
            print('LICENSES_bundled.txt is not what rust/bundled_notices.py writes: run it and commit the file')
            return 1
        print('LICENSES_bundled.txt is current')
        return 0
    with open(NOTICES, 'w', encoding='utf-8') as file:
        file.write(wanted)
    return 0


if __name__ == '__main__':
    sys.exit(main())
