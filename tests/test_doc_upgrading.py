"""The upgrading guide (docs/upgrading.md) is held to the same doctrine as every other page
(mech-3, tests/test_doc_examples.py): each fenced python block executes and each printed output
is asserted. It is a separate file so the guide's claims about 0.15 are checked by name."""
from test_doc_examples import test_doc_examples_execute_and_their_outputs_hold as _run_page


def test_the_upgrading_guide_examples_execute_and_hold(tmp_path, monkeypatch) -> None:
    _run_page('docs/upgrading.md', tmp_path, monkeypatch)


def test_the_upgrading_guide_is_reachable() -> None:
    """The guide is in the site's navigation and linked from the pages a 0.14 user opens first."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'mkdocs.yml')) as fh:
        assert 'upgrading.md' in fh.read()
    for page, link in [('docs/index.md', 'upgrading.md'), ('docs/getting-started.md', 'upgrading.md'),
                       ('docs/compatibility.md', 'upgrading.md'), ('README.md', '/upgrading/')]:
        with open(os.path.join(root, page)) as fh:
            assert link in fh.read(), f'{page} does not link the upgrading guide'
