"""Software-only source refusal check; no solver import or native evidence."""

import hashlib
import sys

import pytest

from runners import run_pressure_diagnostic as diagnostic


def test_descendant_must_preserve_parent_and_current_committed_methods(tmp_path, monkeypatch):
    """An added runner permits a descendant; changed committed physics still fails."""
    head = "1" * 40  # Synthetic metadata only, not a native-run receipt.
    payload = b"unchanged software fixture\n"
    digest = hashlib.sha256(payload).hexdigest()
    sources = {}
    for relative in diagnostic.METHOD_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        sources[relative] = {
            "sha256": digest,
            "committed_sha256": digest,
            "committed_matches": True,
        }
    parent = {
        "component_commit": diagnostic.SOURCE,
        "component": {"commit": diagnostic.SOURCE, "sources": sources},
    }
    changed = False
    requests = []

    def metadata(root, *args):
        assert root == tmp_path
        requests.append(args)
        if args == ("rev-parse", "HEAD"):
            return head.encode()
        if args[0] in {"merge-base", "status"}:
            return b""
        assert args[0] == "show"
        if changed and args[1] == f"{head}:problems/paper_bubble.py":
            return b"changed committed physics\n"
        return payload

    monkeypatch.setattr(diagnostic, "git", metadata)
    before = set(sys.modules)
    assert diagnostic.source_contract(tmp_path, parent) == head
    assert ("merge-base", "--is-ancestor", diagnostic.SOURCE, head) in requests
    changed = True
    with pytest.raises(ValueError, match="source mismatch: problems/paper_bubble.py"):
        diagnostic.source_contract(tmp_path, parent)
    assert not any(
        name == "pyoomph" or name.startswith("pyoomph.") for name in set(sys.modules) - before
    )
