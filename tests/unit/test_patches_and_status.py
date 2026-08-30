"""patches/ and docs/issues/STATUS.md must agree (P11: one source of truth).

Every in-flight fix is three things that drift apart if nothing checks them: a
format-patch under patches/, a gitcode issue+PR whose links live in the patch
header, and a row in STATUS.md. This test is the cheap guard.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STATUS = ROOT / "docs" / "issues" / "STATUS.md"
# in-flight fixes we can file upstream ourselves (P10: gitcode.com/ascend only)
FILEABLE = ("torch_npu", "op-plugin")


def _patches(sub: str) -> list[pathlib.Path]:
    d = ROOT / "patches" / sub
    return sorted(p for p in d.glob("*.patch")) if d.is_dir() else []


@pytest.mark.parametrize("sub", FILEABLE)
def test_every_inflight_patch_carries_its_upstream_links(sub):
    for patch in _patches(sub):
        text = patch.read_text()
        assert "Upstream-PR: https://gitcode.com/" in text, f"{patch.name}: no Upstream-PR header"
        assert "Upstream-Issue: https://gitcode.com/" in text, f"{patch.name}: no Upstream-Issue"


@pytest.mark.parametrize("sub", FILEABLE)
def test_every_inflight_patch_has_a_status_row(sub):
    status = STATUS.read_text()
    for patch in _patches(sub):
        issue_id = patch.name.split("-")[0] + "-" + patch.name.split("-")[1]
        assert re.search(rf"^\|\s*{re.escape(issue_id)}\s*\|", status, re.M), (
            f"{patch.name}: no '{issue_id}' row in docs/issues/STATUS.md"
        )
        assert patch.name in status or issue_id in status


def test_status_links_match_the_patch_headers():
    """A patch whose PR link disagrees with STATUS.md means one of them is stale."""
    status = STATUS.read_text()
    for sub in FILEABLE:
        for patch in _patches(sub):
            m = re.search(r"Upstream-PR: (\S+)", patch.read_text())
            assert m, patch.name
            number = m.group(1).rstrip("/").split("/")[-1]
            assert number in status, f"{patch.name}: PR {number} is not mentioned in STATUS.md"


def test_evidence_patches_are_never_advertised_as_filed():
    """patches/evidence/ is read-only proof for repos we may not touch (P10)."""
    for patch in (ROOT / "patches" / "evidence").rglob("*.patch"):
        assert "Upstream-PR: https://gitcode.com/" not in patch.read_text(), (
            f"{patch}: evidence patches must not claim a gitcode PR"
        )
