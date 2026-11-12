"""Files saved as "UTF-8 with BOM" (Windows Notepad's default for years) read like plain UTF-8.

A byte-order mark is an invisible character at the very start of a file. Left in, it stops the
first line from matching: a policy's "# Title" heading is missed, a regulation's title falls back
to its file name, and the first setting in .env gets a different name, so it is silently ignored.
"""

from policypilot import config
from policypilot.ingest import load_policies, load_regulation

POLICY = "# Acme Privacy Policy\n\n## 1.1 Retention\n\nWe keep data for ten (10) years.\n"
REGULATION = "# Circular 7/2026\n\n## R1. Keep less\n\nPersonal data shall not be kept beyond three (3) years.\n"


def _write_bom(path, text):
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    return path


def test_policy_title_survives_a_bom(tmp_path):
    folder = tmp_path / "policies"
    folder.mkdir()
    _write_bom(folder / "privacy.md", POLICY)
    (folder / "plain.md").write_text(POLICY, encoding="utf-8")
    by_policy = {c.policy: c for c in load_policies(folder)}
    assert by_policy["privacy"].policy_title == by_policy["plain"].policy_title == "Acme Privacy Policy"
    assert by_policy["privacy"].text == "We keep data for ten (10) years."


def test_regulation_title_survives_a_bom(tmp_path):
    reg = load_regulation(_write_bom(tmp_path / "circular.md", REGULATION))
    assert reg.title == "Circular 7/2026"
    assert reg.text.startswith("# Circular")  # quotes are checked against this text, so no stray mark
    assert [r.id for r in reg.requirements] == ["R1"]


def test_first_env_setting_survives_a_bom(tmp_path, monkeypatch):
    monkeypatch.delenv("PP_TEST_FIRST_KEY", raising=False)
    monkeypatch.delenv("﻿PP_TEST_FIRST_KEY", raising=False)
    env = _write_bom(tmp_path / ".env", "PP_TEST_FIRST_KEY=from-file\n")
    config._load_dotenv(env)
    assert config.os.environ.get("PP_TEST_FIRST_KEY") == "from-file"
    monkeypatch.delenv("PP_TEST_FIRST_KEY", raising=False)
