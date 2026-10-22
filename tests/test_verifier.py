from policypilot.agent.verifier import quote_in, verify
from policypilot.ingest import load_regulation
from policypilot.models import Finding


def _reg(settings):
    return load_regulation(settings.regulations_dir / "reg-01-data-protection.md")


def _finding(**kw):
    base = dict(requirement_id="R2", clause_id="data-privacy§3.2", impact="conflict", explanation="x",
                regulation_quote="shall not be kept for longer than three (3) years",
                policy_quote="Transaction logs are retained for two (2) years",
                proposed_text="new text")
    return Finding(**{**base, **kw})


def test_quote_matching_ignores_case_spacing_and_curly_quotes():
    src = "Customers may request   deletion of their “personal data”."
    assert quote_in('customers may request deletion of their "personal data"', src)
    assert quote_in("Customers may request ... personal data", src)
    assert not quote_in("Customers must request deletion", src)


def test_exact_quotes_pass(settings, index):
    [f] = verify([_finding()], _reg(settings), index)
    assert f.verified and not f.issues


def test_paraphrased_quote_fails_with_a_helpful_hint(settings, index):
    [f] = verify([_finding(policy_quote="logs are kept for two years only")], _reg(settings), index)
    assert not f.verified
    assert "not found word for word" in f.issues[0] and "Closest actual sentence" in f.issues[0]


def test_unknown_clause_and_requirement_are_rejected(settings, index):
    [f] = verify([_finding(clause_id="data-privacy§99", requirement_id="R42")], _reg(settings), index)
    assert any("R42" in i for i in f.issues) and any("does not exist" in i for i in f.issues)


def test_gap_may_omit_policy_quote_but_needs_a_draft(settings, index):
    [ok] = verify([_finding(impact="gap", policy_quote="")], _reg(settings), index)
    assert ok.verified
    [bad] = verify([_finding(impact="gap", policy_quote="", proposed_text="")], _reg(settings), index)
    assert any("proposed_text" in i for i in bad.issues)


def test_too_short_quotes_are_rejected(settings, index):
    [f] = verify([_finding(regulation_quote="shall")], _reg(settings), index)
    assert any("too short" in i for i in f.issues)


def test_duplicates_are_dropped(settings, index):
    assert len(verify([_finding(), _finding()], _reg(settings), index)) == 1


def test_unicode_hyphens_and_spaces_match_plain_ones():
    # GPT-OSS writes "board‑approved" with U+2011 (non-breaking hyphen); the document has "-".
    assert quote_in("the entity’s board‑approved policy", "the entity's board-approved policy")
    assert quote_in("T− 1 and 24 hours", "T- 1 and 24 hours")


def test_hint_sentences_do_not_include_headings():
    from policypilot.textutil import sentences

    text = "## R1. Zero liability\n\nA customer shall bear zero liability. Beyond that, rules apply."
    out = sentences(text)
    assert "A customer shall bear zero liability." in out  # the body sentence stands alone
    assert not any("liability A customer" in s for s in out)  # heading not glued onto it
