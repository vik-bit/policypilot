from policypilot.ingest import load_policies, load_regulations, parse_regulation
from policypilot.ingest.loader import split_clauses


def test_policies_split_on_numbered_headings(settings):
    clauses = load_policies(settings.policies_dir)
    ids = {c.id for c in clauses}
    assert "data-privacy§3.2" in ids and "payments-ops§5.1" in ids
    retention = next(c for c in clauses if c.id == "data-privacy§3.2")
    assert retention.title == "Data Retention"
    assert retention.text.startswith("Customer personal data is retained for ten (10) years")
    assert "## " not in retention.text  # the next heading is not swallowed


def test_every_sample_regulation_has_requirements(settings):
    regs = load_regulations(settings.regulations_dir)
    assert len(regs) == 6
    assert all(r.requirements for r in regs)
    assert [q.id for q in regs[0].requirements] == ["R1", "R2", "R3", "R4", "R5"]


def test_unnumbered_text_falls_back_to_chunks():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(5))
    clauses = split_clauses(text, "messy", "Messy Policy")
    assert len(clauses) >= 2 and clauses[0].id == "messy§P1"


def test_pasted_regulation_without_numbering_uses_paragraphs():
    reg = parse_regulation("Entities must encrypt all backups using strong keys.\n\n"
                           "Entities must rotate encryption keys every six months at least.")
    assert [r.id for r in reg.requirements] == ["R1", "R2"]
