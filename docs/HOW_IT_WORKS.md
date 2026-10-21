# How it works: one run, step by step

This walks through what happens when you analyse the sample regulation
**"Amendments to Personal Data Protection Requirements"**
(`data/regulations/reg-01-data-protection.md`), with links to the code for each step.

---

## Step 0 — Building the index (happens once)

📁 `policypilot/ingest/loader.py`, `policypilot/retrieval/index.py`

Before anything else, the five company policies in `data/policies/` are cut into **51 clauses**,
each with its own ID:

```
data-privacy§3.2   "Data Retention"
  Customer personal data is retained for ten (10) years after the account is closed.
  Transaction logs are retained for two (2) years. ...
```

Each clause is indexed two ways:
- **by words**: a BM25 keyword index (`retrieval/bm25.py`)
- **by meaning**: each clause becomes an *embedding*, a list of 384 numbers produced by the
  `bge-small` model (`retrieval/embedders.py`)

The index is saved to `data/index/` and rebuilt automatically when a policy file changes.

---

## Step 1 — Read the regulation

📁 `policypilot/ingest/loader.py → parse_regulation()`

The regulation is split at its numbered headings into five requirements:

| ID | Title |
|---|---|
| R1 | Reporting of personal data breaches |
| R2 | Storage limitation |
| R3 | Localisation of payment data |
| R4 | Granular and revocable consent |
| R5 | Data of minors |

---

## Step 2 — Search

📁 `policypilot/retrieval/index.py → search()`

Take R2: *"Personal data shall not be kept for longer than three (3) years after the end of the
customer relationship…"*

The two searches return different orders:

| Clause | Keyword rank | Meaning rank | Combined |
|---|---|---|---|
| data-privacy§3.2 Data Retention | 2 | **1** | **1** |
| kyc-aml§6.1 Record Keeping | 3 | 4 | 2 |
| data-privacy§3.3 Deletion Requests | 6 | 2 | 3 |

Keyword search ranked "Data Retention" only 2nd: the regulation says "kept", the policy says
"retained". Meaning search understood they're the same idea. Reciprocal Rank Fusion merges the
two lists, and the cross-encoder re-reads the top 20 and settles the final order.

You can try this yourself in the **Search playground** tab.

---

## Step 3 — The agent works

📁 `policypilot/agent/claude_agent.py`, `policypilot/agent/prompts.py`

With an API key, Claude gets the regulation plus three tools and decides its own next steps. A
typical sequence, shown live in the **Agent steps** tab, looks like this (illustrative):

```
🔍 Searched "personal data breach notification timeline" → data-privacy§5.1, data-privacy§5.2, info-security§5.1
📖 Read data-privacy§5.1 (Personal Data Breach Notification)
🔍 Searched "how long customer data is retained after closure" → data-privacy§3.2, data-privacy§3.3, kyc-aml§6.1
📖 Read data-privacy§3.2 (Data Retention)
🔍 Searched "where data is stored, data centre location" → data-privacy§4.1, ...
...
📨 Submitted 7 findings
```

Two things to notice:
- The agent **rephrases** its searches. It doesn't paste the regulation text in; it asks the
  question the way a policy would answer it.
- It **reads the full clause** before judging. Search results only show a snippet.

For each affected clause it fills in a structured finding: the requirement, the clause, the label
(conflict / gap / aligned), a plain-English explanation, an exact quote from each document, and
the proposed new wording.

**Without an API key**, `policypilot/agent/offline.py` does a simple version: it searches each
requirement, takes the best match, and picks the best-matching sentences as quotes. It's a
baseline. It finds clauses well but can't really reason about them.

---

## Step 4 — Fact-check and fix

📁 `policypilot/agent/verifier.py`

Every finding is checked by plain code:

1. Does the clause ID exist?
2. Is the regulation quote really in the regulation, word for word? (Case, spacing and curly
   quotes are ignored; nothing else is.)
3. Is the policy quote really in that clause?
4. Is there a proposed rewrite for every conflict and gap?

Say the AI paraphrased a quote:

```
policy_quote: "Customer data is kept for a decade"
```

The verifier rejects it and sends this back to the AI as the tool's result:

```json
{
  "status": "rejected",
  "errors": [{
    "requirement_id": "R2", "clause_id": "data-privacy§3.2",
    "errors": ["policy_quote was not found word for word in clause data-privacy§3.2. Closest actual sentence: \"Customer personal data is retained for ten (10) years after the account is closed.\""]
  }]
}
```

The AI fixes the quote and resubmits. This happens at most twice. Anything still wrong after
that is shown to the reviewer with a **⚠ needs manual check** badge and is never shown as
verified. The report records how many quotes were valid **before and after** this loop, and that
difference is the verifier's measurable value.

---

## Step 5 — Human review

📁 `policypilot/web/app.js`

Each finding is shown as a card:
- the **regulation quote** that triggers it,
- the **current clause**, with the quoted words highlighted,
- the **proposed clause**, with insertions in green and deletions struck through,
- **Approve / Reject** buttons. Decisions are saved to `data/runs/<run-id>.json`.

---

## Step 6 — Measuring it

📁 `policypilot/eval/run_eval.py`

`python -m policypilot eval` re-runs steps 2–4 for all six sample regulations and compares the
results with the hand-written answers in `data/eval/labels.jsonl`. Results go to
[EVAL_RESULTS.md](EVAL_RESULTS.md). Terms like Recall@k and MRR are explained in the
[GLOSSARY](GLOSSARY.md).
