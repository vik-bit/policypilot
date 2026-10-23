<div align="center">

# PolicyPilot

**An AI agent that reads a new regulation, finds every company policy it affects, drafts the
changes, and proves every claim with a quote that is checked automatically.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Claude%20%7C%20Groq%20(GPT--OSS)-6E56CF)
![RAG](https://img.shields.io/badge/RAG-BM25%20%2B%20embeddings%20%2B%20rerank-2950D6)
[![Tests](https://github.com/vik-bit/policypilot/actions/workflows/tests.yml/badge.svg)](https://github.com/vik-bit/policypilot/actions/workflows/tests.yml)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

[What it does](#what-it-does) · [Results](#results) · [How it works](#how-it-works) ·
[Run it](#run-it-in-3-minutes) · [System design](docs/SYSTEM_DESIGN.md) · [Plain-English tour](docs/HOW_IT_WORKS.md)

<img src="docs/images/report.png" alt="PolicyPilot report: each affected clause shown with the new rule, today's wording, the suggested wording, and a verified-evidence badge" width="900">

<sub>A real report produced by the AI agent (Groq · GPT-OSS 120B). Each card shows the new rule, today's policy wording,
the suggested rewrite (green = added, red = removed), and whether every quote was verified.</sub>

</div>

---

## At a glance

| | |
|---|---|
| **The problem** | When regulators publish new rules, compliance teams spend weeks finding which internal policies are now out of date, and rewriting them. AI can read fast but invents things, which is unacceptable in compliance. |
| **What I built** | A full-stack **agentic RAG** system. An LLM agent searches a policy library, decides what each new rule changes, and drafts new wording. A **deterministic fact-checker** then rejects any quote that isn't word for word in the source, and the agent has to fix it. A human approves every change. |
| **Result (measured)** | Across 25 hand-labelled requirements: **96%** of correct clauses in the top-3 search results; the AI agent reached **100% precision**, **93% recall** and **96% correct impact labels**, against 85% / 89% / 50% for a no-AI baseline, a gain that is statistically significant (p = 0.0002). Nothing unverified is ever shown as verified. Also tested on 38 requirements from real Indian regulations, against prompt injection, and at 10,000 clauses. [Details ↓](#results) |
| **Skills shown** | LLM tool use and agent loops · RAG (hybrid search, RRF fusion, cross-encoder reranking) · hallucination control · evaluation design · FastAPI · front-end UX · testing an LLM loop without an API · multi-provider LLM integration |

---

## What it does

Pick a regulation, click **Run analysis**, and get a report you can review and export:

| Step | What happens | Under the hood |
|---|---|---|
| **1. Read** | The regulation is split into numbered requirements (R1, R2…) | Document parsing |
| **2. Search** | Each requirement is matched against every policy clause, by **exact words** and by **meaning** | Hybrid retrieval: BM25 + embeddings → Reciprocal Rank Fusion → cross-encoder rerank |
| **3. Assess** | The AI agent decides what to search for, reads clauses, labels each one **Needs change**, **Missing from policy** or **Already compliant**, and drafts new wording | Tool-calling agent loop (search / read / submit) |
| **4. Verify** | Code checks every quote exists word for word. Failures go back to the agent with the closest real sentence | Deterministic verifier + self-correction loop |
| **5. Decide** | You compare old and new wording side by side, approve or reject, and export | Human in the loop |

<table>
  <tr>
    <td width="50%"><img src="docs/images/overview.png" alt="Overview page"><br><sub><b>Overview:</b> what it does, in plain English</sub></td>
    <td width="50%"><img src="docs/images/analyze.png" alt="New analysis page"><br><sub><b>New analysis:</b> a guided three-step flow</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/images/library.png" alt="Policy library page"><br><sub><b>Policy library:</b> browse, filter and upload policies</sub></td>
    <td width="50%"><img src="docs/images/help.png" alt="How to use page"><br><sub><b>How to use:</b> a built-in guide for non-technical users</sub></td>
  </tr>
</table>

---

## Results

I wrote the correct answers by hand for **25 requirements** across 6 sample regulations
([`data/eval/labels.jsonl`](data/eval/labels.jsonl)), then measured the system against them.
Full tables: [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md).

**1. Search: does the right clause come up?** (all 25 requirements)

| Method | Right clause in top 3 | Ranked #1 | MRR |
|---|---|---|---|
| Exact words only (BM25, written from scratch) | 92% | 72% | 0.84 |
| Meaning only (bge-small embeddings) | 88% | 84% | 0.88 |
| **Hybrid + cross-encoder rerank** | **96%** | **80%** | **0.89** |

Each method misses things the other catches. "V-CIP" needs exact words; "storage limitation" ≈
"data retention" needs meaning. Combined, they put the right clause in the top 3 results 96% of the time.

**2. End to end: is the final report right?**

All 6 regulations, all 25 requirements:

| Metric | No-AI baseline | AI agent (Groq · GPT-OSS 120B) |
|---|---|---|
| Precision: flagged clauses that are really affected | 85% | **100%** (27/27) |
| Recall: affected clauses found | 89% | **93%** (25/27) |
| Impact label correct (needs change / missing / compliant) | 50% | **96%** (23/24) |
| Quotes verified word for word | 100% | 98% (53/54), and the 1 failure was a checker bug, [now fixed](docs/EVAL_RESULTS.md#a-bug-the-evaluation-caught) |
| Cost | free | 167k tokens · ~17 min on Groq's free tier |

**What this shows:** search finds the clauses; the agent adds the judgement. It flagged no
unrelated clause and labelled 23 of 24 correctly. The evaluation also caught a bug in my own
fact-checker: it rejected a real quote because the model typed a Unicode non-breaking hyphen. The
flaw failed safe (the finding was flagged for manual check, never shown as verified), and it's now
fixed with a regression test. Every miss is analysed in [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md).

**3. How sure can we be, and does it hold up outside the sample set?**

25 requirements is a small sample, and the sample set was built alongside the system. So I wrote
an [evaluation plan](docs/EVAL_PLAN.md) *before* scoring anything new. It fixes the metrics,
statistics and pass bars in advance.

- **Confidence intervals** ([EVAL_STATS.md](docs/EVAL_STATS.md)). The agent's 96% impact accuracy
  is really **87–100%**, and recall is **82–100%**. The gain over the baseline is unlikely to be
  luck. The agent was fully right on 13 requirements where the baseline wasn't, and never the
  reverse (exact McNemar test, **p = 0.0002**).
- **Real regulations** ([EVAL_RESULTS_REAL.md](docs/EVAL_RESULTS_REAL.md)). There are 38
  requirements from CERT-In's 2022 Directions, the DPDP Act 2023 and four RBI circulars, held out
  from all tuning. Three of them don't apply to a payments wallet at all, which tests whether the
  system knows when to say nothing. The labels are **drafts awaiting review**
  ([LABEL_REVIEW.md](docs/LABEL_REVIEW.md)), so these numbers are provisional. Search holds up
  (hybrid recall@3 **89%**). The no-AI baseline does not: 58% precision, 38% impact accuracy, and
  it never abstains. The agent has not been scored on this set yet.
- **Prompt injection.** A regulation could contain `</regulation>` followed by a fake `SYSTEM:`
  line, which escaped the block the model is told is data. That's fixed, and
  [`tests/test_injection.py`](tests/test_injection.py) checks it. The tests also check that
  planted instructions in a policy can't change what the fact-checker accepts and reach the model
  only as tool data. An [adversarial copy](data/regulations_adversarial/) of a regulation, with
  four planted instructions, is ready for an agent run.
- **Scale** ([SCALE_RESULTS.md](docs/SCALE_RESULTS.md)). I grew the library from 51 to 10,051
  clauses with real contract provisions as distractors.

  | Clauses | Recall@3 (hybrid + rerank) | Search time p50 |
  |---|---|---|
  | 51 | 96% | 0.6 s |
  | 10,051 | 84% | 2.3 s |

  The cross-encoder is both the slowest step and, with distractors, the one that *lowers* recall.
  Hybrid without it holds 86% in 0.11 s, about 20 times faster. The real-regulation set points the same way
  (89% without, 86% with). Embedding the large library also exposed a memory bug: fastembed's
  default batch used 6.9 GB and was killed. A smaller batch fixed it (3.5 GB peak).

---

## How it works

```mermaid
flowchart LR
    A[New regulation] --> B[Split into<br/>requirements]
    B --> C{{AI agent}}
    C -- "search_policies" --> D[(Hybrid index<br/>BM25 + embeddings<br/>→ RRF → rerank)]
    D --> C
    C -- "read_clause" --> E[(51 policy clauses)]
    C -- "submit_report" --> F[Fact-checker<br/>verbatim quote check]
    F -- "rejected: closest real<br/>sentence is …" --> C
    F -- "all quotes verified" --> G[Reviewer approves<br/>or rejects]
```

**Engineering decisions worth discussing:**

| Decision | Why |
|---|---|
| Chunk at each policy's own numbering ("3.2 Data Retention"), not fixed token windows | A citation should point at a whole rule a human recognises |
| Hybrid retrieval instead of embeddings alone | Legal text mixes exact terms (V-CIP, T+1, AES-256) with paraphrases; the comparison table above shows the gain |
| Verify quotes with plain code, not a second LLM | A substring check can't hallucinate, costs nothing and is explainable |
| Send failures back *with the closest real sentence* | The model corrects itself instead of guessing again |
| Hand-written agent loop (~150 lines per provider) instead of a framework | Every step is visible and unit-testable |
| One agent interface, two providers (Anthropic and Groq/OpenAI-style) | The shared logic sits in `agent/base.py`; each provider file is just its message format |
| Tests run the real SDKs against a scripted fake API | The agent loop, retries and verify-and-fix flow are tested with no key or network |
| Offline demo mode | Anyone can clone and run it for free, and it doubles as the evaluation baseline |

More in [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md): requirements, sequence diagrams, data model,
failure modes (including prompt injection), and a scaling plan.

---

## Run it in 3 minutes

Needs **Python 3.10+**. Works without any API key, in demo mode.

```bash
git clone https://github.com/vik-bit/policypilot.git
```
```bash
cd policypilot
```
```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
<sub>(macOS / Linux: `source .venv/bin/activate`)</sub>

```bash
pip install -r requirements.txt
```
```bash
python -m policypilot serve
```

Open **http://localhost:8000**. The **How to use** page in the app walks through everything, and
**See an example AI report** opens a real report from the Groq agent without needing an API key.

**Turn on the AI agent:** copy `.env.example` to `.env` and choose a provider:

| Provider | Add to `.env` |
|---|---|
| **Groq** (free tier; open-weight models such as GPT-OSS 120B) | `PP_PROVIDER=groq` · `GROQ_API_KEY=gsk_...` |
| **Claude** (Anthropic) | `PP_PROVIDER=claude` · `ANTHROPIC_API_KEY=sk-ant-...` |

**Or with Docker:** `docker compose up`

**Host a public demo:** [`render.yaml`](render.yaml) deploys a read-only demo on Render's free plan
in one click (no API keys, uploads off, ~70 MB RAM).

| Command | What it does |
|---|---|
| `python -m policypilot analyze reg-01-data-protection` | Analyse a regulation from the terminal |
| `python -m policypilot search "how long do we keep data?"` | Try the search step on its own |
| `python -m policypilot eval` | Re-run the evaluation and regenerate `docs/EVAL_RESULTS.md` |
| `python -m policypilot eval --suite real --agent-runs 3` | Score the held-out real regulations, with 3 agent runs as the plan requires (`--suite adversarial` for the injection set) |
| `python -m policypilot stats` | Confidence intervals and significance tests from the saved results, with no API calls |
| `python -m policypilot scale` | Grow the library to 10,000 clauses and measure search quality and speed |
| `python -m pytest` | Run the 47 tests (no API key or internet needed) |

---

## The problem, in plain English

Banks and payment apps run on rulebooks called **policies**: "we keep customer data for 10 years",
"we report data breaches within 72 hours". Regulators keep publishing **new rules**, and every time
someone has to find what's now out of date and rewrite it.

> **An analogy.** Think of a smart intern with a strict editor. The intern (the AI) researches and
> writes the first draft. The editor (plain code, not AI) checks every quote against the original
> documents and sends the draft back if anything is made up. A manager (you) has the final say.

New to the terms? [docs/GLOSSARY.md](docs/GLOSSARY.md) explains RAG, embeddings, agents and more
in one or two sentences each. [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) follows one real run step by step.

---

## Project structure

```
policypilot/
├── policypilot/
│   ├── pipeline.py          ← start here: the whole flow in one function
│   ├── ingest/              ← split documents into clauses and requirements
│   ├── retrieval/           ← BM25 (from scratch), embeddings, RRF fusion, cross-encoder rerank
│   ├── agent/
│   │   ├── base.py          ← shared tool running + fact-check submission
│   │   ├── claude_agent.py  ← Anthropic Messages API loop
│   │   ├── groq_agent.py    ← Groq (OpenAI-style) loop, with retries for malformed model output
│   │   ├── verifier.py      ← deterministic quote checker
│   │   ├── prompts.py       ← system prompt + tool schemas
│   │   └── offline.py       ← no-AI baseline
│   ├── api/server.py        ← FastAPI: background jobs, search, uploads, decisions
│   ├── web/                 ← UI (HTML/CSS/JS, no build step)
│   └── eval/
│       ├── run_eval.py      ← retrieval and end-to-end scoring for each test set
│       ├── stats.py         ← bootstrap confidence intervals, McNemar test
│       └── scale.py         ← benchmark with 10,000 distractor clauses
├── data/                    ← policies and sample regulations (fictional), real regulation excerpts, labels
├── docs/                    ← system design, eval plan and results, label review, glossary, screenshots
└── tests/                   ← 47 tests, including both agent loops and injection checks against fake APIs
```

---

## Limitations and next steps

- **Small data set, fictional policies:** 51 policy clauses, 63 labelled requirements (25 fictional,
  38 from real regulations with labels still in draft). A production evaluation would need real
  company policies, 100+ requirements and several reviewers.
- **The agent has not yet been scored on the real or adversarial sets.** That needs API credits:
  `eval --suite real --agent-runs 3`.
- **The reranker needs rethinking** for large libraries: it is the slowest step and hurt recall
  once distractors were added.
- **The fact-checker proves a quote exists, not that the interpretation is right.** That's why a human
  approves every change.
- **Next:** analyse only what changed between versions of a regulation; prompt caching to cut cost;
  a vector database past ~10k clauses; an LLM judge for explanation quality.

---

<sub>Sample documents describe a fictional company ("Northwind Pay") and a fictional regulator. Not legal advice. MIT licensed.</sub>
