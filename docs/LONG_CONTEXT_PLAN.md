# Long context vs RAG: comparison plan (pre-registered)

_Written on 26 November 2026, before any model was run in long-context mode. The grid, metrics,
tests and decision rules below are fixed. A later change goes in the [change log](#change-log)
with its reason, and results under the old plan stay as measured. This plan extends
[EVAL_PLAN.md](EVAL_PLAN.md) and uses its metrics and statistics unless it says otherwise._

## Question

PolicyPilot finds the clauses a requirement affects by searching the library (hybrid BM25 +
embeddings) and reading the top results. An open model with a 32K-token window can instead read
the whole library at once. Does that find more affected clauses, at what cost in precision,
latency and tokens, and does it depend on where the relevant clause sits in the prompt?

## What is held fixed

To compare *context*, everything else is the same in every configuration:

- **Reader:** Qwen2.5-7B-Instruct, 4-bit (`qwen2.5:7b-instruct`, Q4_K_M GGUF), native 32,768-token
  window, served by Ollama with `num_ctx 32768`, on one free T4 (Colab or Kaggle). No paid API.
  Before each run the grid sends its largest prompt once and stops if the server read less than
  all of it (Ollama silently drops the start of prompts longer than its context setting).
- **Prompt:** the system prompt and JSON output format in `policypilot/longctx/prompts.py`; one
  call per requirement; clauses tagged with the same ids search uses.
- **Fact-check loop:** the existing verifier, up to 2 retries (`PP_MAX_VERIFY_RETRIES`).
- **Sampling:** temperature 0.2, seed = run number (1, 2, 3).
- **Budget:** a prompt may use at most 27,136 tokens (32,768 minus room for 3 replies of 1,536
  tokens and 2 error messages of 512), counted with the Qwen tokenizer. A prompt over budget is not
  sent; the requirement is recorded as "did not fit" and scored as no finding.

The RAG comparator is therefore **single-shot RAG with the same reader** (hybrid top 5, which is
the agent's default `search_policies` depth). It is not the tool-using agent: the agent also
writes its own queries and runs on a different model (gpt-oss-120b), so comparing with it would
mix model, retrieval and agent loop. The agent's own `real`-set runs are reported separately in
[EVAL_PLAN.md](EVAL_PLAN.md)'s results and are not used in this comparison.

## Test set

The `real` set: 38 requirements from 6 real Indian regulations (35 applicable, 3 not applicable),
labels in `data/eval/labels_real.jsonl`. The labels are still **drafts** (EVAL_PLAN.md), so every
number from this plan is **provisional** until they are reviewed. The `sample` set is not used:
it was seen while building the system. The held-out rule of EVAL_PLAN.md applies: the prompt and
reader are not changed because of anything seen on the `real` set; a bug found while scoring is
fixed and both numbers are reported.

## Library sizes

The 51 Northwind Pay clauses, plus LEDGAR contract provisions (as in the scale benchmark) as
distractors that are never a correct answer, at fixed random positions (seed 0):

| Clauses | Distractors | `full` prompt tokens (mean / max over the 38 requirements) |
|---|---|---|
| 51 | 0 | 4,055 / 4,099 |
| 101 | 50 | 11,421 / 11,465 |
| **184** | 133 | 27,029 / 27,073: the largest library whose `full` prompt fits the 27,136-token budget for every requirement (185 clauses: max 27,347) |

Measured with `python -m policypilot.longctx tokens` (Qwen tokenizer) before any model call.
Libraries above 184 clauses do not fit a 32K window; for them the app falls back to search (see
`auto` below), which is what RAG is for.

## Context variants

| Variant | What is in the prompt |
|---|---|
| `rag5` | Hybrid search top 5, best first. **The RAG baseline.** |
| `hybrid50` | Hybrid search top 50, best first (retrieve wide, then read long). At 51 clauses this is nearly the whole library in relevance order. |
| `full` | The whole library in one fixed order, identical for every requirement (so its prefix is cacheable). |
| `random` | The whole library in a random order drawn per requirement (seeded). |
| `bm25` | The whole library sorted by BM25 score for the requirement, best first. |
| `pos-start`, `pos-middle`, `pos-end` | `random` order, with the labelled relevant clauses moved as one block to the start, the middle (centred at 50% depth) or the end. Applicable requirements only (35). |

`auto` (whole library if it fits, else hybrid top 50/40/30/20/10/5, whichever fits first) is the
mode the app uses. It is not scored separately: at every size in the grid it equals `full`.

## Grid

| Clauses | Variants | Runs |
|---|---|---|
| 51 | `rag5`, `full` | 3 each |
| 51 | `hybrid50`, `random`, `bm25`, `pos-start`, `pos-middle`, `pos-end` | 1 each |
| 101 | `rag5`, `hybrid50`, `full`, `bm25` | 1 each |
| 184 | `rag5`, `full` | 3 each |
| 184 | `hybrid50`, `random`, `bm25`, `pos-start`, `pos-middle`, `pos-end` | 1 each |

About 1,000 model calls. The grid is in `policypilot/eval/long_context.py` (`GRID`) and runs with
`python -m policypilot.longctx eval`; it checkpoints every requirement so it can span several
Kaggle sessions. Before it, `--grid smoke` checks the server on 5 requirements of the `sample`
(development) set, so a setup problem never costs a held-out run.

## Metrics

Per configuration, from EVAL_PLAN.md: clause **recall**, **precision**, **impact accuracy**,
**fully correct requirement**, **abstention**, and **quote-verification pass rate** on the first
reply and after the fact-check loop. Added for this comparison:

- **Context recall:** share of labelled relevant clauses that were in the prompt at all (an upper
  bound on what the reader can find; 100% for the full-library variants).
- **Latency:** wall time per requirement, all calls included; p50 and p95.
- **Prompt tokens per requirement** (as reported by the server, retries included) and the share
  served from the server's prefix cache when the server reports it (vLLM does; Ollama reuses the
  cached prefix but does not report it, so with Ollama this column reads "not reported" and the
  cache shows only in latency).
- **Cost per requirement:** GPU-seconds on the free T4 (= latency, one request at a time) and
  tokens. The dollar cost is zero; tokens are reported so any price list can be applied.
- Requirements that did not fit, server errors, JSON parse failures, and findings left unverified.
  A finding the verifier rejects is never counted as verified (the core safety property).

## Statistics

- 95% percentile bootstrap intervals over requirements (10,000 resamples, seed 0), ratios
  recomputed from summed counts, as in EVAL_PLAN.md.
- **Against RAG:** every variant is paired with `rag5` at the same size and run number. Reported:
  the difference in each metric with a **paired bootstrap** interval (same resampled requirements
  for both), and the **exact McNemar** test on *fully correct requirement*, two-sided.
- **Position sensitivity:** metric = share of applicable requirements with *every* relevant clause
  flagged. `pos-start` vs `pos-middle` and `pos-end` vs `pos-middle`, exact McNemar, Holm-corrected
  for the two tests at each size.
- Repeated runs: `rag5` and `full` at 51 and 184 clauses are run 3 times and every run is reported.

## Pre-registered hypotheses and decision rules

Primary comparison: `full` vs `rag5`, **run 1**, at **51** and at **184** clauses (two tests, Holm
corrected). Runs 2 and 3 are reported to show stability, not tested.

| | Hypothesis | Counts as supported if |
|---|---|---|
| H1 | At 51 clauses, `full` has higher recall than `rag5` (RAG misses clauses search does not surface). | Paired recall difference interval is above 0. |
| H2 | At 184 clauses, `full` loses precision to distractors relative to `rag5`. | Paired precision difference interval is below 0. |
| H3 | A relevant clause in the middle is found less often than at the start or end (lost in the middle). | Either Holm-corrected McNemar p < 0.05 at 184 clauses, with middle lower. |
| H4 | `hybrid50` matches `full` recall at lower latency. | Recall difference interval vs `full` (run 1) includes 0 *and* p50 latency is lower. |

**Recommendation rule** (for the README), fixed now:

- Recommend **long context as the default** at a library size only if `full` is not worse than
  `rag5` on fully-correct requirements (McNemar p ≥ 0.05 or `full` ahead) **and** its recall
  interval difference is above 0.
- Otherwise recommend **RAG as the default** at that size, and long context only where the table
  shows a specific win (named in the README).
- Above the 32K budget, RAG (or `auto`'s fallback) is the only option; this is stated, not tested.

## Threats to validity

- One 7B model, 4-bit. A stronger reader may use long context better (or RAG better).
- Draft labels; 35 applicable requirements, so intervals will be wide and small differences will
  not be detectable. A null result is reported as "no detectable difference", not as "equal".
- LEDGAR distractors are contract clauses, easy to tell apart from payments policy. Real large
  libraries have near-duplicate policies, which would hurt long context more.
- The relevant-clause block in the position test is placed using the labels; it measures position
  sensitivity, not a usable system.
- The 32K window is the model's native one. Extending Qwen2.5 to 128K with YaRN is not part of the
  grid (its static scaling changes behaviour on short prompts, and a T4 is too slow at 100K tokens).
  For reference, a 131,072-token window would hold a 744-clause library under the same budget rule.

## Change log

| Date | Change | Reason |
|---|---|---|
| 26 Nov 2026 | First version | — |
| 26 Nov 2026 | Named the smoke check (`sample` set, 5 requirements) and the 128K reference size | Clarification before any run; no metric, test or grid cell changed |
| 27 Nov 2026 | Server changed from vLLM (AWQ weights) to Ollama (Q4_K_M weights of the same model) on one T4; added the whole-prompt check before each run | vLLM 0.6.6, the last release known to run on a T4, cannot be installed on Colab's Python 3.13 (no matching torch build). Changed before any model call on any set; grid, metrics, tests and rules unchanged |
