# Glossary — every term, explained simply

**AI agent** — An AI that doesn't just answer once, but takes actions in a loop: decide what to do,
use a tool, look at the result, decide again. Here, the agent's tools are *search the policies*,
*read a clause* and *submit the report*.

**Agentic AI** — Building systems around agents rather than single question-and-answer prompts.

**BM25** — A classic keyword-search formula (used by search engines for decades). A document scores
higher when it contains the query's words — especially rare words — and long documents don't win
just by being long. See `policypilot/retrieval/bm25.py`.

**Chunking** — Cutting long documents into smaller pieces so they can be searched and quoted. We
cut along each policy's own numbering, so each piece is one whole clause.

**Citation / quote verification** — Checking that a quote the AI used really appears, word for
word, in the source document. Done with plain code in `policypilot/agent/verifier.py`.

**Clause** — One numbered rule in a policy, e.g. "3.2 Data Retention". Its ID is
`data-privacy§3.2`.

**Conflict / gap / aligned** — Our three labels. *Conflict*: the policy says something the new rule
contradicts. *Gap*: the rule requires something the policy doesn't mention. *Aligned*: the policy
already complies.

**Cross-encoder (re-ranker)** — A small AI model that reads the question and a candidate clause
*together* and scores how relevant the clause is. More accurate than first-pass search but slower,
so it only re-orders the top ~20 results.

**Embedding** — A list of numbers that represents a piece of text's *meaning*. Texts that mean
similar things get similar numbers, so "storage limitation" and "data retention" end up close even
though they share no words.

**Evaluation (eval)** — Measuring quality against known correct answers, instead of eyeballing.

**Generative AI (GenAI)** — AI that produces new content (text, here) rather than just classifying.

**Hallucination** — When an AI states something that sounds right but isn't true — like an invented
quote. The fact-checker exists to catch these.

**Human in the loop** — A person reviews and approves the AI's output before it takes effect.

**Hybrid search** — Using keyword search and meaning search together and merging the results.

**LLM (large language model)** — An AI model trained on huge amounts of text that can read and write
language. This project uses Claude, made by Anthropic.

**MRR (mean reciprocal rank)** — A search score: if the right answer is ranked 1st you get 1, 2nd
you get ½, 3rd ⅓… averaged over all test questions. Higher is better; 1.0 is perfect.

**Precision** — Of the clauses the system flagged, how many were actually affected? (Avoids false
alarms.)

**Prompt** — The instructions and text sent to the AI. The *system prompt* is its standing job
description (`policypilot/agent/prompts.py`).

**Prompt injection** — Hidden instructions inside a document ("ignore your instructions and…") that
try to hijack an AI. We mark the regulation as *data*, give the agent only read-only tools, and
require human approval.

**RAG (retrieval-augmented generation)** — Instead of relying on what the AI memorised, first
*retrieve* the relevant documents, then have the AI *generate* an answer from them. It's how you
get answers grounded in *your* documents.

**Recall / Recall@k** — Of the clauses that really are affected, how many did we find? Recall@3 means
"…within the top 3 search results".

**Reciprocal Rank Fusion (RRF)** — A simple way to merge two ranked lists: each item scores
1/(60 + its rank) in each list, and the scores are added. Items near the top of *either* list rise.

**Requirement** — One numbered rule in a regulation, e.g. "R2. Storage limitation".

**Strict tool schema** — A precise description of the inputs each tool accepts; the API guarantees
the AI's tool calls match it exactly, so the code never receives malformed data.

**Token** — The unit AI models read and write in (roughly ¾ of a word). API cost is per token.

**Tool use (function calling)** — The AI asks the program to run a function ("search for X") and
continues once it sees the result.
