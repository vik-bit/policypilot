"""Scale benchmark: what happens to search when the policy library is 200 times bigger?

Plain English: the sample library has 51 clauses, so every search is easy and instant. A real
company might have thousands. We add real contract provisions (LEDGAR: clauses from SEC filings
about confidentiality, governing law, indemnities and so on) as distractors that are never the
right answer, then re-run the retrieval evaluation and time every search. Two questions:
- Does the right clause still reach the top 3 when it competes with 10,000 others?
- How much slower does each search method get?

LEDGAR (Tuggener et al., 2020) is used through the LexGLUE benchmark (CC BY 4.0). It is downloaded
on first use and cached in data/bench/, which is not committed.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import date
from pathlib import Path

from ..config import PROJECT_ROOT, Settings
from ..ingest import load_policies
from ..models import Clause
from ..retrieval import PolicyIndex
from ..retrieval.embedders import make_embedder
from ..retrieval.index import _search_text
from ..retrieval.rerank import make_reranker
from .run_eval import eval_retrieval

ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=coastalcph%2Flex_glue&config=ledgar"
            "&split=train&offset={offset}&length=100")
DISTRACTOR_POLICY = "ledgar"


def _fetch_page(offset: int, attempts: int = 7) -> dict:
    request = urllib.request.Request(ROWS_URL.format(offset=offset), headers={"User-Agent": "policypilot-bench/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except OSError as exc:
            if attempt == attempts - 1:
                raise
            # The API rate-limits bursts (HTTP 429). Wait as long as it asks, else back off.
            retry_after = getattr(exc, "headers", None) and exc.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else min(5 * 2 ** attempt, 120)
            print(f"  download paused {wait:.0f} s ({exc})", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def load_distractors(cache_dir: Path, n: int) -> list[Clause]:
    """The first n LEDGAR provisions as Clause objects, downloading any not yet cached."""
    cache = cache_dir / "ledgar.jsonl"
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()] if cache.exists() else []
    if len(rows) < n:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {n - len(rows):,} LEDGAR provisions…", flush=True)
        names = None
        with cache.open("a", encoding="utf-8") as out:
            while len(rows) < n:
                page = _fetch_page(len(rows))
                names = names or page["features"][1]["type"]["names"]
                for item in page["rows"]:
                    row = {"text": item["row"]["text"], "label": names[item["row"]["label"]]}
                    rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                time.sleep(0.5)  # stay under the API's rate limit
    return [Clause(id=f"{DISTRACTOR_POLICY}§{i}", policy=DISTRACTOR_POLICY, policy_title="Contract provisions (LEDGAR)",
                   number=str(i), title=row["label"], text=row["text"]) for i, row in enumerate(rows[:n], 1)]


def _distractors_in_top3(index: PolicyIndex, settings: Settings) -> float:
    """Share of top-3 slots (best method) taken by a distractor, over the sample requirements."""
    from ..ingest import load_regulations

    taken = total = 0
    for reg in load_regulations(settings.regulations_dir):
        for req in reg.requirements:
            hits = index.search(f"{req.title}. {req.text}", k=3)
            taken += sum(h.clause.policy == DISTRACTOR_POLICY for h in hits)
            total += len(hits)
    return taken / max(total, 1)


def run(settings: Settings, sizes: list[int]) -> str:
    embedder = make_embedder(settings.embedder, settings.embed_model)
    reranker = make_reranker(settings.rerank, settings.rerank_model)
    policy_clauses = load_policies(settings.policies_dir)
    distractors = load_distractors(settings.data_dir / "bench", max(sizes))
    clauses = policy_clauses + distractors
    texts = [_search_text(c) for c in clauses]
    embedder.fit(texts)

    print(f"Embedding {len(texts):,} clauses with {embedder.name}…", flush=True)
    started = time.perf_counter()
    vectors = embedder.embed_documents(texts)
    embed_seconds = time.perf_counter() - started

    rows = []
    for n in sorted(sizes):
        size = len(policy_clauses) + n
        started = time.perf_counter()
        index = PolicyIndex(clauses[:size], embedder, vectors[:size], reranker)
        build_seconds = time.perf_counter() - started
        retrieval = eval_retrieval(index, settings, suite="sample")
        for r in retrieval.values():
            r.pop("per_query")
        rows.append({"clauses": size, "distractors": n, "index_build_seconds": build_seconds,
                     "distractor_share_top3": _distractors_in_top3(index, settings), "methods": retrieval})
        best = retrieval.get("hybrid+rerank") or retrieval["hybrid"]
        print(f"  {size:>6,} clauses: recall@3 {best['recall@3']:.2f}, p95 {best['ms_p95']:.0f} ms", flush=True)

    results = {"embedder": embedder.name, "reranker": reranker.name if reranker else None,
               "embed_seconds": embed_seconds, "embedded_clauses": len(texts), "rows": rows}
    (settings.data_dir / "eval" / "scale_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = write_markdown(results)
    (PROJECT_ROOT / "docs" / "SCALE_RESULTS.md").write_text(md, encoding="utf-8")
    return md


def write_markdown(results: dict) -> str:
    rows = results["rows"]
    methods = list(rows[0]["methods"])
    names = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid", "hybrid+rerank": "Hybrid + rerank"}
    lines = [
        "# Scale benchmark",
        "",
        f"_Generated by `python -m policypilot scale` on {date.today().isoformat()}. The 51 Northwind Pay "
        f"clauses plus up to {rows[-1]['distractors']:,} real contract provisions (LEDGAR, CC BY 4.0) as "
        f"distractors. Scored on the 25 `sample` requirements. Embedder `{results['embedder']}`, reranker "
        f"`{results['reranker'] or 'off'}`. One run on a laptop (Intel i7-1255U, WSL2)._",
        "",
        "## Does search still find the right clause?",
        "",
        "| Library size | " + " | ".join(f"{names[m]} recall@3" for m in methods)
        + " | Top-3 slots taken by distractors |",
        "|---|" + "---|" * (len(methods) + 1),
    ]
    for r in rows:
        lines.append(f"| {r['clauses']:,} | " + " | ".join(f"{r['methods'][m]['recall@3']:.0%}" for m in methods)
                     + f" | {r['distractor_share_top3']:.0%} |")
    lines += [
        "",
        "## How long does one search take?",
        "",
        "Per search, in milliseconds (p50 / p95 over the 25 queries). Every method computes both the "
        "BM25 and the dense ranking, because the hybrid fuses them.",
        "",
        "| Library size | " + " | ".join(f"{names[m]} p50 / p95" for m in methods) + " | Index build |",
        "|---|" + "---|" * (len(methods) + 1),
    ]
    for r in rows:
        lines.append(f"| {r['clauses']:,} | " + " | ".join(
            f"{r['methods'][m]['ms_p50']:.0f} / {r['methods'][m]['ms_p95']:.0f}" for m in methods)
            + f" | {r['index_build_seconds']:.1f} s |")
    per_1k = 1000 * results["embed_seconds"] / results["embedded_clauses"]
    lines += ["", f"Embedding the {results['embedded_clauses']:,} clauses once took {results['embed_seconds']:.0f} s "
              f"({per_1k:.1f} s per 1,000 clauses). In the app this is paid once per library change, because "
              f"vectors are cached in `data/index/`; this benchmark re-embeds on every run.", ""]
    return "\n".join(lines)
