"""Independent model scores: Epoch AI's Capabilities Index (CC BY 4.0) and Artificial Analysis (own free API key, credit required).

Scores are hints for ordering models, not proof of quality. Each row keeps its source so the page can show where a number came from.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import zipfile

import httpx

EPOCH_URL = "https://epoch.ai/data/benchmark_data.zip"
EPOCH_CITATION = "Epoch AI, 'Capabilities & benchmarking'. Published online at epoch.ai (CC BY 4.0)."
AA_URL = "https://artificialanalysis.ai/api/v2/language/models/free"
AA_CREDIT = "Model intelligence and price data by Artificial Analysis (artificialanalysis.ai)."
AA_PAGE_LIMIT = 5  # the free tier allows 100 requests per 24 hours


def normalise(name: str) -> str:
    """'GLM-5.3-Flash', 'glm_5.3 flash' and 'z-ai/glm-5.3-flash' all become 'glm-5.3-flash'."""
    name = name.strip().lower().rsplit("/", 1)[-1]
    name = re.sub(r"[^a-z0-9.]+", "-", name).strip("-")
    return name


def candidate_keys(model_id: str) -> list[str]:
    key = normalise(model_id)
    keys = [key]
    for suffix in ("-gguf", "-instruct", "-it", "-chat"):
        if key.endswith(suffix):
            keys.append(key[: -len(suffix)])
    return list(dict.fromkeys(keys))


def parse_epoch(zip_bytes: bytes) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        name = next((n for n in z.namelist() if n.endswith("eci_scores.csv")), None)
        if name is None:
            raise ValueError("eci_scores.csv not found in the Epoch download")
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")))
    out = []
    for r in rows:
        try:
            score = float(r["eci"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"name": r.get("Display name") or r.get("Model"), "keys": [normalise(r["Model"]), normalise(r.get("Display name") or r["Model"])],
                    "score": score, "detail": {"low": r.get("eci_ci_low"), "high": r.get("eci_ci_high"), "date": r.get("date"),
                                               "organization": r.get("Organization"), "accessibility": r.get("Model accessibility")}})
    return out


def parse_aa(pages: list[dict]) -> list[dict]:
    out = []
    for page in pages:
        for m in page.get("data", []):
            evals = m.get("evaluations") or {}
            score = evals.get("artificial_analysis_intelligence_index")
            if not isinstance(score, (int, float)):
                continue
            pricing, perf = m.get("pricing") or {}, m.get("performance") or {}
            out.append({"name": m.get("name") or m.get("slug"), "keys": [normalise(m.get("slug") or m.get("name") or ""), normalise(m.get("name") or "")],
                        "score": float(score),
                        "detail": {"creator": (m.get("model_creator") or {}).get("name"), "input_per_m": pricing.get("price_1m_input_tokens"),
                                   "output_per_m": pricing.get("price_1m_output_tokens"), "tokens_per_s": perf.get("median_output_tokens_per_second"),
                                   "version": page.get("intelligence_index_version")}})
    return out


async def fetch_epoch(http: httpx.AsyncClient) -> list[dict]:
    response = await http.get(EPOCH_URL, timeout=120.0, follow_redirects=True)
    response.raise_for_status()
    return parse_epoch(response.content)


async def fetch_aa(http: httpx.AsyncClient, api_key: str) -> list[dict]:
    pages, page = [], 1
    while page <= AA_PAGE_LIMIT:
        response = await http.get(AA_URL, params={"page": page}, headers={"x-api-key": api_key}, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        pages.append(data)
        if not (data.get("pagination") or {}).get("has_more"):
            break
        page += 1
    return parse_aa(pages)


SOURCES = {"epoch": ("Epoch AI Capabilities Index", EPOCH_CITATION), "aa": ("Artificial Analysis Intelligence Index", AA_CREDIT)}


class ScoreStore:
    """Scores live in the same SQLite file as prices (a PriceStore's connection)."""

    def __init__(self, db) -> None:
        self.db = db
        db.executescript("""create table if not exists scores(
            source text, key text, name text, score real, detail text, fetched_at real, primary key(source, key));""")

    def replace(self, source: str, rows: list[dict]) -> int:
        self.db.execute("delete from scores where source=?", (source,))
        now = time.time()
        for r in rows:
            for key in dict.fromkeys(k for k in r["keys"] if k):
                self.db.execute("insert or ignore into scores values(?,?,?,?,?,?)", (source, key, r["name"], r["score"], json.dumps(r["detail"]), now))
        self.db.commit()
        return len(rows)

    def lookup(self, model_id: str, source: str | None = None) -> dict | None:
        """The best-known score for a model id (exact normalised match), with its source."""
        for key in candidate_keys(model_id):
            sql = "select * from scores where key=?" + (" and source=?" if source else "")
            row = self.db.execute(sql, (key, source) if source else (key,)).fetchone()
            if row:
                return {"source": row["source"], "name": row["name"], "score": row["score"], "detail": json.loads(row["detail"] or "{}"), "fetched_at": row["fetched_at"]}
        return None

    def summary(self) -> dict:
        return {s: {"models": self.db.execute("select count(*) from scores where source=?", (s,)).fetchone()[0],
                    "fetched_at": (self.db.execute("select max(fetched_at) from scores where source=?", (s,)).fetchone()[0]),
                    "label": SOURCES[s][0], "credit": SOURCES[s][1]} for s in SOURCES}
