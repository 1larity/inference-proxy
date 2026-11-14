"""Model prices (kept current by refresh-and-approve) and a ledger of what requests really cost.

Prices are per million tokens, in US dollars. A refresh never changes the active table: it
writes proposals (new, changed, retired) that are applied or rejected afterwards. Each request
is costed at the price active when it ran, so history stays correct when prices move.
"""

from __future__ import annotations

import sqlite3
import time

SCHEMA = """
create table if not exists prices(
  id integer primary key, provider text not null, model text not null,
  input_per_m real, output_per_m real, context integer,
  source text, source_url text, fetched_at real, status text not null default 'active',
  old_input real, old_output real, kind text);
create index if not exists prices_lookup on prices(provider, model, status);
create table if not exists usage(
  id integer primary key, ts real, provider text, model text, tier text,
  input_tokens integer, output_tokens integer, cost_usd real, price_fetched_at real);
create index if not exists usage_ts on usage(ts);
"""


def extract_price(raw: dict) -> tuple[float, float, int | None] | None:
    """Price per million tokens from a /models entry in the OpenRouter shape (strings, per token)."""
    pricing = raw.get("pricing") if isinstance(raw, dict) else None
    if not isinstance(pricing, dict):
        return None
    try:
        prompt, completion = float(pricing["prompt"]) * 1e6, float(pricing["completion"]) * 1e6
    except (KeyError, TypeError, ValueError):
        return None
    if prompt < 0 or completion < 0:  # negative marks "dynamic/varies" on some gateways
        return None
    context = raw.get("context_length")
    return round(prompt, 6), round(completion, 6), int(context) if isinstance(context, (int, float)) else None


class PriceStore:
    def __init__(self, path: str | None = None) -> None:
        self.db = sqlite3.connect(path or ":memory:", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---- prices ----
    def active(self, provider: str, model: str) -> sqlite3.Row | None:
        return self.db.execute("select * from prices where provider=? and model=? and status='active'", (provider, model)).fetchone()

    def listing(self, status: str | None = None) -> list[dict]:
        sql = "select * from prices" + (" where status=?" if status else " where status in ('active','proposed')") + " order by provider, model, status"
        return [dict(r) for r in self.db.execute(sql, (status,) if status else ())]

    def propose(self, provider: str, model: str, price: tuple[float, float, int | None] | None, source: str, url: str | None = None) -> str | None:
        """Record a proposal if it differs from the active price. Returns new, changed, retired or None."""
        current = self.active(provider, model)
        self.db.execute("delete from prices where provider=? and model=? and status='proposed'", (provider, model))
        if price is None:
            if current is None:
                return None
            kind, inp, out, ctx = "retired", current["input_per_m"], current["output_per_m"], current["context"]
        else:
            inp, out, ctx = price
            if current is None:
                kind = "new"
            elif abs(current["input_per_m"] - inp) < 1e-9 and abs(current["output_per_m"] - out) < 1e-9:
                self.db.execute("update prices set fetched_at=? where id=?", (time.time(), current["id"]))
                self.db.commit()
                return None
            else:
                kind = "changed"
        self.db.execute(
            "insert into prices(provider,model,input_per_m,output_per_m,context,source,source_url,fetched_at,status,old_input,old_output,kind) "
            "values(?,?,?,?,?,?,?,?, 'proposed', ?,?,?)",
            (provider, model, inp, out, ctx, source, url, time.time(),
             current["input_per_m"] if current else None, current["output_per_m"] if current else None, kind))
        self.db.commit()
        return kind

    def apply(self, ids: list[int] | None = None) -> int:
        """Make proposals active (all of them when ids is None). Returns how many were applied."""
        rows = self.db.execute("select * from prices where status='proposed'" + (f" and id in ({','.join('?' * len(ids))})" if ids else ""), ids or ()).fetchall()
        for r in rows:
            self.db.execute("delete from prices where provider=? and model=? and status='active'", (r["provider"], r["model"]))
            if r["kind"] == "retired":
                self.db.execute("delete from prices where id=?", (r["id"],))
            else:
                self.db.execute("update prices set status='active', kind=null where id=?", (r["id"],))
        self.db.commit()
        return len(rows)

    def reject(self, ids: list[int] | None = None) -> int:
        cur = self.db.execute("delete from prices where status='proposed'" + (f" and id in ({','.join('?' * len(ids))})" if ids else ""), ids or ())
        self.db.commit()
        return cur.rowcount

    def set_manual(self, provider: str, model: str, input_per_m: float, output_per_m: float, context: int | None = None) -> None:
        if input_per_m < 0 or output_per_m < 0:
            raise ValueError("prices cannot be negative")
        self.db.execute("delete from prices where provider=? and model=?", (provider, model))
        self.db.execute("insert into prices(provider,model,input_per_m,output_per_m,context,source,fetched_at,status) values(?,?,?,?,?,'manual',?,'active')",
                        (provider, model, input_per_m, output_per_m, context, time.time()))
        self.db.commit()

    # ---- usage ledger ----
    def record_usage(self, provider: str, model: str | None, tier: str | None, usage: dict | None) -> float | None:
        """Store one request's tokens and cost at the price active now. Returns the cost, or None when unknown."""
        if not isinstance(usage, dict) or not model:
            return None
        inp, out = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        price = self.active(provider, model)
        cost = (inp * price["input_per_m"] + out * price["output_per_m"]) / 1e6 if price else None
        self.db.execute("insert into usage(ts,provider,model,tier,input_tokens,output_tokens,cost_usd,price_fetched_at) values(?,?,?,?,?,?,?,?)",
                        (time.time(), provider, model, tier, inp, out, cost, price["fetched_at"] if price else None))
        self.db.commit()
        return cost

    def usage_summary(self, days: float = 7.0) -> list[dict]:
        rows = self.db.execute(
            "select provider, model, tier, count(*) requests, sum(input_tokens) input_tokens, sum(output_tokens) output_tokens, "
            "sum(cost_usd) cost_usd, sum(cost_usd is null) unpriced from usage where ts >= ? group by provider, model, tier order by cost_usd desc",
            (time.time() - days * 86400,)).fetchall()
        return [dict(r) for r in rows]
