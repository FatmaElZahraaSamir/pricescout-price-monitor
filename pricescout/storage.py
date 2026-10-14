"""SQLite price history and change detection between scraping runs."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from .pipeline import ScrapeResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url    TEXT NOT NULL,
    category      TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    n_products    INTEGER NOT NULL,
    n_errors      INTEGER NOT NULL,
    duration_s    REAL
);
CREATE TABLE IF NOT EXISTS snapshots (
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    product_key   TEXT NOT NULL,
    title         TEXT NOT NULL,
    price         REAL NOT NULL,
    currency      TEXT,
    rating        INTEGER,
    in_stock      INTEGER,
    stock_count   INTEGER,
    category      TEXT,
    upc           TEXT,
    url           TEXT NOT NULL,
    scraped_at    TEXT,
    PRIMARY KEY (run_id, product_key)
);
CREATE INDEX IF NOT EXISTS idx_snapshots_key ON snapshots(product_key);
"""

CHANGE_LABELS = {  # also the display order: most actionable first
    "price_down": "Price drop",
    "price_up": "Price increase",
    "out_of_stock": "Out of stock",
    "back_in_stock": "Back in stock",
    "new": "New product",
    "removed": "Removed",
}


class PriceHistory:
    def __init__(self, db_path: str | Path = "data/pricescout.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # --------------------------------------------------------------- writing
    def save_run(self, result: ScrapeResult) -> int:
        stats = result.stats
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO runs (source_url, category, started_at, finished_at, n_products, n_errors, duration_s)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    result.source_url,
                    result.category,
                    stats.started_at.isoformat(),
                    stats.finished_at.isoformat() if stats.finished_at else None,
                    len(result.products),
                    len(stats.errors),
                    stats.duration_seconds,
                ),
            )
            run_id = int(cur.lastrowid)
            conn.executemany(
                "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        run_id,
                        p.key,
                        p.title,
                        p.price,
                        p.currency,
                        p.rating,
                        int(p.in_stock),
                        p.stock_count,
                        p.category,
                        p.upc,
                        p.url,
                        p.scraped_at.isoformat(),
                    )
                    for p in result.products
                ],
            )
        return run_id

    def delete_all(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM snapshots")
            conn.execute("DELETE FROM runs")

    # --------------------------------------------------------------- reading
    def runs(self) -> pd.DataFrame:
        with closing(self._connect()) as conn:
            return pd.read_sql_query("SELECT * FROM runs ORDER BY id DESC", conn)

    def snapshot(self, run_id: int) -> pd.DataFrame:
        with closing(self._connect()) as conn:
            df = pd.read_sql_query("SELECT * FROM snapshots WHERE run_id = ?", conn, params=(run_id,))
        df["in_stock"] = df["in_stock"].astype(bool)
        return df

    def latest_run_id(self) -> int | None:
        runs = self.runs()
        return int(runs["id"].iloc[0]) if not runs.empty else None

    def price_history(self, product_key: str) -> pd.DataFrame:
        query = (
            "SELECT r.id AS run_id, r.started_at, s.price, s.in_stock FROM snapshots s "
            "JOIN runs r ON r.id = s.run_id WHERE s.product_key = ? ORDER BY r.id"
        )
        with closing(self._connect()) as conn:
            return pd.read_sql_query(query, conn, params=(product_key,))

    def compare(self, run_id: int | None = None, previous_run_id: int | None = None) -> pd.DataFrame:
        """What changed between two runs (defaults: latest vs the one before it)."""
        runs = self.runs()
        columns = ["change", "title", "old_price", "new_price", "price_diff", "pct_change", "category", "url"]
        if len(runs) < 2 and previous_run_id is None:
            return pd.DataFrame(columns=columns)
        run_id = int(run_id if run_id is not None else runs["id"].iloc[0])
        if previous_run_id is None:
            older = runs[runs["id"] < run_id]
            if older.empty:
                return pd.DataFrame(columns=columns)
            previous_run_id = int(older["id"].iloc[0])
        return diff_snapshots(self.snapshot(previous_run_id), self.snapshot(run_id))


def diff_snapshots(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Compare two snapshot tables keyed by ``product_key``."""
    columns = ["change", "title", "old_price", "new_price", "price_diff", "pct_change", "category", "url"]
    merged = old.merge(new, on="product_key", how="outer", suffixes=("_old", "_new"), indicator=True)
    rows = []
    for _, r in merged.iterrows():
        title = r["title_new"] if pd.notna(r.get("title_new")) else r["title_old"]
        url = r["url_new"] if pd.notna(r.get("url_new")) else r["url_old"]
        category = r["category_new"] if pd.notna(r.get("category_new")) else r.get("category_old")
        base = {"title": title, "url": url, "category": category,
                "old_price": r.get("price_old"), "new_price": r.get("price_new")}
        if r["_merge"] == "right_only":
            rows.append({**base, "change": CHANGE_LABELS["new"]})
            continue
        if r["_merge"] == "left_only":
            rows.append({**base, "change": CHANGE_LABELS["removed"]})
            continue
        if abs(float(r["price_new"]) - float(r["price_old"])) >= 0.005:
            kind = "price_down" if r["price_new"] < r["price_old"] else "price_up"
            rows.append({**base, "change": CHANGE_LABELS[kind]})
        if bool(r["in_stock_old"]) != bool(r["in_stock_new"]):
            kind = "back_in_stock" if bool(r["in_stock_new"]) else "out_of_stock"
            rows.append({**base, "change": CHANGE_LABELS[kind]})
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    out["price_diff"] = (out["new_price"] - out["old_price"]).round(2)
    out["pct_change"] = (out["price_diff"] / out["old_price"] * 100).round(1)
    order = {label: i for i, label in enumerate(CHANGE_LABELS.values())}
    out["_order"] = out["change"].map(order)
    return out.sort_values(["_order", "pct_change"]).drop(columns="_order").reset_index(drop=True)
