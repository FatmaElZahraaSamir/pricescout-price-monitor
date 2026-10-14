"""Price history, change detection and every export target."""

import asyncio
import dataclasses
import json

import mongomock
import pandas as pd
import pytest
from openpyxl import load_workbook

from pricescout.exporters import (
    category_summary,
    export_all,
    export_google_sheets,
    export_mongodb,
    products_to_df,
)
from pricescout.pipeline import scrape
from pricescout.storage import PriceHistory


def crawl(url):
    return asyncio.run(scrape(url, delay=0, concurrency=16))


@pytest.fixture
def two_runs(mirror, books, tmp_path):
    history = PriceHistory(tmp_path / "history.db")
    history.save_run(crawl(mirror.url))
    changed = [dataclasses.replace(b) for b in books]
    changed[0].price = round(changed[0].price * 0.8, 2)   # price drop
    changed[1].price = round(changed[1].price + 5, 2)     # price increase
    changed[2].stock = 0                                   # out of stock
    removed = changed.pop(3)                               # removed from site
    mirror.update_books(changed)
    history.save_run(crawl(mirror.url))
    return history, books, removed


def test_history_records_runs(two_runs):
    history, books, _ = two_runs
    runs = history.runs()
    assert len(runs) == 2 and list(runs["n_products"]) == [125, 126]


def test_change_detection(two_runs):
    history, books, removed = two_runs
    changes = history.compare()
    kinds = dict(zip(changes["title"], changes["change"]))
    assert kinds[books[0].title] == "Price drop"
    assert kinds[books[1].title] == "Price increase"
    assert kinds[books[2].title] == "Out of stock"
    assert kinds[removed.title] == "Removed"
    assert len(changes) == 4
    drop = changes[changes["change"] == "Price drop"].iloc[0]
    assert drop["pct_change"] == pytest.approx(-20.0, abs=0.2)


def test_price_history_for_one_product(two_runs):
    history, books, _ = two_runs
    hist = history.price_history(books[0].upc)
    assert len(hist) == 2 and hist["price"].iloc[1] < hist["price"].iloc[0]


def test_compare_with_single_run_is_empty(mirror, tmp_path):
    history = PriceHistory(tmp_path / "h.db")
    history.save_run(crawl(mirror.url))
    assert history.compare().empty


def test_file_exports(mirror, tmp_path):
    result = crawl(mirror.url)
    df = products_to_df(result.products)
    changes = pd.DataFrame([{"change": "Price drop", "title": "x", "old_price": 10.0, "new_price": 8.0,
                             "price_diff": -2.0, "pct_change": -20.0, "category": "Poetry", "url": "u"}])
    files = export_all(df, tmp_path, ["csv", "xlsx", "json"], basename="books", changes=changes)

    csv_text = files["csv"].read_text(encoding="utf-8-sig")
    assert csv_text.splitlines()[0].startswith("title,price,currency,rating")
    assert "£" not in csv_text.splitlines()[1]  # prices exported as numbers, not strings

    records = json.loads(files["json"].read_text(encoding="utf-8"))
    assert len(records) == 126 and records[0]["currency"] == "GBP"

    wb = load_workbook(files["xlsx"])
    assert wb.sheetnames == ["Products", "Summary", "Changes"]
    assert wb["Products"].max_row == 127 and wb["Products"].freeze_panes == "A2"
    assert wb["Summary"]["A1"].value == "category"


def test_category_summary_numbers(mirror):
    df = products_to_df(crawl(mirror.url).products)
    summary = category_summary(df).set_index("category")
    assert summary.loc["Poetry", "products"] == 19
    assert summary["products"].sum() == 126
    assert summary.loc["Poetry", "in_stock_pct"] == 100.0


def test_unknown_export_format_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        export_all(pd.DataFrame({"a": [1]}), tmp_path, ["pdf"])


class FakeWorksheet:
    def __init__(self):
        self.values = None
        self.cleared = False

    def clear(self):
        self.cleared = True

    def update(self, range_name, values):
        self.values = values


class FakeBook:
    def __init__(self):
        self.sheets = {}

    def worksheet(self, name):
        if name not in self.sheets:
            raise KeyError(name)  # stands in for gspread.WorksheetNotFound
        return self.sheets[name]

    def add_worksheet(self, title, rows, cols):
        self.sheets[title] = FakeWorksheet()
        return self.sheets[title]


class FakeGspreadClient:
    def __init__(self):
        self.book = FakeBook()

    def open_by_key(self, key):
        assert key == "SHEET_KEY"
        return self.book


def test_google_sheets_export_creates_and_fills_worksheet(mirror):
    df = products_to_df(crawl(mirror.url).products)
    client = FakeGspreadClient()
    written = export_google_sheets(df, "SHEET_KEY", client=client)
    ws = client.book.sheets["Products"]
    assert written == 126 and ws.cleared
    assert ws.values[0][:3] == ["title", "price", "currency"]
    assert all(isinstance(v, (str, int, float, bool)) for v in ws.values[1])  # JSON-safe cells


def test_mongodb_export_is_idempotent_upsert(mirror):
    df = products_to_df(crawl(mirror.url).products)
    client = mongomock.MongoClient()
    assert export_mongodb(df, client=client) == 126
    assert export_mongodb(df, client=client) == 126  # second run updates, no duplicates
    coll = client["pricescout"]["products"]
    assert coll.count_documents({}) == 126
    assert coll.find_one({"title": "A Light in the Attic"})["price"] == 51.77
