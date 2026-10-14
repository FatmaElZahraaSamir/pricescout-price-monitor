"""Demo of change tracking, fully offline.

Scrapes the local mirror of books.toscrape.com, changes a few prices and stock
levels on the mirror (books.toscrape.com itself never changes its prices), then
scrapes again and prints what PriceScout detected. The dashboard's
"Load sample data" button does the same thing.

    python scripts/demo_price_changes.py
    streamlit run app.py        # then open the "Price changes" tab
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pricescout.mirror import MirrorServer, load_books, with_demo_changes  # noqa: E402
from pricescout.pipeline import run_sync, scrape  # noqa: E402
from pricescout.storage import PriceHistory  # noqa: E402


def main() -> None:
    history = PriceHistory("data/pricescout.db")
    books = load_books()
    with MirrorServer(books) as mirror:
        first = run_sync(scrape(mirror.url, delay=0, concurrency=16))
        history.save_run(first)
        print(f"Run 1: {len(first.products)} products")

        mirror.update_books(with_demo_changes(books))
        second = run_sync(scrape(mirror.url, delay=0, concurrency=16))
        history.save_run(second)
        print(f"Run 2: {len(second.products)} products\n")

    changes = history.compare()
    print(changes[["change", "title", "old_price", "new_price", "pct_change"]].to_string(index=False))


if __name__ == "__main__":
    main()
