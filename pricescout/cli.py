"""Command line interface.

Examples
--------
python -m pricescout scrape --max-pages 2
python -m pricescout scrape --category Poetry --formats csv,xlsx,json
python -m pricescout scrape --engine playwright --max-pages 1
python -m pricescout categories
python -m pricescout runs
python -m pricescout changes
python -m pricescout extract https://books.toscrape.com/catalogue/category/books/travel_2/index.html --provider gemini
python -m pricescout extract --sample 1                     # a built-in demo page with structured data
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .exporters import export_all, export_google_sheets, export_mongodb, products_to_df
from .pipeline import CategoryNotFound, run_sync, scrape
from .storage import PriceHistory

DEFAULT_URL = "https://books.toscrape.com/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pricescout", description="Web scraper & price monitor")
    parser.add_argument("--version", action="version", version=f"pricescout {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scrape", help="scrape a catalogue and export the results")
    s.add_argument("--url", default=DEFAULT_URL, help="start URL (default: %(default)s)")
    s.add_argument("--category", help="only scrape this category (e.g. 'Poetry')")
    s.add_argument("--max-pages", type=int, default=None, help="limit listing pages (default: all)")
    s.add_argument("--no-details", action="store_true", help="skip product pages (faster, fewer fields)")
    s.add_argument("--engine", choices=["http", "playwright"], default="http",
                   help="http = fast; playwright = headless browser for JavaScript sites")
    s.add_argument("--concurrency", type=int, default=8)
    s.add_argument("--delay", type=float, default=0.25, help="min seconds between requests (politeness)")
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--timeout", type=float, default=20.0)
    s.add_argument("--no-robots", action="store_true", help="do not check robots.txt")
    s.add_argument("--site-config", help="JSON file with CSS selectors for a custom site")
    s.add_argument("--out", default="output", help="output folder (default: %(default)s)")
    s.add_argument("--formats", default="csv,xlsx,json", help="comma separated: csv,xlsx,json")
    s.add_argument("--db", default="data/pricescout.db", help="SQLite history file")
    s.add_argument("--gsheet", help="Google Sheet key to update (needs a service account)")
    s.add_argument("--gsheet-credentials", help="service account JSON file")
    s.add_argument("--mongo-uri", help="MongoDB URI, e.g. mongodb://localhost:27017")

    c = sub.add_parser("categories", help="list categories available on the site")
    c.add_argument("--url", default=DEFAULT_URL)
    c.add_argument("--site-config")

    r = sub.add_parser("runs", help="show scraping history")
    r.add_argument("--db", default="data/pricescout.db")

    ch = sub.add_parser("changes", help="show what changed since the previous run")
    ch.add_argument("--db", default="data/pricescout.db")
    ch.add_argument("--out", help="also save the changes to this CSV file")

    e = sub.add_parser("extract", help="AI Extract: products from any store page, no CSS selectors")
    e.add_argument("url", nargs="?", help="product or category page")
    e.add_argument("--sample", type=int, choices=[1, 2, 3], help="use a built-in sample page instead of a URL")
    e.add_argument("--provider", choices=["gemini", "openai", "groq", "ollama", "none"], default="none",
                   help="AI engine (reads GEMINI_API_KEY / OPENAI_API_KEY / GROQ_API_KEY); none = structured data only")
    e.add_argument("--model", default=None)
    e.add_argument("--engine", choices=["http", "playwright"], default="http")
    e.add_argument("--no-structured", action="store_true", help="let the AI read the page even if it has schema.org data")
    e.add_argument("--out", help="save the products to this CSV file")
    return parser


def _progress(event: str, info: dict) -> None:
    if event == "listing_page":
        print(f"  page {info['page']:>3} -> {info['items']} products found", flush=True)
    elif event == "details":
        print(f"  product pages: {info['done']}/{info['total']}", flush=True)
    elif event == "category":
        print(f"  category: {info['category']}", flush=True)


def cmd_scrape(args) -> int:
    print(f"PriceScout {__version__} | {args.url} | engine={args.engine}")
    try:
        result = run_sync(
            scrape(
                args.url,
                engine=args.engine,
                category=args.category,
                max_pages=args.max_pages,
                fetch_details=not args.no_details,
                site_config=args.site_config,
                on_progress=_progress,
                concurrency=args.concurrency,
                delay=args.delay,
                max_retries=args.retries,
                timeout=args.timeout,
                **({"respect_robots": False} if args.no_robots and args.engine == "http" else {}),
            )
        )
    except CategoryNotFound as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # start page unreachable, browser missing, ...
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not result.products:
        print("ERROR: no products were scraped.", file=sys.stderr)
        for line in result.stats.errors[:5]:
            print(f"  - {line}", file=sys.stderr)
        return 1

    history = PriceHistory(args.db)
    run_id = history.save_run(result)
    changes = history.compare(run_id)
    df = products_to_df(result.products)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files = export_all(df, args.out, args.formats.split(","), basename=f"products_{stamp}", changes=changes)

    if args.gsheet:
        n = export_google_sheets(df, args.gsheet, credentials_file=args.gsheet_credentials)
        print(f"  Google Sheets: {n} rows written")
    if args.mongo_uri:
        n = export_mongodb(df, args.mongo_uri)
        print(f"  MongoDB: {n} documents upserted")

    st = result.stats
    print("\nDone")
    print(f"  products      : {len(result.products)}")
    print(f"  listing pages : {st.listing_pages}")
    print(f"  requests      : {st.requests} (retries: {st.retries})")
    print(f"  duration      : {st.duration_seconds}s")
    print(f"  errors        : {len(st.errors)}")
    print(f"  history run id: {run_id}")
    if not changes.empty:
        print(f"  changes since last run: {len(changes)}")
        print(changes.groupby("change").size().to_string())
    for fmt, path in files.items():
        print(f"  {fmt:<5} -> {path}")
    return 0


def cmd_categories(args) -> int:
    from .fetchers import HttpFetcher
    from .pipeline import Scraper
    from .sites import get_adapter

    async def _run():
        async with HttpFetcher() as fetcher:
            return await Scraper(fetcher, get_adapter(args.url, args.site_config)).list_categories(args.url)

    categories = run_sync(_run())
    for name in categories:
        print(name)
    print(f"\n{len(categories)} categories")
    return 0


def cmd_runs(args) -> int:
    runs = PriceHistory(args.db).runs()
    print("No runs yet." if runs.empty else runs.to_string(index=False))
    return 0


def cmd_changes(args) -> int:
    changes = PriceHistory(args.db).compare()
    if changes.empty:
        print("No changes (or fewer than two runs).")
        return 0
    print(changes.to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        changes.to_csv(args.out, index=False)
        print(f"\nsaved -> {args.out}")
    return 0


def cmd_extract(args) -> int:
    from .ai_extract import SAMPLE_PAGES, extract_from_html, extract_url
    from .llm import LLMClient, LLMError

    try:
        llm = None if args.provider == "none" else LLMClient(args.provider, model=args.model)
        if args.sample:
            sample = SAMPLE_PAGES[args.sample - 1]
            result = extract_from_html(sample.html(), sample.url, llm, not args.no_structured)
        elif args.url:
            result = extract_url(args.url, llm, args.engine, use_structured=not args.no_structured, allow_private=True)
        else:
            print("Give a URL or --sample 1|2|3", file=sys.stderr)
            return 2
    except (LLMError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"{result.url}\n  method: {result.method}{' · ' + result.engine if result.engine else ''}"
          f" · {result.seconds:.1f}s")
    for warning in result.warnings:
        print(f"  ! {warning}")
    for p in result.products:
        mark = "OK " if p.verified else "?? "
        was = f" (was {p.old_price:,.2f})" if p.old_price else ""
        stock = {True: "in stock", False: "out of stock"}.get(p.in_stock, "")
        print(f"  {mark} {p.title} · {p.price:,.2f} {p.currency or ''}{was} {stock}".rstrip()
              + (f" · not on the page: {', '.join(p.problems)}" if p.problems else ""))
    if args.out and result.products:
        import pandas as pd

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{**{k: v for k, v in vars(p).items() if k != "checks"}, "verified": p.verified}
                      for p in result.products]).to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\nsaved -> {args.out}")
    return 0 if result.products else 3


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    handlers = {"scrape": cmd_scrape, "categories": cmd_categories, "runs": cmd_runs, "changes": cmd_changes,
                "extract": cmd_extract}
    return handlers[args.command](args)
