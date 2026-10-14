"""Crawl orchestration: pagination -> product pages -> validated records."""

from __future__ import annotations

import asyncio
import difflib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import ValidationError

from .models import Product
from .sites import SiteAdapter

log = logging.getLogger("pricescout")

ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class ScrapeStats:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    listing_pages: int = 0
    products_found: int = 0
    details_fetched: int = 0
    invalid_records: int = 0
    duplicates_removed: int = 0
    errors: list[str] = field(default_factory=list)
    requests: int = 0
    retries: int = 0

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or datetime.now(timezone.utc)
        return round((end - self.started_at).total_seconds(), 2)


@dataclass
class ScrapeResult:
    source_url: str
    products: list[Product]
    stats: ScrapeStats
    category: str | None = None


class CategoryNotFound(ValueError):
    pass


class Scraper:
    """Scrape a catalogue: follow "next" links, then fetch product pages concurrently."""

    def __init__(
        self,
        fetcher,
        adapter: SiteAdapter,
        *,
        max_pages: int | None = None,
        fetch_details: bool = True,
        on_progress: ProgressCallback | None = None,
    ):
        self.fetcher = fetcher
        self.adapter = adapter
        self.max_pages = max_pages
        self.fetch_details = fetch_details
        self.on_progress = on_progress or (lambda event, info: None)

    async def list_categories(self, start_url: str) -> dict[str, str]:
        result = await self.fetcher.fetch(start_url)
        if not result.ok:
            raise RuntimeError(f"Could not load {start_url}: {result.error}")
        return self.adapter.parse_categories(result.text, start_url)

    async def run(self, start_url: str, category: str | None = None) -> ScrapeResult:
        stats = ScrapeStats()
        listing_url: str | None = start_url

        if category:
            categories = await self.list_categories(start_url)
            match = _find_category(category, categories)
            listing_url = categories[match]
            category = match
            self.on_progress("category", {"category": match, "url": listing_url})

        # 1) Listing pages ----------------------------------------------------
        items: list[dict[str, Any]] = []
        seen_pages: set[str] = set()
        while listing_url and (self.max_pages is None or stats.listing_pages < self.max_pages):
            if listing_url in seen_pages:  # protects against pagination loops
                break
            seen_pages.add(listing_url)
            result = await self.fetcher.fetch(listing_url)
            if not result.ok:
                stats.errors.append(f"{listing_url}: {result.error}")
                break
            page = self.adapter.parse_listing(result.text, listing_url)
            stats.listing_pages += 1
            for item in page.items:
                if category and not item.get("category"):
                    item["category"] = category
            items.extend(page.items)
            self.on_progress("listing_page", {"page": stats.listing_pages, "items": len(items), "url": listing_url})
            listing_url = page.next_url
        stats.products_found = len(items)

        # 2) Product pages (concurrent) ---------------------------------------
        if self.fetch_details and items:
            done = 0
            total = len(items)

            async def enrich(item: dict[str, Any]) -> dict[str, Any]:
                nonlocal done
                result = await self.fetcher.fetch(item["url"])
                done += 1
                if done % 10 == 0 or done == total:
                    self.on_progress("details", {"done": done, "total": total})
                if not result.ok:
                    stats.errors.append(f"{item['url']}: {result.error}")
                    return item
                stats.details_fetched += 1
                detail = self.adapter.parse_product(result.text, item["url"])
                return {**item, **detail, "url": item["url"]}

            items = await asyncio.gather(*(enrich(item) for item in items))

        # 3) Validate + de-duplicate ---------------------------------------------
        products: list[Product] = []
        seen_keys: set[str] = set()
        for raw in items:
            try:
                product = Product(**_normalise(raw))
            except ValidationError as exc:
                stats.invalid_records += 1
                stats.errors.append(f"invalid record {raw.get('url')}: {exc.errors()[0]['msg']}")
                continue
            if product.key in seen_keys:
                stats.duplicates_removed += 1
                continue
            seen_keys.add(product.key)
            products.append(product)

        fetch_stats = getattr(self.fetcher, "stats", None)
        if fetch_stats is not None:
            stats.requests = fetch_stats.requests
            stats.retries = fetch_stats.retries
        stats.finished_at = datetime.now(timezone.utc)
        self.on_progress("done", {"products": len(products), "errors": len(stats.errors)})
        log.info(
            "scraped %s products from %s pages in %ss (%s errors)",
            len(products), stats.listing_pages, stats.duration_seconds, len(stats.errors),
        )
        return ScrapeResult(start_url, products, stats, category)


def _normalise(raw: dict[str, Any]) -> dict[str, Any]:
    """Unpack (amount, currency) price tuples and drop unknown/None values."""
    data = {k: v for k, v in raw.items() if v is not None}
    price = data.get("price")
    if isinstance(price, tuple):
        amount, currency = price
        data["price"] = amount
        if currency:
            data["currency"] = currency
    return data


def _find_category(wanted: str, categories: dict[str, str]) -> str:
    lookup = {name.lower(): name for name in categories}
    if wanted.lower() in lookup:
        return lookup[wanted.lower()]
    suggestions = difflib.get_close_matches(wanted, list(categories), n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise CategoryNotFound(f"Category '{wanted}' not found.{hint}")


def run_sync(coro):
    """Run a coroutine from sync code (CLI, Streamlit), even inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is already running (e.g. Jupyter): run in a worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def scrape(
    start_url: str,
    *,
    engine: str = "http",
    category: str | None = None,
    max_pages: int | None = None,
    fetch_details: bool = True,
    site_config: str | None = None,
    on_progress: ProgressCallback | None = None,
    **fetcher_kwargs,
) -> ScrapeResult:
    """One-call convenience API used by the CLI and the dashboard."""
    from .fetchers import make_fetcher
    from .sites import get_adapter

    adapter = get_adapter(start_url, site_config)
    started = time.monotonic()
    async with make_fetcher(engine, **fetcher_kwargs) as fetcher:
        scraper = Scraper(fetcher, adapter, max_pages=max_pages, fetch_details=fetch_details, on_progress=on_progress)
        result = await scraper.run(start_url, category=category)
    log.debug("scrape finished in %.2fs", time.monotonic() - started)
    return result
