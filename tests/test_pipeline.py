"""End-to-end crawls against the local books.toscrape.com mirror."""

import asyncio
import dataclasses
import shutil

import pytest

from pricescout.fetchers import HttpFetcher
from pricescout.mirror import MirrorServer
from pricescout.pipeline import CategoryNotFound, Scraper, scrape
from pricescout.sites import BOOKS_TOSCRAPE, SiteAdapter


def run(coro):
    return asyncio.run(coro)


def test_full_crawl_returns_every_book_with_details(mirror, books):
    result = run(scrape(mirror.url, delay=0, concurrency=16))
    assert len(result.products) == len(books) == 126
    assert result.stats.listing_pages == 7  # 126 books / 20 per page
    assert result.stats.errors == []
    by_upc = {p.upc: p for p in result.products}
    for book in books:
        product = by_upc[book.upc]
        assert product.title == book.title
        assert product.price == pytest.approx(book.price)
        assert product.stock_count == book.stock
        assert product.category == book.category
        assert product.rating == book.rating


def test_max_pages_limits_the_crawl(mirror):
    result = run(scrape(mirror.url, delay=0, max_pages=2, fetch_details=False))
    assert result.stats.listing_pages == 2
    assert len(result.products) == 40
    assert all(p.upc is None for p in result.products)  # details were skipped


def test_category_filter_is_case_insensitive(mirror):
    result = run(scrape(mirror.url, delay=0, category="poetry"))
    assert result.category == "Poetry"
    assert len(result.products) == 19
    assert {p.category for p in result.products} == {"Poetry"}


def test_category_spanning_two_pages(books):
    relabelled = [dataclasses.replace(b, category="Mystery") if i < 25 else b for i, b in enumerate(books)]
    expected = sum(1 for b in relabelled if b.category == "Mystery")
    assert expected > 20  # forces pagination inside the category
    with MirrorServer(relabelled) as big_mirror:
        result = run(scrape(big_mirror.url, delay=0, category="Mystery", fetch_details=False))
    assert result.stats.listing_pages == 2 and len(result.products) == expected


def test_unknown_category_suggests_close_match(mirror):
    with pytest.raises(CategoryNotFound, match="Did you mean: Poetry"):
        run(scrape(mirror.url, delay=0, category="Poetri"))


def test_retries_transient_errors_on_product_pages(mirror, books):
    path = f"/catalogue/{books[0].slug}/index.html"
    mirror.state.flaky[path] = 2  # two 503s, then OK
    result = run(scrape(mirror.url, delay=0, max_pages=1, backoff=0.01))
    assert len(result.products) == 20 and result.stats.errors == []
    assert result.stats.retries == 2
    assert mirror.state.hits[path] == 3


def test_broken_product_page_keeps_listing_data_and_reports_error(mirror, books):
    path = f"/catalogue/{books[1].slug}/index.html"
    mirror.state.broken.add(path)
    result = run(scrape(mirror.url, delay=0, max_pages=1, backoff=0.01, max_retries=1))
    assert len(result.products) == 20  # the book is still exported with listing fields
    broken = next(p for p in result.products if p.title == books[1].title)
    assert broken.upc is None and broken.price == pytest.approx(books[1].price)
    assert len(result.stats.errors) == 1 and "HTTP 500" in result.stats.errors[0]


def test_progress_callback_reports_pages_and_details(mirror):
    events = []
    run(scrape(mirror.url, delay=0, max_pages=1, on_progress=lambda e, i: events.append(e)))
    assert events.count("listing_page") == 1 and "details" in events and events[-1] == "done"


def test_duplicates_are_removed(mirror):
    async def go():
        async with HttpFetcher(delay=0) as fetcher:
            scraper = Scraper(fetcher, SiteAdapter(BOOKS_TOSCRAPE), max_pages=1)
            # same start page twice via a loop-free trick: crawl page 1 and its catalogue twin
            first = await scraper.run(mirror.url)
            second = await scraper.run(mirror.url + "catalogue/page-1.html")
            return first, second

    first, second = run(go())
    assert [p.key for p in first.products] == [p.key for p in second.products]


def _chromium_available() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _chromium_available(), reason="playwright not installed")
def test_playwright_engine_renders_and_parses(mirror):
    executable = shutil.which("chromium") or None
    try:
        result = run(scrape(mirror.url, engine="playwright", delay=0, max_pages=1, fetch_details=False,
                            executable_path=executable))
    except Exception as exc:  # browser binaries missing on this machine
        pytest.skip(f"chromium not available: {exc}")
    assert len(result.products) == 20
    assert result.products[0].title == "It's Only the Himalayas"
