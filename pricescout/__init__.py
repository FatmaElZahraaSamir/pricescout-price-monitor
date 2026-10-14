"""PriceScout: a production-style web scraper and price monitor.

Main building blocks:

* ``fetchers``  - polite async HTTP fetcher (retries, rate limiting, robots.txt)
                  and an optional headless-browser fetcher (Playwright) for
                  JavaScript-heavy sites.
* ``sites``     - config-driven site adapters (CSS selectors, no code needed
                  to add a new site).
* ``pipeline``  - crawls listing pages, follows pagination, fetches product
                  pages concurrently and validates every record.
* ``storage``   - SQLite price history with change detection between runs.
* ``exporters`` - CSV, JSON, formatted Excel, Google Sheets and MongoDB.
* ``ai_extract``- AI Extract: products from any store page (schema.org data
                  first, then an LLM through function calling), every value
                  checked against the page.
"""

from .models import Product
from .pipeline import Scraper, ScrapeResult, ScrapeStats

__all__ = ["Product", "Scraper", "ScrapeResult", "ScrapeStats"]
__version__ = "1.0.0"
