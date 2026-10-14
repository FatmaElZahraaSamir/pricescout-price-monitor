"""Config-driven site adapters.

A site is described with CSS selectors instead of code. Adding a new store is
usually a matter of writing a small JSON file (see ``examples/site_config.json``)
and passing it with ``--site-config``.

Field spec keys
---------------
css    CSS selector, relative to the listing item / detail page
attr   read an attribute instead of the element text (e.g. "href", "title")
table  read a value from a key/value table on the detail page (e.g. "UPC")
type   converter: text | url | price | rating_word | in_stock | int | int_in_text
strip_suffix   remove a trailing marker such as "...more"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .utils import clean_text, first_int, is_in_stock, parse_price, parse_rating_word

try:  # lxml is faster; html.parser is the always-available fallback
    import lxml  # noqa: F401

    _PARSER = "lxml"
except ImportError:  # pragma: no cover - depends on the environment
    _PARSER = "html.parser"


BOOKS_TOSCRAPE: dict[str, Any] = {
    "name": "books.toscrape.com",
    "domains": ["books.toscrape.com"],
    "listing": {
        "item": "article.product_pod",
        "fields": {
            "title": {"css": "h3 a", "attr": "title"},
            "url": {"css": "h3 a", "attr": "href", "type": "url"},
            "price": {"css": "p.price_color", "type": "price"},
            "rating": {"css": "p.star-rating", "attr": "class", "type": "rating_word"},
            "in_stock": {"css": "p.availability", "type": "in_stock"},
            "image_url": {"css": "div.image_container img", "attr": "src", "type": "url"},
        },
        "next_page": "ul.pager li.next a",
    },
    "detail": {
        "fields": {
            "title": {"css": "div.product_main h1"},
            "price": {"css": "div.product_main p.price_color", "type": "price"},
            "in_stock": {"css": "div.product_main p.availability", "type": "in_stock"},
            "stock_count": {"css": "div.product_main p.availability", "type": "int_in_text"},
            "rating": {"css": "div.product_main p.star-rating", "attr": "class", "type": "rating_word"},
            "description": {"css": "#product_description + p", "strip_suffix": "...more"},
            "category": {"css": "ul.breadcrumb li:nth-of-type(3) a"},
            "upc": {"table": "UPC"},
            "num_reviews": {"table": "Number of reviews", "type": "int"},
        },
        "table": {"row": "table tr", "key": "th", "value": "td"},
    },
    "categories": "div.side_categories ul li ul li a",
}


@dataclass
class ListingPage:
    items: list[dict[str, Any]] = field(default_factory=list)
    next_url: str | None = None


class SiteAdapter:
    """Turns raw HTML into dictionaries using a selector config."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name: str = config.get("name", "custom-site")
        self.domains: list[str] = config.get("domains", [])

    # ------------------------------------------------------------------ public
    @classmethod
    def from_file(cls, path: str | Path) -> "SiteAdapter":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def matches(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return any(host == d or host.endswith("." + d) for d in self.domains)

    def parse_listing(self, html: str, page_url: str) -> ListingPage:
        soup = BeautifulSoup(html, _PARSER)
        spec = self.config["listing"]
        page = ListingPage()
        for node in soup.select(spec["item"]):
            item = {name: self._extract(node, fs, page_url) for name, fs in spec["fields"].items()}
            if item.get("url") and item.get("title"):
                page.items.append(item)
        next_sel = spec.get("next_page")
        if next_sel:
            link = soup.select_one(next_sel)
            if link and link.get("href"):
                page.next_url = urljoin(page_url, link["href"])
        return page

    def parse_product(self, html: str, page_url: str) -> dict[str, Any]:
        soup = BeautifulSoup(html, _PARSER)
        spec = self.config.get("detail", {})
        table = self._read_table(soup, spec.get("table"))
        data: dict[str, Any] = {}
        for name, fs in spec.get("fields", {}).items():
            if "table" in fs:
                raw = table.get(fs["table"])
                data[name] = self._convert(raw, fs.get("type", "text"), page_url)
            else:
                data[name] = self._extract(soup, fs, page_url)
        return {k: v for k, v in data.items() if v is not None}

    def parse_categories(self, html: str, page_url: str) -> dict[str, str]:
        selector = self.config.get("categories")
        if not selector:
            return {}
        soup = BeautifulSoup(html, _PARSER)
        result: dict[str, str] = {}
        for link in soup.select(selector):
            name = clean_text(link.get_text())
            if name and link.get("href"):
                result[name] = urljoin(page_url, link["href"])
        return result

    # ----------------------------------------------------------------- helpers
    def _extract(self, root, fs: dict[str, Any], page_url: str):
        node = root.select_one(fs["css"]) if fs.get("css") else root
        if node is None:
            return None
        if fs.get("attr"):
            raw = node.get(fs["attr"])
            if isinstance(raw, list):  # e.g. class="star-rating Three"
                raw = " ".join(raw)
        else:
            raw = node.get_text(" ")
        value = self._convert(raw, fs.get("type", "text"), page_url)
        suffix = fs.get("strip_suffix")
        if suffix and isinstance(value, str) and value.endswith(suffix):
            value = value[: -len(suffix)].rstrip()
        return value

    @staticmethod
    def _convert(raw, kind: str, page_url: str):
        if raw is None:
            return None
        if kind == "url":
            return urljoin(page_url, raw.strip())
        if kind == "price":
            return parse_price(raw)  # (amount, currency) - unpacked by the pipeline
        if kind == "rating_word":
            return parse_rating_word(raw)
        if kind == "in_stock":
            return is_in_stock(raw)
        if kind == "int":
            return first_int(raw)
        if kind == "int_in_text":
            return first_int(raw)
        return clean_text(raw)

    @staticmethod
    def _read_table(soup, spec: dict[str, str] | None) -> dict[str, str]:
        if not spec:
            return {}
        table: dict[str, str] = {}
        for row in soup.select(spec["row"]):
            key, value = row.select_one(spec["key"]), row.select_one(spec["value"])
            if key and value:
                table[clean_text(key.get_text()) or ""] = clean_text(value.get_text()) or ""
        return table


BUILTIN_SITES = [BOOKS_TOSCRAPE]


def get_adapter(url: str, config_path: str | Path | None = None) -> SiteAdapter:
    """Pick the adapter for ``url``: an explicit config file wins, then built-ins."""
    if config_path:
        return SiteAdapter.from_file(config_path)
    for config in BUILTIN_SITES:
        adapter = SiteAdapter(config)
        if adapter.matches(url):
            return adapter
    # Same HTML template (Django-Oscar) works for local mirrors and test servers.
    return SiteAdapter({**BOOKS_TOSCRAPE, "name": "books.toscrape.com (template)"})
