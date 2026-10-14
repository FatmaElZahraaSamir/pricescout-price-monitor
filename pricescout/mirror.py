"""A local, offline replica of books.toscrape.com.

It rebuilds the real site's HTML structure (same templates, same relative
links, same missing charset header) from ``sample_data/books_sample.psv`` -
126 real books captured from books.toscrape.com.

Used for:
* the automated tests (no internet needed, deterministic),
* demos when the real site is unreachable,
* simulating price changes / flaky servers to exercise retries and history.
"""

from __future__ import annotations

import csv
import dataclasses
import html
import math
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SAMPLE_FILE = Path(__file__).resolve().parent.parent / "sample_data" / "books_sample.psv"
PER_PAGE = 20
RATING_WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
WORD_RATING = {v: k for k, v in RATING_WORD.items()}

# Simulated changes for the offline demo (positions in the catalogue)
DEMO_DROPS = [(3, 0.75), (14, 0.85), (40, 0.90), (61, 0.80), (90, 0.70), (110, 0.88)]   # price x factor
DEMO_INCREASES = [(25, 4.50), (77, 2.25), (101, 6.00)]                                 # price + amount
DEMO_SOLD_OUT = [52, 95]
DEMO_REMOVED = 120


@dataclass
class Book:
    category: str
    slug: str
    title: str
    price: float
    rating: int
    upc: str
    stock: int
    description: str

    @property
    def in_stock(self) -> bool:
        return self.stock > 0


def load_books(path: str | Path = SAMPLE_FILE) -> list[Book]:
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="|", quoting=csv.QUOTE_NONE)
        return [
            Book(
                category=row["category"],
                slug=row["slug"],
                title=row["title"],
                price=float(row["price"]),
                rating=WORD_RATING[row["rating"]],
                upc=row["upc"],
                stock=int(row["stock"]),
                description=row["description"],
            )
            for row in reader
        ]


def with_demo_changes(books: list[Book]) -> list[Book]:
    """A copy of the catalogue with a few price drops, increases, sold-out items and one delisted book.

    books.toscrape.com never changes its prices, so the offline demo makes its own changes between two runs
    to show the change tracking. The original list is not modified.
    """
    changed = [dataclasses.replace(b) for b in books]
    n = len(changed)
    for i, factor in DEMO_DROPS:
        if i < n:
            changed[i].price = round(changed[i].price * factor, 2)
    for i, extra in DEMO_INCREASES:
        if i < n:
            changed[i].price = round(changed[i].price + extra, 2)
    for i in DEMO_SOLD_OUT:
        if i < n:
            changed[i].stock = 0
    if DEMO_REMOVED < n:
        changed.pop(DEMO_REMOVED)
    return changed


def _cat_slug(name: str, index: int) -> str:
    return f"{name.lower().replace(' ', '-')}_{index}"


def _pod(book: Book, href: str, img_prefix: str) -> str:
    title = html.escape(book.title, quote=True)
    short = html.escape(book.title if len(book.title) <= 30 else book.title[:27] + "...")
    availability = "In stock" if book.in_stock else "Out of stock"
    return f"""<li class="col-xs-6 col-sm-4 col-md-3 col-lg-3">
<article class="product_pod">
<div class="image_container">
<a href="{href}"><img src="{img_prefix}media/cache/{book.upc[:2]}/{book.upc[2:4]}/{book.upc}.jpg" alt="{title}" class="thumbnail"></a>
</div>
<p class="star-rating {RATING_WORD[book.rating]}">
<i class="icon-star"></i><i class="icon-star"></i><i class="icon-star"></i><i class="icon-star"></i><i class="icon-star"></i>
</p>
<h3><a href="{href}" title="{title}">{short}</a></h3>
<div class="product_price">
<p class="price_color">£{book.price:.2f}</p>
<p class="instock availability">
<i class="icon-ok"></i> {availability} </p>
</div>
</article>
</li>"""


def _page(title: str, home: str, breadcrumb: str, body: str) -> str:
    return f"""<html lang="en-us" class="no-js"><head>
<title> {html.escape(title)} | Books to Scrape - Sandbox </title>
</head>
<body id="default" class="default">
<header class="header container-fluid"><div class="page_inner"><div class="row">
<div class="col-sm-8 h1"><a href="{home}">Books to Scrape</a><small> We love being scraped!</small></div>
</div></div></header>
<div class="container-fluid page"><div class="page_inner">
<ul class="breadcrumb">{breadcrumb}</ul>
{body}
</div></div>
<footer class="footer container-fluid"></footer>
</body></html>"""


def _listing(books, page_no, total_pages, title, home, prod_prefix, img_prefix,
             next_href, prev_href, categories_html, crumb) -> str:
    pods = "\n".join(_pod(b, f"{prod_prefix}{b.slug}/index.html", img_prefix) for b in books)
    pager = ""
    if prev_href:
        pager += f'<li class="previous"><a href="{prev_href}">previous</a></li>\n'
    pager += f'<li class="current"> Page {page_no} of {total_pages} </li>\n'
    if next_href:
        pager += f'<li class="next"><a href="{next_href}">next</a></li>\n'
    body = f"""<div class="row">
<aside class="sidebar col-sm-4 col-md-3"><div class="side_categories"><ul class="nav nav-list">
<li><a href="#"> Books </a><ul>
{categories_html}
</ul></li></ul></div></aside>
<div class="col-sm-8 col-md-9">
<div class="page-header action"><h1>{html.escape(title)}</h1></div>
<section><div><ol class="row">
{pods}
</ol>
<div><ul class="pager">
{pager}</ul></div>
</div></section>
</div></div>"""
    return _page(title, home, crumb, body)


def _product(book: Book, cat_slug: str) -> str:
    title = html.escape(book.title)
    availability = f"In stock ({book.stock} available)" if book.in_stock else "Out of stock (0 available)"
    body = f"""<article class="product_page">
<div class="row">
<div class="col-sm-6"><div id="product_gallery" class="carousel"><div class="thumbnail">
<img src="../../media/cache/{book.upc}.jpg" alt="{title}"></div></div></div>
<div class="col-sm-6 product_main">
<h1>{title}</h1>
<p class="price_color">£{book.price:.2f}</p>
<p class="instock availability">
<i class="icon-ok"></i> {availability} </p>
<p class="star-rating {RATING_WORD[book.rating]}">
<i class="icon-star"></i><i class="icon-star"></i><i class="icon-star"></i><i class="icon-star"></i><i class="icon-star"></i>
</p>
</div></div>
<div id="product_description" class="sub-header"><h2>Product Description</h2></div>
<p>{html.escape(book.description)} ...more</p>
<div class="sub-header"><h2>Product Information</h2></div>
<table class="table table-striped"><tbody>
<tr><th>UPC</th><td>{book.upc}</td></tr>
<tr><th>Product Type</th><td>Books</td></tr>
<tr><th>Price (excl. tax)</th><td>£{book.price:.2f}</td></tr>
<tr><th>Price (incl. tax)</th><td>£{book.price:.2f}</td></tr>
<tr><th>Tax</th><td>£0.00</td></tr>
<tr><th>Availability</th><td>{availability}</td></tr>
<tr><th>Number of reviews</th><td>0</td></tr>
</tbody></table>
</article>"""
    crumb = (
        '<li><a href="../../index.html">Home</a></li>\n'
        '<li><a href="../category/books_1/index.html">Books</a></li>\n'
        f'<li><a href="../category/books/{cat_slug}/index.html">{html.escape(book.category)}</a></li>\n'
        f'<li class="active">{title}</li>'
    )
    return _page(book.title, "../../index.html", crumb, body)


def build_site(books: list[Book]) -> dict[str, str]:
    """Return {path: html} for the whole mirror."""
    pages: dict[str, str] = {}
    cat_names = list(dict.fromkeys(b.category for b in books))
    cat_slugs = {name: _cat_slug(name, i + 2) for i, name in enumerate(cat_names)}

    def cats_html(prefix: str) -> str:
        return "\n".join(f'<li><a href="{prefix}{cat_slugs[n]}/index.html"> {html.escape(n)} </a></li>' for n in cat_names)

    total = max(1, math.ceil(len(books) / PER_PAGE))
    for n in range(1, total + 1):
        chunk = books[(n - 1) * PER_PAGE: n * PER_PAGE]
        if n == 1:
            pages["/index.html"] = _listing(
                chunk, 1, total, "All products", "index.html", "catalogue/", "",
                "catalogue/page-2.html" if total > 1 else None, None,
                cats_html("catalogue/category/books/"),
                '<li><a href="index.html">Home</a></li><li class="active">All products</li>',
            )
        pages[f"/catalogue/page-{n}.html"] = _listing(
            chunk, n, total, "All products", "../index.html", "", "../",
            f"page-{n + 1}.html" if n < total else None, f"page-{n - 1}.html" if n > 1 else None,
            cats_html("category/books/"),
            '<li><a href="../index.html">Home</a></li><li class="active">All products</li>',
        )
    pages["/"] = pages["/index.html"]

    for name in cat_names:
        members = [b for b in books if b.category == name]
        ctotal = max(1, math.ceil(len(members) / PER_PAGE))
        for n in range(1, ctotal + 1):
            chunk = members[(n - 1) * PER_PAGE: n * PER_PAGE]
            filename = "index.html" if n == 1 else f"page-{n}.html"
            prev = None if n == 1 else ("index.html" if n == 2 else f"page-{n - 1}.html")
            pages[f"/catalogue/category/books/{cat_slugs[name]}/{filename}"] = _listing(
                chunk, n, ctotal, name, "../../../../index.html", "../../../", "../../../../",
                f"page-{n + 1}.html" if n < ctotal else None, prev, cats_html("../"),
                f'<li><a href="../../../../index.html">Home</a></li><li class="active">{html.escape(name)}</li>',
            )

    for book in books:
        pages[f"/catalogue/{book.slug}/index.html"] = _product(book, cat_slugs[book.category])
    return pages


@dataclass
class MirrorState:
    pages: dict[str, str]
    flaky: dict[str, int] = field(default_factory=dict)      # path -> remaining 503 responses
    broken: set[str] = field(default_factory=set)            # paths that always return 500
    robots_txt: str | None = None                            # None -> 404 like the real site
    hits: dict[str, int] = field(default_factory=dict)


class _Handler(BaseHTTPRequestHandler):
    state: MirrorState  # injected per server

    def do_GET(self):  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        st = self.server.state  # type: ignore[attr-defined]
        st.hits[path] = st.hits.get(path, 0) + 1
        if path == "/robots.txt":
            if st.robots_txt is None:
                return self._send(404, "<h1>404 Not Found</h1>")
            return self._send(200, st.robots_txt, "text/plain")
        if path in st.broken:
            return self._send(500, "<h1>500 Internal Server Error</h1>")
        if st.flaky.get(path, 0) > 0:
            st.flaky[path] -= 1
            return self._send(503, "<h1>503 Service Unavailable</h1>")
        body = st.pages.get(path)
        if body is None:
            return self._send(404, "<h1>404 Not Found</h1>")
        return self._send(200, body)

    def _send(self, status: int, body: str, ctype: str = "text/html"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)  # deliberately no charset, like the real site
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep test output clean
        pass


class MirrorServer:
    """Run the mirror on a random local port: ``with MirrorServer() as m: m.url``."""

    def __init__(self, books: list[Book] | None = None, port: int = 0):
        self.books = books if books is not None else load_books()
        self.state = MirrorState(pages=build_site(self.books))
        self._server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self._server.state = self.state  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def update_books(self, books: list[Book]) -> None:
        """Swap the catalogue (e.g. to simulate price changes between runs)."""
        self.books = books
        self.state.pages = build_site(books)

    def __enter__(self) -> "MirrorServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


if __name__ == "__main__":  # python -m pricescout.mirror
    import time

    with MirrorServer(port=8765) as mirror:
        print(f"Mirror of books.toscrape.com running at {mirror.url} (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
