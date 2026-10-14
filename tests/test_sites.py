"""Parsing tests against real HTML captured from books.toscrape.com."""

from pathlib import Path

from pricescout.sites import BOOKS_TOSCRAPE, SiteAdapter, get_adapter
from pricescout.utils import decode_html, parse_price, parse_rating_word

BASE = "https://books.toscrape.com/"
adapter = SiteAdapter(BOOKS_TOSCRAPE)


def test_listing_first_page(fixture_html):
    page = adapter.parse_listing(fixture_html("listing_page_1.html"), BASE)
    assert len(page.items) == 4
    first = page.items[0]
    assert first["title"] == "A Light in the Attic"  # full title from the attribute, not "A Light in the ..."
    assert first["url"] == BASE + "catalogue/a-light-in-the-attic_1000/index.html"
    assert first["price"] == (51.77, "GBP")
    assert first["rating"] == 3
    assert first["in_stock"] is True
    assert first["image_url"].startswith(BASE + "media/cache/")
    assert page.next_url == BASE + "catalogue/page-2.html"


def test_listing_last_page_has_no_next_and_resolves_relative_links(fixture_html):
    url = BASE + "catalogue/page-50.html"
    page = adapter.parse_listing(fixture_html("listing_page_50.html"), url)
    assert [i["title"] for i in page.items] == ["Frankenstein", "Forever Rockers (The Rocker #12)"]
    assert page.items[0]["url"] == BASE + "catalogue/frankenstein_20/index.html"
    assert page.next_url is None


def test_product_page_details(fixture_html):
    url = BASE + "catalogue/a-light-in-the-attic_1000/index.html"
    data = adapter.parse_product(fixture_html("product_page.html"), url)
    assert data["title"] == "A Light in the Attic"
    assert data["price"] == (51.77, "GBP")
    assert data["stock_count"] == 22
    assert data["in_stock"] is True
    assert data["rating"] == 3
    assert data["category"] == "Poetry"
    assert data["upc"] == "a897fe39b1053632"
    assert data["num_reviews"] == 0
    assert data["description"].startswith("It's hard to imagine a world without A Light in the Attic.")
    assert not data["description"].endswith("...more")


def test_categories(fixture_html):
    cats = adapter.parse_categories(fixture_html("listing_page_1.html"), BASE)
    assert list(cats) == ["Travel", "Mystery", "Historical Fiction", "Sequential Art"]
    assert cats["Travel"] == BASE + "catalogue/category/books/travel_2/index.html"


def test_price_and_rating_helpers():
    assert parse_price("£51.77") == (51.77, "GBP")
    assert parse_price("Â£51.77") == (51.77, "GBP")  # mis-decoded pound sign still works
    assert parse_price("$1,299.00") == (1299.0, "USD")
    assert parse_price("AED 49") == (49.0, "AED")
    assert parse_price("") == (None, None)
    assert parse_rating_word("star-rating Five") == 5
    assert parse_rating_word("star-rating") is None


def test_decode_without_charset_header():
    raw = "<p>£51.77</p>".encode("utf-8")
    assert decode_html(raw) == "<p>£51.77</p>"  # naive latin-1 decoding would give 'Â£51.77'
    assert decode_html("café".encode("cp1252")) == "café"


def test_custom_site_from_json_config():
    config = Path(__file__).parent.parent / "examples" / "site_config.json"
    shop = SiteAdapter.from_file(config)
    html = """
    <ul class="products">
      <li class="product">
        <a class="woocommerce-LoopProduct-link" href="/product/desk-lamp/"><img src="/img/lamp.jpg"></a>
        <h2 class="woocommerce-loop-product__title">Desk Lamp</h2>
        <span class="price"><del><span class="amount">AED 120</span></del><ins><span class="amount">AED 99</span></ins></span>
      </li>
    </ul>
    <a class="next page-numbers" href="/shop/page/2/">Next</a>
    """
    page = shop.parse_listing(html, "https://shop.example.com/shop/")
    assert page.items[0]["title"] == "Desk Lamp"
    assert page.items[0]["url"] == "https://shop.example.com/product/desk-lamp/"
    assert page.items[0]["price"] == (99.0, "AED")  # sale price, not the crossed-out one
    assert page.next_url == "https://shop.example.com/shop/page/2/"
    assert shop.matches("https://shop.example.com/x") and not shop.matches("https://other.com")


def test_get_adapter_picks_builtin_for_books_toscrape():
    assert get_adapter("https://books.toscrape.com/catalogue/page-2.html").name == "books.toscrape.com"
