"""AI Extract: page text, structured data, the AI tool call, the checks against the page, fetching and the page UI."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from pricescout.ai_extract import (
    SAMPLE_PAGES, SYSTEM_PROMPT, TOOLS, check_public_url, extract_from_html, extract_url, html_to_text, parse_money,
    structured_products,
)
from pricescout.llm import LLMClient, LLMError, ScriptedLLM

ROOT = Path(__file__).resolve().parent.parent
DUNE, OASIS, BOOK = SAMPLE_PAGES

OASIS_PRODUCTS = [
    {"title": "Dune 2 Backpacking Tent", "price_text": "AED 449", "price": 449, "currency": "AED",
     "old_price_text": "AED 599", "in_stock": True, "url": "/products/dune-2-backpacking-tent", "rating": 4.3},
    {"title": "Liwa 4 Family Tent", "price_text": "AED 1,299.00", "price": 1299, "currency": "AED", "in_stock": False,
     "url": "https://oasis-outdoor.example/products/liwa-4-family-tent"},
    {"title": "Hajar Ultralight 1P", "price_text": "AED 2,150", "price": 2150, "currency": "AED", "in_stock": True,
     "url": "/products/hajar-ultralight-1p", "rating": 4.8},
    {"title": "Jebel Pop-Up Beach Shelter", "price_text": "Dhs 89.50", "price": 89.5, "old_price_text": "Dhs 129.00",
     "in_stock": True, "url": "/products/jebel-pop-up-beach-shelter"},
    {"title": "Fujairah 3 Dome Tent", "price_text": "AED 689.00", "price": 689, "currency": "AED", "in_stock": True,
     "url": "/products/fujairah-3-dome-tent"},
    {"title": "Empty Quarter 6 Tunnel Tent", "price_text": "AED 2,899.00", "price": 2899, "currency": "AED",
     "in_stock": True, "url": "/products/empty-quarter-6-tunnel-tent"},
    {"title": "Al Ain Tent Footprint (groundsheet)", "price_text": "AED 75.00", "price": 75, "currency": "AED",
     "in_stock": True, "url": "/products/al-ain-tent-footprint"},
    {"title": "Ras Al Khaimah Roof-Top Tent", "price_text": "AED 4,250.00", "price": 4250, "currency": "AED",
     "old_price_text": "AED 4,999.00", "url": "/products/ras-al-khaimah-roof-top-tent"},
]


def save_call(products, page_type="listing"):
    return {"tool_calls": [{"name": "save_products", "arguments": {"page_type": page_type, "products": products}}]}


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("text, expected", [
    ("£51.77", (51.77, "GBP")), ("AED 1,299.00", (1299.0, "AED")), ("1.299,00 €", (1299.0, "EUR")),
    ("$1,234", (1234.0, "USD")), ("Dhs 89.50", (89.5, "AED")), ("349.50 د.إ", (349.5, "AED")),
    ("US$ 12.5", (12.5, "USD")), ("EUR 2.150", (2150.0, "EUR")), ("1 299,00 €", (1299.0, "EUR")),
    ("SAR 75", (75.0, "SAR")), ("no price", (None, None)),
])
def test_parse_money_reads_store_formats(text, expected):
    assert parse_money(text) == expected


def test_page_text_is_compact_and_keeps_what_matters():
    page = html_to_text(OASIS.html(), OASIS.url)
    text = page.text
    assert "[Dune 2 Backpacking Tent](https://oasis-outdoor.example/products/dune-2-backpacking-tent)" in text
    assert "AED 449 ~~AED 599~~" in text and "Dhs 89.50 ~~Dhs 129.00~~" in text
    assert "[button: Sold out (disabled)]" in text and "# Tents" in text
    assert "line-through" not in text and "app.js" not in text and "Fictional store page" not in text
    assert "https://oasis-outdoor.example/products/liwa-4-family-tent" in page.links
    assert page.title == "Tents | Oasis Outdoor" and not page.truncated and len(text) < 2000

    hidden = html_to_text('<p>Visible</p><p style="display: none">Secret AED 1</p><div hidden>Also hidden</div>'
                          '<section class="old-collection"><p>A whole section, not a price</p></section>',
                          "https://x.example/")
    assert hidden.text == "Visible\nA whole section, not a price"
    long_page = html_to_text("<p>" + "word " * 9000 + "</p>", "https://x.example/", limit=1000)
    assert long_page.truncated and long_page.text.endswith("[... page text cut here ...]")


# ------------------------------------------------------------ structured data
def test_json_ld_product_in_a_graph():
    [item] = structured_products(DUNE.html(), DUNE.url)
    assert item == {"title": "Aurora ANC Wireless Headphones", "price": 649.0, "currency": "AED", "in_stock": True,
                    "url": "https://dune-electronics.example/products/aurora-anc-headphones", "sku": "DE-AUR-ANC-BLK",
                    "rating": 4.6, "image_url": "https://dune-electronics.example/img/aurora-anc-black.jpg"}


def test_json_ld_lists_microdata_open_graph_and_broken_data():
    item_list = {"@type": "ItemList", "itemListElement": [
        {"@type": "ListItem", "position": 1, "item": {"@type": "Product", "name": "Lamp", "url": "/lamp",
                                                      "offers": {"price": "19.90", "priceCurrency": "EUR",
                                                                 "availability": "OutOfStock"}}},
        {"@type": "ListItem", "position": 2, "item": {"@type": ["Product"], "name": "Desk",
                                                      "offers": [{"@type": "AggregateOffer", "lowPrice": 120}]}},
    ]}
    html = ('<script type="application/ld+json">{broken</script>'
            f'<script type="application/ld+json">{json.dumps(item_list)}</script>')
    items = structured_products(html, "https://shop.example/c/")
    assert [(i["title"], i["price"], i["in_stock"], i["url"]) for i in items] == [
        ("Lamp", 19.9, False, "https://shop.example/lamp"), ("Desk", 120.0, None, "https://shop.example/c/")]

    micro = ('<div itemscope itemtype="https://schema.org/Product"><h1 itemprop="name">Kettle</h1>'
             '<span itemprop="price" content="89.00">AED 89</span><meta itemprop="priceCurrency" content="AED">'
             '<link itemprop="availability" href="https://schema.org/InStock"></div>')
    assert structured_products(micro, "https://x.example/")[0] == {
        "title": "Kettle", "price": 89.0, "currency": "AED", "in_stock": True, "url": "https://x.example/",
        "sku": None, "rating": None, "image_url": None}
    og = ('<meta property="og:title" content="Fan"><meta property="product:price:amount" content="1,250.00">'
          '<meta property="product:price:currency" content="SAR">')
    assert structured_products(og, "https://x.example/fan")[0]["price"] == 1250.0
    assert structured_products("<p>No data here</p>", "https://x.example/") == []


def test_structured_page_needs_no_model_and_is_checked_against_the_page():
    result = extract_from_html(DUNE.html(), DUNE.url)
    assert result.method == "structured data" and result.llm_calls == 0 and result.verified == 1
    [product] = result.products
    assert (product.title, product.price, product.currency, product.source) == (
        "Aurora ANC Wireless Headphones", 649.0, "AED", "structured data")
    stale = DUNE.html().replace('"price": "649.00"', '"price": "599.00"')
    result = extract_from_html(stale, DUNE.url)
    assert result.products[0].problems == ["price"] and "not shown on the page" in result.warnings[0]


def test_page_without_structured_data_needs_a_model():
    result = extract_from_html(OASIS.html(), OASIS.url)
    assert result.method == "none" and result.products == [] and "Choose an AI engine" in result.warnings[0]


# --------------------------------------------------------------------- the AI
def test_ai_values_are_checked_against_the_page():
    invented = {"title": "Sahara Camping Chair", "price_text": "AED 120", "price": 120, "url": "/products/chair"}
    wrong_number = {**OASIS_PRODUCTS[4], "price": 698}
    llm = ScriptedLLM([save_call(OASIS_PRODUCTS[:4] + [wrong_number] + OASIS_PRODUCTS[5:] + [invented])])
    result = extract_from_html(OASIS.html(), OASIS.url, llm)
    assert result.method == "AI" and result.page_type == "listing" and result.llm_calls == 1
    assert len(result.products) == 9 and result.verified == 7
    by_title = {p.title: p for p in result.products}
    tent = by_title["Dune 2 Backpacking Tent"]
    assert (tent.price, tent.old_price, tent.in_stock, tent.rating) == (449.0, 599.0, True, 4.3)
    assert tent.url == "https://oasis-outdoor.example/products/dune-2-backpacking-tent" and tent.verified
    assert by_title["Jebel Pop-Up Beach Shelter"].currency == "AED"  # from "Dhs"
    assert by_title["Liwa 4 Family Tent"].in_stock is False
    assert by_title["Fujairah 3 Dome Tent"].problems == ["price"] and by_title["Fujairah 3 Dome Tent"].price == 689.0
    assert by_title["Sahara Camping Chair"].problems == ["title", "price", "link"]
    assert "2 of 9 products" in result.warnings[-1]
    prompt = llm.calls[0]
    assert prompt[0]["content"] == SYSTEM_PROMPT and "AED 449 ~~AED 599~~" in prompt[1]["content"]


def test_unusable_items_are_skipped_and_duplicates_merged():
    llm = ScriptedLLM([save_call([{"title": "", "price_text": "AED 5"}, {"title": "Mystery", "price_text": "free"},
                                  OASIS_PRODUCTS[0], OASIS_PRODUCTS[0], "not a dict"])])
    result = extract_from_html(OASIS.html(), OASIS.url, llm)
    assert [p.title for p in result.products] == ["Dune 2 Backpacking Tent"]
    assert sum("Skipped" in w for w in result.warnings) == 2


def test_json_answer_without_a_tool_call_is_accepted():
    book = {"title": "A Light in the Attic", "price_text": "£51.77", "price": 51.77, "currency": "GBP",
            "in_stock": True}
    llm = ScriptedLLM(["```json\n" + json.dumps({"page_type": "product", "products": [book]}) + "\n```"])
    result = extract_from_html(BOOK.html(), BOOK.url, llm)
    assert result.verified == 1 and result.products[0].currency == "GBP"


def test_model_that_never_saves_products_gets_one_reminder_then_fails():
    llm = ScriptedLLM(["I think there are some books.", "Still no tool call."])
    with pytest.raises(LLMError, match="expected format"):
        extract_from_html(BOOK.html(), BOOK.url, llm)
    assert len(llm.calls) == 2 and "Call the save_products tool now" in llm.calls[1][-1]["content"]


def test_full_loop_over_http_with_function_calling(fake_api):
    book = {"title": "A Light in the Attic", "price_text": "£51.77", "price": 51.77, "currency": "GBP",
            "in_stock": True, "sku": "a897fe39b1053632"}
    fake_api.script = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "save_products", "arguments": json.dumps({"page_type": "product", "products": [book]})},
         "extra_content": {"google": {"thought_signature": "sig-1"}}}]}]
    llm = LLMClient("gemini", api_key="test-key", base_url=fake_api.base_url, backoff=0.01)
    result = extract_from_html(BOOK.html(), BOOK.url, llm)
    assert result.verified == 1 and result.products[0].sku == "a897fe39b1053632"
    body = fake_api.requests[0]["body"]
    assert body["tools"] == TOOLS and "£51.77" in body["messages"][1]["content"]
    assert result.engine.startswith("Google Gemini")


# ------------------------------------------------------------------ fetching
def test_extract_url_downloads_the_page(mirror, books):
    book = books[0]
    llm = ScriptedLLM([save_call([{"title": book.title, "price_text": f"£{book.price:.2f}", "price": book.price,
                                   "currency": "GBP", "in_stock": book.in_stock}], "product")])
    result = extract_url(f"{mirror.url}catalogue/{book.slug}/index.html", llm, allow_private=True)
    assert result.verified == 1 and result.products[0].price == pytest.approx(book.price)
    with pytest.raises(ValueError, match="Could not download"):
        extract_url(f"{mirror.url}missing.html", llm, allow_private=True)


def test_public_url_check(monkeypatch):
    real = socket.getaddrinfo

    def fake(host, port, *args, **kwargs):
        table = {"shop.example": "93.184.216.34", "intranet.example": "10.0.0.5"}
        if host in table:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], port))]
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    assert check_public_url("shop.example/tents") == "https://shop.example/tents"
    for url in ["http://127.0.0.1:8501/", "http://169.254.169.254/latest/meta-data", "http://intranet.example/",
                "http://[::1]/", "http://localhost/"]:
        with pytest.raises(ValueError, match="private or local"):
            check_public_url(url)
    with pytest.raises(ValueError, match="full web address"):
        check_public_url("ftp://files.example/x")


def test_every_request_is_guarded_redirects_included(mirror):
    from pricescout.ai_extract import BlockedURL, fetch_page, public_only_hooks
    from pricescout.pipeline import run_sync

    with pytest.raises(BlockedURL):  # the mirror runs on 127.0.0.1, like a redirect to a private address would
        run_sync(fetch_page(mirror.url, event_hooks=public_only_hooks()))
    assert run_sync(fetch_page(mirror.url)).startswith("<html")


# ------------------------------------------------------------------- the page
def _extract_page(root: str):
    import sys

    sys.path.insert(0, root)
    from pricescout.web_extract import page

    page()


def test_extract_page_in_demo_mode(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_function(_extract_page, args=(str(ROOT),), default_timeout=60).run()
    assert not app.exception
    next(b for b in app.button if b.label == "Extract products").click().run()
    assert not app.exception
    assert app.dataframe[0].value["title"].tolist() == ["Aurora ANC Wireless Headphones"]
    assert app.metric[1].value == "1 of 1"
    sample = next(s for s in app.selectbox if s.label == "Sample page")
    sample.set_value(OASIS.label).run()
    next(b for b in app.button if b.label == "Extract products").click().run()
    assert any("no structured product data" in i.value for i in app.info)


def test_app_starts_on_the_price_monitor_and_sample_data_shows_changes(monkeypatch, tmp_path):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)  # empty history: the monitor shows its start message
    st.cache_resource.clear()    # the price history object is cached per process
    try:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        assert not app.exception
        assert any("Run scraper" in i.value for i in app.info)

        # one click = two runs of the offline copy, with simulated changes in between
        next(b for b in app.button if b.label == "Load sample data").click().run()
        assert not app.exception
        runs = [s.value for s in app.success]
        assert runs[0].startswith("Run #1: 126 products") and runs[1].startswith("Run #2: 125 products")
        metrics = {m.label: m.value for m in app.metric}
        assert {k: metrics.get(k) for k in ("Price drop", "Price increase", "Out of stock", "Removed")} == {
            "Price drop": "6", "Price increase": "3", "Out of stock": "2", "Removed": "1"}
        assert any("simulated" in c.value for c in app.caption)
    finally:
        st.cache_resource.clear()
