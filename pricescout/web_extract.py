"""The "AI Extract" page of the dashboard: products from any store page, no CSS selectors."""

from __future__ import annotations

import io
import json
import os

import pandas as pd
import streamlit as st

from .ai_extract import SAMPLE_PAGES, ExtractResult, extract_from_html, extract_url
from .llm import PROVIDERS, LLMClient, LLMError

ENGINES = {
    "Structured data only (no API key)": None,
    "Google Gemini (free tier)": "gemini",
    "OpenAI": "openai",
    "Groq (free tier)": "groq",
    "Ollama (runs on your computer)": "ollama",
}
SOURCES = ["Web address", "Sample pages", "Paste HTML"]


def secret(name: str | None) -> str | None:
    if not name:
        return None
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file
        value = None
    return value or os.environ.get(name)


def engine_settings():
    """Sidebar: pick the AI engine. Returns (llm or None, error message or None)."""
    st.markdown("**AI engine**")
    engine = st.selectbox("AI engine", list(ENGINES), index=1 if secret("GEMINI_API_KEY") else 0,
                          label_visibility="collapsed", key="extract_engine")
    provider = ENGINES[engine]
    if not provider:
        st.caption("Pages with schema.org product data still work. Pick an AI engine for every other page.")
        return None, None
    cfg = PROVIDERS[provider]
    stored_key = secret(cfg["env"])
    typed_key = ""
    if provider != "ollama":
        typed_key = st.text_input("API key", type="password", key="extract_key",
                                  placeholder="Using the key from the app settings" if stored_key else "Paste your API key",
                                  help="Used only for this session and never saved.")
        if not (typed_key or stored_key):
            st.caption(f"Get a free key: [{cfg['key_url'].split('/')[2]}]({cfg['key_url']})")
    else:
        st.caption("Needs [Ollama](https://ollama.com/download) running on this computer.")
    model = st.text_input("Model", cfg["model"], key="extract_model")
    try:
        return LLMClient(provider, api_key=typed_key.strip() or stored_key, model=model.strip() or None), None
    except LLMError as exc:
        return None, str(exc)


def results_frame(result: ExtractResult) -> pd.DataFrame:
    rows = []
    for p in result.products:
        rows.append({
            "status": "✅" if p.verified else "⚠️", "title": p.title, "price": p.price, "currency": p.currency,
            "old_price": p.old_price, "in_stock": {True: "Yes", False: "No"}.get(p.in_stock, "—"),
            "rating": p.rating, "sku": p.sku, "url": p.url,
            "not_found_on_page": ", ".join(p.problems), "source": p.source,
        })
    frame = pd.DataFrame(rows, columns=["status", "title", "price", "currency", "old_price", "in_stock", "rating",
                                        "sku", "url", "not_found_on_page", "source"])
    for column in ("price", "old_price", "rating"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Products")
        sheet = writer.sheets["Products"]
        for i, column in enumerate(frame.columns, start=1):
            width = max(len(str(column)), *(len(str(v)) for v in frame[column].head(200))) if len(frame) else 12
            sheet.column_dimensions[sheet.cell(1, i).column_letter].width = min(max(width + 2, 8), 60)
        sheet.freeze_panes = "A2"
    return buffer.getvalue()


def show_result(result: ExtractResult) -> None:
    if result.method == "none":
        for warning in result.warnings:
            st.info(warning, icon=":material/info:")
        with st.expander(f"What the AI would read · {len(result.page.text):,} characters"):
            st.text(result.page.text)
        return
    total = len(result.products)
    cols = st.columns([1, 1, 1.4, 0.8])
    cols[0].metric("Products found", total)
    cols[1].metric("Checked on the page", f"{result.verified} of {total}")
    cols[2].metric("Method", "Structured data" if result.method == "structured data" else "AI")
    cols[3].metric("Time", f"{result.seconds:.1f} s")
    with st.container(horizontal=True):
        if result.method == "structured data":
            st.badge("schema.org data on the page · no AI needed", icon=":material/data_object:", color="green")
        else:
            st.badge(result.engine, icon=":material/auto_awesome:", color="violet")
            st.badge(f"{len(result.page.text):,} characters read", icon=":material/article:", color="gray")
        if result.page_type:
            st.badge(f"{result.page_type} page", color="gray")
    for warning in result.warnings:
        st.warning(warning, icon=":material/warning:")
    if not total:
        st.info("No products were found on this page.")
        return
    frame = results_frame(result)
    shown = frame.drop(columns=["source"]).assign(  # blanks instead of "None" for values the page doesn't show
        old_price=frame["old_price"].map(lambda v: "" if pd.isna(v) else f"{v:,.2f}"),
        rating=frame["rating"].map(lambda v: "" if pd.isna(v) else f"{v:.1f} ★"),
        sku=frame["sku"].fillna(""))
    st.dataframe(shown, hide_index=True, width="stretch",
                 height=min(35 * (len(frame) + 1) + 3, 600), column_config={
                     "status": st.column_config.TextColumn("", width=40,
                                                           help="✅ every value was found on the page"),
                     "title": st.column_config.TextColumn("Product", width="large"),
                     "price": st.column_config.NumberColumn("Price", format="%.2f"),
                     "currency": st.column_config.TextColumn("Cur.", width=60),
                     "old_price": st.column_config.TextColumn("Was"),
                     "in_stock": st.column_config.TextColumn("In stock", width=80),
                     "rating": st.column_config.TextColumn("Rating", width=70),
                     "sku": st.column_config.TextColumn("SKU"),
                     "url": st.column_config.LinkColumn("Link", display_text="open", width=70),
                     "not_found_on_page": st.column_config.TextColumn("Not on the page"),
                 })
    a, b, c = st.columns(3)
    a.download_button("CSV", frame.to_csv(index=False).encode("utf-8-sig"), "products.csv", "text/csv",
                      width="stretch", icon=":material/download:")
    b.download_button("Excel", excel_bytes(frame), "products.xlsx",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
                      icon=":material/download:")
    c.download_button("JSON", json.dumps(frame.to_dict("records"), ensure_ascii=False, indent=2).encode("utf-8"),
                      "products.json", "application/json", width="stretch", icon=":material/download:")
    with st.expander("How each value was checked"):
        if result.method == "structured data":
            st.markdown("The products come from the page's schema.org data (the data stores publish for search "
                        "engines). The title and price were also looked up in the visible page text.")
        else:
            st.markdown("The AI returned each price *exactly as written on the page*. PriceScout then checked "
                        "that the title, the price text and the crossed-out price appear on the page, that the "
                        "number matches the price text, and that the link is a real link on the page.")
        problems = [p for p in result.products if not p.verified]
        for p in problems:
            st.markdown(f"⚠️ **{p.title}**: not found on the page: {', '.join(p.problems)}"
                        + (f" (price text: `{p.price_text}`)" if p.price_text else ""))
        if not problems:
            st.markdown("✅ Every value was found on the page.")
    with st.expander(f"What the AI read · {len(result.page.text):,} characters"
                     + (" (cut to the limit)" if result.page.truncated else "")):
        st.text(result.page.text)


def page() -> None:
    state = st.session_state
    with st.sidebar:
        st.title("🔎 PriceScout")
        st.caption("AI Extract reads product data from any store page, with no CSS selectors to write.")
        llm, llm_error = engine_settings()
        st.divider()
        st.caption("Built by **Fatma El Zahraa**. Need data from a website? "
                   "[Let's talk on Upwork](https://www.upwork.com/freelancers/~01f79195ceec502c59).")

    top_left, top_right = st.columns([3, 2], vertical_alignment="bottom")
    with top_left:
        st.header("AI Extract")
    with top_right:
        with st.container(horizontal=True, horizontal_alignment="right"):
            if llm:
                st.badge(f"{llm.label} · {llm.model}", icon=":material/auto_awesome:", color="violet")
            else:
                st.badge("Demo mode · structured data only", icon=":material/data_object:", color="gray")
    st.caption("Paste the link of a product or category page from any online store. When the page carries schema.org "
               "product data, PriceScout reads it directly (exact and free). Otherwise an AI model extracts the "
               "products, and every title and price is checked against the page before you see it.")
    if llm_error:
        st.warning(f"{llm_error} Until then, only pages with structured data can be read.")

    source = st.segmented_control("Page to read", SOURCES, default=SOURCES[0] if llm else SOURCES[1],
                                  key="extract_source") or SOURCES[1]
    html, page_url, url = None, None, None
    engine, robots = "http", True
    if source == "Web address":
        url = st.text_input("Page address", "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",
                            help="A product page or a category / search page.")
        left, right = st.columns([2, 1])
        engine_label = left.radio("Download with", ["Fast HTTP", "Headless browser (for JavaScript pages)"],
                                  horizontal=True)
        engine = "playwright" if engine_label.startswith("Headless") else "http"
        robots = right.checkbox("Respect robots.txt", value=True)
    elif source == "Sample pages":
        labels = [p.label for p in SAMPLE_PAGES]
        chosen = st.selectbox("Sample page", labels, index=0 if not llm else 1)
        sample = SAMPLE_PAGES[labels.index(chosen)]
        st.caption(sample.note)
        html, page_url = sample.html(), sample.url
    else:
        html = st.text_area("Page HTML", height=180, placeholder="Right-click the page > View page source, "
                                                                 "copy everything and paste it here")
        page_url = st.text_input("Page address (to turn relative links into full links)", "https://example.com/")
    use_structured = st.toggle("Use the page's structured data when it has some", value=True,
                               help="Turn off to make the AI read the page even when schema.org data exists.")

    if st.button("Extract products", type="primary", icon=":material/auto_awesome:"):
        try:
            with st.spinner("Reading the page…"):
                if source == "Web address":
                    allow_private = os.environ.get("PRICESCOUT_ALLOW_PRIVATE") == "1"
                    state.extract_result = extract_url(url, llm, engine, robots, use_structured,
                                                       allow_private=allow_private)
                elif not (html or "").strip():
                    st.warning("Paste the page HTML first.")
                else:
                    state.extract_result = extract_from_html(html, page_url or "https://example.com/", llm,
                                                             use_structured)
        except (ValueError, LLMError, RuntimeError) as exc:
            st.error(str(exc))
    if state.get("extract_result") is not None:
        show_result(state.extract_result)
