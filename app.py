"""PriceScout dashboard - run scrapes, explore results, track price changes.

Start with:  streamlit run app.py
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from pricescout import web_extract
from pricescout.ai_extract import check_public_url, public_only_hooks
from pricescout.exporters import category_summary, export_excel
from pricescout.mirror import MirrorServer, load_books, with_demo_changes
from pricescout.pipeline import CategoryNotFound, run_sync, scrape
from pricescout.storage import PriceHistory

DEFAULT_URL = "https://books.toscrape.com/"
DB_PATH = Path("data/pricescout.db")
# Validated, colour-blind-safe categorical order (see README > Design notes)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

st.set_page_config(page_title="PriceScout · Web Scraper & Price Monitor", page_icon="🔎", layout="wide")


def style(fig, height: int = 320):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
        title_font=dict(size=15),
        colorway=PALETTE,
        bargap=0.25,
        barcornerradius=4,
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=False,
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
    )
    axis = dict(gridcolor=GRID, linecolor="#c3c2b7", tickfont=dict(color=MUTED), title_font=dict(size=13, color=MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis, zeroline=False)
    return fig


@st.cache_resource
def history() -> PriceHistory:
    return PriceHistory(DB_PATH)


def to_frame(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot.drop(columns=["run_id"]).rename(columns={"product_key": "key"})
    return frame.sort_values(["category", "title"], na_position="last").reset_index(drop=True)


def monitor_page() -> None:
    """Run scrapes, explore the results and track price changes between runs."""
    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.title("🔎 PriceScout")
        st.caption("Production-style scraper with retries, rate limiting, robots.txt, price history and exports.")
        url = st.text_input("Start URL", DEFAULT_URL)
        category = st.text_input("Category (optional)", placeholder="e.g. Poetry, Travel, Mystery")
        max_pages = st.number_input("Listing pages to crawl (0 = all)", min_value=0, max_value=200, value=3)
        details = st.toggle("Open every product page (UPC, stock, category)", value=True)
        engine_label = st.radio("Engine", ["Fast HTTP (httpx)", "Headless browser (Playwright)"],
                                help="Use the browser engine for sites that render products with JavaScript.")
        with st.expander("Advanced settings"):
            concurrency = st.slider("Concurrent requests", 1, 16, 8)
            delay = st.slider("Min seconds between requests", 0.0, 2.0, 0.25, 0.05)
            retries = st.slider("Retries per page", 0, 5, 3)
            robots = st.checkbox("Respect robots.txt", value=True)
        run_clicked = st.button("🚀 Run scraper", type="primary", width="stretch")
        st.divider()
        st.caption("No internet, or just exploring? **Load sample data** runs the same pipeline twice on a built-in "
                   "offline copy of 126 real books from books.toscrape.com, with a few simulated price and stock "
                   "changes in between, so you can see the change tracking.")
        sample_clicked = st.button("Load sample data", width="stretch")
        if st.button("Clear history", width="stretch"):
            history().delete_all()
            st.toast("History cleared")


    def run_scrape(start_url: str, label: str, guard: bool = False, offline: bool = False) -> int | None:
        """Scrape, save the run and return its id (None if it failed).

        offline: the built-in copy on this machine, so every page, no politeness delay.
        """
        progress = st.progress(0.0, text="Starting…")
        log = st.empty()
        lines: list[str] = []

        def on_progress(event: str, info: dict) -> None:
            if event == "listing_page":
                lines.append(f"Listing page {info['page']} → {info['items']} products so far")
            elif event == "category":
                lines.append(f"Category: {info['category']}")
            elif event == "details":
                progress.progress(info["done"] / info["total"], text=f"Product pages {info['done']}/{info['total']}")
            log.code("\n".join(lines[-8:]) or "…")

        engine = "http" if offline or not engine_label.startswith("Headless") else "playwright"
        kwargs = dict(concurrency=16 if offline else concurrency, delay=0 if offline else delay,
                      max_retries=retries)
        if engine == "http":
            kwargs["respect_robots"] = robots
            if guard:  # every request, redirects included, must stay on the public internet
                kwargs["event_hooks"] = public_only_hooks()
        try:
            with st.spinner(label):
                result = run_sync(scrape(
                    start_url, engine=engine, category=category or None,
                    max_pages=None if offline else (max_pages or None),
                    fetch_details=details, on_progress=on_progress, **kwargs,
                ))
        except CategoryNotFound as exc:
            st.error(str(exc))
            return None
        except Exception as exc:  # network down, browser not installed, invalid URL...
            if engine == "playwright" and "executable doesn't exist" in str(exc).lower():
                st.error("The headless browser isn't installed on this server, so this run can't use the "
                         "Playwright engine. Choose **Fast HTTP** here, or run `playwright install chromium` "
                         "once on your own computer to use it there.")
                return None
            hint = ""
            if engine == "playwright":
                hint = " Tip: run `playwright install chromium` once to download the headless browser."
            st.error(f"Scrape failed: {exc}.{hint}")
            return None
        finally:
            progress.empty()
            log.empty()
        if not result.products:
            st.error("No products were scraped. " + ("; ".join(result.stats.errors[:3]) or "Check the URL."))
            return None
        run_id = history().save_run(result)
        st.session_state["last_errors"] = result.stats.errors
        st.success(f"Run #{run_id}: {len(result.products)} products · {result.stats.listing_pages} pages · "
                   f"{result.stats.requests} requests · {result.stats.duration_seconds}s · "
                   f"{len(result.stats.errors)} errors")
        return run_id


    if run_clicked:
        allow_private = os.environ.get("PRICESCOUT_ALLOW_PRIVATE") == "1"
        try:
            target = url if allow_private else check_public_url(url)
        except ValueError as exc:
            st.error(str(exc))
        else:
            run_scrape(target, f"Scraping {target}…", guard=not allow_private)
    if sample_clicked:
        books = load_books()
        with MirrorServer(books) as mirror:
            if run_scrape(mirror.url, "Scraping the offline sample site (run 1 of 2)…", offline=True):
                mirror.update_books(with_demo_changes(books))
                run_scrape(mirror.url, "Scraping it again after a few simulated price changes (run 2 of 2)…",
                           offline=True)

    # --------------------------------------------------------------------- main
    st.header("Web Scraper & Price Monitor")
    runs = history().runs()
    if runs.empty:
        st.info("👈 Click **Run scraper** to crawl books.toscrape.com (a sandbox site built for scraping), "
                "or **Load sample data** to try it offline.")
        st.stop()

    labels = {int(r.id): f"Run #{int(r.id)} · {r.started_at[:16].replace('T', ' ')} UTC · {int(r.n_products)} products"
              for r in runs.itertuples()}
    run_id = st.selectbox("Showing", list(labels), format_func=labels.get)
    df = to_frame(history().snapshot(run_id))

    overview, data_tab, changes_tab, history_tab = st.tabs(["📊 Overview", "📋 Data", "📈 Price changes", "🕑 History"])

    with overview:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Products", f"{len(df):,}")
        c2.metric("Average price", f"£{df['price'].mean():,.2f}")
        c3.metric("In stock", f"{df['in_stock'].mean() * 100:.0f}%")
        c4.metric("Categories", df["category"].nunique() if df["category"].notna().any() else "—")
        c5.metric("Average rating", f"{df['rating'].mean():.1f} ★" if df["rating"].notna().any() else "—")

        left, right = st.columns(2)
        if df["category"].notna().any():
            summary = category_summary(df).sort_values("avg_price")
            fig = px.bar(summary, x="avg_price", y="category", orientation="h", title="Average price by category",
                         labels={"avg_price": "Average price (£)", "category": ""},
                         hover_data={"products": True, "avg_price": ":.2f"})
            fig.update_traces(marker_color=PALETTE[0])
            left.plotly_chart(style(fig), theme=None, width="stretch")
        fig = px.histogram(df, x="price", nbins=20, title="Price distribution",
                           labels={"price": "Price (£)", "count": "Products"})
        fig.update_traces(marker_color=PALETTE[0], marker_line_width=0)
        fig.update_yaxes(title="Products")
        right.plotly_chart(style(fig), theme=None, width="stretch")

        left, right = st.columns(2)
        if df["rating"].notna().any():
            ratings = df["rating"].value_counts().reindex(range(1, 6), fill_value=0).reset_index()
            ratings.columns = ["rating", "products"]
            ratings["stars"] = ratings["rating"].map(lambda r: "★" * int(r))
            fig = px.bar(ratings, x="stars", y="products", title="Rating distribution",
                         labels={"stars": "Rating", "products": "Products"})
            fig.update_traces(marker_color=PALETTE[0])
            left.plotly_chart(style(fig), theme=None, width="stretch")
        if df["stock_count"].notna().any():
            fig = px.scatter(df, x="stock_count", y="price", title="Stock level vs price",
                             labels={"stock_count": "Units in stock", "price": "Price (£)"},
                             hover_data={"title": True, "category": True})
            fig.update_traces(marker=dict(size=9, color=PALETTE[0], opacity=0.75, line=dict(width=1, color="white")))
            right.plotly_chart(style(fig), theme=None, width="stretch")

        cheap, pricey = st.columns(2)
        cols = ["title", "price", "rating"]
        small_table = {
            "title": st.column_config.TextColumn("Title"),
            "price": st.column_config.NumberColumn("Price", format="£%.2f", width="small"),
            "rating": st.column_config.NumberColumn("Rating", format="%d ★", width="small"),
        }
        cheap.markdown("**Cheapest 5**")
        cheap.dataframe(df.nsmallest(5, "price")[cols], hide_index=True, width="stretch", column_config=small_table)
        pricey.markdown("**Most expensive 5**")
        pricey.dataframe(df.nlargest(5, "price")[cols], hide_index=True, width="stretch", column_config=small_table)

    with data_tab:
        f1, f2, f3, f4 = st.columns([2, 2, 1, 1])
        cats = sorted(df["category"].dropna().unique())
        chosen = f1.multiselect("Category", cats)
        lo, hi = float(df["price"].min()), float(df["price"].max())
        price_range = f2.slider("Price (£)", lo, hi, (lo, hi)) if hi > lo else (lo, hi)
        min_rating = f3.selectbox("Min rating", [1, 2, 3, 4, 5], index=0)
        stock_only = f4.checkbox("In stock only")
        query = st.text_input("Search titles", placeholder="type to filter…")

        view = df[df["price"].between(*price_range)]
        if chosen:
            view = view[view["category"].isin(chosen)]
        if df["rating"].notna().any():
            view = view[view["rating"].fillna(0) >= min_rating]
        if stock_only:
            view = view[view["in_stock"]]
        if query:
            view = view[view["title"].str.contains(query, case=False, regex=False)]

        st.caption(f"{len(view)} of {len(df)} products")
        st.dataframe(
            view[["title", "category", "price", "rating", "in_stock", "stock_count", "upc", "url"]],
            hide_index=True,
            width="stretch",
            column_config={
                "title": st.column_config.TextColumn("Title"),
                "category": st.column_config.TextColumn("Category"),
                "upc": st.column_config.TextColumn("UPC"),
                "price": st.column_config.NumberColumn("Price", format="£%.2f"),
                "rating": st.column_config.NumberColumn("Rating", format="%d ★"),
                "in_stock": st.column_config.CheckboxColumn("In stock"),
                "stock_count": st.column_config.NumberColumn("Units"),
                "url": st.column_config.LinkColumn("Link", display_text="open"),
            },
        )
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ CSV", view.to_csv(index=False).encode("utf-8-sig"), "products.csv", "text/csv",
                           width="stretch")
        buffer = io.BytesIO()
        export_excel(view.drop(columns=["key"]), buffer)
        d2.download_button("⬇️ Excel (formatted)", buffer.getvalue(), "products.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
        d3.download_button("⬇️ JSON", view.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
                           "products.json", "application/json", width="stretch")

    with changes_tab:
        if len(runs) < 2:
            st.info("Price changes appear from the second run onwards. Schedule the scraper (see README) "
                    "to get daily price-drop and stock alerts.")
        else:
            ids = list(labels)
            a, b = st.columns(2)
            newer = a.selectbox("Compare run", ids, index=0, format_func=labels.get, key="newer")
            older_options = [i for i in ids if i < newer] or ids
            older = b.selectbox("with run", older_options, index=0, format_func=labels.get, key="older")
            changes = history().compare(newer, older)
            sample_runs = set(runs.loc[runs["source_url"].str.startswith("http://127.0.0.1"), "id"].astype(int))
            if {newer, older} <= sample_runs:
                st.caption("Sample data: these changes were simulated on the offline copy between the two runs "
                           "(books.toscrape.com itself never changes its prices).")
            if changes.empty:
                st.success("No price or stock changes between these runs.")
            else:
                counts = changes["change"].value_counts()
                m = st.columns(len(counts))
                for col, (kind, n) in zip(m, counts.items()):
                    col.metric(kind, n)
                st.dataframe(
                    changes, hide_index=True, width="stretch",
                    column_config={
                        "change": st.column_config.TextColumn("Change"),
                        "title": st.column_config.TextColumn("Title"),
                        "category": st.column_config.TextColumn("Category"),
                        "old_price": st.column_config.NumberColumn("Old price", format="£%.2f"),
                        "new_price": st.column_config.NumberColumn("New price", format="£%.2f"),
                        "price_diff": st.column_config.NumberColumn("Diff (£)", format="%+.2f"),
                        "pct_change": st.column_config.NumberColumn("% change", format="%+.1f%%"),
                        "url": st.column_config.LinkColumn("Link", display_text="open"),
                    },
                )
                st.download_button("⬇️ Download changes (CSV)", changes.to_csv(index=False).encode("utf-8-sig"),
                                   "price_changes.csv", "text/csv")

    with history_tab:
        st.dataframe(runs, hide_index=True, width="stretch")
        errors = st.session_state.get("last_errors") or []
        if errors:
            with st.expander(f"Errors in the last run ({len(errors)})"):
                st.code("\n".join(errors[:200]))


navigation = st.navigation([
    st.Page(monitor_page, title="Price monitor", icon=":material/monitoring:", default=True),
    st.Page(web_extract.page, title="AI Extract", icon=":material/auto_awesome:", url_path="ai-extract"),
], position="top")
navigation.run()
