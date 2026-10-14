"""Capture README screenshots of the running dashboard with headless Chromium.

    streamlit run app.py &          # in another terminal
    python scripts/take_screenshots.py --url http://localhost:8501
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
WIDTH = 1440


async def shoot(page, name: str) -> None:
    """Resize the viewport to the content height, then take the screenshot."""
    await page.wait_for_timeout(1500)
    bottom = await page.evaluate(
        "() => { const el = document.querySelector('[data-testid=\"stMainBlockContainer\"]');"
        " return el ? el.getBoundingClientRect().bottom + window.scrollY : document.body.scrollHeight; }"
    )
    await page.set_viewport_size({"width": WIDTH, "height": int(min(max(bottom + 24, 700), 3000))})
    await page.wait_for_timeout(1200)
    await page.screenshot(path=OUT / name)


async def main(url: str, executable: str | None) -> None:
    from playwright.async_api import async_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(**({"executable_path": executable} if executable else {}))
        page = await browser.new_page(viewport={"width": WIDTH, "height": 1600}, device_scale_factor=2)
        await page.goto(url, wait_until="networkidle")
        await page.get_by_text("Web Scraper & Price Monitor").wait_for(timeout=60_000)
        await page.wait_for_timeout(2000)
        await page.mouse.move(WIDTH - 10, 10)
        await shoot(page, "dashboard-overview.png")

        await page.get_by_role("tab", name="📋 Data").click()
        await shoot(page, "dashboard-data.png")

        await page.get_by_role("tab", name="📈 Price changes").click()
        await shoot(page, "dashboard-price-changes.png")

        await page.set_viewport_size({"width": WIDTH, "height": 1600})
        await page.get_by_role("link", name="AI Extract").click()
        await page.get_by_text("Page to read").wait_for(timeout=30_000)
        await page.get_by_role("button", name="Extract products").click()
        await page.get_by_text("Products found").wait_for(timeout=30_000)
        await page.get_by_text("How each value was checked").click()
        await page.mouse.move(WIDTH - 10, 10)
        await shoot(page, "ai-extract.png")
        await browser.close()
    print(f"saved to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--executable", help="path to a Chromium binary (optional)")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.executable))
