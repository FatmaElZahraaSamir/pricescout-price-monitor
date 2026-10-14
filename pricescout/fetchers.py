"""Page fetchers.

``HttpFetcher``       fast async HTTP client with retries, exponential backoff,
                      Retry-After support, a global rate limit and robots.txt.
``PlaywrightFetcher`` real headless Chromium for JavaScript-rendered sites.

Both expose the same ``fetch(url) -> FetchResult`` coroutine, so the pipeline
does not care which one is used.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from .utils import decode_html

log = logging.getLogger("pricescout")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 PriceScout/1.0"
)
RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class FetchResult:
    url: str
    status: int | None
    text: str | None
    error: str | None = None
    elapsed: float = 0.0
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.text is not None


@dataclass
class FetchStats:
    requests: int = 0
    retries: int = 0
    failures: int = 0
    blocked_by_robots: int = 0
    bytes: int = 0
    by_status: dict[int, int] = field(default_factory=dict)


class RateLimiter:
    """Allow at most one request start every ``interval`` seconds (globally)."""

    def __init__(self, interval: float):
        self.interval = max(0.0, interval)
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next = max(now, self._next) + self.interval


class HttpFetcher:
    """Polite, resilient async HTTP fetcher (use as ``async with``)."""

    def __init__(
        self,
        concurrency: int = 8,
        delay: float = 0.25,
        timeout: float = 20.0,
        max_retries: int = 3,
        backoff: float = 0.75,
        user_agent: str = DEFAULT_USER_AGENT,
        respect_robots: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
        event_hooks: dict | None = None,
    ):
        self.concurrency = max(1, concurrency)
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff = backoff
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self._transport = transport
        self._event_hooks = event_hooks or {}  # e.g. a guard that checks every request, redirects included
        self._sem = asyncio.Semaphore(self.concurrency)
        self._limiter = RateLimiter(delay)
        self._robots: dict[str, RobotFileParser | None] = {}
        self._robots_lock = asyncio.Lock()
        self.client: httpx.AsyncClient | None = None
        self.stats = FetchStats()

    async def __aenter__(self) -> "HttpFetcher":
        self.client = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent, "Accept-Language": "en"},
            timeout=self.timeout,
            follow_redirects=True,
            transport=self._transport,
            event_hooks=self._event_hooks,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self.client is not None:
            await self.client.aclose()

    # ---------------------------------------------------------------- robots
    async def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        async with self._robots_lock:
            if origin not in self._robots:
                self._robots[origin] = await self._load_robots(origin)
        parser = self._robots[origin]
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    async def _load_robots(self, origin: str) -> RobotFileParser | None:
        assert self.client is not None
        try:
            response = await self.client.get(origin + "/robots.txt")
        except httpx.HTTPError as exc:
            log.warning("robots.txt unreachable for %s (%s) - assuming allowed", origin, exc)
            return None
        if response.status_code != 200:
            return None  # no robots.txt -> everything allowed
        parser = RobotFileParser()
        parser.parse(decode_html(response.content).splitlines())
        return parser

    # ----------------------------------------------------------------- fetch
    async def fetch(self, url: str) -> FetchResult:
        assert self.client is not None, "use 'async with HttpFetcher() as f'"
        if not await self.allowed(url):
            self.stats.blocked_by_robots += 1
            return FetchResult(url, None, None, "blocked by robots.txt", attempts=0)

        async with self._sem:
            last_error = "unknown error"
            status: int | None = None
            started = time.monotonic()
            for attempt in range(1, self.max_retries + 2):
                await self._limiter.wait()
                self.stats.requests += 1
                retry_after: float | None = None
                try:
                    response = await self.client.get(url)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error, status = f"{type(exc).__name__}: {exc}", None
                else:
                    status = response.status_code
                    self.stats.by_status[status] = self.stats.by_status.get(status, 0) + 1
                    if status == 200:
                        self.stats.bytes += len(response.content)
                        text = decode_html(response.content, response.charset_encoding)
                        return FetchResult(url, status, text, None, time.monotonic() - started, attempt)
                    last_error = f"HTTP {status}"
                    if status not in RETRY_STATUSES:
                        break
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))

                if attempt <= self.max_retries:
                    self.stats.retries += 1
                    wait = retry_after if retry_after is not None else self.backoff * 2 ** (attempt - 1)
                    wait += random.uniform(0, self.backoff / 2)  # jitter avoids thundering herd
                    log.info("retry %s/%s for %s in %.1fs (%s)", attempt, self.max_retries, url, wait, last_error)
                    await asyncio.sleep(wait)

            self.stats.failures += 1
            return FetchResult(url, status, None, last_error, time.monotonic() - started, attempt)


class PlaywrightFetcher:
    """Headless Chromium fetcher for pages that build their content with JavaScript.

    Requires ``pip install playwright`` and ``playwright install chromium``.
    """

    def __init__(
        self,
        concurrency: int = 3,
        delay: float = 0.25,
        timeout: float = 30.0,
        max_retries: int = 2,
        wait_until: str = "networkidle",
        user_agent: str = DEFAULT_USER_AGENT,
        headless: bool = True,
        executable_path: str | None = None,
    ):
        self.concurrency = max(1, concurrency)
        self.timeout_ms = int(timeout * 1000)
        self.max_retries = max(0, max_retries)
        self.wait_until = wait_until
        self.user_agent = user_agent
        self.headless = headless
        self.executable_path = executable_path
        self._sem = asyncio.Semaphore(self.concurrency)
        self._limiter = RateLimiter(delay)
        self.stats = FetchStats()
        self._pw = self._browser = self._context = None

    async def __aenter__(self) -> "PlaywrightFetcher":
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            ) from exc
        self._pw = await async_playwright().start()
        launch_kwargs = {"headless": self.headless}
        if self.executable_path:
            launch_kwargs["executable_path"] = self.executable_path
        try:
            self._browser = await self._pw.chromium.launch(**launch_kwargs)
            self._context = await self._browser.new_context(user_agent=self.user_agent)
        except Exception as exc:  # e.g. the browser was never downloaded: stop the driver, explain the fix
            await self.__aexit__(None, None, None)
            raise RuntimeError("Could not start the headless browser. Run `playwright install chromium` once, "
                               f"or use the fast HTTP engine. ({type(exc).__name__})") from exc
        return self

    async def __aexit__(self, *exc) -> None:
        for closer in (self._context, self._browser):
            if closer is not None:
                await closer.close()
        if self._pw is not None:
            await self._pw.stop()
        self._pw = self._browser = self._context = None

    async def fetch(self, url: str) -> FetchResult:
        async with self._sem:
            started = time.monotonic()
            last_error, status = "unknown error", None
            for attempt in range(1, self.max_retries + 2):
                await self._limiter.wait()
                self.stats.requests += 1
                page = await self._context.new_page()
                try:
                    response = await page.goto(url, wait_until=self.wait_until, timeout=self.timeout_ms)
                    status = response.status if response else None
                    if status is not None:
                        self.stats.by_status[status] = self.stats.by_status.get(status, 0) + 1
                    if status == 200:
                        html = await page.content()
                        self.stats.bytes += len(html)
                        return FetchResult(url, status, html, None, time.monotonic() - started, attempt)
                    last_error = f"HTTP {status}"
                    if status not in RETRY_STATUSES:
                        break
                except Exception as exc:  # playwright raises its own TimeoutError etc.
                    last_error = f"{type(exc).__name__}: {exc}"
                finally:
                    await page.close()
                if attempt <= self.max_retries:
                    self.stats.retries += 1
                    await asyncio.sleep(0.75 * 2 ** (attempt - 1))
            self.stats.failures += 1
            return FetchResult(url, status, None, last_error, time.monotonic() - started, attempt)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return min(60.0, max(0.0, float(value)))
    except ValueError:
        return None


def make_fetcher(engine: str = "http", **kwargs):
    """Factory used by the CLI and the dashboard."""
    engine = engine.lower()
    if engine in {"http", "httpx", "fast"}:
        return HttpFetcher(**kwargs)
    if engine in {"playwright", "browser", "js"}:
        allowed = {"concurrency", "delay", "timeout", "max_retries", "user_agent", "executable_path"}
        return PlaywrightFetcher(**{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"Unknown engine '{engine}'. Use 'http' or 'playwright'.")
