"""Retries, backoff, robots.txt and rate limiting (no network needed)."""

import asyncio
import time

import httpx
import pytest

from pricescout.fetchers import HttpFetcher, RateLimiter


def run(coro):
    return asyncio.run(coro)


def make_transport(responses: dict[str, list]):
    """responses: path -> list of (status, body, headers) returned in order."""
    calls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls[path] = calls.get(path, 0) + 1
        queue = responses.get(path, [(404, "", {})])
        status, body, headers = queue[min(calls[path], len(queue)) - 1]
        return httpx.Response(status, content=body.encode("utf-8"), headers={"content-type": "text/html", **headers})

    return httpx.MockTransport(handler), calls


def test_retries_on_503_then_succeeds():
    transport, calls = make_transport({"/p": [(503, "busy", {}), (503, "busy", {}), (200, "<p>£10</p>", {})]})

    async def go():
        async with HttpFetcher(transport=transport, delay=0, backoff=0.01, respect_robots=False) as f:
            return await f.fetch("https://x.test/p"), f.stats

    result, stats = run(go())
    assert result.ok and result.text == "<p>£10</p>"
    assert result.attempts == 3 and calls["/p"] == 3
    assert stats.retries == 2 and stats.failures == 0


def test_does_not_retry_404():
    transport, calls = make_transport({"/missing": [(404, "nope", {})]})

    async def go():
        async with HttpFetcher(transport=transport, delay=0, backoff=0.01, respect_robots=False) as f:
            return await f.fetch("https://x.test/missing")

    result = run(go())
    assert not result.ok and result.error == "HTTP 404" and calls["/missing"] == 1


def test_gives_up_after_max_retries():
    transport, calls = make_transport({"/down": [(500, "err", {})]})

    async def go():
        async with HttpFetcher(transport=transport, delay=0, backoff=0.01, max_retries=2, respect_robots=False) as f:
            return await f.fetch("https://x.test/down"), f.stats

    result, stats = run(go())
    assert not result.ok and calls["/down"] == 3 and stats.failures == 1


def test_honours_retry_after_header():
    transport, _ = make_transport({"/slow": [(429, "", {"Retry-After": "0"}), (200, "ok", {})]})

    async def go():
        async with HttpFetcher(transport=transport, delay=0, backoff=5, respect_robots=False) as f:
            started = time.monotonic()
            res = await f.fetch("https://x.test/slow")
            return res, time.monotonic() - started

    result, elapsed = run(go())
    assert result.ok
    assert elapsed < 4  # used Retry-After: 0 (+ small jitter) instead of the 5s backoff


def test_respects_robots_txt():
    robots = "User-agent: *\nDisallow: /private/\n"
    transport, calls = make_transport({"/robots.txt": [(200, robots, {})], "/private/x": [(200, "secret", {})],
                                       "/public": [(200, "hello", {})]})

    async def go():
        async with HttpFetcher(transport=transport, delay=0) as f:
            return await f.fetch("https://x.test/private/x"), await f.fetch("https://x.test/public"), f.stats

    blocked, allowed, stats = run(go())
    assert blocked.error == "blocked by robots.txt" and "/private/x" not in calls
    assert allowed.ok and stats.blocked_by_robots == 1
    assert calls["/robots.txt"] == 1  # cached per origin


def test_missing_robots_txt_means_allowed():
    transport, _ = make_transport({"/page": [(200, "fine", {})]})  # /robots.txt -> 404

    async def go():
        async with HttpFetcher(transport=transport, delay=0) as f:
            return await f.fetch("https://x.test/page")

    assert run(go()).ok


def test_rate_limiter_spaces_requests():
    async def go():
        limiter = RateLimiter(0.05)
        started = time.monotonic()
        for _ in range(5):
            await limiter.wait()
        return time.monotonic() - started

    assert run(go()) >= 0.19


def test_fetch_requires_context_manager():
    with pytest.raises(AssertionError):
        run(HttpFetcher().fetch("https://x.test/"))
