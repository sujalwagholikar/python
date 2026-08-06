"""
web_search.py
=============
Real-time internet search for JARVIS, using only the Python standard
library for HTTP/XML (no extra pip installs required for the free
sources) plus an optional Tavily API call if a key is configured.

Sources (all attempted, merged, deduplicated by URL):
  1. Tavily API      — used first if TAVILY_API_KEY is set in .env;
                        purpose-built for LLM-facing search, gives the
                        cleanest short answers.
  2. DuckDuckGo HTML  — scrapes duckduckgo.com/html (their JS-free
                        endpoint, no API key needed).
  3. Google News RSS  — news.google.com/rss/search, good for current
                        events / breaking news phrasing ("latest",
                        "today", "news").

Every source degrades independently: if one fails (network error, no
key, layout change) the others still return results. Only if every
source fails does `web_search()` raise WebSearchError, which the
caller should catch and speak a graceful fallback line.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional

try:
    from env_loader import load_env
    load_env()
except ImportError:
    pass

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_TIMEOUT = 8


class WebSearchError(Exception):
    pass


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    source: str   # "tavily" | "duckduckgo" | "google_news"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def web_search(query: str, max_results: int = 6) -> List[SearchResult]:
    """
    Run the query against every available source and return a merged,
    deduplicated list. Raises WebSearchError only if ALL sources fail
    or return nothing.
    """
    query = query.strip()
    if not query:
        raise WebSearchError("Empty search query.")

    results: List[SearchResult] = []
    errors: List[str] = []

    for fn, name in (
        (_search_tavily, "tavily"),
        (_search_duckduckgo, "duckduckgo"),
        (_search_google_news, "google_news"),
    ):
        try:
            found = fn(query, max_results)
            results.extend(found)
        except Exception as e:  # noqa: BLE001 - genuinely want to swallow & continue
            errors.append(f"{name}: {e}")

    deduped = _dedupe(results)

    if not deduped:
        raise WebSearchError(
            "All search sources failed or returned nothing. " + "; ".join(errors)
        )

    return deduped[:max_results]


def summarize_for_speech(query: str, results: List[SearchResult], max_items: int = 3) -> str:
    """
    Build a short, speakable plain-text summary of search results for
    JARVIS to read aloud / hand to the LLM brain to paraphrase further.
    """
    if not results:
        return f"I couldn't find anything on '{query}', sir."
    lines = [f"Top results for '{query}':"]
    for r in results[:max_items]:
        snippet = r.snippet.strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        lines.append(f"- {r.title}: {snippet} ({r.url})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Source: Tavily API (optional, needs TAVILY_API_KEY)
# --------------------------------------------------------------------------- #

def _search_tavily(query: str, max_results: int) -> List[SearchResult]:
    if not TAVILY_API_KEY or TAVILY_API_KEY == "paste_your_tavily_key_here":
        return []

    payload = json.dumps({
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.URLError as e:
        raise WebSearchError(f"Tavily request failed: {e}") from e

    out = []
    for item in data.get("results", [])[:max_results]:
        out.append(SearchResult(
            title=item.get("title", "").strip() or item.get("url", ""),
            snippet=item.get("content", "").strip(),
            url=item.get("url", "").strip(),
            source="tavily",
        ))
    return out


# --------------------------------------------------------------------------- #
# Source: DuckDuckGo HTML endpoint (no key required)
# --------------------------------------------------------------------------- #

_DDG_RESULT_BLOCK = re.compile(
    r'<a rel="nofollow" class="result__a" href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_TAG_STRIP = re.compile(r"<[^>]+>")


def _clean_html(text: str) -> str:
    text = _TAG_STRIP.sub("", text)
    text = text.replace("&amp;", "&").replace("&#x27;", "'").replace("&quot;", '"')
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    return text.strip()


def _search_duckduckgo(query: str, max_results: int) -> List[SearchResult]:
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise WebSearchError(f"DuckDuckGo request failed: {e}") from e

    out = []
    for m in _DDG_RESULT_BLOCK.finditer(html):
        raw_url = m.group("url")
        # DDG wraps outbound links in a redirect with the real target in
        # the "uddg" query param — unwrap it so we hand back a real URL.
        real_url = raw_url
        if "uddg=" in raw_url:
            parsed = urllib.parse.urlparse(raw_url)
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs:
                real_url = urllib.parse.unquote(qs["uddg"][0])

        out.append(SearchResult(
            title=_clean_html(m.group("title")),
            snippet=_clean_html(m.group("snippet")),
            url=real_url,
            source="duckduckgo",
        ))
        if len(out) >= max_results:
            break
    return out


# --------------------------------------------------------------------------- #
# Source: Google News RSS (no key required)
# --------------------------------------------------------------------------- #

def _search_google_news(query: str, max_results: int) -> List[SearchResult]:
    url = ("https://news.google.com/rss/search?" +
           urllib.parse.urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        raise WebSearchError(f"Google News RSS request failed: {e}") from e

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise WebSearchError(f"Google News RSS parse failed: {e}") from e

    out = []
    for item in root.findall(".//item")[:max_results]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = _clean_html(item.findtext("description") or "")
        source_el = item.find("source")
        pub_source = source_el.text.strip() if source_el is not None and source_el.text else ""
        snippet = f"{desc} ({pub_source})" if pub_source else desc
        out.append(SearchResult(title=title, snippet=snippet, url=link, source="google_news"))
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _dedupe(results: List[SearchResult]) -> List[SearchResult]:
    seen = set()
    out = []
    # Prefer tavily > duckduckgo > google_news when the same URL appears
    # in multiple sources, by processing in that priority order first.
    priority = {"tavily": 0, "duckduckgo": 1, "google_news": 2}
    for r in sorted(results, key=lambda r: priority.get(r.source, 9)):
        key = r.url.strip().rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "latest AI news"
    try:
        res = web_search(q)
        print(summarize_for_speech(q, res))
    except WebSearchError as e:
        print(f"Search failed: {e}")
