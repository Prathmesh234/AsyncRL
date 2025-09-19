import os
import asyncio
from typing import List, Dict, Any

from bs4 import BeautifulSoup


class WebTool:
    """Simple async web search + fetch tool.

    - First tries Brave Search API for top-k links.
    - On error or no links, falls back to Bing (Playwright).
    - Then fetches each page and extracts a short text summary.
    """

    def __init__(self, *, max_chars: int = 20000, request_timeout: float = 8.0, concurrency: int = 5) -> None:
        self.max_chars = max_chars
        self.request_timeout = request_timeout
        self.concurrency = max(1, concurrency)

    # --- Link discovery via Bing (Playwright) ---
    async def _fetch_links_bing(self, query: str, k: int) -> List[str]:
        from playwright.async_api import async_playwright

        links: List[str] = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"])
            try:
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
                    ),
                    locale="en-US",
                )
                page = await context.new_page()
                q = query.replace(" ", "+")
                await page.goto(
                    f"https://www.bing.com/search?q={q}&setlang=en-US",
                    wait_until="domcontentloaded",
                    timeout=20000,
                )
                try:
                    await page.wait_for_selector("li.b_algo h2 a", timeout=5000)
                except Exception:
                    pass
                anchors = await page.query_selector_all("li.b_algo h2 a")
                if not anchors:
                    anchors = await page.query_selector_all("li.b_algo a, ol#b_results a[href]")

                for a in anchors:
                    href = await a.get_attribute("href")
                    if href and href.startswith("http") and href not in links:
                        links.append(href)
                    if len(links) >= max(1, int(k)):
                        break
            finally:
                await browser.close()
        return links

    # --- Link discovery via Brave API (fallback) ---
    async def _fetch_links_brave(self, query: str, k: int) -> List[str]:
        import aiohttp

        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            return []

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
        params = {"q": query}
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, params=params, timeout=self.request_timeout) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
        except Exception:
            return []

        results = data.get("web", {}).get("results", [])
        links: List[str] = []
        for item in results:
            u = item.get("url")
            if u and u.startswith("http") and u not in links:
                links.append(u)
            if len(links) >= max(1, int(k)):
                break
        return links

    # --- Page fetching ---
    async def _fetch_content(self, session: "aiohttp.ClientSession", url: str) -> Dict[str, Any]:
        import aiohttp
        try:
            async with session.get(url, timeout=self.request_timeout) as resp:
                if resp.status != 200:
                    return {"url": url, "error": f"HTTP {resp.status}"}
                html = await resp.text(errors="ignore")
        except Exception as e:
            return {"url": url, "error": str(e)}

        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        paragraphs = " ".join(p.get_text(separator=" ", strip=True) for p in soup.find_all("p"))
        content = " ".join(paragraphs.split())[: self.max_chars]
        return {"url": url, "title": title, "content": content}

    # --- Orchestrator ---
    async def run(self, query: str, k: int) -> List[Dict[str, Any]]:
        import aiohttp

        # 1) Try Brave first
        links: List[str] = []
        try:
            links = await self._fetch_links_brave(query, k)
        except Exception:
            links = []

        # 2) Fallback to Bing if needed
        if not links:
            try:
                links = await self._fetch_links_bing(query, k)
            except Exception:
                links = []
        if not links:
            return [{"error": "No results"}]

        connector = aiohttp.TCPConnector(limit=self.concurrency, ttl_dns_cache=60)
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36"}
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            sem = asyncio.Semaphore(self.concurrency)

            async def _guarded(u: str) -> Dict[str, Any]:
                async with sem:
                    return await self._fetch_content(session, u)

            tasks = [asyncio.create_task(_guarded(u)) for u in links]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        out: List[Dict[str, Any]] = []
        for r in results:
            out.append({"error": str(r)} if isinstance(r, Exception) else r)
        return out
