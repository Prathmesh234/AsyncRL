## Web env 

## how is our web env going to be shaped 

So our web env is going to be relaitvely simple, once we get the query and the k value this is what we are going to do. 
Also this web call to the playwright browser is going to be async for faster implementation 

1) Use a headless browser and search for our query k 
2) In future optimization, we can tell the llm to generate multiple queries for a parallel web search

Implementation

Installs 

uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install playwright requests beautifulsoup4
playwright install


Code implementation 

import asyncio
from playwright.async_api import async_playwright
import aiohttp
from bs4 import BeautifulSoup

# 1. Extract top k links using Playwright headless browser
async def fetch_top_k_links(query, k):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(f"https://www.google.com/search?q={query.replace(' ', '+')}")
        await page.wait_for_selector('a')
        anchors = await page.query_selector_all('a')
        links = []
        for a in anchors:
            href = await a.get_attribute('href')
            if href and href.startswith('http'):
                links.append(href)
            if len(links) >= k:
                break
        await browser.close()
        return links

# 2. Async fetch page content using aiohttp
async def fetch_content(session, url, max_chars=3000):
    try:
        async with session.get(url, timeout=8) as resp:
            if resp.status == 200:
                html = await resp.text()
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.title.string if soup.title else ''
                paragraphs = " ".join(p.get_text() for p in soup.find_all("p"))
                return {"url": url, "title": title, "content": paragraphs[:max_chars]}
            else:
                return {"url": url, "error": f"HTTP {resp.status}"}
    except Exception as e:
        return {"url": url, "error": str(e)}

# 3. Orchestrator: fetch links + fetch content in parallel
async def query_and_get_contents(query, k):
    links = await fetch_top_k_links(query, k)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_content(session, link) for link in links]
        results = await asyncio.gather(*tasks)
        return results

# 4. Example usage
if __name__ == "__main__":
    query = "latest AI research papers"
    k = 5
    results = asyncio.run(query_and_get_contents(query, k))
    for r in results:
        print(f"\nURL: {r.get('url')}\nTitle: {r.get('title')}\nContent: {r.get('content')[:200]}...")
